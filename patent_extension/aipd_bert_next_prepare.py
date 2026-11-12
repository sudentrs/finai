"""Build a GPU benchmark after separating known U.S. continuity groups."""
import zipfile
from pathlib import Path

import joblib
import pandas as pd

from aipd_claims_benchmark import SOURCE, claims_for
from aipd_claims_expanded import OUT as EXPANDED, TRAIN_YEARS, WEIGHTS, blend, predict, train_text
from aipd_continuity_audit import PATH as RELATIONSHIPS, TYPES, components
from aipd_extension import ROOT, TARGET, fingerprint, metrics, normalized_hash, operating_threshold, write_json
from aipd_v2_prepare import V2

OUT = ROOT / "aipd_bert_next"
SEED = 20261113
IDS = {"pgpub_id": str, "application_id": str}
TEXT = ["pgpub_id", "published_date", "application_abstract", "claim_text"]


def add_groups(frame):
    pub_to_app = frame.set_index("pgpub_id", verify_integrity=True).application_id
    edges = set()
    columns = ["pgpub_id", "related_doc_number", "published_country", "related_doc_type"]
    with zipfile.ZipFile(RELATIONSHIPS) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".tsv")]
        with archive.open(member) as stream:
            for chunk in pd.read_csv(stream, sep="\t", dtype=str, chunksize=150_000,
                                     keep_default_na=False, usecols=columns):
                hit = chunk.loc[chunk.pgpub_id.isin(pub_to_app.index) & chunk.published_country.eq("US")
                                & chunk.related_doc_type.isin(TYPES)
                                & chunk.related_doc_number.str.fullmatch(r"\d{8}")]
                edges.update((f"a:{a}", f"a:{b}") for a, b in zip(
                    hit.pgpub_id.map(pub_to_app), hit.related_doc_number) if a != b)
    for column in ("hash", "claims_hash"):
        valid = frame[column].ne("")
        edges.update((f"a:{app}", f"{column}:{digest}") for app, digest in zip(
            frame.loc[valid, "application_id"], frame.loc[valid, column]))
    groups = components(sorted(edges))
    frame["known_group"] = frame.application_id.map(lambda app: groups.get(f"a:{app}", f"a:{app}"))
    return frame


def select_roles(frame):
    test = frame.loc[frame.new_test & frame.claims_hash.ne("") & frame.hash.ne("")].copy()
    available = frame.loc[~frame.new_test & frame.year.le(2022) & frame.hash.ne("")
                          & ~frame.known_group.isin(test.known_group)].copy()
    cal_pool = available.loc[available.year.eq(2022) & available.claims_hash.ne("")]
    cal = cal_pool.sample(n=2500, random_state=SEED)
    fit = available.loc[~available.known_group.isin(cal.known_group)].copy()
    if (not set(test.known_group).isdisjoint(cal.known_group)
            or not set(fit.known_group).isdisjoint(test.known_group)
            or not set(fit.known_group).isdisjoint(cal.known_group)):
        raise ValueError("Known relationship or exact text crosses roles")
    neural = fit.loc[fit.claims_hash.ne("") & fit.year.isin(TRAIN_YEARS)].sample(n=10000, random_state=SEED)
    return fit, neural, cal, test


def prepare():
    hist = pd.read_csv(V2 / "historical_abstracts_labels.csv", dtype=IDS, keep_default_na=False)
    prior = pd.read_csv(EXPANDED / "new_2023_frame.csv", dtype=IDS, keep_default_na=False)
    test = pd.read_csv(OUT / "new_2023_frame.csv", dtype=IDS, keep_default_na=False)
    previous = pd.concat([hist, prior], ignore_index=True)
    test = test.loc[test.hash.ne("") & ~test.hash.isin(previous.hash)].copy()
    if not set(test.application_id).isdisjoint(previous.application_id):
        raise ValueError("Previously used application in new test")
    hist_claims = [pd.read_csv(EXPANDED / f"claims_{year}_joined.csv", dtype=IDS, keep_default_na=False)
                   for year in (*TRAIN_YEARS, 2023)]
    archive = SOURCE / "pg_claims_2023.tsv.zip"
    new_claims = claims_for(archive, set(test.pgpub_id))
    new_claims["claims_hash"] = new_claims.claim_text.map(normalized_hash)
    claims = pd.concat([*hist_claims, new_claims], ignore_index=True)
    prior_claims = set(pd.concat(hist_claims).claims_hash) - {""}
    test = test.loc[~test.pgpub_id.isin(new_claims.loc[new_claims.claims_hash.isin(prior_claims), "pgpub_id"])]
    frame = pd.concat([previous.assign(new_test=False), test.assign(new_test=True)], ignore_index=True)
    frame = frame.merge(claims, on="pgpub_id", how="left", validate="one_to_one").fillna(
        {"claims_hash": "", "claim_text": ""})
    fit, neural, cal, test = select_roles(add_groups(frame))
    full_claims = fit.loc[fit.claims_hash.ne("") & fit.year.isin(TRAIN_YEARS)]
    bundle = OUT / "gpu_bundle"
    bundle.mkdir(exist_ok=False)
    for name, part in (("fit", neural), ("calibration", cal), ("test", test)):
        cols = TEXT if name == "test" else TEXT + [TARGET]
        part[cols].to_csv(bundle / f"{name}.csv.gz", index=False, compression="gzip")
    test[["pgpub_id", TARGET]].to_csv(OUT / "CURATOR_test_labels.csv", index=False)
    pd.concat([part[["pgpub_id", "application_id", "known_group"]].assign(role=role)
               for part, role in ((fit, "fit"), (cal, "calibration"), (test, "test"))]).to_csv(
                   OUT / "split_manifest.csv", index=False)
    report = {"full_abstract_fit": len(fit), "full_claims_fit": len(full_claims),
              "gpu_fit": len(neural), "calibration": len(cal), "test": len(test),
              "known_group_cross_split": 0, "complete_patent_family_clearance": False,
              "source_sha256": {p.name: fingerprint(p) for p in (RELATIONSHIPS, archive)},
              "bundle_sha256": {p.name: fingerprint(p) for p in bundle.glob("*.csv.gz")},
              "test_labels_in_bundle": False}
    write_json(OUT / "split_audit.json", report)
    print(report, flush=True)

    av, ah, ap = train_text(fit, cal, "application_abstract")
    cv, ch, cp = train_text(full_claims, cal, "claim_text")
    options = []
    for w in WEIGHTS:
        p = blend(ap, cp, w)
        t = operating_threshold(cal[TARGET], p)
        options.append((metrics(cal[TARGET], p, t)["f1"], -w, t))
    _, neg_weight, threshold = max(options)
    weight = -neg_weight
    path = OUT / "frozen_sparse_reference.joblib"
    joblib.dump(dict(abstract_vectorizer=av, abstract_head=ah, claims_vectorizer=cv, claims_head=ch,
                     weight=weight, threshold=threshold), path, compress=3)
    write_json(OUT / "sparse_reference_freeze.json", {"sha256": fingerprint(path), "threshold": threshold,
                                                     "claims_weight": weight})
    scores = blend(predict(av, ah, test, "application_abstract"), predict(cv, ch, test, "claim_text"), weight)
    test[["pgpub_id"]].assign(sparse_score=scores).to_csv(OUT / "sparse_test_scores.csv", index=False)
    print("Frozen sparse reference scored the sealed test; no test metrics opened.", flush=True)


if __name__ == "__main__":
    prepare()
