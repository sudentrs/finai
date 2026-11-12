"""Compare abstract-only and abstract-plus-pre-grant-claims on identical patents."""
import argparse
import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from aipd_extension import ROOT, TARGET, fingerprint, metrics, normalized_hash, operating_threshold, write_json
from aipd_v2_prepare import V2, SEED

SOURCE = ROOT / "claims_source"
OUT = ROOT / "aipd_claims_v1"
YEARS = (2021, 2022, 2023)


def claims_for(path, wanted):
    columns = ["pgpub_id", "claim_sequence", "claim_text"]
    rows = []
    with zipfile.ZipFile(path) as archive:
        member, = [name for name in archive.namelist() if name.endswith(".tsv")]
        with archive.open(member) as stream:
            for chunk in pd.read_csv(stream, sep="\t", usecols=columns, dtype=str,
                                     chunksize=100_000, keep_default_na=False):
                rows.append(chunk.loc[chunk.pgpub_id.isin(wanted)])
    data = pd.concat(rows, ignore_index=True)
    data = data[data.claim_text.str.contains(r"\w", regex=True)].copy()
    data["claim_sequence"] = pd.to_numeric(data.claim_sequence, errors="raise")
    data = data.drop_duplicates()
    if data.duplicated(["pgpub_id", "claim_sequence"]).any():
        raise ValueError(f"Conflicting text for the same publication and claim sequence in {path}")
    data["claim_text"] = data.claim_text.str.replace(r"\s+", " ", regex=True).str.strip()
    agg = data.sort_values(["pgpub_id", "claim_sequence"]).groupby("pgpub_id").claim_text.agg(
        claim_text=lambda texts: " ".join(texts)[:12000], claim_count="size")
    return agg.reset_index()


def matched_data(source, out):
    frame = pd.read_csv(V2 / "historical_abstracts_labels.csv", dtype={"pgpub_id": str, "application_id": str},
                        keep_default_na=False)
    frame = frame[(frame.year.isin(YEARS) & frame.hash.ne("")) &
                  (~frame.year.eq(2023) | frame.fresh_benchmark)].copy()
    blocks, metadata = [], []
    for year in YEARS:
        archive = source / f"pg_claims_{year}.tsv.zip"
        part = frame[frame.year.eq(year)]
        claims = claims_for(archive, set(part.pgpub_id))
        merged = part.merge(claims, on="pgpub_id", how="left", validate="one_to_one")
        metadata.append(dict(year=year, eligible=len(part), matched=len(claims),
                             source=str(archive), sha256=fingerprint(archive)))
        print(f"{year}: {len(claims):,}/{len(part):,} with usable claims", flush=True)
        blocks.append(merged)
    combined = pd.concat(blocks, ignore_index=True)
    combined["claims_hash"] = combined.claim_text.fillna("").map(normalized_hash)
    combined = combined[combined.claims_hash.ne("")].copy()
    test = combined[combined.year.eq(2023)].copy()
    cal = combined[combined.year.eq(2022) & ~combined.hash.isin(test.hash)
                   & ~combined.claims_hash.isin(test.claims_hash)].copy()
    fit = combined[combined.year.eq(2021) & ~combined.hash.isin(test.hash)
                   & ~combined.claims_hash.isin(test.claims_hash) & ~combined.hash.isin(cal.hash)
                   & ~combined.claims_hash.isin(cal.claims_hash)].copy()
    write_json(out / "coverage.json", {"claims": metadata,
               "split_counts_after_exact_text_exclusions": {"fit": len(fit), "calibration": len(cal), "test": len(test)},
               "family_overlap_cleared": False, "test_reused_from_v2": True, "claim_text_cap_chars": 12000})
    return fit, cal, test


def vectorizer():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=160000,
                           sublinear_tf=True, dtype=np.float32)


def run(source, out):
    missing = [year for year in YEARS if not (source / f"pg_claims_{year}.tsv.zip").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing verified pre-grant archives for {missing} in {source}")
    out.mkdir(parents=True, exist_ok=False)
    fit, cal, test = matched_data(source, out)
    if any(part.application_id.duplicated().any() or part[TARGET].nunique() != 2 for part in (fit, cal, test)):
        raise ValueError("Invalid split: duplicate applications or missing target class")
    if len(set(pd.concat([fit, cal, test]).application_id)) != len(fit) + len(cal) + len(test):
        raise ValueError("Application overlap across splits")
    abstracts = vectorizer()
    a_train = abstracts.fit_transform(fit.application_abstract)
    a_cal, a_test = (abstracts.transform(data.application_abstract) for data in (cal, test))
    claims = vectorizer()
    c_train = claims.fit_transform(fit.claim_text)
    c_cal, c_test = (claims.transform(data.claim_text) for data in (cal, test))
    report, preds = {}, test[["pgpub_id", "application_id", "published_date", TARGET, "claim_count"]].copy()
    for name, xf, xc, xt in (
        ("abstract_only", a_train, a_cal, a_test),
        ("abstract_plus_claims", hstack([a_train, c_train], format="csr"),
         hstack([a_cal, c_cal], format="csr"), hstack([a_test, c_test], format="csr")),
    ):
        model = LogisticRegression(solver="liblinear", C=2, max_iter=300, random_state=SEED)
        model.fit(xf, fit[TARGET])
        threshold = operating_threshold(cal[TARGET], model.predict_proba(xc)[:, 1])
        artifact = out / f"{name}.joblib"
        joblib.dump({"abstracts": abstracts, "claims": claims if name.endswith("claims") else None,
                     "model": model, "threshold": threshold}, artifact, compress=3)
        probabilities = model.predict_proba(xt)[:, 1]
        report[name] = dict(threshold=threshold, model_sha256=fingerprint(artifact),
                            **metrics(test[TARGET], probabilities, threshold))
        preds[f"{name}_score"] = probabilities
        preds[f"{name}_label"] = (probabilities >= threshold).astype(int)
        print(name, report[name], flush=True)
    preds.to_csv(out / "paired_2023_claims_predictions.csv", index=False)
    write_json(out / "paired_2023_claims_benchmark.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.source, args.out)
