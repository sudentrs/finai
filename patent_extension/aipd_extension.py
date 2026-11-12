"""Train an abstract-only AIPD surrogate, then score recent first utility publications."""
import argparse
import hashlib
import json
import re
import sqlite3
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve)
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
AIPD = BASE / "ai_model_predictions.csv/ai_model_predictions.csv"
META = BASE / "pg_published_application.tsv/pg_published_application.tsv"
ABSTRACTS = BASE / "publication pack/pg_published_application_abstract.tsv.zip"
FRAME = ROOT / "independent_validation_v2/outputs/patent_only_sampling_frame.sqlite"
REVIEW = ROOT / "independent_validation_v2/outputs_v2_1/CURATOR_sample_manifest.csv"
COMPONENTS = "ml evo nlp speech vision planning kr hardware".split()
TARGET = "predict93_any_ai"
SEED = 20261109
CHUNK = 100_000


def fingerprint(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def normalized_hash(text):
    normalized = re.sub(r"\W+", " ", text.casefold()).strip()
    return hashlib.sha256(normalized.encode()).hexdigest() if normalized else ""


def select_frame():
    excluded = set(pd.read_csv(REVIEW, usecols=["application_id"], dtype=str).application_id)
    rng = np.random.default_rng(SEED)
    historical, recent = [], []
    with sqlite3.connect(FRAME.as_uri() + "?mode=ro", uri=True) as db:
        frame_hash, = db.execute("SELECT value FROM provenance WHERE key='metadata_sha256'").fetchone()
        if frame_hash != fingerprint(META):
            raise ValueError("Cached first-publication frame does not match metadata")
        sql = """SELECT application_id, pgpub_id, published_date, filing_date FROM applications
                 WHERE published_date BETWEEN '2015-01-01' AND '2025-12-31'
                 ORDER BY application_id"""
        for chunk in pd.read_sql_query(sql, db, chunksize=CHUNK):
            year = chunk.published_date.str[:4].astype(int)
            p = np.select([year.le(2021), year.eq(2022), year.eq(2023)], [.03, .06, .08])
            keep = (rng.random(len(chunk)) < p) & ~chunk.application_id.isin(excluded)
            historical.append(chunk.loc[keep].assign(year=year[keep]))
            recent.append(chunk.loc[year.ge(2024)].copy())
    hist, new = map(lambda chunks: pd.concat(chunks, ignore_index=True), (historical, recent))
    if pd.concat([hist, new]).application_id.duplicated().any():
        raise ValueError("Repeated application in frame")
    return hist, new, frame_hash


def attach_aipd(hist):
    flags = [f"predict93_{c}" for c in COMPONENTS]
    scores = [f"ai_score_{c}" for c in COMPONENTS]
    cols = ["doc_id", "flag_patent", "pub_dt", "appl_id", TARGET] + flags + scores
    pieces, audit = [], dict(pregrant_rows_checked=0, any_vs_component_or_mismatches=0,
                             component_vs_093_mismatches_outside_tolerance=0,
                             component_vs_08993_mismatches_outside_tolerance=0,
                             invalid_label_or_score_rows=0)
    ids = set(hist.pgpub_id)
    for chunk in pd.read_csv(AIPD, usecols=cols, dtype=str, chunksize=CHUNK):
        chunk = chunk[chunk.flag_patent.eq("0")]
        valid_flags = chunk[[TARGET] + flags].isin(["0", "1"]).all(axis=1)
        s = chunk[scores].apply(pd.to_numeric, errors="coerce").to_numpy()
        valid_scores = ((s >= 0) & (s <= 1)).all(axis=1)
        f = chunk[flags].eq("1").to_numpy()
        audit["pregrant_rows_checked"] += len(chunk)
        audit["invalid_label_or_score_rows"] += int((~(valid_flags & valid_scores)).sum())
        audit["any_vs_component_or_mismatches"] += int((f.any(axis=1) != chunk[TARGET].eq("1")).sum())
        for cutoff, key in [(0.93, "093"), (0.8993, "08993")]:
            bad = (f != (s >= cutoff)) & (np.abs(s - cutoff) > 1e-6)
            audit[f"component_vs_{key}_mismatches_outside_tolerance"] += int(bad.sum())
        pieces.append(chunk.loc[chunk.doc_id.isin(ids), ["doc_id", "appl_id", "pub_dt", TARGET]])
    labels = pd.concat(pieces, ignore_index=True)
    joined = hist.merge(labels, left_on="pgpub_id", right_on="doc_id", how="left", validate="one_to_one")
    matched = joined.doc_id.notna()
    audit["sampled_historical"] = len(joined)
    audit["matched_by_publication_id"] = int(matched.sum())
    audit["application_id_disagreements"] = int((matched & joined.appl_id.ne(joined.application_id)).sum())
    audit["publication_date_disagreements"] = int((matched & joined.pub_dt.ne(joined.published_date)).sum())
    audit["sample_join_by_year"] = joined.assign(
        matched=matched.astype(int),
        usable=(matched & joined.appl_id.eq(joined.application_id) &
                joined.pub_dt.eq(joined.published_date) & joined[TARGET].isin(["0", "1"])).astype(int)
    ).groupby("year").agg(sampled=("pgpub_id", "size"), matched=("matched", "sum"),
                          usable=("usable", "sum")).reset_index().to_dict("records")
    valid = matched & joined.appl_id.eq(joined.application_id) & joined.pub_dt.eq(joined.published_date)
    valid &= joined[TARGET].isin(["0", "1"])
    return joined.loc[valid, list(hist.columns) + [TARGET]].copy(), audit


def extract_abstracts(hist, recent, stage):
    train_ids, recent_ids = set(hist.pgpub_id), set(recent.pgpub_id)
    parts, first = [], True
    seen = set()
    with zipfile.ZipFile(ABSTRACTS) as archive:
        members = [name for name in archive.namelist() if name.endswith(".tsv")]
        if len(members) != 1:
            raise ValueError("Unexpected abstract archive contents")
        with archive.open(members[0]) as stream:
            for i, chunk in enumerate(pd.read_csv(stream, sep="\t", dtype=str,
                                                  keep_default_na=False, chunksize=CHUNK), 1):
                h = chunk[chunk.pgpub_id.isin(train_ids)]
                r = chunk[chunk.pgpub_id.isin(recent_ids)]
                selected = pd.concat([h.pgpub_id, r.pgpub_id])
                if selected.duplicated().any() or not seen.isdisjoint(selected):
                    raise ValueError("Repeated publication key in abstract source")
                seen.update(selected)
                parts.append(h)
                if len(r):
                    r.to_csv(stage, index=False, mode="w" if first else "a", header=first)
                    first = False
                if i % 20 == 0:
                    print(f"Abstract source rows scanned: {i * CHUNK:,}", flush=True)
    return hist.merge(pd.concat(parts, ignore_index=True), on="pgpub_id",
                      how="left", validate="one_to_one")


def separate_historical(hist):
    hist = hist.copy()
    hist["hash"] = hist.application_abstract.fillna("").map(normalized_hash)
    hist["split"] = np.select([hist.year.le(2021), hist.year.eq(2022)], ["train", "development"],
                              default="test")
    hist["text_available"] = hist.hash.ne("")
    test = hist.loc[hist.split.eq("test") & hist.text_available].copy()
    dev = hist.loc[hist.split.eq("development") & hist.text_available & ~hist.hash.isin(test.hash)].copy()
    train = hist.loc[hist.split.eq("train") & hist.text_available & ~hist.hash.isin(test.hash)
                     & ~hist.hash.isin(dev.hash)].copy()
    kept = set(pd.concat([train, dev, test]).pgpub_id)
    hist["included"] = hist.pgpub_id.isin(kept)
    return train, dev, test, hist


def operating_threshold(y, probability):
    precision, recall, thresholds = precision_recall_curve(y, probability)
    total = precision[:-1] + recall[:-1]
    f1 = np.divide(2 * precision[:-1] * recall[:-1], total,
                   out=np.zeros_like(total), where=total > 0)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def metrics(y, probability, threshold):
    pred = probability >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(n=len(y), positives=tp+fn, predicted_positive=tp+fp, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None, recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                average_precision=float(average_precision_score(y, probability)),
                brier=float(brier_score_loss(y, probability)))


def score_recent(model, threshold, stage, recent, out):
    path = out / "publications_2024_2025.csv"
    universe = recent.set_index("pgpub_id", verify_integrity=True)
    seen, first, counts = set(), True, []
    for chunk in pd.read_csv(stage, dtype=str, keep_default_na=False, chunksize=10_000):
        chunk = chunk.reset_index(drop=True)
        if chunk.pgpub_id.duplicated().any() or not seen.isdisjoint(chunk.pgpub_id):
            raise ValueError("Repeated key in staged recent abstracts")
        scored = universe.loc[chunk.pgpub_id].reset_index()
        seen.update(chunk.pgpub_id)
        usable = chunk.application_abstract.str.contains(r"\w", regex=True)
        probability = model.predict_proba(chunk.loc[usable, "application_abstract"])[:, 1] if usable.any() else []
        scored["aipd_surrogate_score"] = np.nan
        scored["aipd_surrogate_label"] = pd.Series(pd.NA, index=scored.index, dtype="Int64")
        scored["score_status"] = "empty_abstract"
        scored.loc[usable, "aipd_surrogate_score"] = probability
        scored.loc[usable, "aipd_surrogate_label"] = (np.asarray(probability) >= threshold).astype(int)
        scored.loc[usable, "score_status"] = "scored"
        counts.append(scored.assign(year=scored.published_date.str[:4],
                                    positive=scored.aipd_surrogate_label.eq(1).fillna(False).astype(int))
                      .groupby(["year", "score_status"]).positive.agg(["size", "sum"]).reset_index())
        scored.to_csv(path, index=False, mode="w" if first else "a", header=first)
        first = False
    missing = universe.loc[~universe.index.isin(seen)].reset_index()
    if len(missing):
        missing["aipd_surrogate_score"] = np.nan
        missing["aipd_surrogate_label"] = pd.NA
        missing["score_status"] = "abstract_not_joined"
        missing.to_csv(path, index=False, mode="a", header=False)
        counts.append(missing.assign(year=missing.published_date.str[:4], positive=0)
                      .groupby(["year", "score_status"]).positive.agg(["size", "sum"]).reset_index())
    report = pd.concat(counts).groupby(["year", "score_status"])[["size", "sum"]].sum().reset_index()
    report = report.rename(columns={"size": "publications", "sum": "predicted_ai"})
    if report.publications.sum() != len(recent):
        raise ValueError("Recent coverage reconciliation failed")
    report.to_csv(out / "recent_coverage.csv", index=False)
    return report.to_dict("records")


def resume_scoring(out):
    model_path = out / "frozen_surrogate.joblib"
    freeze = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    if fingerprint(model_path) != freeze["model_sha256"] or not (out / "benchmark.json").is_file():
        raise ValueError("Frozen model or completed benchmark could not be verified")
    bundle = joblib.load(model_path)
    audit = json.loads((out / "source_audit.json").read_text(encoding="utf-8"))
    if fingerprint(FRAME) != audit["sources"][str(FRAME)]:
        raise ValueError("Cached publication frame changed since training")
    with sqlite3.connect(FRAME.as_uri() + "?mode=ro", uri=True) as db:
        recent = pd.read_sql_query(
            """SELECT application_id, pgpub_id, published_date, filing_date FROM applications
               WHERE published_date BETWEEN '2024-01-01' AND '2025-12-31'""", db
        )
    return score_recent(bundle["model"], bundle["threshold"], out / "staged_recent_abstracts.csv", recent, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "aipd_surrogate_v1")
    parser.add_argument("--resume-score", action="store_true")
    args = parser.parse_args()
    out = args.out
    if args.resume_score:
        print(json.dumps(resume_scoring(out), indent=2), flush=True)
        return
    out.mkdir(parents=True, exist_ok=False)
    print("Selecting patent-only historical and recent publication frames", flush=True)
    hist, recent, meta_hash = select_frame()
    print(f"Historical draw: {len(hist):,}; recent universe: {len(recent):,}", flush=True)
    hist, audit = attach_aipd(hist)
    audit["sources"] = {str(path): fingerprint(path) for path in (AIPD, ABSTRACTS, REVIEW)}
    audit["sources"][str(META)] = meta_hash
    audit["sources"][str(FRAME)] = fingerprint(FRAME)
    write_json(out / "source_audit.json", audit)
    if (audit["pregrant_rows_checked"] == 0 or audit["invalid_label_or_score_rows"]
            or audit["any_vs_component_or_mismatches"]
            or audit["component_vs_093_mismatches_outside_tolerance"]):
        raise ValueError("AIPD source failed the corrected-threshold checks; inspect source_audit.json")
    print(f"AIPD joins passing ID and date checks: {len(hist):,}; extracting abstracts", flush=True)
    stage = out / "staged_recent_abstracts.csv"
    hist = extract_abstracts(hist, recent, stage)
    train, dev, test, split_audit = separate_historical(hist)
    split_audit.drop(columns="application_abstract").to_csv(out / "historical_split_manifest.csv", index=False)
    print(f"Train / development / test: {len(train):,} / {len(dev):,} / {len(test):,}", flush=True)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=180_000,
                        sublinear_tf=True, dtype=np.float32),
        LogisticRegression(solver="liblinear", C=2, max_iter=300, random_state=SEED),
    )
    model.fit(train.application_abstract, train[TARGET].astype(int))
    dev_prob = model.predict_proba(dev.application_abstract)[:, 1]
    threshold = operating_threshold(dev[TARGET].astype(int), dev_prob)
    model_path = out / "frozen_surrogate.joblib"
    joblib.dump({"model": model, "threshold": threshold, "target": TARGET}, model_path, compress=3)
    freeze = dict(model_sha256=fingerprint(model_path), threshold=threshold, training="2015-2021",
                  development="2022", test="2023", sklearn_version=sklearn.__version__,
                  seed=SEED, threshold_rule="maximum development F1; highest threshold breaks ties",
                  family_overlap="not cleared; application IDs and normalized exact text disjoint",
                  validation_target="AIPD teacher-label agreement, not human accuracy")
    write_json(out / "freeze.json", freeze)
    test_prob = model.predict_proba(test.application_abstract)[:, 1]
    test.drop(columns="application_abstract").assign(
        surrogate_score=test_prob, surrogate_label=(test_prob >= threshold).astype(int)
    ).to_csv(out / "heldout_2023_predictions.csv", index=False)
    report = {"development": metrics(dev[TARGET].astype(int), dev_prob, threshold),
              "heldout_2023": metrics(test[TARGET].astype(int), test_prob, threshold)}
    write_json(out / "benchmark.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("Scoring 2024-2025 first utility publications", flush=True)
    print(json.dumps(score_recent(model, threshold, stage, recent, out), indent=2), flush=True)


if __name__ == "__main__":
    main()
