"""Verify frozen AIPD surrogate outputs and produce the filing/publication cross-tab."""
import argparse
import json
from pathlib import Path

import pandas as pd

from aipd_extension import ROOT, TARGET, fingerprint, metrics, write_json


def verify(out):
    freeze = json.loads((out / "freeze.json").read_text(encoding="utf-8"))
    if fingerprint(out / "frozen_surrogate.joblib") != freeze["model_sha256"]:
        raise ValueError("Model fingerprint differs from the freeze record")
    threshold = freeze["threshold"]
    historical = pd.read_csv(out / "historical_split_manifest.csv", dtype={"pgpub_id": str, "application_id": str})
    heldout = pd.read_csv(out / "heldout_2023_predictions.csv", dtype={"pgpub_id": str},
                          float_precision="round_trip")
    test = historical.loc[historical.included & historical.split.eq("test")]
    if (historical.application_id.duplicated().any() or heldout.pgpub_id.duplicated().any()
            or set(heldout.pgpub_id) != set(test.pgpub_id)
            or historical.loc[historical.included].groupby("hash").split.nunique().gt(1).any()):
        raise ValueError("Historical split reconciliation failed")
    observed = metrics(heldout[TARGET], heldout.surrogate_score, threshold)
    benchmark = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))["heldout_2023"]
    if any(observed[k] != benchmark[k] for k in ("n", "tp", "tn", "fp", "fn")):
        raise ValueError("Held-out confusion matrix differs from report")

    recent_path = out / "publications_2024_2025.csv"
    recent = pd.read_csv(recent_path, dtype={"pgpub_id": str, "application_id": str}, float_precision="round_trip")
    scored = recent.score_status.eq("scored")
    if (recent.pgpub_id.duplicated().any() or recent.application_id.duplicated().any()
            or not set(recent.application_id).isdisjoint(historical.application_id)
            or not recent.published_date.str[:4].isin(["2024", "2025"]).all()
            or not recent.loc[scored, "aipd_surrogate_score"].between(0, 1).all()
            or not recent.loc[scored, "aipd_surrogate_label"].eq(
                recent.loc[scored, "aipd_surrogate_score"].ge(threshold).astype(int)).all()
            or recent.loc[~scored, ["aipd_surrogate_score", "aipd_surrogate_label"]].notna().any().any()):
        raise ValueError("Recent prediction integrity check failed")
    coverage = pd.read_csv(out / "recent_coverage.csv", dtype={"year": str}).set_index(["year", "score_status"])
    recent["publication_year"] = recent.published_date.str[:4]
    recent["filing_year"] = pd.to_datetime(recent.filing_date, errors="coerce").dt.year.astype("Int64")
    recent["scored"] = scored.astype(int)
    recent["predicted_ai"] = recent.aipd_surrogate_label.eq(1).astype(int)
    counts = recent.groupby(["publication_year", "score_status"]).agg(
        publications=("pgpub_id", "size"), predicted_ai=("predicted_ai", "sum"))
    counts.index.names = coverage.index.names
    pd.testing.assert_frame_equal(counts.sort_index(), coverage.sort_index(), check_dtype=False)
    recent.groupby(["publication_year", "filing_year"], dropna=False).agg(
        publications=("pgpub_id", "size"), scored=("scored", "sum"), predicted_ai=("predicted_ai", "sum")
    ).reset_index().to_csv(out / "filing_publication_year_coverage.csv", index=False)
    verification = dict(rows=len(recent), scored=int(scored.sum()),
                        predicted_ai=int(recent.predicted_ai.sum()), prediction_sha256=fingerprint(recent_path),
                        historical_scored_by_split=historical.groupby("split").included.sum().astype(int).to_dict(),
                        historical_missing_abstract=int((~historical.text_available).sum()),
                        historical_cross_split_exact_text_exclusions=int((historical.text_available & ~historical.included).sum()),
                        status="output integrity verified; no human or family validation implied")
    write_json(out / "output_verification.json", verification)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "aipd_surrogate_v1")
    verify(parser.parse_args().out)
