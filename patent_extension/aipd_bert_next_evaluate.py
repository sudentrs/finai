"""One-time evaluator: open test labels only after both models are frozen."""
import argparse
import json
from pathlib import Path

import pandas as pd

from aipd_claims_expanded import paired_week_bootstrap
from aipd_extension import ROOT, TARGET, fingerprint, metrics, write_json

DEFAULT = ROOT / "aipd_bert_next"


def evaluate(base, neural):
    destination = base / "paired_benchmark.json"
    if destination.exists():
        raise FileExistsError("Benchmark already evaluated; preserve the original result")
    split = json.loads((base / "split_audit.json").read_text())
    sparse = json.loads((base / "sparse_reference_freeze.json").read_text())
    bert = json.loads((neural / "freeze.json").read_text())
    if (bert["bundle_sha256"] != split["bundle_sha256"]
            or fingerprint(neural / "frozen_head.joblib") != bert["head_sha256"]
            or fingerprint(base / "frozen_sparse_reference.joblib") != sparse["sha256"]):
        raise ValueError("Mismatched benchmark inputs or frozen model artifacts")
    paths = [base / "gpu_bundle/test.csv.gz", base / "sparse_test_scores.csv",
             neural / "neural_test_scores.csv", base / "CURATOR_test_labels.csv"]
    dates = pd.read_csv(paths[0], usecols=["pgpub_id", "published_date"], dtype=str)
    frame = dates
    for path in paths[1:]:
        part = pd.read_csv(path, dtype={"pgpub_id": str}, float_precision="round_trip")
        if part.pgpub_id.duplicated().any() or set(part.pgpub_id) != set(dates.pgpub_id):
            raise ValueError(f"Missing, extra or duplicate test IDs: {path}")
        frame = frame.merge(part, on="pgpub_id", validate="one_to_one")
    for field in ("sparse_score", "neural_score"):
        if not frame[field].between(0, 1).all():
            raise ValueError("Invalid prediction scores")
    report = {"sparse": metrics(frame[TARGET], frame.sparse_score, sparse["threshold"]),
              "neural": metrics(frame[TARGET], frame.neural_score, bert["threshold"]),
              "paired_week_bootstrap": paired_week_bootstrap(
                  frame, frame.sparse_score.ge(sparse["threshold"]).to_numpy(),
                  frame.neural_score.ge(bert["threshold"]).to_numpy()),
              "known_group_cross_split": split["known_group_cross_split"],
              "complete_patent_family_clearance": False, "recent_predictions_replaced": False}
    write_json(destination, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT)
    parser.add_argument("--neural", type=Path, default=DEFAULT / "neural_run")
    args = parser.parse_args()
    evaluate(args.base, args.neural)
