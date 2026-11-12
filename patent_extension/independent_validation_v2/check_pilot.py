"""Check pilot annotations without modifying reviewer files."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_review(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"review_id", "title", "application_abstract", "abstract_ai_label", "reason", "evidence_span"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Missing rows or required columns: {path}")
    ids = [row["review_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError(f"Duplicate or empty review IDs: {path}")
    return {row["review_id"]: row for row in rows}


def summarize(left, right):
    if set(left) != set(right):
        raise ValueError("Reviewer document IDs differ")
    errors, disagreements = [], []
    complete, counts_left, counts_right = [], Counter(), Counter()
    for key in sorted(left):
        a, b = left[key], right[key]
        if any(a[field] != b[field] for field in ("title", "application_abstract")):
            raise ValueError(f"Reviewer texts differ for {key}")
        labels = []
        for reviewer, row in ((1, a), (2, b)):
            label = row["abstract_ai_label"].strip().upper()
            labels.append(label)
            if label not in {"", "0", "1", "U"}:
                errors.append(f"{key}: reviewer {reviewer} invalid label")
            if label in {"0", "1", "U"} and not row["reason"].strip():
                errors.append(f"{key}: reviewer {reviewer} missing reason")
            if label in {"0", "1"} and not row["evidence_span"].strip():
                errors.append(f"{key}: reviewer {reviewer} missing evidence")
        x, y = labels
        counts_left[x] += 1
        counts_right[y] += 1
        if x in {"0", "1", "U"} and y in {"0", "1", "U"}:
            complete.append((x, y))
            if x != y or "U" in (x, y):
                disagreements.append(key)
    n = len(complete)
    observed = sum(x == y for x, y in complete) / n if n else None
    c1, c2 = Counter(x for x, _ in complete), Counter(y for _, y in complete)
    expected = sum(c1[k] * c2[k] for k in ("0", "1", "U")) / n**2 if n else None
    kappa = (observed - expected) / (1 - expected) if n and expected < 1 else None
    return {
        "documents": len(left), "paired_labels": n,
        "reviewer_1_label_counts": dict(counts_left),
        "reviewer_2_label_counts": dict(counts_right),
        "three_category_agreement": observed, "three_category_cohen_kappa": kappa,
        "adjudication_review_ids": disagreements, "validation_errors": errors,
        "independent_annotation_complete": n == len(left) and not errors,
        "rubric_frozen": False, "publication_validation_complete": False,
        "note": "Pilot agreement is descriptive, not classifier accuracy. U remains a separate category. Human independence cannot be verified by this script.",
    }


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-1", type=Path, default=root / "outputs_v2_1/pilot/reviewer_1_BLIND_UNLABELLED.csv")
    parser.add_argument("--reviewer-2", type=Path, default=root / "outputs_v2_1/pilot/reviewer_2_BLIND_UNLABELLED.csv")
    args = parser.parse_args()
    report = summarize(read_review(args.reviewer_1), read_review(args.reviewer_2))
    print(json.dumps(report, indent=2))
    return 0 if report["independent_annotation_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
