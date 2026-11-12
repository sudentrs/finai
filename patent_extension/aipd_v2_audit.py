"""Verify recent outputs and draw a weighted, blind recent-publication review sample."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from aipd_extension import REVIEW, fingerprint, write_json
from aipd_v2_prepare import SEED, V1, V2


def audit(out):
    recent_path = out / "publications_2024_2025.csv"
    columns = {"application_id": str, "pgpub_id": str}
    recent = pd.read_csv(recent_path, dtype=columns, float_precision="round_trip")
    old = pd.read_csv(V1 / recent_path.name, dtype=columns, usecols=[*columns, "aipd_surrogate_label"])
    freeze = json.loads((out / "deployment_freeze.json").read_text())
    cutoff = freeze["threshold"]
    if fingerprint(out / "frozen_deployment.joblib") != freeze["sha256"]:
        raise ValueError("Deployment model fingerprint mismatch")
    if set(recent.application_id) != set(old.application_id) or recent.application_id.duplicated().any():
        raise ValueError("Deployment frame differs from v1")
    recent = recent.merge(old.rename(columns={"aipd_surrogate_label": "v1_label"}),
                          on=list(columns), validate="one_to_one")
    recent["year"] = recent.published_date.str[:4]
    recent["filing_year"] = pd.to_datetime(recent.filing_date, errors="coerce").dt.year.astype("Int64")
    recent["scored"] = recent.score_status.eq("scored")
    scored = recent.loc[recent.scored].copy()
    if not scored.aipd_surrogate_score.between(0, 1).all() or not scored.aipd_surrogate_label.eq(
            scored.aipd_surrogate_score.ge(cutoff)).all() or recent.loc[
            ~recent.scored, ["aipd_surrogate_score", "aipd_surrogate_label"]].notna().any().any():
        raise ValueError("Inconsistent scores and labels")
    actual = recent.groupby(["year", "score_status"]).agg(
        publications=("pgpub_id", "size"), predicted_ai=("aipd_surrogate_label", "sum")).reset_index()
    expected = pd.read_csv(out / "recent_coverage.csv", dtype={"year": str})
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
    pd.crosstab([scored.year, scored.v1_label], scored.aipd_surrogate_label).to_csv(out / "v1_v2_label_comparison.csv")
    recent.groupby(["year", "filing_year"], dropna=False).agg(
        publications=("pgpub_id", "size"), scored=("scored", "sum"),
        predicted_ai=("aipd_surrogate_label", "sum")).reset_index().to_csv(
            out / "filing_publication_year_coverage.csv", index=False)
    reserved = set(pd.read_csv(REVIEW, dtype=str, usecols=["application_id"]).application_id)
    eligible = scored.loc[~scored.application_id.isin(reserved)].copy()
    eligible["distance"] = (eligible.aipd_surrogate_score - cutoff).abs()
    parts = []
    for (year, label), group in eligible.groupby(["year", "aipd_surrogate_label"]):
        ordered = group.sort_values(["distance", "pgpub_id"])
        mid = len(ordered) // 2
        for band, stratum in (("nearer", ordered.iloc[:mid]), ("farther", ordered.iloc[mid:])):
            draw = stratum.sample(n=50, random_state=SEED + len(parts)).copy()
            draw["band"] = band
            draw["stratum_size"] = len(stratum)
            draw["inclusion_probability"] = 50 / len(stratum)
            draw["weight"] = len(stratum) / 50
            parts.append(draw)
    sample = pd.concat(parts, ignore_index=True)
    sample["review_id"] = sample.pgpub_id.map(
        lambda key: "A2-" + hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()[:16])
    ids = set(sample.pgpub_id)
    texts = pd.concat([part[part.pgpub_id.isin(ids)] for part in pd.read_csv(
        V1 / "staged_recent_abstracts.csv", dtype=str, chunksize=100_000, keep_default_na=False)])
    blind = sample[["pgpub_id", "review_id"]].merge(texts, on="pgpub_id", validate="one_to_one")
    if len(blind) != 400 or blind.review_id.duplicated().any():
        raise ValueError("Incomplete or nonunique review sample")
    folder = out / "recent_blind_audit"
    folder.mkdir(exist_ok=False)
    sample[["review_id", "pgpub_id", "application_id", "year", "aipd_surrogate_label",
            "aipd_surrogate_score", "band", "stratum_size", "inclusion_probability", "weight"]].to_csv(
                folder / "CURATOR_ONLY.csv", index=False)
    blind = blind.drop(columns="pgpub_id").assign(abstract_ai_label="", reason="", evidence_span="")
    for reviewer in (1, 2):
        blind.sample(frac=1, random_state=SEED + reviewer).to_csv(
            folder / f"reviewer_{reviewer}_BLIND.csv", index=False, encoding="utf-8-sig")
    write_json(out / "output_verification.json", {
        "rows": len(recent), "scored": len(scored), "predicted_ai": int(scored.aipd_surrogate_label.sum()),
        "prediction_sha256": fingerprint(recent_path), "blind_sample_n": len(blind),
        "audit_population": len(eligible), "excluded_prior_review_ids": len(scored)-len(eligible),
        "human_labels_complete": False, "family_overlap_cleared": False,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=V2)
    audit(parser.parse_args().out)
