from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SECTION_DIR = ROOT / "analysis" / "results_sections_sample_style"
DOC_TOPIC_DIR = ROOT / "analysis" / "document_ai_exposure"
OUT_DIR = ROOT / "analysis" / "mutually_exclusive_ai_constructs"
SECTIONS = ["ITEM_1", "ITEM_1A", "ITEM_1C"]


def safe_div(numerator: float | pd.Series, denominator: float | pd.Series):
    return np.where(denominator != 0, numerator / denominator, 0.0)


def normalize_cik(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")


def construct_category(row: pd.Series) -> str:
    dimension = str(row.get("dimension") or "")
    if dimension == "expertise":
        return "ai_expertise"
    if dimension.startswith("risk_"):
        return "ai_risk_governance_exposure"
    if dimension == "exclude_noise":
        return "noise"
    if bool(row.get("counts_as_exposure")):
        return "ai_general_exposure"
    return "noise"


def read_topic_dictionary() -> pd.DataFrame:
    topics = pd.read_csv(DOC_TOPIC_DIR / "document_ai_topics.csv")
    topics["mutually_exclusive_category"] = topics.apply(construct_category, axis=1)
    topics.to_csv(OUT_DIR / "mutually_exclusive_topic_dictionary.csv", index=False)

    examples = pd.read_csv(DOC_TOPIC_DIR / "document_ai_topic_examples.csv")
    examples = examples.merge(
        topics[["topic_id", "mutually_exclusive_category"]],
        on="topic_id",
        how="left",
    )
    examples.to_csv(OUT_DIR / "mutually_exclusive_topic_examples.csv", index=False)
    return topics


def read_assignments(topics: pd.DataFrame) -> pd.DataFrame:
    usecols = [
        "company_name",
        "cik",
        "year",
        "period_end",
        "filed_date",
        "accession",
        "section",
        "sentence",
        "topic_id",
        "topic_weight",
    ]
    assignments = pd.read_csv(DOC_TOPIC_DIR / "document_ai_topic_assignments.csv", usecols=usecols, dtype={"cik": str, "year": str})
    assignments["cik"] = normalize_cik(assignments["cik"])
    assignments = assignments.merge(
        topics[["topic_id", "manual_label", "dimension", "mutually_exclusive_category"]],
        on="topic_id",
        how="left",
    )
    return assignments


def read_boilerplate_counts() -> pd.DataFrame:
    usecols = ["accession", "section", "boilerplate_noise", "sentence"]
    raw = pd.read_csv(DOC_TOPIC_DIR / "document_ai_sentences.csv", usecols=usecols)
    grouped = (
        raw.groupby(["accession", "section"], dropna=False)
        .agg(
            raw_ai_sentence_rows=("sentence", "count"),
            boilerplate_ai_sentence_rows=("boilerplate_noise", "sum"),
        )
        .reset_index()
    )
    return grouped


def read_denominator(scope: str) -> pd.DataFrame:
    if scope == "DOCUMENT":
        path = SECTION_DIR / "document_level.csv"
        total_col = "document_total_sentences"
        ai_col = "document_ai_sentences_count"
        risk_col = "document_risk_sentences_count"
        ai_risk_col = "document_ai_risk_sentences_count"
    else:
        path = SECTION_DIR / f"{scope}.csv"
        prefix = scope.lower()
        total_col = f"{prefix}_total_sentences"
        ai_col = f"{prefix}_ai_sentences_count"
        risk_col = f"{prefix}_risk_sentences_count"
        ai_risk_col = f"{prefix}_ai_risk_sentences_count"

    usecols = [
        "company_name",
        "cik",
        "fiscal_year_end",
        "form_type",
        "period_end",
        "filed_date",
        "quarter",
        "form_from_path",
        "accession",
        "ticker",
        "sic",
        "state",
        "total_sentences",
        "ai_related_sentences_count",
        "risk_related_sentences_count",
        "ai_risk_sentences_count",
        "crsp_match_accepted",
        "crsp_match_gap_days",
        "crsp_match_method",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in usecols, dtype={"cik": str, "period_end": str}, low_memory=False)
    df = df.drop_duplicates(subset=["accession"]).copy()
    df["cik"] = normalize_cik(df["cik"])
    df["year"] = df["period_end"].astype(str).str.slice(0, 4)
    df = df.rename(
        columns={
            "total_sentences": total_col,
            "ai_related_sentences_count": ai_col,
            "risk_related_sentences_count": risk_col,
            "ai_risk_sentences_count": ai_risk_col,
        }
    )
    df["scope"] = scope
    return df


def category_counts(assignments: pd.DataFrame, boilerplate: pd.DataFrame, scope: str) -> pd.DataFrame:
    subset = assignments if scope == "DOCUMENT" else assignments[assignments["section"] == scope]
    counts = (
        subset.groupby(["accession", "mutually_exclusive_category"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["ai_expertise", "ai_risk_governance_exposure", "ai_general_exposure", "noise"]:
        if col not in counts.columns:
            counts[col] = 0

    raw = boilerplate if scope == "DOCUMENT" else boilerplate[boilerplate["section"] == scope]
    raw_counts = (
        raw.groupby("accession", dropna=False)
        .agg(
            raw_ai_sentence_rows=("raw_ai_sentence_rows", "sum"),
            boilerplate_ai_sentence_rows=("boilerplate_ai_sentence_rows", "sum"),
        )
        .reset_index()
    )

    counts = counts.merge(raw_counts, on="accession", how="outer")
    for col in [
        "ai_expertise",
        "ai_risk_governance_exposure",
        "ai_general_exposure",
        "noise",
        "raw_ai_sentence_rows",
        "boilerplate_ai_sentence_rows",
    ]:
        counts[col] = counts[col].fillna(0).astype(int)
    counts["noise"] = counts["noise"] + counts["boilerplate_ai_sentence_rows"]
    counts["mutually_exclusive_ai_sentence_rows"] = (
        counts["ai_expertise"] + counts["ai_risk_governance_exposure"] + counts["ai_general_exposure"] + counts["noise"]
    )
    return counts


def build_scope_outputs(assignments: pd.DataFrame, boilerplate: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    denom = read_denominator(scope)
    counts = category_counts(assignments, boilerplate, scope)
    filing = denom.merge(counts, on="accession", how="left")
    for col in [
        "ai_expertise",
        "ai_risk_governance_exposure",
        "ai_general_exposure",
        "noise",
        "raw_ai_sentence_rows",
        "boilerplate_ai_sentence_rows",
        "mutually_exclusive_ai_sentence_rows",
    ]:
        filing[col] = filing[col].fillna(0).astype(int)

    if scope == "DOCUMENT":
        total_col = "document_total_sentences"
        raw_ai_col = "document_ai_sentences_count"
        out_prefix = "document"
    else:
        out_prefix = scope.lower()
        total_col = f"{out_prefix}_total_sentences"
        raw_ai_col = f"{out_prefix}_ai_sentences_count"

    filing[f"{out_prefix}_ai_expertise_ratio"] = safe_div(filing["ai_expertise"], filing[total_col])
    filing[f"{out_prefix}_ai_risk_governance_ratio"] = safe_div(filing["ai_risk_governance_exposure"], filing[total_col])
    filing[f"{out_prefix}_ai_general_exposure_ratio"] = safe_div(filing["ai_general_exposure"], filing[total_col])
    filing[f"{out_prefix}_ai_noise_ratio"] = safe_div(filing["noise"], filing[total_col])
    filing[f"{out_prefix}_ai_expertise_share_of_exclusive_ai"] = safe_div(
        filing["ai_expertise"], filing["mutually_exclusive_ai_sentence_rows"]
    )
    filing[f"{out_prefix}_ai_risk_governance_share_of_exclusive_ai"] = safe_div(
        filing["ai_risk_governance_exposure"], filing["mutually_exclusive_ai_sentence_rows"]
    )
    filing[f"{out_prefix}_exclusive_rows_share_of_raw_ai_count"] = safe_div(
        filing["mutually_exclusive_ai_sentence_rows"], filing[raw_ai_col]
    )

    topic_breadth = (
        (assignments if scope == "DOCUMENT" else assignments[assignments["section"] == scope])
        .groupby("accession", dropna=False)["topic_id"]
        .nunique()
        .reset_index(name=f"{out_prefix}_topic_breadth")
    )
    filing = filing.merge(topic_breadth, on="accession", how="left")
    filing[f"{out_prefix}_topic_breadth"] = filing[f"{out_prefix}_topic_breadth"].fillna(0).astype(int)

    company_year = (
        filing.groupby(["cik", "company_name", "year"], dropna=False)
        .agg(
            filings=("accession", "nunique"),
            total_sentences=(total_col, "sum"),
            raw_ai_count=(raw_ai_col, "sum"),
            ai_expertise=("ai_expertise", "sum"),
            ai_risk_governance_exposure=("ai_risk_governance_exposure", "sum"),
            ai_general_exposure=("ai_general_exposure", "sum"),
            noise=("noise", "sum"),
            raw_ai_sentence_rows=("raw_ai_sentence_rows", "sum"),
            boilerplate_ai_sentence_rows=("boilerplate_ai_sentence_rows", "sum"),
            mutually_exclusive_ai_sentence_rows=("mutually_exclusive_ai_sentence_rows", "sum"),
            crsp_match_accepted=("crsp_match_accepted", "max"),
            topic_breadth=(f"{out_prefix}_topic_breadth", "max"),
        )
        .reset_index()
    )
    company_year["scope"] = scope
    company_year[f"{out_prefix}_ai_expertise_ratio"] = safe_div(company_year["ai_expertise"], company_year["total_sentences"])
    company_year[f"{out_prefix}_ai_risk_governance_ratio"] = safe_div(
        company_year["ai_risk_governance_exposure"], company_year["total_sentences"]
    )
    company_year[f"{out_prefix}_ai_general_exposure_ratio"] = safe_div(
        company_year["ai_general_exposure"], company_year["total_sentences"]
    )
    company_year[f"{out_prefix}_ai_noise_ratio"] = safe_div(company_year["noise"], company_year["total_sentences"])
    company_year[f"{out_prefix}_ai_expertise_share_of_exclusive_ai"] = safe_div(
        company_year["ai_expertise"], company_year["mutually_exclusive_ai_sentence_rows"]
    )
    company_year[f"{out_prefix}_ai_risk_governance_share_of_exclusive_ai"] = safe_div(
        company_year["ai_risk_governance_exposure"], company_year["mutually_exclusive_ai_sentence_rows"]
    )
    company_year[f"{out_prefix}_exclusive_rows_share_of_raw_ai_count"] = safe_div(
        company_year["mutually_exclusive_ai_sentence_rows"], company_year["raw_ai_count"]
    )

    filing_path = OUT_DIR / f"{scope}_filing_mutually_exclusive.csv"
    company_year_path = OUT_DIR / f"{scope}_company_year_mutually_exclusive.csv"
    filing.to_csv(filing_path, index=False)
    company_year.to_csv(company_year_path, index=False)

    summary = {
        "scope": scope,
        "filing_rows": len(filing),
        "company_year_rows": len(company_year),
        "total_sentences": int(company_year["total_sentences"].sum()),
        "raw_ai_count": int(company_year["raw_ai_count"].sum()),
        "mutually_exclusive_ai_sentence_rows": int(company_year["mutually_exclusive_ai_sentence_rows"].sum()),
        "ai_expertise": int(company_year["ai_expertise"].sum()),
        "ai_risk_governance_exposure": int(company_year["ai_risk_governance_exposure"].sum()),
        "ai_general_exposure": int(company_year["ai_general_exposure"].sum()),
        "noise": int(company_year["noise"].sum()),
        "boilerplate_ai_sentence_rows": int(company_year["boilerplate_ai_sentence_rows"].sum()),
        "mean_expertise_ratio": float(company_year[f"{out_prefix}_ai_expertise_ratio"].mean()),
        "mean_risk_governance_ratio": float(company_year[f"{out_prefix}_ai_risk_governance_ratio"].mean()),
    }
    return filing, company_year, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topics = read_topic_dictionary()
    assignments = read_assignments(topics)
    boilerplate = read_boilerplate_counts()

    summaries = []
    for scope in ["DOCUMENT", *SECTIONS]:
        _, _, summary = build_scope_outputs(assignments, boilerplate, scope)
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(OUT_DIR / "mutually_exclusive_summary.csv", index=False)
    shutil.copy2(DOC_TOPIC_DIR / "document_ai_topics.csv", OUT_DIR / "source_document_ai_topics.csv")
    print(f"Wrote mutually exclusive AI construct outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
