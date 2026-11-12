from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_NDJSON = ROOT / "results_sections.ndjson"
SECTION_DIR = ROOT / "analysis" / "results_sections_sample_style"
DOC_TOPIC_DIR = ROOT / "analysis" / "document_ai_exposure"
OUT_DIR = ROOT / "analysis" / "mutually_exclusive_ai_constructs_occurrence"
SECTIONS = ["ITEM_1", "ITEM_1A", "ITEM_1C"]

BOILERPLATE_PATTERN = re.compile(
    r"\b(xbrl|taxonomy|linkbase|schema document|calculation linkbase|definition linkbase|"
    r"label linkbase|presentation linkbase|inline xbrl|exhibit\s+\d|"
    r"101\.(sch|cal|def|lab|pre|ins)|104 cover page)\b",
    re.IGNORECASE,
)


def clean_text(text: object) -> str:
    value = str(text or "").replace("\n", " ").replace("\r", " ")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_cik(value: object) -> str:
    return str(value or "").replace(".0", "").lstrip("0")


def normalize_cik_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")


def safe_div(numerator: float | pd.Series, denominator: float | pd.Series):
    return np.where(denominator != 0, numerator / denominator, 0.0)


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
    topics.to_csv(OUT_DIR / "occurrence_topic_dictionary.csv", index=False)

    examples = pd.read_csv(DOC_TOPIC_DIR / "document_ai_topic_examples.csv")
    examples = examples.merge(
        topics[["topic_id", "mutually_exclusive_category"]],
        on="topic_id",
        how="left",
    )
    examples.to_csv(OUT_DIR / "occurrence_topic_examples.csv", index=False)
    return topics


def read_dedup_assignment_lookup(topics: pd.DataFrame) -> pd.DataFrame:
    usecols = ["accession", "section", "sentence", "topic_id", "topic_weight"]
    lookup = pd.read_csv(DOC_TOPIC_DIR / "document_ai_topic_assignments.csv", usecols=usecols)
    lookup = lookup.merge(
        topics[["topic_id", "manual_label", "dimension", "mutually_exclusive_category"]],
        on="topic_id",
        how="left",
    )
    return lookup.drop_duplicates(subset=["accession", "section", "sentence"])


def extract_original_ai_sentence_occurrences() -> pd.DataFrame:
    rows = []
    with SOURCE_NDJSON.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            accession = record.get("accession", "")
            cik = normalize_cik(record.get("cik"))
            year = str(record.get("period_end") or record.get("filed_date") or "")[:4]
            for section_name, section in (record.get("sections") or {}).items():
                for idx, sentence in enumerate(section.get("ai_sentences") or []):
                    text = clean_text(sentence.get("sentence", ""))
                    if len(text) < 25:
                        continue
                    rows.append(
                        {
                            "company_name": record.get("company_name", ""),
                            "cik": cik,
                            "year": year,
                            "period_end": record.get("period_end", ""),
                            "filed_date": record.get("filed_date", ""),
                            "accession": accession,
                            "section": section_name,
                            "occurrence_index": idx,
                            "sentence": text,
                            "boilerplate_noise": bool(BOILERPLATE_PATTERN.search(text)),
                        }
                    )
    return pd.DataFrame(rows)


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

    df = pd.read_csv(path, dtype={"cik": str, "period_end": str}, low_memory=False)
    df = df.drop_duplicates(subset=["accession"]).copy()
    df["cik"] = normalize_cik_series(df["cik"])
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


def assign_occurrence_topics(occurrences: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    substantive = occurrences.loc[~occurrences["boilerplate_noise"]].copy()
    assigned = substantive.merge(lookup, on=["accession", "section", "sentence"], how="left")
    missing = assigned["mutually_exclusive_category"].isna()
    assigned.loc[missing, "mutually_exclusive_category"] = "noise"
    assigned.loc[missing, "manual_label"] = "Unmatched substantive occurrence"
    assigned.loc[missing, "dimension"] = "exclude_noise"

    boilerplate = occurrences.loc[occurrences["boilerplate_noise"]].copy()
    boilerplate["topic_id"] = np.nan
    boilerplate["topic_weight"] = np.nan
    boilerplate["manual_label"] = "XBRL/exhibit boilerplate"
    boilerplate["dimension"] = "exclude_noise"
    boilerplate["mutually_exclusive_category"] = "noise"

    out = pd.concat([assigned, boilerplate], ignore_index=True, sort=False)
    out.to_csv(OUT_DIR / "original_ai_sentence_occurrence_assignments.csv", index=False)
    return out


def category_counts(assignments: pd.DataFrame, scope: str) -> pd.DataFrame:
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

    boilerplate = (
        subset.groupby("accession", dropna=False)["boilerplate_noise"]
        .sum()
        .reset_index(name="boilerplate_ai_sentence_rows")
    )
    counts = counts.merge(boilerplate, on="accession", how="left")
    for col in ["ai_expertise", "ai_risk_governance_exposure", "ai_general_exposure", "noise", "boilerplate_ai_sentence_rows"]:
        counts[col] = counts[col].fillna(0).astype(int)
    counts["original_ai_sentence_occurrences"] = (
        counts["ai_expertise"] + counts["ai_risk_governance_exposure"] + counts["ai_general_exposure"] + counts["noise"]
    )
    return counts


def build_scope_outputs(assignments: pd.DataFrame, scope: str) -> dict:
    denom = read_denominator(scope)
    counts = category_counts(assignments, scope)
    filing = denom.merge(counts, on="accession", how="left")
    for col in [
        "ai_expertise",
        "ai_risk_governance_exposure",
        "ai_general_exposure",
        "noise",
        "boilerplate_ai_sentence_rows",
        "original_ai_sentence_occurrences",
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

    topic_breadth = (
        (assignments if scope == "DOCUMENT" else assignments[assignments["section"] == scope])
        .loc[lambda d: d["mutually_exclusive_category"] != "noise"]
        .groupby("accession", dropna=False)["topic_id"]
        .nunique()
        .reset_index(name=f"{out_prefix}_topic_breadth")
    )
    filing = filing.merge(topic_breadth, on="accession", how="left")
    filing[f"{out_prefix}_topic_breadth"] = filing[f"{out_prefix}_topic_breadth"].fillna(0).astype(int)

    filing[f"{out_prefix}_ai_expertise_ratio"] = safe_div(filing["ai_expertise"], filing[total_col])
    filing[f"{out_prefix}_ai_risk_governance_ratio"] = safe_div(filing["ai_risk_governance_exposure"], filing[total_col])
    filing[f"{out_prefix}_ai_general_exposure_ratio"] = safe_div(filing["ai_general_exposure"], filing[total_col])
    filing[f"{out_prefix}_ai_noise_ratio"] = safe_div(filing["noise"], filing[total_col])
    filing[f"{out_prefix}_ai_expertise_share_of_original_ai"] = safe_div(filing["ai_expertise"], filing["original_ai_sentence_occurrences"])
    filing[f"{out_prefix}_ai_risk_governance_share_of_original_ai"] = safe_div(
        filing["ai_risk_governance_exposure"], filing["original_ai_sentence_occurrences"]
    )
    filing[f"{out_prefix}_original_occurrences_share_of_raw_ai_count"] = safe_div(
        filing["original_ai_sentence_occurrences"], filing[raw_ai_col]
    )

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
            boilerplate_ai_sentence_rows=("boilerplate_ai_sentence_rows", "sum"),
            original_ai_sentence_occurrences=("original_ai_sentence_occurrences", "sum"),
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
    company_year[f"{out_prefix}_ai_expertise_share_of_original_ai"] = safe_div(
        company_year["ai_expertise"], company_year["original_ai_sentence_occurrences"]
    )
    company_year[f"{out_prefix}_ai_risk_governance_share_of_original_ai"] = safe_div(
        company_year["ai_risk_governance_exposure"], company_year["original_ai_sentence_occurrences"]
    )
    company_year[f"{out_prefix}_original_occurrences_share_of_raw_ai_count"] = safe_div(
        company_year["original_ai_sentence_occurrences"], company_year["raw_ai_count"]
    )

    filing.to_csv(OUT_DIR / f"{scope}_filing_occurrence_mutually_exclusive.csv", index=False)
    company_year.to_csv(OUT_DIR / f"{scope}_company_year_occurrence_mutually_exclusive.csv", index=False)

    return {
        "scope": scope,
        "filing_rows": len(filing),
        "company_year_rows": len(company_year),
        "total_sentences": int(company_year["total_sentences"].sum()),
        "raw_ai_count": int(company_year["raw_ai_count"].sum()),
        "original_ai_sentence_occurrences": int(company_year["original_ai_sentence_occurrences"].sum()),
        "ai_expertise": int(company_year["ai_expertise"].sum()),
        "ai_risk_governance_exposure": int(company_year["ai_risk_governance_exposure"].sum()),
        "ai_general_exposure": int(company_year["ai_general_exposure"].sum()),
        "noise": int(company_year["noise"].sum()),
        "boilerplate_ai_sentence_rows": int(company_year["boilerplate_ai_sentence_rows"].sum()),
        "mean_expertise_ratio": float(company_year[f"{out_prefix}_ai_expertise_ratio"].mean()),
        "mean_risk_governance_ratio": float(company_year[f"{out_prefix}_ai_risk_governance_ratio"].mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topics = read_topic_dictionary()
    lookup = read_dedup_assignment_lookup(topics)
    occurrences = extract_original_ai_sentence_occurrences()
    assignments = assign_occurrence_topics(occurrences, lookup)

    summaries = []
    for scope in ["DOCUMENT", *SECTIONS]:
        summaries.append(build_scope_outputs(assignments, scope))

    pd.DataFrame(summaries).to_csv(OUT_DIR / "occurrence_mutually_exclusive_summary.csv", index=False)
    shutil.copy2(DOC_TOPIC_DIR / "document_ai_topics.csv", OUT_DIR / "source_document_ai_topics.csv")
    print(f"Wrote occurrence-based mutually exclusive AI construct outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
