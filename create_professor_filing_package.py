from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE_DIR = ROOT / "analysis" / "mutually_exclusive_ai_constructs_occurrence"
OUT_DIR = ROOT / "analysis" / "professor_filing_level_package_20260804"

SCOPE_FILES = {
    "DOCUMENT": ("DOCUMENT_filing_occurrence_mutually_exclusive.csv", "document"),
    "ITEM_1": ("ITEM_1_filing_occurrence_mutually_exclusive.csv", "item_1"),
    "ITEM_1A": ("ITEM_1A_filing_occurrence_mutually_exclusive.csv", "item_1a"),
    "ITEM_1C": ("ITEM_1C_filing_occurrence_mutually_exclusive.csv", "item_1c"),
}

BASE_COLUMNS = [
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
    "matched_date",
    "crsp_permno",
    "crsp_permco",
    "crsp_cusip",
    "ret",
    "retx",
    "crsp_match_gap_days",
    "crsp_match_method",
    "crsp_match_accepted",
    "crsp_error",
    "compustat_ticker",
    "compustat_cusip",
    "compustat_gvkey",
    "compustat_date",
    "compustat_match_gap_days",
    "compustat_source",
    "sic",
    "state",
    "filename",
    "filing_html_index",
    "htm_filing_link",
    "complete_text_filing_link",
    "year",
    "scope",
]

COUNT_COLUMNS = [
    "ai_expertise",
    "ai_risk_governance_exposure",
    "ai_general_exposure",
    "noise",
    "boilerplate_ai_sentence_rows",
    "original_ai_sentence_occurrences",
]


def standardize_scope(scope: str, filename: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(SOURCE_DIR / filename, dtype={"cik": str, "period_end": str, "filed_date": str})
    rename_map = {
        f"{prefix}_total_sentences": "total_sentences",
        f"{prefix}_ai_sentences_count": "ai_sentences_count",
        f"{prefix}_risk_sentences_count": "risk_sentences_count",
        f"{prefix}_ai_risk_sentences_count": "ai_risk_sentences_count",
        f"{prefix}_topic_breadth": "topic_breadth",
        f"{prefix}_ai_expertise_ratio": "ai_expertise_ratio",
        f"{prefix}_ai_risk_governance_ratio": "ai_risk_governance_ratio",
        f"{prefix}_ai_general_exposure_ratio": "ai_general_exposure_ratio",
        f"{prefix}_ai_noise_ratio": "ai_noise_ratio",
        f"{prefix}_ai_expertise_share_of_original_ai": "ai_expertise_share",
        f"{prefix}_ai_risk_governance_share_of_original_ai": "ai_risk_governance_share",
        f"{prefix}_original_occurrences_share_of_raw_ai_count": "original_occurrences_share_of_raw_ai_count",
    }
    df = df.rename(columns=rename_map)
    df["scope"] = scope

    preferred = [
        *BASE_COLUMNS,
        "total_sentences",
        "ai_sentences_count",
        "risk_sentences_count",
        "ai_risk_sentences_count",
        *COUNT_COLUMNS,
        "topic_breadth",
        "ai_expertise_ratio",
        "ai_risk_governance_ratio",
        "ai_general_exposure_ratio",
        "ai_noise_ratio",
        "ai_expertise_share",
        "ai_risk_governance_share",
        "original_occurrences_share_of_raw_ai_count",
    ]
    available = [col for col in preferred if col in df.columns]
    remaining = [col for col in df.columns if col not in available]
    return df[available + remaining]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for scope, (filename, prefix) in SCOPE_FILES.items():
        out = standardize_scope(scope, filename, prefix)
        out_path = OUT_DIR / f"{scope}_filing_standardized.csv"
        out.to_csv(out_path, index=False)
        summary_rows.append(
            {
                "scope": scope,
                "file": out_path.name,
                "rows": len(out),
                "has_filed_date": "filed_date" in out.columns,
                "missing_filed_date": int(out["filed_date"].isna().sum()),
                "columns": len(out.columns),
                "ai_sentences_count_sum": int(out["ai_sentences_count"].sum()),
                "original_ai_sentence_occurrences_sum": int(out["original_ai_sentence_occurrences"].sum()),
                "ai_expertise_sum": int(out["ai_expertise"].sum()),
                "ai_risk_governance_exposure_sum": int(out["ai_risk_governance_exposure"].sum()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "standardized_filing_package_summary.csv", index=False)

    readme = """# Filing-Level Standardized AI Disclosure Package

This folder contains filing-level files for the professor's CRSP matching and return tests.

Each row is one 10-K filing/accession. All four files use the same standardized variable names, including `filed_date`, and preserve the original metadata/CRSP columns such as `crsp_permno`, `crsp_permco`, `crsp_cusip`, `matched_date`, `ret`, and `retx`.

Main files:

- `DOCUMENT_filing_standardized.csv`: whole 10-K document.
- `ITEM_1_filing_standardized.csv`: ITEM_1 / business section.
- `ITEM_1A_filing_standardized.csv`: ITEM_1A / risk factors section.
- `ITEM_1C_filing_standardized.csv`: ITEM_1C / cybersecurity disclosure section.

Main explanatory variables:

- `ai_expertise_ratio`: AI expertise sentence occurrences divided by total sentences in the scope.
- `ai_risk_governance_ratio`: AI risk/governance sentence occurrences divided by total sentences in the scope.
- `ai_expertise_share`: AI expertise sentence occurrences divided by all original AI sentence occurrences in the filing/scope.
- `ai_risk_governance_share`: AI risk/governance sentence occurrences divided by all original AI sentence occurrences in the filing/scope.

Other useful controls/audit columns:

- `ai_sentences_count`: raw AI sentence count from the section/document counters.
- `original_ai_sentence_occurrences`: original AI sentence occurrences used in the mutually exclusive classification.
- `noise`: false positives and XBRL/exhibit boilerplate rows.
- `ai_noise_ratio`: noise rows divided by total sentences.
- `topic_breadth`: number of non-noise AI topics detected in the filing/scope.

Note: The previous scope-specific files are still available in the occurrence folder. These standardized copies only rename section-prefixed columns into common names so the same regression/matching code can be used across DOCUMENT, ITEM_1, ITEM_1A, and ITEM_1C.
"""
    (OUT_DIR / "README_standardized_filing_package.md").write_text(readme, encoding="utf-8")
    print(f"Wrote standardized professor package to {OUT_DIR}")


if __name__ == "__main__":
    main()
