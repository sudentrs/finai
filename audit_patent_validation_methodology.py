from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


ROOT = Path(".")
AIPD_PATH = ROOT / "ai_model_predictions.csv" / "ai_model_predictions.csv"
PG_APP_PATH = ROOT / "pg_published_application.tsv" / "pg_published_application.tsv"
PG_CROSSWALK_PATH = ROOT / "pg_granted_pgpubs_crosswalk.tsv" / "pg_granted_pgpubs_crosswalk.tsv"
PATENT_DIR = ROOT / "analysis" / "patent_validation_improved"
COUNTS_PATH = PATENT_DIR / "firm_year_ai_patent_application_counts.csv"
DETAIL_PATH = PATENT_DIR / "matched_patent_applications_detail.csv"
ITEM1_FINAL = (
    PATENT_DIR
    / "extended_validation_analysis"
    / "company_characteristics_ai_analysis"
    / "ITEM_1_filing_with_company_characteristic_quintiles.csv"
)
DOCUMENT_FINAL = (
    PATENT_DIR
    / "extended_validation_analysis"
    / "company_characteristics_ai_analysis"
    / "DOCUMENT_filing_with_company_characteristic_quintiles.csv"
)


def year_from_date(series):
    return pd.to_datetime(series, errors="coerce").dt.year


def audit_aipd(chunksize=750_000):
    cols = [
        "doc_id",
        "flag_patent",
        "pub_dt",
        "predict50_any_ai",
        "predict86_any_ai",
        "predict93_any_ai",
        "predict93_ml",
        "predict93_evo",
        "predict93_nlp",
        "predict93_speech",
        "predict93_vision",
        "predict93_planning",
        "predict93_kr",
        "predict93_hardware",
    ]
    stats = []
    mismatches = 0
    rows = 0
    for chunk in pd.read_csv(AIPD_PATH, usecols=cols, chunksize=chunksize):
        rows += len(chunk)
        chunk["year"] = year_from_date(chunk["pub_dt"])
        comp_cols = [
            "predict93_ml",
            "predict93_evo",
            "predict93_nlp",
            "predict93_speech",
            "predict93_vision",
            "predict93_planning",
            "predict93_kr",
            "predict93_hardware",
        ]
        comp_or = chunk[comp_cols].fillna(0).astype(int).max(axis=1)
        mismatches += int((chunk["predict93_any_ai"].fillna(0).astype(int) != comp_or).sum())
        stats.append(
            chunk.groupby(["flag_patent", "year"], dropna=False)
            .agg(
                rows=("doc_id", "size"),
                any50=("predict50_any_ai", "sum"),
                any86=("predict86_any_ai", "sum"),
                any93=("predict93_any_ai", "sum"),
            )
            .reset_index()
        )
    out = pd.concat(stats, ignore_index=True)
    out = out.groupby(["flag_patent", "year"], dropna=False).sum(numeric_only=True).reset_index()
    return rows, mismatches, out


def audit_pg_app(chunksize=750_000):
    stats = []
    rows = 0
    for chunk in pd.read_csv(PG_APP_PATH, sep="\t", usecols=["pgpub_id", "filing_date", "published_date"], chunksize=chunksize):
        rows += len(chunk)
        chunk["filing_year"] = year_from_date(chunk["filing_date"])
        chunk["publication_year"] = year_from_date(chunk["published_date"])
        stats.append(
            chunk.groupby(["filing_year", "publication_year"], dropna=False)
            .agg(rows=("pgpub_id", "size"))
            .reset_index()
        )
    out = pd.concat(stats, ignore_index=True)
    out = out.groupby(["filing_year", "publication_year"], dropna=False).sum(numeric_only=True).reset_index()
    return rows, out


def audit_counts_and_final():
    counts = pd.read_csv(COUNTS_PATH)
    counts_summary = (
        counts.groupby("year", as_index=False)
        .agg(
            firm_years=("cik", "size"),
            total_patent_app_count=("total_patent_app_count", "sum"),
            aipd_matched_patent_app_count=("aipd_matched_patent_app_count", "sum"),
            ai_patent_app_count_93=("ai_patent_app_count_93", "sum"),
            has_ai_patent_app_93=("has_ai_patent_app_93", "sum"),
        )
        .sort_values("year")
    )
    counts_summary["aipd_coverage_share"] = np.where(
        counts_summary["total_patent_app_count"] > 0,
        counts_summary["aipd_matched_patent_app_count"] / counts_summary["total_patent_app_count"],
        np.nan,
    )

    final_summaries = []
    for label, path in [("ITEM_1", ITEM1_FINAL), ("DOCUMENT", DOCUMENT_FINAL)]:
        usecols = [
            "year",
            "form_type",
            "total_patent_app_count",
            "aipd_matched_patent_app_count",
            "ai_patent_app_count_93",
            "ai_patent_app_share_93",
            "has_ai_patent_app_93",
            "log1p_ai_patent_app_count_93",
        ]
        df = pd.read_csv(path, usecols=usecols)
        check = pd.DataFrame(
            {
                "scope": [label],
                "rows": [len(df)],
                "min_year": [df["year"].min()],
                "max_year": [df["year"].max()],
                "non_10k_rows": [(df["form_type"] != "10-K").sum()],
                "missing_ai_patent_count_93": [df["ai_patent_app_count_93"].isna().sum()],
                "has_indicator_mismatch": [
                    (
                        df["has_ai_patent_app_93"].fillna(0).astype(int)
                        != (df["ai_patent_app_count_93"].fillna(0) > 0).astype(int)
                    ).sum()
                ],
                "log1p_mismatch_max": [
                    (
                        df["log1p_ai_patent_app_count_93"].fillna(0)
                        - np.log1p(df["ai_patent_app_count_93"].fillna(0))
                    )
                    .abs()
                    .max()
                ],
            }
        )
        yearly = (
            df.groupby("year", as_index=False)
            .agg(
                rows=("form_type", "size"),
                total_patents=("total_patent_app_count", "sum"),
                aipd_matched=("aipd_matched_patent_app_count", "sum"),
                ai_patents_93=("ai_patent_app_count_93", "sum"),
                has_ai_patent_93=("has_ai_patent_app_93", "sum"),
                mean_ai_patent_count_93=("ai_patent_app_count_93", "mean"),
                share_has_ai_patent_93=("has_ai_patent_app_93", "mean"),
            )
            .assign(scope=label)
        )
        final_summaries.append((check, yearly))
    return counts_summary, final_summaries


def main():
    out_dir = PATENT_DIR / "methodology_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("AIPD path:", AIPD_PATH)
    print("AIPD modified:", datetime.fromtimestamp(AIPD_PATH.stat().st_mtime))
    aipd_rows, predict93_or_mismatches, aipd_years = audit_aipd()
    aipd_years.to_csv(out_dir / "aipd_year_flag_summary.csv", index=False)
    print("AIPD rows:", f"{aipd_rows:,}")
    print("predict93_any_ai vs OR(component predict93*) mismatches:", f"{predict93_or_mismatches:,}")
    print("AIPD year min/max:", int(aipd_years["year"].min()), int(aipd_years["year"].max()))
    print("AIPD rows by flag:")
    print(aipd_years.groupby("flag_patent")["rows"].sum().to_string())
    print("AIPD pre-grant rows tail:")
    print(aipd_years[aipd_years["flag_patent"].eq(0)].tail(8).to_string(index=False))

    pg_rows, pg_years = audit_pg_app()
    pg_years.to_csv(out_dir / "patentsview_pg_application_year_summary.csv", index=False)
    print("\nPatentsView application rows:", f"{pg_rows:,}")
    print("PatentsView filing year min/max:", int(pg_years["filing_year"].min()), int(pg_years["filing_year"].max()))
    print("PatentsView publication year min/max:", int(pg_years["publication_year"].min()), int(pg_years["publication_year"].max()))

    if PG_CROSSWALK_PATH.exists():
        cw = pd.read_csv(PG_CROSSWALK_PATH, sep="\t", nrows=5)
        print("\nGranted/PGPub crosswalk exists:", PG_CROSSWALK_PATH)
        print("Crosswalk columns:", cw.columns.tolist())
    else:
        print("\nGranted/PGPub crosswalk missing.")

    counts_summary, final_summaries = audit_counts_and_final()
    counts_summary.to_csv(out_dir / "firm_year_patent_counts_year_summary.csv", index=False)
    print("\nFirm-year patent count years:", int(counts_summary["year"].min()), int(counts_summary["year"].max()))
    print(counts_summary.tail(8).to_string(index=False))

    checks = []
    yearlies = []
    for check, yearly in final_summaries:
        checks.append(check)
        yearlies.append(yearly)
    checks_df = pd.concat(checks, ignore_index=True)
    yearly_df = pd.concat(yearlies, ignore_index=True)
    checks_df.to_csv(out_dir / "final_dataset_patent_variable_integrity.csv", index=False)
    yearly_df.to_csv(out_dir / "final_dataset_patent_year_summary.csv", index=False)
    print("\nFinal dataset checks:")
    print(checks_df.to_string(index=False))
    print("\nFinal dataset yearly patent summary tail:")
    print(yearly_df[yearly_df["year"].between(2020, 2026)].to_string(index=False))
    print("\nAudit outputs:", out_dir)


if __name__ == "__main__":
    main()
