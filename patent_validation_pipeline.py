from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_AIPD = ROOT / "ai_model_predictions.csv" / "ai_model_predictions.csv"
DEFAULT_PG_APP = ROOT / "pg_published_application.tsv" / "pg_published_application.tsv"
DEFAULT_PG_ASSIGNEE = ROOT / "pg_assignee_disambiguated.tsv" / "pg_assignee_disambiguated.tsv"
DEFAULT_TENK = ROOT / "analysis" / "professor_filing_level_package_20260804" / "DOCUMENT_filing_standardized.csv"
DEFAULT_COMPUSTAT = ROOT / "ncw9xm2krtoqavrc.csv"
DEFAULT_OUT = ROOT / "analysis" / "patent_validation"

LEGAL_SUFFIXES = {
    "THE",
    "INC",
    "INCORPORATED",
    "CORP",
    "CORPORATION",
    "CO",
    "COMPANY",
    "COS",
    "COMPANIES",
    "LTD",
    "LIMITED",
    "PLC",
    "LLC",
    "LLP",
    "LP",
    "L P",
    "SA",
    "NV",
    "AG",
    "SE",
    "BV",
    "GMBH",
    "HOLDING",
    "HOLDINGS",
    "HLDG",
    "HLDGS",
    "GROUP",
}


def normalize_name(value: object) -> str:
    text = str(value or "").upper()
    text = text.replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]+", " ", text)
    tokens = [token for token in text.split() if token not in LEGAL_SUFFIXES]
    return " ".join(tokens)


def normalize_cik(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")


def normalize_cik_value(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value or "").replace(".0", "").lstrip("0")


def year_from_date(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return dates.dt.year.astype("Int64").astype(str).replace("<NA>", "")


def read_firm_universe(tenk_path: Path, compustat_path: Path) -> pd.DataFrame:
    tenk_cols = ["cik", "company_name", "ticker", "compustat_gvkey", "compustat_cusip", "crsp_permno"]
    tenk = pd.read_csv(tenk_path, usecols=lambda c: c in tenk_cols, dtype={"cik": str}, low_memory=False)
    tenk["cik"] = normalize_cik(tenk["cik"])
    tenk = tenk.drop_duplicates()
    tenk_names = tenk.rename(columns={"company_name": "firm_name", "ticker": "firm_ticker"})

    comp_cols = ["cik", "conm", "tic", "gvkey", "cusip"]
    comp = pd.read_csv(compustat_path, usecols=lambda c: c in comp_cols, dtype={"cik": str}, low_memory=False)
    comp["cik"] = normalize_cik(comp["cik"])
    comp_names = comp.rename(
        columns={
            "conm": "firm_name",
            "tic": "firm_ticker",
            "gvkey": "compustat_gvkey",
            "cusip": "compustat_cusip",
        }
    )

    firm = pd.concat([tenk_names, comp_names], ignore_index=True, sort=False)
    firm = firm.loc[firm["cik"].notna() & (firm["cik"] != "") & firm["firm_name"].notna()].copy()
    firm["normalized_firm_name"] = firm["firm_name"].map(normalize_name)
    firm = firm.loc[firm["normalized_firm_name"] != ""].copy()

    # Prefer rows with richer identifiers if the same normalized name maps once.
    firm["identifier_score"] = firm[["compustat_gvkey", "compustat_cusip", "crsp_permno", "firm_ticker"]].notna().sum(axis=1)
    firm = (
        firm.sort_values(["normalized_firm_name", "identifier_score"], ascending=[True, False])
        .drop_duplicates(subset=["normalized_firm_name", "cik"])
        .reset_index(drop=True)
    )
    name_counts = firm.groupby("normalized_firm_name")["cik"].nunique().reset_index(name="candidate_ciks")
    firm = firm.merge(name_counts, on="normalized_firm_name", how="left")
    return firm


def read_ai_predictions(aipd_path: Path, chunksize: int) -> pd.DataFrame:
    cols = ["doc_id", "flag_patent", "pub_dt", "appl_id", "predict50_any_ai", "predict86_any_ai", "predict93_any_ai"]
    pieces = []
    for chunk in pd.read_csv(aipd_path, usecols=cols, chunksize=chunksize, low_memory=False):
        # For applications, AIPD uses patent flag 0 and doc_id aligns with PatentsView pgpub_id.
        chunk = chunk.loc[chunk["flag_patent"] == 0].copy()
        if chunk.empty:
            continue
        chunk["pgpub_id"] = pd.to_numeric(chunk["doc_id"], errors="coerce").astype("Int64")
        chunk = chunk.loc[chunk["pgpub_id"].notna()].copy()
        pieces.append(
            chunk[
                ["pgpub_id", "appl_id", "pub_dt", "predict50_any_ai", "predict86_any_ai", "predict93_any_ai"]
            ]
        )
    if not pieces:
        return pd.DataFrame(columns=["pgpub_id", "appl_id", "pub_dt", "predict50_any_ai", "predict86_any_ai", "predict93_any_ai"])
    out = pd.concat(pieces, ignore_index=True)
    for col in ["predict50_any_ai", "predict86_any_ai", "predict93_any_ai"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out.drop_duplicates(subset=["pgpub_id"])


def read_applications(pg_app_path: Path) -> pd.DataFrame:
    cols = ["pgpub_id", "application_id", "filing_date", "published_date", "application_title"]
    apps = pd.read_csv(pg_app_path, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    apps["pgpub_id"] = pd.to_numeric(apps["pgpub_id"], errors="coerce").astype("Int64")
    apps = apps.loc[apps["pgpub_id"].notna()].drop_duplicates(subset=["pgpub_id"]).copy()
    apps["application_year"] = year_from_date(apps["filing_date"])
    apps["publication_year"] = year_from_date(apps["published_date"])
    apps["patent_year"] = np.where(apps["application_year"] != "", apps["application_year"], apps["publication_year"])
    return apps


def read_assignees(pg_assignee_path: Path) -> pd.DataFrame:
    cols = ["pgpub_id", "assignee_id", "disambig_assignee_organization", "assignee_type"]
    assignees = pd.read_csv(pg_assignee_path, sep="\t", usecols=lambda c: c in cols, low_memory=False)
    assignees["pgpub_id"] = pd.to_numeric(assignees["pgpub_id"], errors="coerce").astype("Int64")
    assignees = assignees.loc[
        assignees["pgpub_id"].notna() & assignees["disambig_assignee_organization"].notna()
    ].copy()
    assignees["normalized_assignee_name"] = assignees["disambig_assignee_organization"].map(normalize_name)
    assignees = assignees.loc[assignees["normalized_assignee_name"] != ""].copy()
    return assignees.drop_duplicates(subset=["pgpub_id", "assignee_id", "normalized_assignee_name"])


def match_assignees_to_firms(assignees: pd.DataFrame, firm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    firm_lookup = firm.loc[firm["candidate_ciks"] == 1].copy()
    firm_lookup = firm_lookup.drop_duplicates(subset=["normalized_firm_name"])
    matched = assignees.merge(
        firm_lookup,
        left_on="normalized_assignee_name",
        right_on="normalized_firm_name",
        how="left",
        suffixes=("", "_firm"),
    )
    matched["cik"] = matched["cik"].map(normalize_cik_value)
    matched.loc[matched["cik"] == "", "cik"] = np.nan
    matched["firm_match_method"] = np.where(matched["cik"].notna(), "exact_normalized_name", "")

    audit = (
        matched.groupby(["normalized_assignee_name", "disambig_assignee_organization"], dropna=False)
        .agg(
            patent_application_rows=("pgpub_id", "nunique"),
            matched_cik=("cik", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
            matched_firm_name=("firm_name", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
            firm_match_method=("firm_match_method", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
        )
        .reset_index()
        .sort_values("patent_application_rows", ascending=False)
    )
    return matched, audit


def build_patent_counts(apps: pd.DataFrame, ai: pd.DataFrame, matched_assignees: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    app_ai = apps.merge(ai, on="pgpub_id", how="left")
    for col in ["predict50_any_ai", "predict86_any_ai", "predict93_any_ai"]:
        app_ai[col] = app_ai[col].fillna(0).astype(int)
    app_ai["has_aipd_prediction"] = app_ai["pub_dt"].notna()

    firm_apps = matched_assignees.loc[matched_assignees["cik"].notna()].merge(app_ai, on="pgpub_id", how="inner")
    firm_apps = firm_apps.loc[firm_apps["patent_year"] != ""].copy()
    firm_apps["cik"] = firm_apps["cik"].map(normalize_cik_value)
    firm_apps["year"] = firm_apps["patent_year"].astype(str)
    firm_apps = firm_apps.drop_duplicates(subset=["cik", "year", "pgpub_id"])

    counts = (
        firm_apps.groupby(["cik", "firm_name", "firm_ticker", "compustat_gvkey", "compustat_cusip", "crsp_permno", "year"], dropna=False)
        .agg(
            total_patent_app_count=("pgpub_id", "nunique"),
            aipd_matched_patent_app_count=("has_aipd_prediction", "sum"),
            ai_patent_app_count_50=("predict50_any_ai", "sum"),
            ai_patent_app_count_86=("predict86_any_ai", "sum"),
            ai_patent_app_count_93=("predict93_any_ai", "sum"),
        )
        .reset_index()
    )
    counts["ai_patent_app_share_50"] = np.where(
        counts["total_patent_app_count"] > 0, counts["ai_patent_app_count_50"] / counts["total_patent_app_count"], 0.0
    )
    counts["ai_patent_app_share_86"] = np.where(
        counts["total_patent_app_count"] > 0, counts["ai_patent_app_count_86"] / counts["total_patent_app_count"], 0.0
    )
    counts["ai_patent_app_share_93"] = np.where(
        counts["total_patent_app_count"] > 0, counts["ai_patent_app_count_93"] / counts["total_patent_app_count"], 0.0
    )
    counts["has_ai_patent_app_93"] = (counts["ai_patent_app_count_93"] > 0).astype(int)
    return counts, firm_apps


def merge_with_10k_outputs(out_dir: Path, counts: pd.DataFrame, tenk_package_dir: Path) -> None:
    files = [
        "DOCUMENT_filing_standardized.csv",
        "ITEM_1_filing_standardized.csv",
        "ITEM_1A_filing_standardized.csv",
        "ITEM_1C_filing_standardized.csv",
    ]
    patent_cols = [
        "cik",
        "year",
        "total_patent_app_count",
        "aipd_matched_patent_app_count",
        "ai_patent_app_count_50",
        "ai_patent_app_count_86",
        "ai_patent_app_count_93",
        "ai_patent_app_share_50",
        "ai_patent_app_share_86",
        "ai_patent_app_share_93",
        "has_ai_patent_app_93",
    ]
    patent = counts[patent_cols].copy()
    for filename in files:
        path = tenk_package_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype={"cik": str, "year": str}, low_memory=False)
        df["cik"] = normalize_cik(df["cik"])
        merged = df.merge(patent, on=["cik", "year"], how="left")
        fill_cols = [col for col in patent_cols if col not in ["cik", "year"]]
        merged[fill_cols] = merged[fill_cols].fillna(0)
        merged.to_csv(out_dir / filename.replace(".csv", "_with_patents.csv"), index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build firm-year AI patent application counts and merge with 10-K AI variables.")
    parser.add_argument("--aipd", default=str(DEFAULT_AIPD), help="USPTO AIPD ai_model_predictions.csv")
    parser.add_argument("--pg-app", default=str(DEFAULT_PG_APP), help="PatentsView pg_published_application.tsv")
    parser.add_argument("--pg-assignee", default=str(DEFAULT_PG_ASSIGNEE), help="PatentsView pg_assignee_disambiguated.tsv")
    parser.add_argument("--tenk", default=str(DEFAULT_TENK), help="10-K document filing file used as firm universe")
    parser.add_argument("--compustat", default=str(DEFAULT_COMPUSTAT), help="Compustat annual file with CIK/company names")
    parser.add_argument("--tenk-package-dir", default=str(DEFAULT_TENK.parent), help="Folder of standardized filing-level 10-K files")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--chunksize", type=int, default=500_000, help="AIPD CSV chunk size")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    firm = read_firm_universe(Path(args.tenk), Path(args.compustat))
    apps = read_applications(Path(args.pg_app))
    assignees = read_assignees(Path(args.pg_assignee))
    ai = read_ai_predictions(Path(args.aipd), args.chunksize)
    matched_assignees, match_audit = match_assignees_to_firms(assignees, firm)
    counts, firm_apps = build_patent_counts(apps, ai, matched_assignees)

    firm.to_csv(out_dir / "firm_name_universe_for_patent_matching.csv", index=False)
    match_audit.to_csv(out_dir / "patent_assignee_name_match_audit.csv", index=False)
    firm_apps.to_csv(out_dir / "matched_patent_applications_detail.csv", index=False)
    counts.to_csv(out_dir / "firm_year_ai_patent_application_counts.csv", index=False)
    merge_with_10k_outputs(out_dir, counts, Path(args.tenk_package_dir))

    summary = pd.DataFrame(
        [
            {"metric": "firm_name_universe_rows", "value": len(firm)},
            {"metric": "pg_application_rows", "value": len(apps)},
            {"metric": "pg_assignee_rows", "value": len(assignees)},
            {"metric": "aipd_pregrant_prediction_rows", "value": len(ai)},
            {"metric": "matched_patent_application_rows", "value": len(firm_apps)},
            {"metric": "firm_year_rows_with_patents", "value": len(counts)},
            {"metric": "total_patent_app_count_matched", "value": int(counts["total_patent_app_count"].sum())},
            {"metric": "ai_patent_app_count_93_matched", "value": int(counts["ai_patent_app_count_93"].sum())},
            {
                "metric": "assignee_rows_matched_exact_name_share",
                "value": float(matched_assignees["cik"].notna().mean()),
            },
        ]
    )
    summary.to_csv(out_dir / "patent_validation_summary.csv", index=False)

    readme = """# Patent Validation Outputs

This folder links PatentsView pre-grant patent applications to the 10-K AI disclosure variables.

Main patent count file:

- `firm_year_ai_patent_application_counts.csv`: firm-year patent application counts matched to public firms by normalized assignee/company name.

Main merged files:

- `DOCUMENT_filing_standardized_with_patents.csv`
- `ITEM_1_filing_standardized_with_patents.csv`
- `ITEM_1A_filing_standardized_with_patents.csv`
- `ITEM_1C_filing_standardized_with_patents.csv`

Patent variables:

- `total_patent_app_count`: total pre-grant patent applications matched to the firm-year.
- `ai_patent_app_count_93`: conservative AI patent application count using AIPD `predict93_any_ai`.
- `ai_patent_app_share_93`: AI patent applications divided by total patent applications.
- `has_ai_patent_app_93`: indicator equal to 1 if the firm-year has at least one AI patent application.
- `ai_patent_app_count_86` and `ai_patent_app_count_50`: less conservative robustness thresholds.

Matching:

PatentsView assignee organization names were normalized and matched to the 10-K/Compustat firm universe by exact normalized name. The file `patent_assignee_name_match_audit.csv` should be reviewed for match coverage and possible high-volume unmatched organizations.

Important limitation:

This first version uses exact normalized name matching. It is conservative and avoids many false matches, but it can miss firms where the patent assignee name differs from the Compustat/SEC name. A later version can add fuzzy matching plus manual review.
"""
    (out_dir / "README_patent_validation.md").write_text(readme, encoding="utf-8")
    print(f"Wrote patent validation outputs to {out_dir}")


if __name__ == "__main__":
    main()
