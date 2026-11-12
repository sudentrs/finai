from __future__ import annotations

import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from patent_validation_pipeline_improved import DEFAULT_COMPUSTAT, DEFAULT_TENK, read_firm_universe


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "analysis" / "patent_validation_improved"
OUT_DIR = INPUT_DIR / "extended_validation_analysis"
COMPUSTAT_PATH = DEFAULT_COMPUSTAT

SCOPE_FILES = {
    "DOCUMENT": INPUT_DIR / "DOCUMENT_filing_standardized_with_patents.csv",
    "ITEM_1": INPUT_DIR / "ITEM_1_filing_standardized_with_patents.csv",
    "ITEM_1A": INPUT_DIR / "ITEM_1A_filing_standardized_with_patents.csv",
}

BASE_AI_VARS = [
    "ai_expertise_ratio",
    "ai_risk_governance_ratio",
    "ai_expertise_share",
    "ai_risk_governance_share",
    "ai_noise_ratio",
]

PATENT_BASE = [
    "ai_patent_app_count_93",
    "ai_patent_app_share_93",
    "has_ai_patent_app_93",
    "log1p_ai_patent_app_count_93",
]

PATENT_LAG_LEAD = [
    "ai_patent_app_count_93_t_minus_2",
    "ai_patent_app_count_93_t_minus_1",
    "ai_patent_app_count_93_t_plus_1",
    "ai_patent_app_count_93_t_plus_2",
    "ai_patent_app_share_93_t_minus_2",
    "ai_patent_app_share_93_t_minus_1",
    "ai_patent_app_share_93_t_plus_1",
    "ai_patent_app_share_93_t_plus_2",
    "has_ai_patent_app_93_t_minus_1",
    "has_ai_patent_app_93_t_plus_1",
]


def normalize_cik(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")


def safe_qcut(series: pd.Series, q: int = 5) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=series.index)
    valid = numeric.notna()
    if valid.sum() < q or numeric[valid].nunique() < 2:
        return out
    ranks = numeric[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranks, q, labels=list(range(1, q + 1))).astype(float)
    return out


def correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    sub = pd.DataFrame({"left": left, "right": right}).apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 30 or sub["left"].nunique() < 2 or sub["right"].nunique() < 2:
        return np.nan
    return float(sub["left"].corr(sub["right"], method=method))


def read_compustat_characteristics() -> pd.DataFrame:
    cols = ["cik", "fyear", "at", "ni", "revt", "sale", "mkvalt", "prcc_f", "sic", "naics"]
    df = pd.read_csv(COMPUSTAT_PATH, usecols=cols, dtype={"cik": str, "fyear": str}, low_memory=False)
    df["cik"] = normalize_cik(df["cik"])
    df["year"] = df["fyear"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in ["at", "ni", "revt", "sale", "mkvalt", "prcc_f"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = (
        df.groupby(["cik", "year"], dropna=False)
        .agg(
            at=("at", "mean"),
            ni=("ni", "mean"),
            revt=("revt", "mean"),
            sale=("sale", "mean"),
            mkvalt=("mkvalt", "mean"),
            prcc_f=("prcc_f", "mean"),
            compustat_sic=("sic", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
            compustat_naics=("naics", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
        )
        .reset_index()
    )
    grouped["log_market_value"] = np.log1p(grouped["mkvalt"].where(grouped["mkvalt"] > 0))
    grouped["log_assets"] = np.log1p(grouped["at"].where(grouped["at"] > 0))
    grouped["market_to_assets"] = grouped["mkvalt"] / grouped["at"].replace(0, np.nan)
    grouped["roa"] = grouped["ni"] / grouped["at"].replace(0, np.nan)
    grouped["profit_margin"] = grouped["ni"] / grouped["sale"].replace(0, np.nan)
    grouped["sales_growth"] = grouped.sort_values(["cik", "year"]).groupby("cik")["sale"].pct_change()
    grouped.loc[~np.isfinite(grouped["sales_growth"]), "sales_growth"] = np.nan
    return grouped


def read_patent_panel_with_leads() -> pd.DataFrame:
    p = INPUT_DIR / "firm_year_ai_patent_application_counts.csv"
    patent = pd.read_csv(p, dtype={"cik": str, "year": str}, low_memory=False)
    patent["cik"] = normalize_cik(patent["cik"])
    patent["year_num"] = pd.to_numeric(patent["year"], errors="coerce")
    for col in ["ai_patent_app_count_93", "ai_patent_app_share_93", "has_ai_patent_app_93", "total_patent_app_count"]:
        patent[col] = pd.to_numeric(patent[col], errors="coerce").fillna(0)
    patent["log1p_ai_patent_app_count_93"] = np.log1p(patent["ai_patent_app_count_93"])
    patent["log1p_total_patent_app_count"] = np.log1p(patent["total_patent_app_count"])

    keep = ["cik", "year_num", "ai_patent_app_count_93", "ai_patent_app_share_93", "has_ai_patent_app_93"]
    shifted_pieces = []
    for shift in [-2, -1, 1, 2]:
        shifted = patent[keep].copy()
        # If disclosure is in year t, t+1 patenting is observed in patent year t+1.
        # Therefore the row should merge back to disclosure year patent_year - 1.
        shifted["year_num"] = shifted["year_num"] - shift
        shifted["year"] = shifted["year_num"].astype("Int64").astype(str).replace("<NA>", "")
        suffix = f"t_plus_{shift}" if shift > 0 else f"t_minus_{abs(shift)}"
        shifted = shifted.rename(
            columns={
                "ai_patent_app_count_93": f"ai_patent_app_count_93_{suffix}",
                "ai_patent_app_share_93": f"ai_patent_app_share_93_{suffix}",
                "has_ai_patent_app_93": f"has_ai_patent_app_93_{suffix}",
            }
        )
        shifted_pieces.append(shifted)

    panel = None
    for shifted in shifted_pieces:
        use = shifted.drop(columns=["year_num"]).drop_duplicates(subset=["cik", "year"])
        panel = use if panel is None else panel.merge(use, on=["cik", "year"], how="outer")
    return panel if panel is not None else pd.DataFrame(columns=["cik", "year"])


def add_industry_adjusted(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sic2"] = out["sic"].astype(str).str.extract(r"(\d{2})", expand=False)
    for var in BASE_AI_VARS:
        out[f"{var}_industry_year_mean"] = out.groupby(["sic2", "year"], dropna=False)[var].transform("mean")
        out[f"{var}_industry_adjusted"] = out[var] - out[f"{var}_industry_year_mean"]
    return out


def load_scope(scope: str, characteristics: pd.DataFrame, patent_leads: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(SCOPE_FILES[scope], dtype={"cik": str, "year": str}, low_memory=False)
    df["scope"] = scope
    df["cik"] = normalize_cik(df["cik"])
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    for col in BASE_AI_VARS + ["ai_sentences_count", "crsp_match_accepted", "total_patent_app_count", "ai_patent_app_count_93", "ai_patent_app_share_93"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["log1p_ai_patent_app_count_93"] = np.log1p(df["ai_patent_app_count_93"].fillna(0))
    df["log1p_total_patent_app_count"] = np.log1p(df["total_patent_app_count"].fillna(0))

    lead_cols = ["cik", "year"] + PATENT_LAG_LEAD
    df = df.drop(columns=[c for c in PATENT_LAG_LEAD if c in df.columns], errors="ignore")
    df = df.merge(patent_leads[lead_cols].drop_duplicates(subset=["cik", "year"]), on=["cik", "year"], how="left")
    for col in PATENT_LAG_LEAD:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    df = df.merge(characteristics, on=["cik", "year"], how="left")
    df = add_industry_adjusted(df)

    df["market_cap_quintile"] = df.groupby("year", dropna=False)["mkvalt"].transform(safe_qcut)
    df["total_patent_count_quintile"] = df.groupby("year", dropna=False)["total_patent_app_count"].transform(safe_qcut)
    return df


def make_samples(df: pd.DataFrame, top_patent_ciks: set[str]) -> dict[str, pd.DataFrame]:
    years = df[(df["year_num"] >= 2015) & (df["year_num"] <= 2023)]
    years_ai_positive = years[years["ai_sentences_count"].fillna(0) > 0]
    return {
        "all": df,
        "ai_positive": df[df["ai_sentences_count"].fillna(0) > 0],
        "years_2015_2023": years,
        "ai_positive_years_2015_2023": years_ai_positive,
        "ai_positive_years_2015_2023_excluding_top_20_patenters": years_ai_positive[
            ~years_ai_positive["cik"].isin(top_patent_ciks)
        ],
        "crsp_matched": df[df["crsp_match_accepted"].fillna(0).astype(bool)],
        "excluding_top_20_patenters": df[~df["cik"].isin(top_patent_ciks)],
        "large_market_cap": df[df["market_cap_quintile"] >= 4],
        "small_market_cap": df[df["market_cap_quintile"] <= 2],
        "high_total_patenting": df[df["total_patent_count_quintile"] >= 4],
        "low_total_patenting": df[df["total_patent_count_quintile"] <= 2],
    }


def correlation_table(scope: str, sample_name: str, data: pd.DataFrame) -> list[dict]:
    x_vars = BASE_AI_VARS + [f"{v}_industry_adjusted" for v in BASE_AI_VARS]
    y_vars = PATENT_BASE + PATENT_LAG_LEAD
    rows = []
    for x in x_vars:
        for y in y_vars:
            sub = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            rows.append(
                {
                    "scope": scope,
                    "sample": sample_name,
                    "explanatory_variable": x,
                    "patent_outcome": y,
                    "n": len(sub),
                    "pearson": correlation(sub[x], sub[y], "pearson") if len(sub) else np.nan,
                    "spearman": correlation(sub[x], sub[y], "spearman") if len(sub) else np.nan,
                    "x_mean": float(sub[x].mean()) if len(sub) else np.nan,
                    "y_mean": float(sub[y].mean()) if len(sub) else np.nan,
                    "x_nonzero_share": float((sub[x] != 0).mean()) if len(sub) else np.nan,
                    "y_nonzero_share": float((sub[y] != 0).mean()) if len(sub) else np.nan,
                }
            )
    return rows


def quintile_table(scope: str, sample_name: str, data: pd.DataFrame) -> list[dict]:
    sort_vars = [
        "ai_expertise_ratio",
        "ai_risk_governance_ratio",
        "ai_expertise_ratio_industry_adjusted",
        "ai_risk_governance_ratio_industry_adjusted",
    ]
    outcomes = [
        "ai_patent_app_count_93",
        "ai_patent_app_share_93",
        "has_ai_patent_app_93",
        "log1p_ai_patent_app_count_93",
        "ai_patent_app_count_93_t_plus_1",
        "ai_patent_app_count_93_t_plus_2",
    ]
    rows = []
    for var in sort_vars:
        work = data.copy()
        work["quintile"] = safe_qcut(work[var])
        work = work.dropna(subset=["quintile"])
        for quintile, group in work.groupby("quintile", dropna=False):
            row = {
                "scope": scope,
                "sample": sample_name,
                "sort_variable": var,
                "quintile": int(quintile),
                "n": len(group),
                "sort_variable_mean": group[var].mean(),
                "sort_variable_median": group[var].median(),
            }
            for outcome in outcomes:
                row[f"{outcome}_mean"] = pd.to_numeric(group[outcome], errors="coerce").mean()
                row[f"{outcome}_median"] = pd.to_numeric(group[outcome], errors="coerce").median()
            rows.append(row)
    return rows


def industry_distribution(scope: str, data: pd.DataFrame) -> pd.DataFrame:
    group = (
        data.groupby(["scope", "sic2", "year"], dropna=False)
        .agg(
            filings=("accession", "count"),
            firms=("cik", "nunique"),
            mean_ai_expertise_ratio=("ai_expertise_ratio", "mean"),
            mean_ai_risk_governance_ratio=("ai_risk_governance_ratio", "mean"),
            mean_ai_patent_app_count_93=("ai_patent_app_count_93", "mean"),
            total_ai_patent_app_count_93=("ai_patent_app_count_93", "sum"),
        )
        .reset_index()
    )
    return group


def characteristic_splits(scope: str, data: pd.DataFrame) -> list[dict]:
    rows = []
    split_vars = ["market_cap_quintile", "total_patent_count_quintile"]
    for split in split_vars:
        for bucket, group in data.dropna(subset=[split]).groupby(split, dropna=False):
            rows.append(
                {
                    "scope": scope,
                    "split_variable": split,
                    "bucket": int(bucket),
                    "n": len(group),
                    "mean_mkvalt": group["mkvalt"].mean(),
                    "mean_market_to_assets": group["market_to_assets"].mean(),
                    "mean_ai_expertise_ratio": group["ai_expertise_ratio"].mean(),
                    "mean_ai_risk_governance_ratio": group["ai_risk_governance_ratio"].mean(),
                    "mean_ai_patent_app_count_93": group["ai_patent_app_count_93"].mean(),
                    "mean_ai_patent_app_share_93": group["ai_patent_app_share_93"].mean(),
                }
            )
    return rows


def compact_professor_summary(correlations: pd.DataFrame, quintiles: pd.DataFrame) -> pd.DataFrame:
    rows = []
    focus_sample = "ai_positive_years_2015_2023"
    for scope in ["DOCUMENT", "ITEM_1", "ITEM_1A"]:
        for var in ["ai_expertise_ratio", "ai_risk_governance_ratio", "ai_expertise_share", "ai_risk_governance_share"]:
            row = {"scope": scope, "sample": focus_sample, "main_variable": var}
            for outcome in ["ai_patent_app_count_93", "ai_patent_app_share_93", "has_ai_patent_app_93"]:
                sub = correlations[
                    (correlations["scope"] == scope)
                    & (correlations["sample"] == focus_sample)
                    & (correlations["explanatory_variable"] == var)
                    & (correlations["patent_outcome"] == outcome)
                ]
                row[f"{outcome}_spearman"] = sub["spearman"].iloc[0] if len(sub) else np.nan
                row[f"{outcome}_pearson"] = sub["pearson"].iloc[0] if len(sub) else np.nan
            q = quintiles[
                (quintiles["scope"] == scope)
                & (quintiles["sample"] == focus_sample)
                & (quintiles["sort_variable"] == var)
            ]
            q1 = q[q["quintile"] == 1]
            q5 = q[q["quintile"] == 5]
            if len(q1) and len(q5):
                row["q1_ai_patent_app_count_93_mean"] = q1["ai_patent_app_count_93_mean"].iloc[0]
                row["q5_ai_patent_app_count_93_mean"] = q5["ai_patent_app_count_93_mean"].iloc[0]
                row["q5_minus_q1_ai_patent_app_count_93_mean"] = (
                    row["q5_ai_patent_app_count_93_mean"] - row["q1_ai_patent_app_count_93_mean"]
                )
                row["q1_ai_patent_app_share_93_mean"] = q1["ai_patent_app_share_93_mean"].iloc[0]
                row["q5_ai_patent_app_share_93_mean"] = q5["ai_patent_app_share_93_mean"].iloc[0]
                row["q5_minus_q1_ai_patent_app_share_93_mean"] = (
                    row["q5_ai_patent_app_share_93_mean"] - row["q1_ai_patent_app_share_93_mean"]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def fuzzy_candidates() -> pd.DataFrame:
    audit = pd.read_csv(INPUT_DIR / "patent_assignee_name_match_audit.csv", low_memory=False)
    unmatched = audit[audit["matched_cik"].isna() | (audit["matched_cik"].astype(str).str.strip() == "")].copy()
    unmatched = unmatched.sort_values("patent_application_rows", ascending=False).head(300)
    firm = read_firm_universe(DEFAULT_TENK, DEFAULT_COMPUSTAT)
    firm = firm[firm["cik"].notna() & (firm["cik"].astype(str).str.strip() != "")].copy()
    choices = firm["normalized_firm_name"].dropna().drop_duplicates().tolist()
    lookup = firm.drop_duplicates("normalized_firm_name").set_index("normalized_firm_name")
    rows = []
    for _, row in unmatched.iterrows():
        name = row["normalized_assignee_name"]
        matches = difflib.get_close_matches(name, choices, n=5, cutoff=0.72)
        for rank, match in enumerate(matches, start=1):
            f = lookup.loc[match]
            rows.append(
                {
                    "normalized_assignee_name": name,
                    "disambig_assignee_organization": row["disambig_assignee_organization"],
                    "patent_application_rows": row["patent_application_rows"],
                    "candidate_rank": rank,
                    "candidate_normalized_firm_name": match,
                    "candidate_firm_name": f["firm_name"],
                    "candidate_ticker": f.get("firm_ticker", ""),
                    "candidate_cik": f["cik"],
                    "similarity": difflib.SequenceMatcher(None, name, match).ratio(),
                }
            )
    return pd.DataFrame(rows).sort_values(["patent_application_rows", "candidate_rank"], ascending=[False, True])


def top_patenting_ciks(n: int = 20) -> set[str]:
    p = INPUT_DIR / "firm_year_ai_patent_application_counts.csv"
    patent = pd.read_csv(p, dtype={"cik": str}, usecols=["cik", "total_patent_app_count"], low_memory=False)
    patent["cik"] = normalize_cik(patent["cik"])
    return set(
        patent.groupby("cik")["total_patent_app_count"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index.astype(str)
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    characteristics = read_compustat_characteristics()
    patent_leads = read_patent_panel_with_leads()
    top_patent_ciks = top_patenting_ciks()

    all_corr = []
    all_quint = []
    all_industry = []
    all_splits = []

    for scope in SCOPE_FILES:
        df = load_scope(scope, characteristics, patent_leads)
        df.to_csv(OUT_DIR / f"{scope}_filing_with_lags_industry_adjusted.csv", index=False)
        for sample_name, sample in make_samples(df, top_patent_ciks).items():
            all_corr.extend(correlation_table(scope, sample_name, sample))
            if sample_name in {
                "all",
                "years_2015_2023",
                "ai_positive_years_2015_2023",
                "ai_positive_years_2015_2023_excluding_top_20_patenters",
                "excluding_top_20_patenters",
            }:
                all_quint.extend(quintile_table(scope, sample_name, sample))
        all_industry.append(industry_distribution(scope, df))
        all_splits.extend(characteristic_splits(scope, df))

    correlations = pd.DataFrame(all_corr)
    quintiles = pd.DataFrame(all_quint)
    industry = pd.concat(all_industry, ignore_index=True)
    splits = pd.DataFrame(all_splits)
    professor = compact_professor_summary(correlations, quintiles)
    candidates = fuzzy_candidates()

    correlations.to_csv(OUT_DIR / "extended_patent_validation_correlations.csv", index=False)
    quintiles.to_csv(OUT_DIR / "extended_patent_validation_quintiles.csv", index=False)
    industry.to_csv(OUT_DIR / "industry_year_ai_patent_distribution.csv", index=False)
    splits.to_csv(OUT_DIR / "company_characteristic_split_summary.csv", index=False)
    professor.to_csv(OUT_DIR / "professor_compact_validation_summary.csv", index=False)
    candidates.to_csv(OUT_DIR / "top_unmatched_assignee_fuzzy_candidates_for_manual_review.csv", index=False)

    readme = """# Extended Patent Validation Analysis

This folder extends the patent validation checks.

Added checks:

- Same-year, future, and past patent outcomes: t-2, t-1, t, t+1, and t+2.
- Industry-adjusted AI disclosure variables, subtracting the SIC2-year mean.
- Quintile tables for raw and industry-adjusted AI expertise/risk variables.
- Size and patent-intensity split summaries using available Compustat variables.
- Excluding top 20 total patenting firms as a robustness sample.
- Fuzzy candidate file for manual review of high-volume unmatched PatentsView assignees.

Data limitation:

The local Compustat file contains market value, assets, sales, net income, SIC and NAICS. It does not contain R&D expense or CapEx, so R&D intensity and CapEx intensity screens cannot be computed from the current WRDS file.
"""
    (OUT_DIR / "README_extended_patent_validation.md").write_text(readme, encoding="utf-8")
    print(f"Wrote extended patent validation outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
