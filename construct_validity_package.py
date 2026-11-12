from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "analysis" / "construct_validity"
ITEM1_DIR = ROOT / "analysis" / "item1_ai_exposure"
SECTION_DIR = ROOT / "analysis" / "results_sections_sample_style"
COMPUSTAT_ANNUAL = ROOT / "ncw9xm2krtoqavrc.csv"


NEW_MEASURES = [
    "item1_ai_exposure_ratio",
    "explicit_ai_exposure_ratio",
    "ai_exposure_clean_ratio",
    "ai_expertise_ratio",
    "ai_expertise_share_of_ai",
    "topic_breadth",
    "noise_share_of_ai",
]

OLD_MEASURES = [
    "doc_ai_ratio",
    "doc_ai_and_risk_ratio",
    "doc_ai_risk_index",
    "doc_negative_ratio",
    "item1a_ai_ratio",
    "item1a_ai_and_risk_ratio",
    "item1a_ai_risk_index",
    "item1a_negative_ratio",
]

VALIDATION_VARS = [
    "log_assets",
    "log_market_value",
    "roa",
    "sales_growth",
    "profit_margin",
    "market_to_assets",
]


def safe_div(numerator: float | pd.Series, denominator: float | pd.Series):
    return np.where(denominator != 0, numerator / denominator, 0.0)


def year_from_yyyymmdd(series: pd.Series) -> pd.Series:
    return series.astype(str).str.slice(0, 4)


def read_item1_measures() -> pd.DataFrame:
    df = pd.read_csv(ITEM1_DIR / "item1_company_year_exposure_expertise.csv", dtype={"cik": str, "year": str})
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    return df


def read_document_measures() -> pd.DataFrame:
    usecols = [
        "cik",
        "company_name",
        "period_end",
        "total_sentences",
        "ai_related_sentences_count",
        "risk_related_sentences_count",
        "ai_risk_sentences_count",
        "positive_count",
        "negative_count",
        "neutral_count",
        "crsp_match_accepted",
    ]
    df = pd.read_csv(SECTION_DIR / "document_level.csv", usecols=usecols, dtype={"cik": str, "period_end": str})
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    df["year"] = year_from_yyyymmdd(df["period_end"])
    grouped = (
        df.groupby(["cik", "year"], dropna=False)
        .agg(
            doc_company_name=("company_name", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
            doc_filings=("period_end", "count"),
            doc_total_sentences=("total_sentences", "sum"),
            doc_ai_count=("ai_related_sentences_count", "sum"),
            doc_risk_count=("risk_related_sentences_count", "sum"),
            doc_ai_risk_count=("ai_risk_sentences_count", "sum"),
            doc_positive_count=("positive_count", "sum"),
            doc_negative_count=("negative_count", "sum"),
            doc_neutral_count=("neutral_count", "sum"),
            crsp_match_accepted=("crsp_match_accepted", "max"),
        )
        .reset_index()
    )
    sentiment_total = grouped["doc_positive_count"] + grouped["doc_negative_count"] + grouped["doc_neutral_count"]
    grouped["doc_ai_ratio"] = safe_div(grouped["doc_ai_count"], grouped["doc_total_sentences"])
    grouped["doc_ai_and_risk_ratio"] = safe_div(grouped["doc_ai_risk_count"], grouped["doc_total_sentences"])
    grouped["doc_ai_risk_index"] = safe_div(grouped["doc_ai_risk_count"], grouped["doc_ai_count"])
    grouped["doc_negative_ratio"] = safe_div(grouped["doc_negative_count"], sentiment_total)
    return grouped


def read_section_year(section_file: str, prefix: str) -> pd.DataFrame:
    usecols = [
        "cik",
        "company_name",
        "period_end",
        "total_sentences",
        "ai_related_sentences_count",
        "risk_related_sentences_count",
        "ai_risk_sentences_count",
        "positive_count",
        "negative_count",
        "neutral_count",
    ]
    path = SECTION_DIR / section_file
    df = pd.read_csv(path, usecols=usecols, dtype={"cik": str, "period_end": str}, low_memory=False)
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    df["year"] = year_from_yyyymmdd(df["period_end"])
    grouped = (
        df.groupby(["cik", "year"], dropna=False)
        .agg(
            section_company_name=("company_name", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
            total=("total_sentences", "sum"),
            ai=("ai_related_sentences_count", "sum"),
            risk=("risk_related_sentences_count", "sum"),
            ai_risk=("ai_risk_sentences_count", "sum"),
            positive=("positive_count", "sum"),
            negative=("negative_count", "sum"),
            neutral=("neutral_count", "sum"),
        )
        .reset_index()
    )
    sentiment_total = grouped["positive"] + grouped["negative"] + grouped["neutral"]
    out = grouped[["cik", "year"]].copy()
    out[f"{prefix}_total_sentences"] = grouped["total"]
    out[f"{prefix}_ai_count"] = grouped["ai"]
    out[f"{prefix}_ai_risk_count"] = grouped["ai_risk"]
    out[f"{prefix}_ai_ratio"] = safe_div(grouped["ai"], grouped["total"])
    out[f"{prefix}_ai_and_risk_ratio"] = safe_div(grouped["ai_risk"], grouped["total"])
    out[f"{prefix}_ai_risk_index"] = safe_div(grouped["ai_risk"], grouped["ai"])
    out[f"{prefix}_negative_ratio"] = safe_div(grouped["negative"], sentiment_total)
    return out


def read_compustat_validation() -> pd.DataFrame:
    cols = ["cik", "fyear", "at", "ni", "revt", "sale", "mkvalt", "prcc_f", "sic", "naics"]
    df = pd.read_csv(COMPUSTAT_ANNUAL, usecols=cols, dtype={"cik": str, "fyear": str, "sic": str, "naics": str})
    df["cik"] = df["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.lstrip("0")
    df["year"] = df["fyear"].astype(str).str.replace(r"\.0$", "", regex=True)
    for col in ["at", "ni", "revt", "sale", "mkvalt", "prcc_f"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Collapse occasional duplicate CIK-years conservatively by averaging numeric fields.
    grouped = (
        df.groupby(["cik", "year"], dropna=False)
        .agg(
            compustat_obs=("year", "count"),
            at=("at", "mean"),
            ni=("ni", "mean"),
            revt=("revt", "mean"),
            sale=("sale", "mean"),
            mkvalt=("mkvalt", "mean"),
            prcc_f=("prcc_f", "mean"),
            sic=("sic", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
            naics=("naics", lambda s: s.dropna().astype(str).iloc[0] if s.dropna().size else ""),
        )
        .reset_index()
        .sort_values(["cik", "year"])
    )
    grouped["log_assets"] = np.log1p(grouped["at"].where(grouped["at"] > 0))
    grouped["log_market_value"] = np.log1p(grouped["mkvalt"].where(grouped["mkvalt"] > 0))
    grouped["roa"] = grouped["ni"] / grouped["at"].replace(0, np.nan)
    grouped["profit_margin"] = grouped["ni"] / grouped["sale"].replace(0, np.nan)
    grouped["market_to_assets"] = grouped["mkvalt"] / grouped["at"].replace(0, np.nan)
    grouped["sales_growth"] = grouped.groupby("cik")["sale"].pct_change()
    grouped.loc[~np.isfinite(grouped["sales_growth"]), "sales_growth"] = np.nan
    return grouped


def correlation_long(df: pd.DataFrame, left_cols: list[str], right_cols: list[str], group_col: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("all", df)] if group_col is None else list(df.groupby(group_col, dropna=False))
    for group, data in groups:
        for left in left_cols:
            for right in right_cols:
                sub = data[[left, right]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(sub) < 30 or sub[left].nunique() < 2 or sub[right].nunique() < 2:
                    pearson = np.nan
                    spearman = np.nan
                    n = len(sub)
                else:
                    pearson = sub[left].corr(sub[right], method="pearson")
                    spearman = sub[left].corr(sub[right], method="spearman")
                    n = len(sub)
                rows.append(
                    {
                        "group": group,
                        "left_measure": left,
                        "right_measure": right,
                        "n": n,
                        "pearson": pearson,
                        "spearman": spearman,
                    }
                )
    return pd.DataFrame(rows)


def measure_descriptives_by_year(df: pd.DataFrame) -> pd.DataFrame:
    measures = NEW_MEASURES + ["doc_ai_ratio", "doc_ai_and_risk_ratio", "doc_ai_risk_index", "doc_negative_ratio"]
    rows = []
    for year, data in df.groupby("year", dropna=False):
        row = {"year": year, "firm_years": len(data)}
        for measure in measures:
            s = pd.to_numeric(data[measure], errors="coerce")
            row[f"{measure}_mean"] = s.mean()
            row[f"{measure}_median"] = s.median()
            row[f"{measure}_p90"] = s.quantile(0.9)
            row[f"{measure}_nonzero_share"] = (s > 0).mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("year")


def decile_validation_table(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    data = df.copy()
    positive = data[pd.to_numeric(data[measure], errors="coerce") > 0].copy()
    if positive.empty:
        return pd.DataFrame()
    positive["decile"] = pd.qcut(positive[measure].rank(method="first"), 10, labels=False) + 1
    rows = []
    for decile, sub in positive.groupby("decile"):
        row = {
            "measure": measure,
            "decile": int(decile),
            "firm_years": len(sub),
            f"{measure}_mean": sub[measure].mean(),
        }
        for var in VALIDATION_VARS:
            row[f"{var}_mean"] = pd.to_numeric(sub[var], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def top_firm_years(df: pd.DataFrame, measure: str, n: int = 50) -> pd.DataFrame:
    keep = [
        "cik",
        "company_name",
        "year",
        "item1_total_sentences",
        "item1_ai_sentences_count",
        "topic_model_ai_sentences",
        "explicit_ai_sentences",
        "expertise_ai_sentences",
        "exposure_ai_sentences",
        "topic_breadth",
        "excluded_noise_sentences",
        "item1_ai_exposure_ratio",
        "explicit_ai_exposure_ratio",
        "ai_exposure_clean_ratio",
        "ai_expertise_ratio",
        "ai_expertise_share_of_ai",
        "noise_share_of_ai",
        "doc_ai_ratio",
        "doc_ai_and_risk_ratio",
        "doc_ai_risk_index",
        "sic",
        "naics",
        "at",
        "mkvalt",
        "roa",
        "sales_growth",
    ]
    return df.sort_values(measure, ascending=False)[keep].head(n)


def write_summary(df: pd.DataFrame, corr_old_new: pd.DataFrame, corr_validation: pd.DataFrame) -> None:
    topic_df = pd.read_csv(ITEM1_DIR / "item1_ai_topics.csv")
    latest_year = df["year"].dropna().astype(str).max()
    lines = [
        "# Construct Validity Package",
        "",
        "## Purpose",
        "",
        "This package reframes the strongest disclosure result as AI exposure / AI capability rather than pure AI risk.",
        "It focuses on ITEM_1 because that section describes the business model, products, operations, customers, and capabilities.",
        "",
        "## Inputs",
        "",
        "- `analysis/item1_ai_exposure/`: ITEM_1 AI sentence topic assignments and exposure/expertise measures.",
        "- `analysis/results_sections_sample_style/document_level.csv`: filing-level AI/risk/sentiment measures.",
        "- `ncw9xm2krtoqavrc.csv`: Compustat annual accounting and market fields available locally.",
        "",
        "## Main Output Files",
        "",
        "- `construct_validity_firm_year.csv`: merged firm-year dataset for professor-side regressions.",
        "- `measure_correlations_old_vs_new.csv`: correlations between new ITEM_1 measures and old AI/risk measures.",
        "- `validation_compustat_correlations.csv`: correlations with local accounting/market proxies.",
        "- `validation_deciles.csv`: validation variables by deciles of clean exposure and expertise.",
        "- `top_firm_years_by_expertise.csv` and `top_firm_years_by_clean_exposure.csv`: sanity-check lists.",
        "- `topic_construct_map.csv` and `topic_examples_for_audit.csv`: manual construct audit material.",
        "",
        "## Key Counts",
        "",
        f"- Firm-year observations: {len(df):,}",
        f"- Latest year in package: {latest_year}",
        f"- Firm-years with positive ITEM_1 AI exposure: {(df['item1_ai_exposure_ratio'] > 0).sum():,}",
        f"- Firm-years with positive clean AI exposure: {(df['ai_exposure_clean_ratio'] > 0).sum():,}",
        f"- Firm-years with positive AI expertise: {(df['ai_expertise_ratio'] > 0).sum():,}",
        "",
        "## Topic Interpretation",
        "",
    ]
    for _, row in topic_df.sort_values("topic_id").iterrows():
        lines.append(
            f"- Topic {int(row['topic_id'])}: {row['manual_label']} "
            f"({row['dimension']}; expertise={row['counts_as_expertise']}; exposure={row['counts_as_exposure']}; "
            f"sentences={int(row['sentence_count']):,})"
        )
    lines.extend(
        [
            "",
            "## Recommended Next Step For Professor-Side Tests",
            "",
            "Run the existing Fama-MacBeth and portfolio pipeline with these variables:",
            "",
            "1. `item1_ai_exposure_ratio` as the broad baseline.",
            "2. `ai_exposure_clean_ratio` as the noise-filtered exposure proxy.",
            "3. `ai_expertise_ratio` as the capability/expertise proxy.",
            "4. `ai_expertise_share_of_ai` as the composition proxy among AI-discussing firms.",
            "5. `noise_share_of_ai` as a rhetoric/noise control or exclusion screen.",
            "",
            "A strong result on `ai_exposure_clean_ratio` or `ai_expertise_ratio` would support the revised mispricing/opportunity story more directly than the old risk framing.",
            "",
            "## Limitation",
            "",
            "The local Compustat files do not contain R&D or patent variables. Those should be added later if available, because they are better construct-validity validators for true AI capability.",
        ]
    )
    (OUT_DIR / "construct_validity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    item1 = read_item1_measures()
    document = read_document_measures()
    item1a = read_section_year("ITEM_1A.csv", "item1a")
    item1c = read_section_year("ITEM_1C.csv", "item1c")
    compustat = read_compustat_validation()

    df = item1.merge(
        document,
        on=["cik", "year"],
        how="left",
        suffixes=("", "_doc"),
    )
    df = df.merge(item1a, on=["cik", "year"], how="left")
    df = df.merge(item1c, on=["cik", "year"], how="left")
    df = df.merge(compustat, on=["cik", "year"], how="left")

    ratio_cols = [col for col in df.columns if col.endswith("_ratio") or col.endswith("_index") or col in ["topic_breadth"]]
    for col in ratio_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df.to_csv(OUT_DIR / "construct_validity_firm_year.csv", index=False)

    topic_df = pd.read_csv(ITEM1_DIR / "item1_ai_topics.csv")
    topic_df.to_csv(OUT_DIR / "topic_construct_map.csv", index=False)
    examples = pd.read_csv(ITEM1_DIR / "item1_ai_topic_examples.csv")
    examples.to_csv(OUT_DIR / "topic_examples_for_audit.csv", index=False)

    measure_descriptives_by_year(df).to_csv(OUT_DIR / "measure_descriptives_by_year.csv", index=False)

    corr_old_new = correlation_long(df, NEW_MEASURES, OLD_MEASURES)
    corr_old_new.to_csv(OUT_DIR / "measure_correlations_old_vs_new.csv", index=False)

    corr_validation = correlation_long(df, NEW_MEASURES, VALIDATION_VARS)
    corr_validation.to_csv(OUT_DIR / "validation_compustat_correlations.csv", index=False)

    by_year_corr = correlation_long(df, ["ai_exposure_clean_ratio", "ai_expertise_ratio"], OLD_MEASURES + VALIDATION_VARS, "year")
    by_year_corr.to_csv(OUT_DIR / "yearly_correlations_key_measures.csv", index=False)

    deciles = pd.concat(
        [
            decile_validation_table(df, "ai_exposure_clean_ratio"),
            decile_validation_table(df, "ai_expertise_ratio"),
        ],
        ignore_index=True,
    )
    deciles.to_csv(OUT_DIR / "validation_deciles.csv", index=False)

    top_firm_years(df, "ai_expertise_ratio").to_csv(OUT_DIR / "top_firm_years_by_expertise.csv", index=False)
    top_firm_years(df, "ai_exposure_clean_ratio").to_csv(OUT_DIR / "top_firm_years_by_clean_exposure.csv", index=False)
    top_firm_years(df[df["year"].isin(["2024", "2025", "2026"])], "ai_expertise_ratio").to_csv(
        OUT_DIR / "top_recent_firm_years_by_expertise.csv", index=False
    )

    write_summary(df, corr_old_new, corr_validation)
    print(f"Wrote construct-validity outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
