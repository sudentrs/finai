from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "analysis" / "patent_validation_improved"
OUT_DIR = ROOT / "analysis" / "patent_validation_improved" / "validation_analysis"

SCOPE_FILES = {
    "DOCUMENT": INPUT_DIR / "DOCUMENT_filing_standardized_with_patents.csv",
    "ITEM_1": INPUT_DIR / "ITEM_1_filing_standardized_with_patents.csv",
    "ITEM_1A": INPUT_DIR / "ITEM_1A_filing_standardized_with_patents.csv",
}

EXPLANATORY_VARS = [
    "ai_expertise_ratio",
    "ai_risk_governance_ratio",
    "ai_expertise_share",
    "ai_risk_governance_share",
    "ai_noise_ratio",
]

PATENT_OUTCOMES = [
    "ai_patent_app_count_93",
    "ai_patent_app_share_93",
    "has_ai_patent_app_93",
    "total_patent_app_count",
    "log1p_ai_patent_app_count_93",
    "log1p_total_patent_app_count",
]


def safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_scope(scope: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"cik": str, "year": str}, low_memory=False)
    df["scope"] = scope
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    numeric_cols = EXPLANATORY_VARS + [
        "ai_sentences_count",
        "crsp_match_accepted",
        "ai_patent_app_count_93",
        "ai_patent_app_share_93",
        "has_ai_patent_app_93",
        "total_patent_app_count",
    ]
    df = safe_numeric(df, numeric_cols)
    df["log1p_ai_patent_app_count_93"] = np.log1p(df["ai_patent_app_count_93"].fillna(0))
    df["log1p_total_patent_app_count"] = np.log1p(df["total_patent_app_count"].fillna(0))
    return df


def subsamples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {
        "all": df,
        "ai_positive": df[df["ai_sentences_count"].fillna(0) > 0],
        "crsp_matched": df[df["crsp_match_accepted"].fillna(0).astype(bool)],
        "years_2015_2023": df[(df["year_num"] >= 2015) & (df["year_num"] <= 2023)],
    }
    out["ai_positive_years_2015_2023"] = out["years_2015_2023"][
        out["years_2015_2023"]["ai_sentences_count"].fillna(0) > 0
    ]
    return out


def correlation_rows(scope: str, sample_name: str, data: pd.DataFrame) -> list[dict]:
    rows = []
    for x in EXPLANATORY_VARS:
        for y in PATENT_OUTCOMES:
            sub = data[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < 30 or sub[x].nunique() < 2 or sub[y].nunique() < 2:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = sub[x].corr(sub[y], method="pearson")
                spearman = sub[x].corr(sub[y], method="spearman")
            rows.append(
                {
                    "scope": scope,
                    "sample": sample_name,
                    "explanatory_variable": x,
                    "patent_outcome": y,
                    "n": len(sub),
                    "pearson": pearson,
                    "spearman": spearman,
                    "x_nonzero_share": float((sub[x] > 0).mean()) if len(sub) else np.nan,
                    "y_nonzero_share": float((sub[y] > 0).mean()) if len(sub) else np.nan,
                    "x_mean": float(sub[x].mean()) if len(sub) else np.nan,
                    "y_mean": float(sub[y].mean()) if len(sub) else np.nan,
                }
            )
    return rows


def make_quintiles(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    out = pd.Series(np.nan, index=series.index)
    if valid.sum() < 5 or numeric[valid].nunique() < 2:
        return out
    ranks = numeric[valid].rank(method="first")
    out.loc[valid] = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5]).astype(float)
    return out


def quintile_rows(scope: str, sample_name: str, data: pd.DataFrame) -> list[dict]:
    rows = []
    for x in EXPLANATORY_VARS:
        work = data.copy()
        work["quintile"] = make_quintiles(work[x])
        work = work.dropna(subset=["quintile"])
        if work.empty:
            continue
        for q, group in work.groupby("quintile", dropna=False):
            row = {
                "scope": scope,
                "sample": sample_name,
                "sort_variable": x,
                "quintile": int(q),
                "n": len(group),
                "sort_variable_mean": group[x].mean(),
                "sort_variable_median": group[x].median(),
                "sort_variable_nonzero_share": (group[x] > 0).mean(),
            }
            for y in PATENT_OUTCOMES:
                row[f"{y}_mean"] = group[y].mean()
                row[f"{y}_median"] = group[y].median()
                row[f"{y}_nonzero_share"] = (group[y] > 0).mean()
            rows.append(row)
    return rows


def scope_summary(scope: str, sample_name: str, data: pd.DataFrame) -> dict:
    row = {
        "scope": scope,
        "sample": sample_name,
        "rows": len(data),
        "unique_ciks": data["cik"].nunique() if "cik" in data.columns else np.nan,
        "min_year": data["year_num"].min(),
        "max_year": data["year_num"].max(),
        "ai_positive_share": (data["ai_sentences_count"].fillna(0) > 0).mean(),
        "crsp_match_share": data["crsp_match_accepted"].fillna(0).astype(bool).mean(),
        "has_ai_patent_app_93_share": data["has_ai_patent_app_93"].fillna(0).astype(bool).mean(),
        "mean_ai_patent_app_count_93": data["ai_patent_app_count_93"].mean(),
        "mean_ai_patent_app_share_93": data["ai_patent_app_share_93"].mean(),
    }
    for x in EXPLANATORY_VARS:
        row[f"{x}_mean"] = data[x].mean()
        row[f"{x}_nonzero_share"] = (data[x] > 0).mean()
    return row


def top_signal_table(correlations: pd.DataFrame) -> pd.DataFrame:
    focus = correlations[
        (correlations["sample"].isin(["all", "ai_positive", "years_2015_2023", "ai_positive_years_2015_2023"]))
        & (correlations["patent_outcome"].isin(["ai_patent_app_count_93", "ai_patent_app_share_93", "has_ai_patent_app_93"]))
    ].copy()
    focus["abs_spearman"] = focus["spearman"].abs()
    return focus.sort_values(["scope", "sample", "abs_spearman"], ascending=[True, True, False])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_correlations = []
    all_quintiles = []
    all_summaries = []

    for scope, path in SCOPE_FILES.items():
        df = load_scope(scope, path)
        for sample_name, sample_df in subsamples(df).items():
            all_summaries.append(scope_summary(scope, sample_name, sample_df))
            all_correlations.extend(correlation_rows(scope, sample_name, sample_df))
            all_quintiles.extend(quintile_rows(scope, sample_name, sample_df))

    summary = pd.DataFrame(all_summaries)
    correlations = pd.DataFrame(all_correlations)
    quintiles = pd.DataFrame(all_quintiles)
    top = top_signal_table(correlations)

    summary.to_csv(OUT_DIR / "patent_validation_by_scope_summary.csv", index=False)
    correlations.to_csv(OUT_DIR / "patent_validation_correlations.csv", index=False)
    quintiles.to_csv(OUT_DIR / "patent_validation_quintiles.csv", index=False)
    top.to_csv(OUT_DIR / "patent_validation_top_correlations.csv", index=False)

    readme = """# Patent Validation Analysis

This folder summarizes whether the 10-K AI disclosure variables line up with AI patenting.

Inputs are the improved patent-merged filing-level files from `analysis/patent_validation_improved`.

Main outputs:

- `patent_validation_correlations.csv`: Pearson and Spearman correlations between 10-K AI variables and patent outcomes.
- `patent_validation_quintiles.csv`: patent outcomes by quintile of each AI disclosure variable.
- `patent_validation_by_scope_summary.csv`: row counts, year ranges, AI-positive shares, CRSP match shares, and mean values by scope/sample.
- `patent_validation_top_correlations.csv`: compact ranked view of the most relevant correlations.

Scopes:

- `DOCUMENT`
- `ITEM_1`
- `ITEM_1A`

Samples:

- `all`: all filing rows.
- `ai_positive`: rows where `ai_sentences_count > 0`.
- `crsp_matched`: rows with accepted CRSP match.
- `years_2015_2023`: restricted to 2015-2023.
- `ai_positive_years_2015_2023`: AI-positive rows within 2015-2023.

Main explanatory variables:

- `ai_expertise_ratio`
- `ai_risk_governance_ratio`
- `ai_expertise_share`
- `ai_risk_governance_share`
- `ai_noise_ratio`

Main patent outcomes:

- `ai_patent_app_count_93`: conservative AIPD AI patent application count.
- `ai_patent_app_share_93`: AI patent applications divided by total patent applications.
- `has_ai_patent_app_93`: indicator for at least one conservative AI patent application.
- `total_patent_app_count`: total matched patent applications.
- `log1p_ai_patent_app_count_93`: log-transformed conservative AI patent application count.
- `log1p_total_patent_app_count`: log-transformed total patent application count.
"""
    (OUT_DIR / "README_patent_validation_analysis.md").write_text(readme, encoding="utf-8")
    print(f"Wrote patent validation analysis outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
