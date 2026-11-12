from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
OUT = BASE / "paper_figure_tables"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
}

AI_VARS = ["ai_ratio", "ai_expertise_ratio"]

CHARACTERISTICS = [
    ("advertising_intensity_sales", "Advertising / sales", "advertising_intensity_sales", False),
    ("assets", "Log assets", "at", True),
    ("capex_intensity_assets", "CapEx / assets", "capex_intensity_assets", False),
    ("capex_intensity_sales", "CapEx / sales", "capex_intensity_sales", False),
    ("cash_ratio", "Cash / assets", "cash_ratio", False),
    ("employees", "Log employees", "emp", True),
    ("leverage", "Debt / assets", "leverage", False),
    ("market_cap", "Log market cap", "mkvalt", True),
    ("market_to_assets", "Market value / assets", "market_to_assets", False),
    ("market_to_book", "Market-to-book", "market_to_book", False),
    ("profit_margin", "Profit margin", "profit_margin", False),
    ("rd_intensity_assets", "R&D / assets", "rd_intensity_assets", False),
    ("rd_intensity_sales", "R&D / sales", "rd_intensity_sales", False),
    ("roa", "ROA", "roa", False),
    ("sales", "Log sales", "sale", True),
    ("sales_growth", "Sales growth", "sales_growth", False),
]


def clean_series(s: pd.Series, use_log: bool) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if use_log:
        s = np.log1p(s.where(s >= 0))
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)


def stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def make_table(scope: str, path: Path, ai_var: str) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path, low_memory=False)
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year_num"] >= 2015) & (df["year_num"] <= 2025)].copy()
    df[ai_var] = pd.to_numeric(df[ai_var], errors="coerce")

    for key, _label, col, use_log in CHARACTERISTICS:
        df[f"clean_{key}"] = clean_series(df[col], use_log)

    base = df[df[ai_var].notna()].copy()
    zero = base[base[ai_var].fillna(0) <= 0].copy()
    positive = base[base[ai_var] > 0].copy()
    positive["ai_positive_quintile"] = pd.qcut(
        positive[ai_var],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    rows = []
    for key, label, _col, _use_log in CHARACTERISTICS:
        clean_col = f"clean_{key}"
        row = {
            "scope": scope,
            "ai_variable": ai_var,
            "characteristic": key,
            "label": label,
            "n_total": len(base),
            "n_no_ai": zero[clean_col].notna().sum(),
            "n_ai_positive": positive[clean_col].notna().sum(),
            "no_ai": zero[clean_col].mean(),
        }
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            q_values = positive.loc[positive["ai_positive_quintile"].eq(q), clean_col]
            row[f"n_{q.lower()}"] = q_values.notna().sum()
            row[q.lower()] = q_values.mean()

        q5 = row["q5"]
        q1 = row["q1"]
        row["q5_minus_no_ai"] = q5 - row["no_ai"]
        row["q5_minus_q1"] = q5 - q1

        q5_values = positive.loc[positive["ai_positive_quintile"].eq("Q5"), clean_col].dropna()
        q1_values = positive.loc[positive["ai_positive_quintile"].eq("Q1"), clean_col].dropna()
        zero_values = zero[clean_col].dropna()

        if len(q5_values) > 2 and len(zero_values) > 2:
            t_no_ai, p_no_ai = stats.ttest_ind(q5_values, zero_values, equal_var=False, nan_policy="omit")
        else:
            t_no_ai, p_no_ai = np.nan, np.nan
        if len(q5_values) > 2 and len(q1_values) > 2:
            t_q1, p_q1 = stats.ttest_ind(q5_values, q1_values, equal_var=False, nan_policy="omit")
        else:
            t_q1, p_q1 = np.nan, np.nan

        row["t_q5_vs_no_ai"] = t_no_ai
        row["p_q5_vs_no_ai"] = p_no_ai
        row["sig_q5_vs_no_ai"] = stars(p_no_ai)
        row["t_q5_vs_q1"] = t_q1
        row["p_q5_vs_q1"] = p_q1
        row["sig_q5_vs_q1"] = stars(p_q1)
        rows.append(row)

    meta = {
        "scope": scope,
        "ai_variable": ai_var,
        "n_total": len(base),
        "n_no_ai": len(zero),
        "n_ai_positive": len(positive),
    }
    return meta, pd.DataFrame(rows)


def main() -> None:
    all_rows = []
    meta_rows = []
    for scope, path in FILES.items():
        for ai_var in AI_VARS:
            meta, table = make_table(scope, path, ai_var)
            meta_rows.append(meta)
            all_rows.append(table)

    result = pd.concat(all_rows, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    result.to_csv(OUT / "ai_positive_quintile_profiles_item1_document.csv", index=False)
    meta.to_csv(OUT / "ai_positive_quintile_profile_sample_counts.csv", index=False)

    display_cols = [
        "scope",
        "ai_variable",
        "label",
        "n_no_ai",
        "n_ai_positive",
        "n_q1",
        "n_q2",
        "n_q3",
        "n_q4",
        "n_q5",
        "no_ai",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q5_minus_no_ai",
        "sig_q5_vs_no_ai",
        "q5_minus_q1",
        "sig_q5_vs_q1",
    ]
    result[display_cols].to_csv(OUT / "ai_positive_quintile_profiles_display.csv", index=False)
    print(f"Wrote tables to {OUT.resolve()}")


if __name__ == "__main__":
    main()
