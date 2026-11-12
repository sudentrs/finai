from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


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


def normalize_cik(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str.lstrip("0")
        .replace("", np.nan)
    )


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


def run_fe_regression(data: pd.DataFrame, y_col: str, indicator_col: str) -> dict:
    needed = data[[y_col, indicator_col, "year_num", "cik"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(needed) < 20 or needed[indicator_col].nunique() < 2 or needed[y_col].nunique() < 2:
        return {
            "n": len(needed),
            "n_firms": needed["cik"].nunique(),
            "estimate": np.nan,
            "std_error": np.nan,
            "t_value": np.nan,
            "p_value": np.nan,
            "cov_type": "cluster_cik",
        }

    year_dummies = pd.get_dummies(needed["year_num"].astype(int).astype(str), prefix="year", drop_first=True)
    x = pd.concat([needed[[indicator_col]].astype(float), year_dummies.astype(float)], axis=1)
    x = sm.add_constant(x)
    model = sm.OLS(needed[y_col].astype(float), x)
    try:
        result = model.fit(cov_type="cluster", cov_kwds={"groups": needed["cik"]})
        cov_type = "cluster_cik"
    except Exception:
        result = model.fit(cov_type="HC3")
        cov_type = "HC3_fallback"

    return {
        "n": int(result.nobs),
        "n_firms": needed["cik"].nunique(),
        "estimate": result.params.get(indicator_col, np.nan),
        "std_error": result.bse.get(indicator_col, np.nan),
        "t_value": result.tvalues.get(indicator_col, np.nan),
        "p_value": result.pvalues.get(indicator_col, np.nan),
        "cov_type": cov_type,
    }


def build_scope(scope: str, path: Path, ai_var: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year_num"] >= 2015) & (df["year_num"] <= 2025)].copy()
    df["cik"] = normalize_cik(df["cik"])
    df[ai_var] = pd.to_numeric(df[ai_var], errors="coerce")

    for key, _label, col, use_log in CHARACTERISTICS:
        df[f"clean_{key}"] = clean_series(df[col], use_log)

    base = df[df[ai_var].notna()].copy()
    base["ai_group"] = np.where(base[ai_var].fillna(0) <= 0, "No AI", "AI positive")
    positive = base[base[ai_var] > 0].copy()
    positive["ai_positive_quintile"] = pd.qcut(
        positive[ai_var],
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )
    base = base.merge(
        positive[["cik", "year", ai_var, "ai_positive_quintile"]],
        on=["cik", "year", ai_var],
        how="left",
    )
    return base


def regression_rows(scope: str, path: Path, ai_var: str) -> list[dict]:
    base = build_scope(scope, path, ai_var)
    rows = []
    for key, label, _col, _use_log in CHARACTERISTICS:
        y_col = f"clean_{key}"

        q5_vs_noai = base[
            (base["ai_group"].eq("No AI")) | (base["ai_positive_quintile"].eq("Q5"))
        ].copy()
        q5_vs_noai["high_ai_group"] = q5_vs_noai["ai_positive_quintile"].eq("Q5").astype(int)
        reg = run_fe_regression(q5_vs_noai, y_col, "high_ai_group")
        rows.append(
            {
                "scope": scope,
                "ai_variable": ai_var,
                "characteristic": key,
                "label": label,
                "comparison": "Q5 vs No AI",
                **reg,
                "significance": stars(reg["p_value"]),
            }
        )

        q5_vs_q1 = base[
            base["ai_positive_quintile"].isin(["Q1", "Q5"])
        ].copy()
        q5_vs_q1["high_ai_group"] = q5_vs_q1["ai_positive_quintile"].eq("Q5").astype(int)
        reg = run_fe_regression(q5_vs_q1, y_col, "high_ai_group")
        rows.append(
            {
                "scope": scope,
                "ai_variable": ai_var,
                "characteristic": key,
                "label": label,
                "comparison": "Q5 vs Q1 among AI-positive",
                **reg,
                "significance": stars(reg["p_value"]),
            }
        )
    return rows


def main() -> None:
    rows = []
    for scope, path in FILES.items():
        for ai_var in AI_VARS:
            rows.extend(regression_rows(scope, path, ai_var))

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ai_positive_quintile_profile_year_fe_cluster_tests.csv", index=False)
    print(f"Wrote regression robustness table to {OUT.resolve()}")


if __name__ == "__main__":
    main()
