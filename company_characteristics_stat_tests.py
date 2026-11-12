from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


BASE_DIR = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
OUT_DIR = BASE_DIR / "statistical_tests"

SCOPE_FILES = {
    "DOCUMENT": BASE_DIR / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
    "ITEM_1": BASE_DIR / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "ITEM_1A": BASE_DIR / "ITEM_1A_filing_with_company_characteristic_quintiles.csv",
}

AI_VARIABLES = [
    "ai_ratio",
    "ai_expertise_ratio",
    "ai_risk_governance_ratio",
]

CHARACTERISTICS = {
    "market_cap": "mkvalt",
    "market_to_book": "market_to_book",
    "r_and_d_intensity_assets": "rd_intensity_assets",
    "capex_intensity_assets": "capex_intensity_assets",
    "sales": "sale",
    "assets": "at",
}


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


def read_sample(path: Path) -> pd.DataFrame:
    cols = ["cik", "year", "year_num", *AI_VARIABLES]
    for label, col in CHARACTERISTICS.items():
        cols.extend([col, f"{label}_quintile"])
    cols = list(dict.fromkeys(cols))
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    df["year_num"] = pd.to_numeric(df["year_num"], errors="coerce")
    df = df[(df["year_num"] >= 2015) & (df["year_num"] <= 2023)].copy()
    for col in df.columns:
        if col not in {"cik", "year"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def spearman_tests(scope: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for char_label, char_col in CHARACTERISTICS.items():
        for ai_var in AI_VARIABLES:
            sub = df[[char_col, ai_var]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 3 or sub[char_col].nunique() < 2 or sub[ai_var].nunique() < 2:
                rho, pval = np.nan, np.nan
            else:
                rho, pval = stats.spearmanr(sub[char_col], sub[ai_var])
            rows.append(
                {
                    "scope": scope,
                    "sample": "years_2015_2023",
                    "test": "spearman",
                    "characteristic": char_label,
                    "ai_variable": ai_var,
                    "n": len(sub),
                    "estimate": rho,
                    "p_value": pval,
                    "significance": stars(pval),
                }
            )
    return rows


def q5_q1_regressions(scope: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for char_label in CHARACTERISTICS:
        qcol = f"{char_label}_quintile"
        if qcol not in df.columns:
            continue
        two_tail = df[df[qcol].isin([1, 5])].copy()
        two_tail["top_quintile"] = (two_tail[qcol] == 5).astype(int)
        for ai_var in AI_VARIABLES:
            sub = two_tail[["year", "top_quintile", ai_var]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(sub) < 20 or sub["top_quintile"].nunique() < 2 or sub[ai_var].nunique() < 2:
                rows.append(
                    {
                        "scope": scope,
                        "sample": "years_2015_2023_q1_q5",
                        "test": "ols_year_fe_hc3",
                        "characteristic": char_label,
                        "ai_variable": ai_var,
                        "n": len(sub),
                        "estimate": np.nan,
                        "std_error": np.nan,
                        "t_value": np.nan,
                        "p_value": np.nan,
                        "significance": "",
                    }
                )
                continue
            model = smf.ols(f"{ai_var} ~ top_quintile + C(year)", data=sub).fit(cov_type="HC3")
            pval = model.pvalues.get("top_quintile", np.nan)
            rows.append(
                {
                    "scope": scope,
                    "sample": "years_2015_2023_q1_q5",
                    "test": "ols_year_fe_hc3",
                    "characteristic": char_label,
                    "ai_variable": ai_var,
                    "n": int(model.nobs),
                    "estimate": model.params.get("top_quintile", np.nan),
                    "std_error": model.bse.get("top_quintile", np.nan),
                    "t_value": model.tvalues.get("top_quintile", np.nan),
                    "p_value": pval,
                    "significance": stars(pval),
                }
            )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scope, path in SCOPE_FILES.items():
        df = read_sample(path)
        rows.extend(spearman_tests(scope, df))
        rows.extend(q5_q1_regressions(scope, df))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "company_characteristic_ai_statistical_tests.csv", index=False)
    print(f"Wrote statistical tests to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
