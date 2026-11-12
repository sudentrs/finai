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

DEPENDENT_VARIABLES = ["ai_ratio", "ai_expertise_ratio"]

CHARACTERISTICS = [
    ("log_market_cap", "Log market cap", "mkvalt", True),
    ("market_to_book", "Market-to-book", "market_to_book", False),
    ("market_to_assets", "Market value / assets", "market_to_assets", False),
    ("rd_intensity_assets", "R&D / assets", "rd_intensity_assets", False),
    ("capex_intensity_assets", "CapEx / assets", "capex_intensity_assets", False),
    ("cash_ratio", "Cash / assets", "cash_ratio", False),
    ("log_sales", "Log sales", "sale", True),
    ("log_assets", "Log assets", "at", True),
    ("log_employees", "Log employees", "emp", True),
    ("leverage", "Debt / assets", "leverage", False),
    ("roa", "ROA", "roa", False),
    ("profit_margin", "Profit margin", "profit_margin", False),
    ("sales_growth", "Sales growth", "sales_growth", False),
    ("advertising_intensity_sales", "Advertising / sales", "advertising_intensity_sales", False),
]

MULTIVARIATE_SPECS = {
    "core_growth_innovation": [
        "log_market_cap",
        "market_to_book",
        "rd_intensity_assets",
        "capex_intensity_assets",
        "cash_ratio",
    ],
    "core_plus_scale": [
        "log_market_cap",
        "market_to_book",
        "rd_intensity_assets",
        "capex_intensity_assets",
        "cash_ratio",
        "log_sales",
        "log_employees",
    ],
}


def normalize_cik(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str.lstrip("0")
        .replace("", np.nan)
    )


def clean_series(s: pd.Series, use_log: bool) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if use_log:
        s = np.log1p(s.where(s >= 0))
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std()
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


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


def read_scope(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["cik"] = normalize_cik(df["cik"])
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year_num"] >= 2015) & (df["year_num"] <= 2025)].copy()
    df["sic2_fe"] = pd.to_numeric(df["sic2"], errors="coerce").astype("Int64").astype(str)
    df.loc[df["sic2_fe"].isin(["<NA>", "nan"]), "sic2_fe"] = np.nan

    for dep in DEPENDENT_VARIABLES:
        df[dep] = pd.to_numeric(df[dep], errors="coerce")

    for key, _label, col, use_log in CHARACTERISTICS:
        df[f"x_{key}"] = clean_series(df[col], use_log)
        df[f"z_{key}"] = zscore(df[f"x_{key}"])

    return df


def fixed_effect_design(data: pd.DataFrame, x_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    fe = pd.get_dummies(
        data[["year_num", "sic2_fe"]].astype({"year_num": "Int64"}).astype(str),
        columns=["year_num", "sic2_fe"],
        drop_first=True,
    )
    x = pd.concat([data[x_cols].astype(float), fe.astype(float)], axis=1)
    x = sm.add_constant(x)
    return x, fe


def fit_clustered(data: pd.DataFrame, y_col: str, x_cols: list[str]) -> dict:
    needed = data[[y_col, "cik", "year_num", "sic2_fe", *x_cols]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(needed) < 100 or needed[y_col].nunique() < 2:
        return {"ok": False, "n": len(needed), "n_firms": needed["cik"].nunique()}
    x, _fe = fixed_effect_design(needed, x_cols)
    y = needed[y_col].astype(float)
    model = sm.OLS(y, x)
    try:
        res = model.fit(cov_type="cluster", cov_kwds={"groups": needed["cik"]})
        cov_type = "cluster_cik"
    except Exception:
        res = model.fit(cov_type="HC3")
        cov_type = "HC3_fallback"
    return {
        "ok": True,
        "result": res,
        "n": int(res.nobs),
        "n_firms": needed["cik"].nunique(),
        "mean_y": y.mean(),
        "sd_y": y.std(),
        "r2": res.rsquared,
        "cov_type": cov_type,
    }


def univariate_rows(scope: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for dep in DEPENDENT_VARIABLES:
        for key, label, _col, _use_log in CHARACTERISTICS:
            x_col = f"z_{key}"
            fit = fit_clustered(df, dep, [x_col])
            if not fit.get("ok"):
                rows.append(
                    {
                        "scope": scope,
                        "dependent_variable": dep,
                        "specification": "univariate_year_sic2_fe",
                        "characteristic": key,
                        "label": label,
                        "n": fit.get("n"),
                        "n_firms": fit.get("n_firms"),
                    }
                )
                continue
            res = fit["result"]
            p = res.pvalues.get(x_col, np.nan)
            rows.append(
                {
                    "scope": scope,
                    "dependent_variable": dep,
                    "specification": "univariate_year_sic2_fe",
                    "characteristic": key,
                    "label": label,
                    "n": fit["n"],
                    "n_firms": fit["n_firms"],
                    "coefficient": res.params.get(x_col, np.nan),
                    "std_error": res.bse.get(x_col, np.nan),
                    "t_value": res.tvalues.get(x_col, np.nan),
                    "p_value": p,
                    "significance": stars(p),
                    "r_squared": fit["r2"],
                    "mean_dependent_variable": fit["mean_y"],
                    "sd_dependent_variable": fit["sd_y"],
                    "x_standardized": True,
                    "fixed_effects": "year + sic2",
                    "cov_type": fit["cov_type"],
                }
            )
    return rows


def multivariate_rows(scope: str, df: pd.DataFrame) -> list[dict]:
    rows = []
    for dep in DEPENDENT_VARIABLES:
        for spec_name, keys in MULTIVARIATE_SPECS.items():
            x_cols = [f"z_{key}" for key in keys]
            fit = fit_clustered(df, dep, x_cols)
            if not fit.get("ok"):
                continue
            res = fit["result"]
            for key in keys:
                x_col = f"z_{key}"
                label = next(label for k, label, _col, _use_log in CHARACTERISTICS if k == key)
                p = res.pvalues.get(x_col, np.nan)
                rows.append(
                    {
                        "scope": scope,
                        "dependent_variable": dep,
                        "specification": spec_name,
                        "characteristic": key,
                        "label": label,
                        "n": fit["n"],
                        "n_firms": fit["n_firms"],
                        "coefficient": res.params.get(x_col, np.nan),
                        "std_error": res.bse.get(x_col, np.nan),
                        "t_value": res.tvalues.get(x_col, np.nan),
                        "p_value": p,
                        "significance": stars(p),
                        "r_squared": fit["r2"],
                        "mean_dependent_variable": fit["mean_y"],
                        "sd_dependent_variable": fit["sd_y"],
                        "x_standardized": True,
                        "fixed_effects": "year + sic2",
                        "cov_type": fit["cov_type"],
                    }
                )
    return rows


def main() -> None:
    univar = []
    multivar = []
    for scope, path in FILES.items():
        df = read_scope(path)
        univar.extend(univariate_rows(scope, df))
        multivar.extend(multivariate_rows(scope, df))

    pd.DataFrame(univar).to_csv(OUT / "ai_disclosure_validation_univariate_year_industry_fe.csv", index=False)
    pd.DataFrame(multivar).to_csv(OUT / "ai_disclosure_validation_multivariate_year_industry_fe.csv", index=False)
    print(f"Wrote AI disclosure validation regressions to {OUT.resolve()}")


if __name__ == "__main__":
    main()
