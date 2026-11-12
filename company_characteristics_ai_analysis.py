from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("analysis/patent_validation_improved/extended_validation_analysis")
OUT_DIR = BASE_DIR / "company_characteristics_ai_analysis"
COMPUSTAT_PATH = Path("x9m8tdipe4kxamf0.csv") / "x9m8tdipe4kxamf0.csv"

SCOPE_FILES = {
    "DOCUMENT": BASE_DIR / "DOCUMENT_filing_with_lags_industry_adjusted.csv",
    "ITEM_1": BASE_DIR / "ITEM_1_filing_with_lags_industry_adjusted.csv",
    "ITEM_1A": BASE_DIR / "ITEM_1A_filing_with_lags_industry_adjusted.csv",
}

AI_VARIABLES = [
    "ai_sentences_count",
    "ai_ratio",
    "ai_expertise_ratio",
    "ai_risk_governance_ratio",
    "ai_expertise_share",
    "ai_risk_governance_share",
    "ai_noise_ratio",
    "ai_general_exposure_ratio",
]

CHARACTERISTICS = {
    "market_cap": "mkvalt",
    "assets": "at",
    "sales": "sale",
    "r_and_d_intensity_assets": "rd_intensity_assets",
    "r_and_d_intensity_sales": "rd_intensity_sales",
    "capex_intensity_assets": "capex_intensity_assets",
    "capex_intensity_sales": "capex_intensity_sales",
    "market_to_book": "market_to_book",
    "market_to_assets": "market_to_assets",
    "roa": "roa",
    "profit_margin": "profit_margin",
    "sales_growth": "sales_growth",
    "leverage": "leverage",
    "cash_ratio": "cash_ratio",
    "intangibles_ratio": "intangibles_ratio",
    "advertising_intensity_sales": "advertising_intensity_sales",
    "employees": "emp",
    "total_patent_app_count": "total_patent_app_count",
}


def safe_qcut(s: pd.Series, q: int = 5) -> pd.Series:
    valid = pd.to_numeric(s, errors="coerce")
    if valid.notna().sum() < q:
        return pd.Series(np.nan, index=s.index)
    try:
        return pd.qcut(valid, q=q, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.Series(np.nan, index=s.index)


def pearson_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    sub = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(sub) < 3:
        return np.nan, np.nan, len(sub)
    if sub.iloc[:, 0].nunique() < 2 or sub.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(sub)
    return sub.iloc[:, 0].corr(sub.iloc[:, 1], method="pearson"), sub.iloc[:, 0].corr(sub.iloc[:, 1], method="spearman"), len(sub)


def normalize_cik(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d+)", expand=False)
        .str.lstrip("0")
        .replace("", np.nan)
    )


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")
    out = num / den.replace({0: np.nan})
    return out.replace([np.inf, -np.inf], np.nan)


def read_compustat_characteristics() -> pd.DataFrame:
    comp = pd.read_csv(COMPUSTAT_PATH, low_memory=False)
    comp.columns = [c.lower() for c in comp.columns]
    comp["cik"] = normalize_cik(comp["cik"])
    comp["year"] = pd.to_numeric(comp["fyear"], errors="coerce").astype("Int64").astype(str)

    numeric_cols = [
        "at",
        "ceq",
        "che",
        "dlc",
        "dltt",
        "intan",
        "itcb",
        "lt",
        "pstk",
        "pstkl",
        "pstkrv",
        "seq",
        "txdb",
        "ni",
        "revt",
        "sale",
        "xad",
        "xrd",
        "xsga",
        "capx",
        "csho",
        "emp",
        "mkvalt",
        "prcc_f",
    ]
    for col in numeric_cols:
        if col in comp.columns:
            comp[col] = pd.to_numeric(comp[col], errors="coerce")

    preferred = comp["pstkrv"].combine_first(comp["pstkl"]).combine_first(comp["pstk"]).fillna(0)
    stockholders_equity = comp["seq"].combine_first(comp["ceq"] + comp["pstk"]).combine_first(comp["at"] - comp["lt"])
    comp["book_equity"] = stockholders_equity + comp["txdb"].fillna(0) + comp["itcb"].fillna(0) - preferred
    comp.loc[comp["book_equity"] <= 0, "book_equity"] = np.nan

    comp["market_cap_calc"] = comp["prcc_f"] * comp["csho"]
    comp["mkvalt"] = comp["mkvalt"].combine_first(comp["market_cap_calc"])
    comp["market_to_book"] = ratio(comp["mkvalt"], comp["book_equity"])
    comp["market_to_assets"] = ratio(comp["mkvalt"], comp["at"])
    comp["rd_intensity_assets"] = ratio(comp["xrd"], comp["at"])
    comp["rd_intensity_sales"] = ratio(comp["xrd"], comp["sale"])
    comp["capex_intensity_assets"] = ratio(comp["capx"], comp["at"])
    comp["capex_intensity_sales"] = ratio(comp["capx"], comp["sale"])
    comp["roa"] = ratio(comp["ni"], comp["at"])
    comp["profit_margin"] = ratio(comp["ni"], comp["sale"])
    comp["leverage"] = ratio(comp["dltt"].fillna(0) + comp["dlc"].fillna(0), comp["at"])
    comp["cash_ratio"] = ratio(comp["che"], comp["at"])
    comp["intangibles_ratio"] = ratio(comp["intan"], comp["at"])
    comp["advertising_intensity_sales"] = ratio(comp["xad"], comp["sale"])

    comp = comp.sort_values(["cik", "year"])
    comp["sales_growth"] = comp.groupby("cik", dropna=False)["sale"].pct_change()

    keep_cols = [
        "cik",
        "year",
        "gvkey",
        "conm",
        "cusip",
        "tic",
        "sic",
        "naics",
        "at",
        "sale",
        "revt",
        "ni",
        "mkvalt",
        "prcc_f",
        "csho",
        "xrd",
        "capx",
        "ceq",
        "seq",
        "book_equity",
        "market_to_book",
        "market_to_assets",
        "rd_intensity_assets",
        "rd_intensity_sales",
        "capex_intensity_assets",
        "capex_intensity_sales",
        "roa",
        "profit_margin",
        "sales_growth",
        "leverage",
        "cash_ratio",
        "intangibles_ratio",
        "advertising_intensity_sales",
        "emp",
    ]
    comp = comp[keep_cols].dropna(subset=["cik", "year"]).drop_duplicates(subset=["cik", "year"], keep="last")
    return comp.add_prefix("new_compustat_").rename(
        columns={"new_compustat_cik": "cik", "new_compustat_year": "year"}
    )


def read_scope(scope: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["scope"] = scope
    df["cik"] = normalize_cik(df["cik"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64").astype(str)
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    for col in set(AI_VARIABLES + list(CHARACTERISTICS.values())):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def attach_compustat(df: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in comp.columns if col not in {"cik", "year"} and col.replace("new_compustat_", "") in df.columns]
    df = df.drop(columns=[col.replace("new_compustat_", "") for col in drop_cols], errors="ignore")
    out = df.merge(comp, on=["cik", "year"], how="left")
    rename_map = {col: col.replace("new_compustat_", "") for col in out.columns if col.startswith("new_compustat_")}
    out = out.rename(columns=rename_map)
    for col in set(list(CHARACTERISTICS.values()) + ["xrd", "capx", "book_equity"]):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for label, col in CHARACTERISTICS.items():
        if col in out.columns:
            out[f"{label}_quintile"] = out.groupby("year", dropna=False)[col].transform(safe_qcut)
    return out


def split_summary(scope: str, df: pd.DataFrame, sample_name: str) -> list[dict]:
    rows = []
    for label, col in CHARACTERISTICS.items():
        qcol = f"{label}_quintile"
        if col not in df.columns or qcol not in df.columns:
            continue
        for bucket, sub in df.dropna(subset=[qcol]).groupby(qcol):
            row = {
                "scope": scope,
                "sample": sample_name,
                "characteristic": label,
                "quintile": int(bucket),
                "n_filings": len(sub),
                "n_firms": sub["cik"].nunique() if "cik" in sub.columns else np.nan,
                f"mean_{col}": sub[col].mean(),
            }
            for ai_var in AI_VARIABLES:
                if ai_var in sub.columns:
                    row[f"mean_{ai_var}"] = sub[ai_var].mean()
                    row[f"share_positive_{ai_var}"] = (sub[ai_var].fillna(0) > 0).mean()
            rows.append(row)
    return rows


def correlation_summary(scope: str, df: pd.DataFrame, sample_name: str) -> list[dict]:
    rows = []
    for label, col in CHARACTERISTICS.items():
        if col not in df.columns:
            continue
        for ai_var in AI_VARIABLES:
            if ai_var not in df.columns:
                continue
            pearson, spearman, n = pearson_spearman(df[col], df[ai_var])
            rows.append(
                {
                    "scope": scope,
                    "sample": sample_name,
                    "characteristic": label,
                    "ai_variable": ai_var,
                    "n": n,
                    "pearson": pearson,
                    "spearman": spearman,
                    "characteristic_mean": df[col].mean(),
                    "ai_variable_mean": df[ai_var].mean(),
                }
            )
    return rows


def q5_minus_q1_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in split_df.groupby(["scope", "sample", "characteristic"], dropna=False):
        q1 = sub[sub["quintile"] == 1]
        q5 = sub[sub["quintile"] == 5]
        if q1.empty or q5.empty:
            continue
        q1 = q1.iloc[0]
        q5 = q5.iloc[0]
        row = {
            "scope": keys[0],
            "sample": keys[1],
            "characteristic": keys[2],
            "q1_n_filings": q1["n_filings"],
            "q5_n_filings": q5["n_filings"],
        }
        for ai_var in AI_VARIABLES:
            mean_col = f"mean_{ai_var}"
            positive_col = f"share_positive_{ai_var}"
            if mean_col in split_df.columns:
                row[f"q1_{mean_col}"] = q1.get(mean_col, np.nan)
                row[f"q5_{mean_col}"] = q5.get(mean_col, np.nan)
                row[f"q5_minus_q1_{mean_col}"] = q5.get(mean_col, np.nan) - q1.get(mean_col, np.nan)
            if positive_col in split_df.columns:
                row[f"q1_{positive_col}"] = q1.get(positive_col, np.nan)
                row[f"q5_{positive_col}"] = q5.get(positive_col, np.nan)
                row[f"q5_minus_q1_{positive_col}"] = q5.get(positive_col, np.nan) - q1.get(positive_col, np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def make_samples(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    years = df[(df["year_num"] >= 2015) & (df["year_num"] <= 2023)]
    return {
        "all": df,
        "years_2015_2023": years,
        "ai_positive_years_2015_2023": years[years["ai_sentences_count"].fillna(0) > 0],
    }


def availability_table() -> pd.DataFrame:
    rows = []
    comp_cols = set(pd.read_csv(COMPUSTAT_PATH, nrows=1, low_memory=False).columns.str.lower())
    derived_cols = {
        "market_to_book",
        "market_to_assets",
        "rd_intensity_assets",
        "rd_intensity_sales",
        "capex_intensity_assets",
        "capex_intensity_sales",
        "roa",
        "profit_margin",
        "sales_growth",
        "leverage",
        "cash_ratio",
        "intangibles_ratio",
        "advertising_intensity_sales",
    }
    for scope in SCOPE_FILES:
        for label, col in CHARACTERISTICS.items():
            rows.append(
                {
                    "scope": scope,
                    "requested_characteristic": label,
                    "source_column": col,
                    "available": col in comp_cols or col in derived_cols or col == "total_patent_app_count",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_split_rows = []
    all_corr_rows = []
    comp = read_compustat_characteristics()

    for scope, path in SCOPE_FILES.items():
        df = add_quintiles(attach_compustat(read_scope(scope, path), comp))
        df.to_csv(OUT_DIR / f"{scope}_filing_with_company_characteristic_quintiles.csv", index=False)
        for sample_name, sample in make_samples(df).items():
            all_split_rows.extend(split_summary(scope, sample, sample_name))
            all_corr_rows.extend(correlation_summary(scope, sample, sample_name))

    split_df = pd.DataFrame(all_split_rows)
    split_df.to_csv(OUT_DIR / "company_characteristic_ai_quintile_summary.csv", index=False)
    q5_minus_q1_summary(split_df).to_csv(OUT_DIR / "company_characteristic_q5_minus_q1_summary.csv", index=False)
    pd.DataFrame(all_corr_rows).to_csv(OUT_DIR / "company_characteristic_ai_correlations.csv", index=False)
    availability_table().to_csv(OUT_DIR / "company_characteristic_variable_availability.csv", index=False)

    readme = """# Company Characteristics AI Analysis

This folder examines whether AI disclosure variables are more prominent for certain types of firms.

The analysis is run separately for DOCUMENT, ITEM_1, and ITEM_1A. For each scope, firms are sorted into within-year quintiles by available company characteristics, then the mean AI variables are computed inside each quintile.

Available characteristics:

- `market_cap`: Compustat `mkvalt`.
- `assets`: Compustat `at`.
- `sales`: Compustat `sale`.
- `r_and_d_intensity_assets`: `xrd / at`.
- `r_and_d_intensity_sales`: `xrd / sale`.
- `capex_intensity_assets`: `capx / at`.
- `capex_intensity_sales`: `capx / sale`.
- `market_to_book`: market value divided by book equity.
- `market_to_assets`: `mkvalt / at`.
- `roa`: `ni / at`.
- `profit_margin`: `ni / sale`.
- `sales_growth`: year-over-year sales growth by CIK.
- `leverage`: `(dltt + dlc) / at`.
- `cash_ratio`: `che / at`.
- `intangibles_ratio`: `intan / at`.
- `advertising_intensity_sales`: `xad / sale`.
- `employees`: Compustat `emp`.
- `total_patent_app_count`: total matched patent applications for the firm-year.

Note:

- Missing `xrd` and `capx` are treated as missing in the main intensity ratios, not automatically converted to zero.

Main outputs:

- `company_characteristic_ai_quintile_summary.csv`: mean AI variables by characteristic quintile.
- `company_characteristic_q5_minus_q1_summary.csv`: compact top-minus-bottom quintile differences for AI variables.
- `company_characteristic_ai_correlations.csv`: Pearson and Spearman correlations between continuous firm characteristics and AI variables.
- `company_characteristic_variable_availability.csv`: audit of which requested characteristics are available.
- `*_filing_with_company_characteristic_quintiles.csv`: filing-level data with within-year characteristic quintile labels added.
"""
    (OUT_DIR / "README_company_characteristics_ai_analysis.md").write_text(readme, encoding="utf-8")
    print(f"Wrote company characteristics AI analysis to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
