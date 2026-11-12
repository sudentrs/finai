from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "analysis" / "advanced"
RESEARCH = BASE / "research_extensions"
OUT = RESEARCH / "industry_market"
PLOTS = OUT / "plots"

COMPUSTAT_QUARTERLY_PATH = ROOT / "g7htfs0feibx39p4.csv"
COMPUSTAT_ANNUAL_PATH = ROOT / "ncw9xm2krtoqavrc.csv"
FF12_PATH = ROOT / "Siccodes12.txt"
FF48_PATH = ROOT / "Siccodes48.txt"


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def parse_ff_file(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    groups = []
    current = None
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        header = re.match(r"^\s*(\d+)\s+([A-Za-z0-9]+)\s+(.*)$", line)
        range_line = re.match(r"^\s*(\d{4})-(\d{4})\s*(.*)$", line)
        if header:
            if current:
                groups.append(current)
            current = {
                "ff_num": int(header.group(1)),
                "ff_code": header.group(2).strip(),
                "ff_name": header.group(3).strip(),
                "ranges": [],
            }
        elif range_line and current:
            current["ranges"].append((int(range_line.group(1)), int(range_line.group(2))))
    if current:
        groups.append(current)
    return groups


FF12_RULES = parse_ff_file(FF12_PATH)
FF48_RULES = parse_ff_file(FF48_PATH)


def map_sic_to_ff(sic: float | int | str | None, rules: list[dict]) -> tuple[str, str]:
    try:
        sic_int = int(float(sic))
    except Exception:
        return "", ""
    for rule in rules:
        for low, high in rule["ranges"]:
            if low <= sic_int <= high:
                return rule["ff_code"], rule["ff_name"]
    return "", ""


def load_base() -> pd.DataFrame:
    df = pd.read_csv(BASE / "company_year_metrics_enriched.csv", dtype={"cik": str, "year": str})
    return df[df["year"] != "2026"].copy()


def finalize_industry_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["ff12_code"], df["ff12_name"] = zip(*df["sic"].map(lambda x: map_sic_to_ff(x, FF12_RULES)))
    df["ff48_code"], df["ff48_name"] = zip(*df["sic"].map(lambda x: map_sic_to_ff(x, FF48_RULES)))
    df["firm_label"] = np.where(df["tic"].notna() & df["tic"].ne(""), df["tic"], df["conm"])
    return df


def load_compustat_annual() -> pd.DataFrame:
    cols = [
        "tic",
        "datadate",
        "gvkey",
        "conm",
        "cusip",
        "cik",
        "naics",
        "sic",
        "apdedate",
        "fyear",
        "at",
        "ni",
        "revt",
        "sale",
        "csho",
        "mkvalt",
        "prcc_f",
    ]
    comp = pd.read_csv(COMPUSTAT_ANNUAL_PATH, usecols=cols, dtype={"cik": str, "tic": str, "gvkey": str})
    comp["cik"] = comp["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10).str.lstrip("0")
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["fyear"] = pd.to_numeric(comp["fyear"], errors="coerce").astype("Int64")
    comp["merge_year"] = comp["fyear"].astype(str)
    numeric_cols = ["naics", "sic", "at", "ni", "revt", "sale", "csho", "mkvalt", "prcc_f"]
    for col in numeric_cols:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")
    comp = comp.sort_values(["cik", "merge_year", "datadate"]).groupby(["cik", "merge_year"], as_index=False).tail(1).copy()
    comp = finalize_industry_labels(comp)
    comp.to_csv(OUT / "compustat_annual_primary.csv", index=False)
    return comp


def load_compustat_quarterly_annualized() -> pd.DataFrame:
    cols = [
        "tic",
        "datadate",
        "gvkey",
        "conm",
        "cik",
        "naics",
        "sic",
        "fyearq",
        "atq",
        "cheq",
        "cshoq",
        "dlcq",
        "dlttq",
        "niq",
        "revtq",
        "saleq",
        "mkvaltq",
        "prccq",
    ]
    comp = pd.read_csv(COMPUSTAT_QUARTERLY_PATH, usecols=cols, dtype={"cik": str, "tic": str, "gvkey": str})
    comp["cik"] = comp["cik"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10).str.lstrip("0")
    comp["datadate"] = pd.to_datetime(comp["datadate"], errors="coerce")
    comp["calendar_year"] = comp["datadate"].dt.year.astype("Int64").astype(str)
    comp["fyearq"] = pd.to_numeric(comp["fyearq"], errors="coerce").astype("Int64")
    comp["fiscal_year"] = comp["fyearq"].astype(str)
    comp["merge_year"] = np.where(comp["fiscal_year"].ne("<NA>"), comp["fiscal_year"], comp["calendar_year"])
    numeric_cols = ["naics", "sic", "atq", "cheq", "cshoq", "dlcq", "dlttq", "niq", "revtq", "saleq", "mkvaltq", "prccq"]
    for col in numeric_cols:
        comp[col] = pd.to_numeric(comp[col], errors="coerce")
    comp = comp.sort_values(["cik", "merge_year", "datadate"]).groupby(["cik", "merge_year"], as_index=False).tail(1).copy()
    comp = finalize_industry_labels(comp)
    comp.to_csv(OUT / "compustat_annualized_from_quarterly.csv", index=False)
    return comp


def merge_data(base: pd.DataFrame, comp: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(comp, left_on=["cik", "year"], right_on=["cik", "merge_year"], how="left", suffixes=("", "_comp"))
    for raw, alias in [("at", "assets"), ("mkvalt", "mktcap"), ("sale", "sales"), ("ni", "income")]:
        merged[f"log_{alias}"] = np.log1p(pd.to_numeric(merged[raw], errors="coerce").clip(lower=0))
    merged["profit_margin"] = np.where(pd.to_numeric(merged["sale"], errors="coerce") > 0, pd.to_numeric(merged["ni"], errors="coerce") / pd.to_numeric(merged["sale"], errors="coerce"), np.nan)
    merged.to_csv(OUT / "company_year_with_industry_market.csv", index=False)
    return merged


def industry_summaries(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ff12 = (
        df[df["ff12_name"].notna() & df["ff12_name"].ne("")]
        .groupby(["year", "ff12_code", "ff12_name"], as_index=False)
        .agg(
            firm_years=("cik", "size"),
            avg_ai_ratio=("ai_ratio", "mean"),
            avg_ai_risk_ratio=("ai_risk_ratio", "mean"),
            avg_risk_framing=("risk_framing_ratio", "mean"),
            avg_ai_count=("ai_count", "mean"),
            avg_mktcap=("mkvalt", "mean"),
        )
    )
    ff48 = (
        df[df["ff48_name"].notna() & df["ff48_name"].ne("")]
        .groupby(["year", "ff48_code", "ff48_name"], as_index=False)
        .agg(
            firm_years=("cik", "size"),
            avg_ai_ratio=("ai_ratio", "mean"),
            avg_ai_risk_ratio=("ai_risk_ratio", "mean"),
            avg_risk_framing=("risk_framing_ratio", "mean"),
        )
    )
    ff12.to_csv(OUT / "industry_ff12_yearly.csv", index=False)
    ff48.to_csv(OUT / "industry_ff48_yearly.csv", index=False)
    return ff12, ff48


def size_buckets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    valid = out["mkvalt"].notna() & (out["mkvalt"] > 0)
    out.loc[valid, "size_bucket"] = out.loc[valid].groupby("year")["mkvalt"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 4, labels=["Small", "Mid", "Large", "Mega"])
    )
    grouped = (
        out[out["size_bucket"].notna()]
        .groupby(["year", "size_bucket"], as_index=False)
        .agg(
            firm_years=("cik", "size"),
            avg_ai_ratio=("ai_ratio", "mean"),
            avg_ai_risk_ratio=("ai_risk_ratio", "mean"),
            avg_risk_framing=("risk_framing_ratio", "mean"),
        )
    )
    grouped.to_csv(OUT / "size_bucket_yearly.csv", index=False)
    return grouped


def cross_sectional_profiles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent = df[df["year"].isin(["2024", "2025"])].copy()
    profile = (
        recent.groupby("ff12_name", dropna=False)
        .agg(
            firm_years=("cik", "size"),
            avg_ai_ratio=("ai_ratio", "mean"),
            avg_ai_risk_ratio=("ai_risk_ratio", "mean"),
            avg_risk_framing=("risk_framing_ratio", "mean"),
            avg_log_mktcap=("log_mktcap", "mean"),
            avg_log_assets=("log_assets", "mean"),
        )
        .reset_index()
        .sort_values("avg_ai_risk_ratio", ascending=False)
    )
    profile.to_csv(OUT / "industry_profiles_2024_2025.csv", index=False)

    corr_cols = ["ai_ratio", "ai_risk_ratio", "risk_framing_ratio", "log_mktcap", "log_assets", "profit_margin"]
    corr = recent[corr_cols].corr(numeric_only=True)
    corr.to_csv(OUT / "market_characteristic_correlation_matrix.csv")
    return profile, corr


def leaders_by_industry(df: pd.DataFrame) -> pd.DataFrame:
    recent = df[df["year"].isin(["2024", "2025"])].copy()
    if recent.empty:
        out = pd.DataFrame()
        out.to_csv(OUT / "leaders_by_industry.csv", index=False)
        return out
    ranked = (
        recent.sort_values(["ff12_name", "ai_risk_ratio", "ai_ratio"], ascending=[True, False, False])
        .groupby("ff12_name", as_index=False)
        .head(5)
    )
    ranked.to_csv(OUT / "leaders_by_industry.csv", index=False)
    return ranked


def event_industry_shift(ff12: pd.DataFrame) -> pd.DataFrame:
    df = ff12.copy()
    df["window"] = np.where(df["year"].isin(["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]), "Pre-2023", "2023-2025")
    out = df.groupby(["ff12_name", "window"], as_index=False)[["avg_ai_ratio", "avg_ai_risk_ratio", "avg_risk_framing"]].mean()
    pivot = out.pivot(index="ff12_name", columns="window", values="avg_ai_risk_ratio")
    pivot["delta_ai_risk_ratio"] = pivot.get("2023-2025", 0) - pivot.get("Pre-2023", 0)
    pivot = pivot.sort_values("delta_ai_risk_ratio", ascending=False).reset_index()
    pivot.to_csv(OUT / "industry_event_shift.csv", index=False)
    return pivot


def plot_industry_heatmap(ff12: pd.DataFrame) -> None:
    top = ff12.groupby("ff12_name")["avg_ai_risk_ratio"].mean().sort_values(ascending=False).head(10).index.tolist()
    plot_df = ff12[ff12["ff12_name"].isin(top)].copy()
    pivot = plot_df.pivot(index="ff12_name", columns="year", values="avg_ai_risk_ratio").fillna(0)
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("ff12", ["#fff5f0", "#fb6a4a", "#67000d"])
    sns.heatmap(pivot, cmap=cmap, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("AI-Risk Intensity by Fama-French 12 Industry")
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "industry_ff12_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_size_effects(size_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    sns.lineplot(data=size_df, x="year", y="avg_ai_ratio", hue="size_bucket", marker="o", linewidth=2.8, ax=axes[0])
    axes[0].set_title("AI Intensity by Market-Cap Bucket")
    axes[0].set_ylabel("Average AI ratio")
    axes[0].set_xlabel("Year")
    axes[0].legend(title="", frameon=False)

    sns.lineplot(data=size_df, x="year", y="avg_ai_risk_ratio", hue="size_bucket", marker="o", linewidth=2.8, ax=axes[1])
    axes[1].set_title("AI-Risk Intensity by Market-Cap Bucket")
    axes[1].set_ylabel("Average AI-risk ratio")
    axes[1].set_xlabel("Year")
    axes[1].legend(title="", frameon=False)
    fig.savefig(PLOTS / "size_bucket_trends.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_market_scatter(df: pd.DataFrame) -> None:
    recent = df[df["year"].isin(["2024", "2025"]) & df["mkvalt"].notna() & (df["mkvalt"] > 0)].copy()
    recent["label"] = np.where(recent["ticker"].notna() & recent["ticker"].ne(""), recent["ticker"], recent["cik"])
    top = recent.sort_values("ai_risk_ratio", ascending=False).head(150).copy()
    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    sns.scatterplot(data=top, x="log_mktcap", y="ai_risk_ratio", hue="ff12_name", size="ai_ratio", sizes=(30, 300), alpha=0.75, ax=ax, legend=False)
    for _, row in top.nlargest(20, "ai_risk_ratio").iterrows():
        ax.text(row["log_mktcap"], row["ai_risk_ratio"], row["label"], fontsize=8)
    ax.set_title("AI-Risk Intensity vs Market Size, 2024-2025")
    ax.set_xlabel("Log market value")
    ax.set_ylabel("AI-risk ratio")
    fig.savefig(PLOTS / "market_size_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_industry_shift(shift_df: pd.DataFrame) -> None:
    top = shift_df.head(12).copy()
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    sns.barplot(data=top.sort_values("delta_ai_risk_ratio"), x="delta_ai_risk_ratio", y="ff12_name", color="#c23b22", ax=ax)
    ax.set_title("Industries with the Largest Post-2023 AI-Risk Shift")
    ax.set_xlabel("Change in average AI-risk ratio")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "industry_shift_bar.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(ff12: pd.DataFrame, size_df: pd.DataFrame, shift_df: pd.DataFrame) -> None:
    lines = ["# Industry and Market Extensions", ""]
    if not ff12.empty:
        top_2025 = ff12[ff12["year"] == "2025"].sort_values("avg_ai_risk_ratio", ascending=False).head(5)
        lines.append("## Top FF12 Industries in 2025 by AI-Risk Intensity")
        lines.append("")
        for _, row in top_2025.iterrows():
            lines.append(f"- {row['ff12_name']}: AI-risk ratio {row['avg_ai_risk_ratio']:.4f}, risk framing {row['avg_risk_framing']:.3f}")
        lines.append("")
    if not size_df.empty:
        latest = size_df[size_df["year"] == "2025"].copy()
        if not latest.empty:
            lines.append("## Size Effects in 2025")
            lines.append("")
            for _, row in latest.sort_values("avg_ai_risk_ratio", ascending=False).iterrows():
                lines.append(f"- {row['size_bucket']}: AI-risk ratio {row['avg_ai_risk_ratio']:.4f}, AI ratio {row['avg_ai_ratio']:.4f}")
            lines.append("")
    if not shift_df.empty:
        lines.append("## Biggest Post-2023 Industry Shifts")
        lines.append("")
        for _, row in shift_df.head(5).iterrows():
            lines.append(f"- {row['ff12_name']}: delta AI-risk ratio {row['delta_ai_risk_ratio']:.4f}")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    base = load_base()
    comp = load_compustat_annual()
    _ = load_compustat_quarterly_annualized()
    merged = merge_data(base, comp)
    ff12, ff48 = industry_summaries(merged)
    size_df = size_buckets(merged)
    profile, corr = cross_sectional_profiles(merged)
    leaders = leaders_by_industry(merged)
    shift_df = event_industry_shift(ff12)

    plot_industry_heatmap(ff12)
    plot_size_effects(size_df)
    plot_market_scatter(merged)
    plot_industry_shift(shift_df)
    write_summary(ff12, size_df, shift_df)
    print(f"Industry and market extensions written to {OUT}")


if __name__ == "__main__":
    main()
