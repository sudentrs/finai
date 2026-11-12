from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
OUT_DIR = BASE / "paper_figures" / "patent_validation_publication"
TABLE_DIR = BASE / "paper_figure_tables" / "patent_validation_publication"

INPUTS = {
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
}

SCOPE_LABELS = {
    "ITEM_1": "ITEM 1",
    "DOCUMENT": "Full document",
}

AI_VARS = {
    "ai_ratio": "AI ratio",
    "ai_expertise_ratio": "AI expertise ratio",
}

AI_COLORS = {
    "ai_ratio": "#2f6f9f",
    "ai_expertise_ratio": "#b65c2a",
}

AI_MARKERS = {
    "ai_ratio": "o",
    "ai_expertise_ratio": "s",
}

FF12_LABELS = {
    1: "Consumer Nondurables",
    2: "Consumer Durables",
    3: "Manufacturing",
    4: "Energy",
    5: "Chemicals",
    6: "Business Equipment",
    7: "Telecommunications",
    8: "Utilities",
    9: "Wholesale and Retail",
    10: "Healthcare",
    11: "Finance",
    12: "Other",
}

# Conservative validation window: AIPD labels patent documents through 2023.
# Because pre-grant applications publish with lag, using 10-K filing years
# through 2021 preserves t+2 patent outcomes without relying on uncovered 2024+ labels.
YEARS = range(2015, 2022)
GROUP_ORDER = ["No AI", "Q1", "Q2", "Q3", "Q4", "Q5"]
PERIOD_LABELS = ["t-2", "t-1", "t", "t+1", "t+2"]


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 9,
            "axes.labelsize": 7.6,
            "xtick.labelsize": 6.9,
            "ytick.labelsize": 6.9,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def in_any(sic, ranges):
    return any(lo <= sic <= hi for lo, hi in ranges)


def ff12_from_sic(value):
    if pd.isna(value):
        return np.nan
    try:
        sic = int(float(value))
    except (TypeError, ValueError):
        return np.nan
    if in_any(sic, [(100, 999), (2000, 2399), (2700, 2749), (2770, 2799), (3100, 3199), (3940, 3989)]):
        return 1
    if in_any(sic, [(2500, 2519), (2590, 2599), (3630, 3659), (3710, 3711), (3714, 3714), (3716, 3716), (3750, 3751), (3792, 3792), (3900, 3939), (3990, 3999)]):
        return 2
    if in_any(sic, [(2520, 2589), (2600, 2699), (2750, 2769), (3000, 3099), (3200, 3569), (3580, 3629), (3700, 3709), (3712, 3713), (3715, 3715), (3717, 3749), (3752, 3791), (3793, 3799), (3830, 3839), (3860, 3899)]):
        return 3
    if in_any(sic, [(1200, 1399), (2900, 2999)]):
        return 4
    if in_any(sic, [(2800, 2829), (2840, 2899)]):
        return 5
    if in_any(sic, [(3570, 3579), (3660, 3692), (3694, 3699), (3810, 3829), (7370, 7379)]):
        return 6
    if in_any(sic, [(4800, 4899)]):
        return 7
    if in_any(sic, [(4900, 4949)]):
        return 8
    if in_any(sic, [(5000, 5999), (7200, 7299), (7600, 7699)]):
        return 9
    if in_any(sic, [(2830, 2839), (3693, 3693), (3840, 3859), (8000, 8099)]):
        return 10
    if in_any(sic, [(6000, 6999)]):
        return 11
    return 12


def safe_qcut_positive_by_year(df, value_col):
    group = pd.Series(index=df.index, dtype="object")
    group[df[value_col].fillna(0).eq(0)] = "No AI"
    for _, idx in df[df[value_col].fillna(0).gt(0)].groupby("year").groups.items():
        values = df.loc[idx, value_col]
        try:
            q = pd.qcut(values, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
            if len(q.cat.categories) == 5:
                group.loc[idx] = q.astype(str)
            else:
                group.loc[idx] = pd.qcut(values.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
        except ValueError:
            group.loc[idx] = "Q5"
    return pd.Categorical(group, categories=GROUP_ORDER, ordered=True)


def load_data():
    usecols = [
        "year",
        "form_type",
        "accession",
        "ai_ratio",
        "ai_expertise_ratio",
        "ai_risk_governance_ratio",
        "total_sentences",
        "ai_patent_app_count_93",
        "ai_patent_app_share_93",
        "has_ai_patent_app_93",
        "log1p_ai_patent_app_count_93",
        "ai_patent_app_count_93_t_minus_2",
        "ai_patent_app_count_93_t_minus_1",
        "ai_patent_app_count_93_t_plus_1",
        "ai_patent_app_count_93_t_plus_2",
        "compustat_sic",
        "sic",
    ]
    frames = []
    for scope, path in INPUTS.items():
        df = pd.read_csv(path, usecols=usecols)
        df = df[df["form_type"].eq("10-K") & df["year"].isin(YEARS)].copy()
        df["scope"] = scope
        df["scope_label"] = SCOPE_LABELS[scope]
        for ai_var in AI_VARS:
            df[f"{ai_var}_group"] = safe_qcut_positive_by_year(df, ai_var)
        df["sic_for_ff12"] = df["compustat_sic"].where(df["compustat_sic"].notna(), df["sic"])
        df["ff12"] = df["sic_for_ff12"].apply(ff12_from_sic)
        df["ff12_label"] = df["ff12"].map(FF12_LABELS)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def summarize_quintiles(df):
    rows = []
    for scope in INPUTS:
        sdf = df[df["scope"].eq(scope)]
        for ai_var, ai_label in AI_VARS.items():
            grouped = (
                sdf.groupby(f"{ai_var}_group", observed=False)
                .agg(
                    n=("accession", "size"),
                    mean_ai_measure=(ai_var, "mean"),
                    mean_log1p_ai_patents=("log1p_ai_patent_app_count_93", "mean"),
                    share_has_ai_patent=("has_ai_patent_app_93", "mean"),
                    mean_ai_patent_share=("ai_patent_app_share_93", "mean"),
                    mean_ai_patent_count=("ai_patent_app_count_93", "mean"),
                )
                .reset_index()
                .rename(columns={f"{ai_var}_group": "ai_group"})
            )
            grouped["scope"] = scope
            grouped["scope_label"] = SCOPE_LABELS[scope]
            grouped["ai_variable"] = ai_var
            grouped["ai_label"] = ai_label
            rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    out["ai_group"] = pd.Categorical(out["ai_group"], categories=GROUP_ORDER, ordered=True)
    return out.sort_values(["scope", "ai_variable", "ai_group"])


def axis_clean(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.55)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9a9a9a")
        ax.spines[spine].set_linewidth(0.55)
    ax.tick_params(axis="both", length=2.5, width=0.55, color="#555555", pad=2)


def draw_quintile_figure(summary, scope):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.95), dpi=350)
    outcomes = [
        ("mean_log1p_ai_patents", "Mean log(1 + AI patent applications)"),
        ("share_has_ai_patent", "Share with AI patent application"),
    ]
    x = np.arange(len(GROUP_ORDER))
    for ax, (outcome, title) in zip(axes, outcomes):
        for ai_var, ai_label in AI_VARS.items():
            sub = summary[(summary["scope"].eq(scope)) & (summary["ai_variable"].eq(ai_var))].sort_values("ai_group")
            y = sub[outcome].to_numpy()
            if outcome == "share_has_ai_patent":
                y = y * 100
            ax.plot(
                x,
                y,
                color=AI_COLORS[ai_var],
                marker=AI_MARKERS[ai_var],
                linewidth=1.5,
                markersize=3.2,
                label=ai_label,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_title(title, fontstyle="italic", fontweight="normal", pad=5)
        ax.set_xlabel("AI measure group")
        if outcome == "share_has_ai_patent":
            ax.set_ylabel("Percent")
        axis_clean(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.035), handlelength=1.8)
    fig.suptitle(f"{SCOPE_LABELS[scope]}: patent outcomes by AI-measure group", fontsize=9, fontstyle="italic", fontweight="normal", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.25, wspace=0.25)
    slug = scope.lower()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_{slug}_patent_outcomes_by_ai_group.{ext}", bbox_inches="tight")
    plt.close(fig)


def summarize_lag_lead(df):
    lag_cols = {
        "t-2": "ai_patent_app_count_93_t_minus_2",
        "t-1": "ai_patent_app_count_93_t_minus_1",
        "t": "ai_patent_app_count_93",
        "t+1": "ai_patent_app_count_93_t_plus_1",
        "t+2": "ai_patent_app_count_93_t_plus_2",
    }
    rows = []
    for scope in INPUTS:
        sdf = df[df["scope"].eq(scope)]
        for ai_var, ai_label in AI_VARS.items():
            for group in ["No AI", "Q5"]:
                gdf = sdf[sdf[f"{ai_var}_group"].astype(str).eq(group)].copy()
                for period, col in lag_cols.items():
                    rows.append(
                        {
                            "scope": scope,
                            "scope_label": SCOPE_LABELS[scope],
                            "ai_variable": ai_var,
                            "ai_label": ai_label,
                            "ai_group": group,
                            "period": period,
                            "mean_log1p_ai_patents": np.log1p(gdf[col].fillna(0)).mean(),
                            "mean_ai_patent_count": gdf[col].fillna(0).mean(),
                            "n": len(gdf),
                        }
                    )
    out = pd.DataFrame(rows)
    out["period"] = pd.Categorical(out["period"], categories=PERIOD_LABELS, ordered=True)
    return out.sort_values(["scope", "ai_variable", "ai_group", "period"])


def draw_lag_lead_figure(lag_summary, scope):
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.95), dpi=350, sharey=True)
    x = np.arange(len(PERIOD_LABELS))
    for ax, ai_var in zip(axes, AI_VARS):
        for group, color, marker in [("No AI", "#777777", "o"), ("Q5", AI_COLORS[ai_var], "s")]:
            sub = lag_summary[
                lag_summary["scope"].eq(scope)
                & lag_summary["ai_variable"].eq(ai_var)
                & lag_summary["ai_group"].eq(group)
            ].sort_values("period")
            ax.plot(
                x,
                sub["mean_log1p_ai_patents"],
                color=color,
                marker=marker,
                linewidth=1.45,
                markersize=3.0,
                label=group,
                markeredgecolor="white",
                markeredgewidth=0.35,
            )
        ax.axvline(2, color="#bfbfbf", linewidth=0.7, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(PERIOD_LABELS)
        ax.set_title(AI_VARS[ai_var], fontstyle="italic", fontweight="normal", pad=5)
        ax.set_xlabel("Patent application year relative to filing year")
        axis_clean(ax)
    axes[0].set_ylabel("Mean log(1 + AI patent applications)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.035), handlelength=1.8)
    fig.suptitle(f"{SCOPE_LABELS[scope]}: AI patent applications around filing year", fontsize=9, fontstyle="italic", fontweight="normal", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.27, wspace=0.16)
    slug = scope.lower()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_{slug}_patent_lag_lead_q5_vs_no_ai.{ext}", bbox_inches="tight")
    plt.close(fig)


def summarize_sectors(df):
    rows = []
    for scope in INPUTS:
        sdf = df[df["scope"].eq(scope)].dropna(subset=["ff12"])
        for ai_var, ai_label in AI_VARS.items():
            grouped = (
                sdf.groupby(["ff12", "ff12_label"], as_index=False)
                .agg(
                    n=("accession", "size"),
                    mean_ai_measure=(ai_var, "mean"),
                    mean_log1p_ai_patents=("log1p_ai_patent_app_count_93", "mean"),
                    share_has_ai_patent=("has_ai_patent_app_93", "mean"),
                )
            )
            grouped["scope"] = scope
            grouped["scope_label"] = SCOPE_LABELS[scope]
            grouped["ai_variable"] = ai_var
            grouped["ai_label"] = ai_label
            rows.append(grouped)
    out = pd.concat(rows, ignore_index=True)
    out["mean_ai_measure_pct"] = out["mean_ai_measure"] * 100
    out["share_has_ai_patent_pct"] = out["share_has_ai_patent"] * 100
    return out


def draw_sector_scatter(sector_summary, scope):
    scope_data = sector_summary[sector_summary["scope"].eq(scope)].copy()
    sector_order = [FF12_LABELS[i] for i in sorted(FF12_LABELS)]
    cmap = plt.get_cmap("tab20")
    sector_colors = {label: cmap(i) for i, label in enumerate(sector_order)}

    fig, axes = plt.subplots(1, 2, figsize=(8.15, 3.65), dpi=350)
    for ax, ai_var in zip(axes, AI_VARS):
        sub = scope_data[scope_data["ai_variable"].eq(ai_var)].copy()
        for _, row in sub.iterrows():
            ax.scatter(
                row["mean_ai_measure_pct"],
                row["share_has_ai_patent_pct"],
                s=np.clip(row["n"] / 85, 16, 82),
                color=sector_colors[row["ff12_label"]],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
        ax.set_title(AI_VARS[ai_var], fontstyle="italic", fontweight="normal", pad=5)
        ax.set_xlabel("Average AI measure (% of sentences)")
        ax.set_ylabel("Firms with AI patent application (%)")
        axis_clean(ax)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=sector_colors[label],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=5.0,
            label=label,
        )
        for label in sector_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        handletextpad=0.35,
        columnspacing=0.95,
        borderaxespad=0,
    )
    fig.suptitle(f"{SCOPE_LABELS[scope]}: sector-level AI text and AI patenting", fontsize=9, fontstyle="italic", fontweight="normal", y=0.98)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.80, bottom=0.31, wspace=0.23)
    slug = scope.lower()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_{slug}_ff12_ai_text_vs_patenting_scatter.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_caption_file():
    text = """Patent validation figure notes.

All main figures use 10-K filing-level observations from 2015-2021. This is the conservative patent-validation window: AIPD 2023 AI classifications cover patent documents published through 2023, and pre-grant patent applications publish with lag. Restricting filing years to 2015-2021 avoids treating later, not-yet-covered application years as true zero AI patent activity and preserves t+2 patent outcomes through 2023. Patent outcomes use AIPD-classified pre-grant patent applications. The main conservative AI patent variable is ai_patent_app_count_93, based on AIPD predict93_any_ai.

AI-measure group figures form a separate No AI group for filings with zero AI measure and form Q1-Q5 within positive observations by year. Lag/lead figures compare the No AI group with Q5 and plot mean log(1 + AI patent applications) from t-2 to t+2 relative to the filing year.
"""
    (OUT_DIR / "patent_validation_figure_notes.txt").write_text(text, encoding="utf-8")


def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    audit = (
        df.groupby(["scope", "year"], as_index=False)
        .agg(
            rows=("accession", "size"),
            ai_patents_93=("ai_patent_app_count_93", "sum"),
            has_ai_patent_93=("has_ai_patent_app_93", "sum"),
            mean_ai_patent_count_93=("ai_patent_app_count_93", "mean"),
            share_has_ai_patent_93=("has_ai_patent_app_93", "mean"),
        )
    )
    audit.to_csv(TABLE_DIR / "patent_validation_figure_sample_audit.csv", index=False)

    quintiles = summarize_quintiles(df)
    quintiles.to_csv(TABLE_DIR / "patent_validation_ai_group_summary.csv", index=False)
    for scope in INPUTS:
        draw_quintile_figure(quintiles, scope)

    lag_summary = summarize_lag_lead(df)
    lag_summary.to_csv(TABLE_DIR / "patent_validation_lag_lead_summary.csv", index=False)
    for scope in INPUTS:
        draw_lag_lead_figure(lag_summary, scope)

    sector_summary = summarize_sectors(df)
    sector_summary.to_csv(TABLE_DIR / "patent_validation_ff12_sector_summary.csv", index=False)
    for scope in INPUTS:
        draw_sector_scatter(sector_summary, scope)

    write_caption_file()
    print(OUT_DIR)
    print(TABLE_DIR)


if __name__ == "__main__":
    main()
