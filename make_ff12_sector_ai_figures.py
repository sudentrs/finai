from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
OUT_DIR = BASE / "paper_figures" / "ff12_sector_publication"
TABLE_DIR = BASE / "paper_figure_tables"

INPUTS = {
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
}

SCOPE_LABELS = {
    "ITEM_1": "ITEM 1",
    "DOCUMENT": "Full document",
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


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 5.8,
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


def load_sector_summary():
    frames = []
    for scope, path in INPUTS.items():
        cols = ["year", "form_type", "ai_ratio", "ai_expertise_ratio", "compustat_sic", "sic"]
        df = pd.read_csv(path, usecols=cols)
        df = df[(df["year"].between(2015, 2025)) & (df["form_type"].eq("10-K"))].copy()
        df["sic_for_ff12"] = df["compustat_sic"].where(df["compustat_sic"].notna(), df["sic"])
        df["ff12"] = df["sic_for_ff12"].apply(ff12_from_sic)
        df["ff12_label"] = df["ff12"].map(FF12_LABELS)

        grouped = (
            df.dropna(subset=["ff12"])
            .groupby(["ff12", "ff12_label"], as_index=False)
            .agg(
                n_filings=("year", "size"),
                n_firm_years=("sic_for_ff12", "size"),
                mean_ai_ratio=("ai_ratio", "mean"),
                mean_ai_expertise_ratio=("ai_expertise_ratio", "mean"),
            )
        )
        grouped["scope"] = scope
        grouped["scope_label"] = SCOPE_LABELS[scope]
        grouped["mean_ai_ratio_pct"] = grouped["mean_ai_ratio"] * 100
        grouped["mean_ai_expertise_ratio_pct"] = grouped["mean_ai_expertise_ratio"] * 100
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def draw_sector_figure(summary, scope):
    df = summary[summary["scope"] == scope].copy()
    df = df.sort_values("mean_ai_ratio_pct", ascending=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(6.2, 3.05), dpi=350)
    blue = "#2f6f9f"
    orange = "#b65c2a"
    offset = 0.105

    ax.hlines(y - offset, 0, df["mean_ai_ratio_pct"], color=blue, linewidth=2.15)
    ax.plot(df["mean_ai_ratio_pct"], y - offset, "o", color=blue, markersize=2.7, markeredgecolor="white", markeredgewidth=0.35)

    ax.hlines(y + offset, 0, df["mean_ai_expertise_ratio_pct"], color=orange, linewidth=2.15)
    ax.plot(df["mean_ai_expertise_ratio_pct"], y + offset, "s", color=orange, markersize=2.55, markeredgecolor="white", markeredgewidth=0.35)

    xmax = max(df["mean_ai_ratio_pct"].max(), df["mean_ai_expertise_ratio_pct"].max()) * 1.13
    ax.set_xlim(0, xmax)
    ax.set_yticks(y)
    ax.set_yticklabels(df["ff12_label"])
    ax.set_xlabel("")
    ax.text(
        0.0,
        1.065,
        f"{SCOPE_LABELS[scope]} AI measures",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontstyle="italic",
        color="#222222",
    )
    ax.grid(axis="x", color="#e3e3e3", linewidth=0.55)
    ax.grid(axis="y", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9a9a9a")
        ax.spines[spine].set_linewidth(0.55)
    ax.tick_params(axis="x", length=2.5, width=0.55, color="#555555", pad=2)
    ax.tick_params(axis="y", length=2.5, width=0.55, color="#555555", pad=3)

    for yi, (_, row) in zip(y, df.iterrows()):
        ax.text(
            xmax * 1.012,
            yi,
            f"[{int(row['n_filings']):,}]",
            ha="left",
            va="center",
            fontsize=5.7,
            color="#5f5f5f",
            clip_on=False,
        )

    ax.text(
        xmax * 1.012,
        len(df) - 0.15,
        "N",
        ha="left",
        va="bottom",
        fontsize=5.9,
        color="#5f5f5f",
        clip_on=False,
    )

    handles = [
        Line2D([0], [0], color=blue, marker="o", linewidth=2.15, markersize=2.9, label="AI ratio"),
        Line2D([0], [0], color=orange, marker="s", linewidth=2.15, markersize=2.7, label="AI expertise ratio"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.205), ncol=2, frameon=False, handlelength=1.7, columnspacing=1.5, borderaxespad=0.0)
    fig.subplots_adjust(left=0.31, right=0.88, top=0.88, bottom=0.19)

    slug = scope.lower()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_{slug}_ff12_ai_ratio_expertise.{ext}", bbox_inches="tight")
    plt.close(fig)


def draw_combined(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.35), dpi=350)
    for ax, scope in zip(axes, ["ITEM_1", "DOCUMENT"]):
        df = summary[summary["scope"] == scope].copy()
        df = df.sort_values("mean_ai_ratio_pct", ascending=True)
        y = np.arange(len(df))
        blue = "#2f6f9f"
        orange = "#b65c2a"
        offset = 0.105
        ax.hlines(y - offset, 0, df["mean_ai_ratio_pct"], color=blue, linewidth=1.9)
        ax.plot(df["mean_ai_ratio_pct"], y - offset, "o", color=blue, markersize=2.5, markeredgecolor="white", markeredgewidth=0.35)
        ax.hlines(y + offset, 0, df["mean_ai_expertise_ratio_pct"], color=orange, linewidth=1.9)
        ax.plot(df["mean_ai_expertise_ratio_pct"], y + offset, "s", color=orange, markersize=2.35, markeredgecolor="white", markeredgewidth=0.35)
        xmax = max(df["mean_ai_ratio_pct"].max(), df["mean_ai_expertise_ratio_pct"].max()) * 1.10
        ax.set_xlim(0, xmax)
        ax.set_yticks(y)
        ax.set_yticklabels(df["ff12_label"])
        ax.set_xlabel("")
        ax.set_title(SCOPE_LABELS[scope], fontstyle="italic", fontweight="normal", pad=5, fontsize=8.2)
        ax.grid(axis="x", color="#e3e3e3", linewidth=0.5)
        ax.grid(axis="y", visible=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#9a9a9a")
            ax.spines[spine].set_linewidth(0.55)
        ax.tick_params(axis="x", length=2.5, width=0.55, color="#555555", pad=2)
        ax.tick_params(axis="y", length=2.5, width=0.55, color="#555555", pad=3)
    handles = [
        Line2D([0], [0], color=blue, marker="o", linewidth=1.9, markersize=2.5, label="AI ratio"),
        Line2D([0], [0], color=orange, marker="s", linewidth=1.9, markersize=2.35, label="AI expertise ratio"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.055), handlelength=1.7, columnspacing=1.5)
    fig.suptitle("AI measures by Fama-French 12 industry", fontweight="normal", fontsize=8.8, fontstyle="italic", y=0.98)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.84, bottom=0.19, wspace=0.35)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_item1_document_ff12_ai_ratio_expertise.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_caption():
    caption = (
        "Figure X. AI measures by Fama-French 12 industry. "
        "The figure reports average AI ratio and AI expertise ratio across 10-K filings by Fama-French 12 industry group. "
        "Industries are assigned using Compustat SIC codes, with filing SIC used only when Compustat SIC is missing. "
        "Ratios are expressed as percentages of sentences. Bracketed values report the number of filings in each industry group. "
        "The sample is restricted to 2015-2025."
    )
    (OUT_DIR / "figure_ff12_ai_ratio_expertise_caption.txt").write_text(caption, encoding="utf-8")


def main():
    set_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_sector_summary()
    summary.to_csv(TABLE_DIR / "ff12_sector_ai_ratio_expertise_summary.csv", index=False)
    draw_sector_figure(summary, "ITEM_1")
    draw_sector_figure(summary, "DOCUMENT")
    draw_combined(summary)
    write_caption()
    print(TABLE_DIR / "ff12_sector_ai_ratio_expertise_summary.csv")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
