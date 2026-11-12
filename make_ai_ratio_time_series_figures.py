from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
OUT_DIR = BASE / "paper_figures"
TABLE_DIR = BASE / "paper_figure_tables"
STYLE_DIR = OUT_DIR / "ai_time_series_publication"

INPUTS = {
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
}

SCOPE_LABELS = {
    "ITEM_1": "ITEM 1",
    "DOCUMENT": "Full document",
}

YEARS = list(range(2015, 2026))
MEASURES = {
    "ai_ratio": {
        "label": "AI ratio",
        "color": "#2f6f9f",
        "marker": "o",
        "linestyle": "-",
    },
    "ai_expertise_ratio": {
        "label": "AI expertise ratio",
        "color": "#b65c2a",
        "marker": "s",
        "linestyle": "-",
    },
}


def set_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_yearly_data():
    frames = []
    for scope, path in INPUTS.items():
        cols = ["year", "ai_ratio", "ai_expertise_ratio"]
        df = pd.read_csv(path, usecols=cols)
        df = df[df["year"].isin(YEARS)].copy()
        grouped = (
            df.groupby("year", as_index=False)
            .agg(
                ai_ratio=("ai_ratio", "mean"),
                ai_expertise_ratio=("ai_expertise_ratio", "mean"),
                n_filings=("year", "size"),
            )
        )
        grouped["scope"] = scope
        grouped["scope_label"] = SCOPE_LABELS[scope]
        frames.append(grouped)
    out = pd.concat(frames, ignore_index=True)
    out["ai_ratio_pct"] = out["ai_ratio"] * 100
    out["ai_expertise_ratio_pct"] = out["ai_expertise_ratio"] * 100
    return out


def style_axis(ax, title, y_max, y_label="Average ratio (% of sentences)"):
    ax.set_title(title, fontsize=11, fontweight="normal", fontstyle="italic", pad=9)
    ax.set_xlim(min(YEARS) - 0.35, max(YEARS) + 0.35)
    ax.set_ylim(0, y_max)
    ax.set_xticks(YEARS)
    ax.tick_params(axis="x", labelrotation=45, length=3, width=0.7, color="#333333")
    ax.tick_params(axis="y", length=3, width=0.7, color="#333333")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.65)
    ax.grid(axis="x", visible=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#9a9a9a")
        ax.spines[spine].set_linewidth(0.75)


def add_n_labels(ax, df, y_max):
    for _, row in df.iterrows():
        ax.text(
            row["year"],
            y_max * 0.985,
            f"[{int(row['n_filings']):,}]",
            ha="center",
            va="top",
            fontsize=6.4,
            color="#5f5f5f",
        )


def draw_scope(ax, yearly, scope, title, y_max=None, show_legend=True):
    df = yearly[yearly["scope"] == scope].sort_values("year")
    if y_max is None:
        max_val = max(df["ai_ratio_pct"].max(), df["ai_expertise_ratio_pct"].max())
        y_max = max_val * 1.22
    style_axis(ax, title, y_max)
    for measure, spec in MEASURES.items():
        y_col = f"{measure}_pct"
        ax.plot(
            df["year"],
            df[y_col],
            color=spec["color"],
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            markersize=3.8,
            linewidth=1.55,
            label=spec["label"],
            markerfacecolor=spec["color"],
            markeredgecolor="white",
            markeredgewidth=0.45,
        )
    add_n_labels(ax, df, y_max)
    if show_legend:
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, -0.31),
            ncol=2,
            frameon=False,
            handlelength=2.5,
        )


def draw_measure(ax, yearly, scope, measure, title, y_max=None):
    df = yearly[yearly["scope"] == scope].sort_values("year")
    y_col = f"{measure}_pct"
    if y_max is None:
        y_max = df[y_col].max() * 1.22
    spec = MEASURES[measure]
    style_axis(ax, title, y_max)
    ax.plot(
        df["year"],
        df[y_col],
        color=spec["color"],
        marker=spec["marker"],
        linestyle=spec["linestyle"],
        markersize=3.8,
        linewidth=1.55,
        markerfacecolor=spec["color"],
        markeredgecolor="white",
        markeredgewidth=0.45,
    )
    add_n_labels(ax, df, y_max)


def save_single_scope(yearly, scope):
    df = yearly[yearly["scope"] == scope]
    max_val = max(df["ai_ratio_pct"].max(), df["ai_expertise_ratio_pct"].max())
    y_max = max_val * 1.22
    fig, ax = plt.subplots(figsize=(6.25, 3.95), dpi=300)
    draw_scope(ax, yearly, scope, f"{SCOPE_LABELS[scope]} AI disclosure by year", y_max)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.87, bottom=0.24)
    slug = scope.lower()
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_{slug}_ai_ratio_expertise_by_year.{ext}", bbox_inches="tight")
        fig.savefig(STYLE_DIR / f"figure_{slug}_ai_ratio_expertise_by_year.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_single_measure(yearly, scope, measure):
    df = yearly[yearly["scope"] == scope]
    y_col = f"{measure}_pct"
    y_max = df[y_col].max() * 1.22
    fig, ax = plt.subplots(figsize=(6.25, 3.95), dpi=300)
    draw_measure(
        ax,
        yearly,
        scope,
        measure,
        f"{SCOPE_LABELS[scope]} {MEASURES[measure]['label']} by year",
        y_max,
    )
    fig.subplots_adjust(left=0.13, right=0.985, top=0.87, bottom=0.18)
    slug = scope.lower()
    measure_slug = measure.replace("_", "-")
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_{slug}_{measure_slug}_by_year.{ext}", bbox_inches="tight")
        fig.savefig(STYLE_DIR / f"figure_{slug}_{measure_slug}_by_year.{ext}", bbox_inches="tight")
    plt.close(fig)


def save_combined(yearly):
    max_val = max(yearly["ai_ratio_pct"].max(), yearly["ai_expertise_ratio_pct"].max())
    y_max = max_val * 1.20
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.95), dpi=300, sharey=True)
    draw_scope(axes[0], yearly, "ITEM_1", "ITEM 1", y_max, show_legend=False)
    draw_scope(axes[1], yearly, "DOCUMENT", "Full document", y_max, show_legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.015),
    )
    fig.suptitle("AI disclosure intensity by year", fontsize=12, fontweight="normal", y=0.98)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.83, bottom=0.22, wspace=0.14)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"figure_item1_document_ai_ratio_expertise_by_year.{ext}", bbox_inches="tight")
        fig.savefig(STYLE_DIR / f"figure_item1_document_ai_ratio_expertise_by_year.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_caption():
    caption = (
        "Figure X. AI disclosure intensity by year. "
        "The figure plots the average AI ratio and AI expertise ratio across 10-K filings by fiscal year. "
        "Ratios are expressed as percentages of sentences. AI ratio is the share of sentences classified as AI-related; "
        "AI expertise ratio is the share of sentences classified as AI expertise/capability disclosure. "
        "Panel A reports ITEM 1 and Panel B reports the full 10-K document. Bracketed values above each year report "
        "the number of filings used to compute the annual average. The plotted sample is restricted to 2015-2025, "
        "the study window; 2026 filings are excluded because the year is incomplete in the current corpus."
    )
    (OUT_DIR / "figure_ai_ratio_expertise_by_year_caption.txt").write_text(caption, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    set_publication_style()
    yearly = load_yearly_data()
    yearly.to_csv(TABLE_DIR / "ai_ratio_expertise_by_year_summary.csv", index=False)
    save_single_scope(yearly, "ITEM_1")
    save_single_scope(yearly, "DOCUMENT")
    for scope in ["ITEM_1", "DOCUMENT"]:
        for measure in ["ai_ratio", "ai_expertise_ratio"]:
            save_single_measure(yearly, scope, measure)
    save_combined(yearly)
    write_caption()
    print(TABLE_DIR / "ai_ratio_expertise_by_year_summary.csv")
    print(OUT_DIR / "figure_item1_ai_ratio_expertise_by_year.png")
    print(OUT_DIR / "figure_document_ai_ratio_expertise_by_year.png")
    print(OUT_DIR / "figure_item1_document_ai_ratio_expertise_by_year.png")
    print(STYLE_DIR)


if __name__ == "__main__":
    main()
