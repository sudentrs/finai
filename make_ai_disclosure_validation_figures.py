from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
TABLE_DIR = BASE / "paper_figure_tables"
FIG_DIR = BASE / "paper_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

REGRESSION_FILE = TABLE_DIR / "ai_disclosure_validation_univariate_year_industry_fe.csv"

PANEL_ORDER = [
    ("ITEM_1", "ai_ratio", "ITEM 1: AI ratio"),
    ("ITEM_1", "ai_expertise_ratio", "ITEM 1: AI expertise ratio"),
    ("DOCUMENT", "ai_ratio", "Document: AI ratio"),
    ("DOCUMENT", "ai_expertise_ratio", "Document: AI expertise ratio"),
]

CHARACTERISTICS = [
    ("log_market_cap", "Log market cap"),
    ("market_to_book", "Market-to-book"),
    ("rd_intensity_assets", "R&D / assets"),
    ("capex_intensity_assets", "CapEx / assets"),
    ("cash_ratio", "Cash / assets"),
    ("log_sales", "Log sales"),
    ("log_employees", "Log employees"),
]


def significance_label(sig: str) -> str:
    return "" if pd.isna(sig) else str(sig)


def main() -> None:
    df = pd.read_csv(REGRESSION_FILE)
    df = df[df["characteristic"].isin([x[0] for x in CHARACTERISTICS])].copy()
    df["coef_bps"] = df["coefficient"] * 10000
    df["se_bps"] = df["std_error"] * 10000
    df["ci_low"] = df["coef_bps"] - 1.96 * df["se_bps"]
    df["ci_high"] = df["coef_bps"] + 1.96 * df["se_bps"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    axes = axes.ravel()
    y_labels = [label for _key, label in CHARACTERISTICS]
    y_positions = np.arange(len(CHARACTERISTICS))

    for ax, (scope, dep, title) in zip(axes, PANEL_ORDER):
        panel = df[(df["scope"] == scope) & (df["dependent_variable"] == dep)].set_index("characteristic")
        coefs = []
        lows = []
        highs = []
        sigs = []
        for key, _label in CHARACTERISTICS:
            row = panel.loc[key]
            coefs.append(row["coef_bps"])
            lows.append(row["ci_low"])
            highs.append(row["ci_high"])
            sigs.append(significance_label(row["significance"]))

        coefs = np.array(coefs)
        lows = np.array(lows)
        highs = np.array(highs)
        err_left = coefs - lows
        err_right = highs - coefs
        colors = ["#1f77b4" if value >= 0 else "#b23a48" for value in coefs]

        ax.barh(y_positions, coefs, color=colors, alpha=0.82)
        ax.errorbar(
            coefs,
            y_positions,
            xerr=[err_left, err_right],
            fmt="none",
            ecolor="#333333",
            elinewidth=1,
            capsize=3,
        )
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        for y, value, sig in zip(y_positions, coefs, sigs):
            if sig:
                offset = 0.12 if value >= 0 else -0.12
                ha = "left" if value >= 0 else "right"
                ax.text(value + offset, y, sig, va="center", ha=ha, fontsize=9)

    fig.suptitle(
        "Do larger, growth-oriented, and innovation-intensive firms disclose more AI?",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.supxlabel(
        "Coefficient: change in AI disclosure ratio, basis points, for a one-standard-deviation increase in firm characteristic",
        fontsize=10,
        y=0.04,
    )
    fig.text(
        0.5,
        0.01,
        "Each regression includes year and SIC2 industry fixed effects. Standard errors are clustered by CIK. "
        "*** p<0.01, ** p<0.05, * p<0.10.",
        ha="center",
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 0.94))

    png = FIG_DIR / "figure_ai_disclosure_predictors_year_industry_fe.png"
    pdf = FIG_DIR / "figure_ai_disclosure_predictors_year_industry_fe.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    caption = """Figure: Firm characteristics and AI disclosure

This figure asks whether larger, growth-oriented, and innovation-intensive firms disclose more AI. Each bar reports a univariate regression coefficient from:

AI disclosure = alpha + beta * firm characteristic + year fixed effects + SIC2 industry fixed effects + error.

Firm characteristics are standardized, so each coefficient is the change in the AI disclosure ratio associated with a one-standard-deviation increase in the characteristic. Coefficients are shown in basis points. Standard errors are clustered by CIK.

Interpretation:
- The strongest and most stable predictors are log market cap, cash/assets, log sales, and log employees.
- Market-to-book is positive in the univariate fixed-effect specifications.
- R&D/assets is positive for AI expertise, weaker than size/cash variables.
- CapEx/assets is not a robust predictor.

Limitations:
- These regressions support construct validity and firm profiling, not causal interpretation.
- The specifications control for year and broad industry fixed effects, but they do not prove that firm characteristics cause AI disclosure.
- The descriptive no-AI/Q1-Q5 figure and this regression figure answer related but different questions.
"""
    (FIG_DIR / "figure_ai_disclosure_predictors_caption.txt").write_text(caption, encoding="utf-8")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
