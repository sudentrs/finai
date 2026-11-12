from pathlib import Path
import math
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
FIG_DIR = BASE / "paper_figures" / "additional_validation"
TAB_DIR = BASE / "paper_figure_tables" / "additional_validation"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
}
SCOPE_LABEL = {"ITEM_1": "ITEM 1", "DOCUMENT": "Full document"}
AI_VARS = {"ai_ratio": "AI ratio", "ai_expertise_ratio": "AI expertise ratio"}
COLORS = {"ai_ratio": "#2f6f9f", "ai_expertise_ratio": "#b65c2a"}
YEARS_DISCLOSURE = range(2015, 2026)
YEARS_PATENT = range(2015, 2022)

USECOLS = [
    "company_name", "cik", "ticker", "accession", "filed_date", "year", "form_type",
    "total_sentences", "ai_sentences_count", "ai_expertise", "ai_ratio", "ai_expertise_ratio",
    "ai_patent_app_count_93", "has_ai_patent_app_93", "log1p_ai_patent_app_count_93",
    "sic2", "compustat_sic", "gvkey", "conm", "tic"
]


def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load(scope):
    df = pd.read_csv(INPUTS[scope], usecols=lambda c: c in USECOLS)
    df = df[df["form_type"].eq("10-K")].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].isin(YEARS_DISCLOSURE)].copy()
    df["scope"] = scope
    return df


def load_all():
    return pd.concat([load(s) for s in INPUTS], ignore_index=True)


def distribution_summary(df):
    rows = []
    for scope in INPUTS:
        sdf = df[df["scope"].eq(scope)]
        for ai_var, label in AI_VARS.items():
            x = pd.to_numeric(sdf[ai_var], errors="coerce")
            valid = x.notna()
            pos = x[valid & x.gt(0)]
            rows.append({
                "scope": scope,
                "scope_label": SCOPE_LABEL[scope],
                "ai_variable": ai_var,
                "ai_label": label,
                "n": int(valid.sum()),
                "zero_n": int((valid & x.eq(0)).sum()),
                "zero_share": float((valid & x.eq(0)).sum() / valid.sum()) if valid.sum() else np.nan,
                "positive_n": int(pos.shape[0]),
                "positive_mean": float(pos.mean()) if len(pos) else np.nan,
                "positive_median": float(pos.median()) if len(pos) else np.nan,
                "positive_p90": float(pos.quantile(0.90)) if len(pos) else np.nan,
                "positive_p99": float(pos.quantile(0.99)) if len(pos) else np.nan,
                "overall_mean": float(x[valid].mean()) if valid.sum() else np.nan,
            })
    return pd.DataFrame(rows)


def draw_distribution(df, scope):
    sdf = df[df["scope"].eq(scope)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.05), dpi=350)
    for ax, (ai_var, label) in zip(axes, AI_VARS.items()):
        x = pd.to_numeric(sdf[ai_var], errors="coerce").dropna()
        zero_share = (x.eq(0).sum() / len(x)) * 100 if len(x) else 0
        pos = x[x.gt(0)] * 100
        cap = pos.quantile(0.99) if len(pos) else 1
        bins = np.linspace(0, cap, 32)
        ax.hist(pos.clip(upper=cap), bins=bins, color=COLORS[ai_var], alpha=0.72, edgecolor="white", linewidth=0.35)
        ax.axvline(pos.median() if len(pos) else 0, color="#333333", linewidth=1.0, linestyle="--", label="Positive median")
        ax.text(0.98, 0.92, f"Zero filings: {zero_share:.1f}%", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_title(label, fontstyle="italic", fontweight="normal", pad=5)
        ax.set_xlabel("Positive AI measure (% of sentences)")
        ax.set_ylabel("Number of filings")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.55)
        ax.grid(axis="x", visible=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#999999")
            ax.spines[spine].set_linewidth(0.55)
        ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.98, 0.80))
    fig.suptitle(f"{SCOPE_LABEL[scope]}: distribution of AI disclosure measures", fontsize=9, fontstyle="italic", y=0.98)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.20, wspace=0.24)
    slug = scope.lower()
    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"figure_{slug}_ai_measure_distribution_zero_positive.{ext}", bbox_inches="tight")
    plt.close(fig)


def stars(p):
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def fit_patent_regs(df):
    rows = []
    reg_data = df[df["year"].isin(YEARS_PATENT)].copy()
    reg_data["sic2"] = reg_data["sic2"].fillna("Missing").astype(str)
    for scope in INPUTS:
        sdf = reg_data[reg_data["scope"].eq(scope)].copy()
        for ai_var, ai_label in AI_VARS.items():
            x = pd.to_numeric(sdf[ai_var], errors="coerce")
            sd = x.std(ddof=0)
            if not sd or pd.isna(sd):
                continue
            sdf["z_ai_measure"] = (x - x.mean()) / sd
            for outcome, outcome_label in [
                ("log1p_ai_patent_app_count_93", "Log(1 + AI patent apps)"),
                ("has_ai_patent_app_93", "Has AI patent app"),
            ]:
                r = sdf.dropna(subset=["z_ai_measure", outcome, "year", "sic2", "cik"]).copy()
                formula = f"{outcome} ~ z_ai_measure + C(year) + C(sic2)"
                model = smf.ols(formula, data=r).fit(cov_type="cluster", cov_kwds={"groups": r["cik"]})
                coef = model.params.get("z_ai_measure", np.nan)
                se = model.bse.get("z_ai_measure", np.nan)
                p = model.pvalues.get("z_ai_measure", np.nan)
                rows.append({
                    "scope": scope,
                    "scope_label": SCOPE_LABEL[scope],
                    "ai_variable": ai_var,
                    "ai_label": ai_label,
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "coef": coef,
                    "se": se,
                    "p_value": p,
                    "stars": stars(p),
                    "n": int(model.nobs),
                    "firms": int(r["cik"].nunique()),
                    "r2": model.rsquared,
                    "year_fe": "Yes",
                    "industry_fe": "SIC2",
                    "cluster": "Firm CIK",
                })
    return pd.DataFrame(rows)


def fmt_num(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def regression_latex(reg):
    order = [
        ("ITEM_1", "ai_ratio"), ("ITEM_1", "ai_expertise_ratio"),
        ("DOCUMENT", "ai_ratio"), ("DOCUMENT", "ai_expertise_ratio"),
    ]
    labels = ["ITEM 1 AI ratio", "ITEM 1 AI expertise", "Document AI ratio", "Document AI expertise"]
    lines = []
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Patent validation regressions}")
    lines.append("\\label{tab:patent_validation_regressions}")
    lines.append("\\scriptsize")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Outcome & " + " & ".join(labels) + " \\\\")
    lines.append("\\midrule")
    for outcome_label in ["Log(1 + AI patent apps)", "Has AI patent app"]:
        vals, ses = [], []
        for scope, ai_var in order:
            row = reg[(reg.scope.eq(scope)) & (reg.ai_variable.eq(ai_var)) & (reg.outcome_label.eq(outcome_label))]
            if row.empty:
                vals.append("")
                ses.append("")
            else:
                rr = row.iloc[0]
                vals.append(f"{rr.coef:.3f}{rr.stars}")
                ses.append(f"({rr.se:.3f})")
        lines.append(outcome_label + " & " + " & ".join(vals) + " \\\\")
        lines.append(" & " + " & ".join(ses) + " \\\\")
    lines.append("\\midrule")
    # Use max because N can differ slightly by missingness, but here likely same within col.
    for stat, col in [("Observations", "n"), ("Firms", "firms")]:
        vals=[]
        for scope, ai_var in order:
            sub=reg[(reg.scope.eq(scope)) & (reg.ai_variable.eq(ai_var))]
            vals.append(f"{int(sub[col].max()):,}" if not sub.empty else "")
        lines.append(stat + " & " + " & ".join(vals) + " \\\\")
    lines.append("Year fixed effects & Yes & Yes & Yes & Yes \\\\")
    lines.append("Industry fixed effects & SIC2 & SIC2 & SIC2 & SIC2 \\\\")
    lines.append("SE clustered by firm & Yes & Yes & Yes & Yes \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{0.25em}")
    lines.append("\\begin{minipage}{0.92\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} This table formally tests the patent-validation relation shown in the figures. The sample contains 10-K filing-level observations from 2015--2021. The dependent variables are AIPD-classified AI patent applications measured in the filing year: log one plus the number of AI patent applications, and an indicator equal to one if the firm has at least one AI patent application. Each column uses one standardized text-based AI disclosure measure as the explanatory variable. All specifications include year fixed effects and two-digit SIC industry fixed effects, and standard errors are clustered by firm CIK. Coefficients therefore report the association between a one-standard-deviation increase in the AI disclosure measure and AI patenting, conditional on year and industry. The indicator outcome is estimated as a linear probability model for interpretability with fixed effects. Significance levels are *** p<0.01, ** p<0.05, and * p<0.10. Patent applications use the conservative AIPD \\texttt{predict93\\_any\\_ai} definition. Because AIPD coverage ends in 2023 and pre-grant applications are observed with publication lag, the validation window is restricted to 2015--2021; the estimates should be interpreted as external validation rather than causal effects.")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def top_companies(df):
    rows = []
    firm_rows = []
    for scope in INPUTS:
        sdf = df[(df["scope"].eq(scope)) & (df["year"].isin(YEARS_DISCLOSURE))].copy()
        sdf = sdf[sdf["total_sentences"].fillna(0).ge(25)].copy()
        for ai_var, label in AI_VARS.items():
            cols = ["company_name", "ticker", "cik", "accession", "year", "filed_date", "total_sentences", "ai_sentences_count", "ai_expertise", "ai_ratio", "ai_expertise_ratio"]
            top = sdf.sort_values([ai_var, "total_sentences"], ascending=[False, False]).head(20)[cols].copy()
            top.insert(0, "rank", range(1, len(top)+1))
            top.insert(1, "scope", scope)
            top.insert(2, "ai_variable", ai_var)
            top.insert(3, "ai_label", label)
            rows.append(top)

            g = sdf.groupby(["company_name", "ticker", "cik"], dropna=False).agg(
                filings=("accession", "nunique"),
                first_year=("year", "min"),
                last_year=("year", "max"),
                mean_ai_ratio=("ai_ratio", "mean"),
                mean_ai_expertise_ratio=("ai_expertise_ratio", "mean"),
                max_ai_ratio=("ai_ratio", "max"),
                max_ai_expertise_ratio=("ai_expertise_ratio", "max"),
                total_ai_sentences=("ai_sentences_count", "sum"),
                total_ai_expertise_sentences=("ai_expertise", "sum"),
                total_sentences=("total_sentences", "sum"),
            ).reset_index()
            g = g[g["filings"].ge(2)].copy()
            rank_col = "mean_ai_ratio" if ai_var == "ai_ratio" else "mean_ai_expertise_ratio"
            topg = g.sort_values([rank_col, "total_sentences"], ascending=[False, False]).head(20).copy()
            topg.insert(0, "rank", range(1, len(topg)+1))
            topg.insert(1, "scope", scope)
            topg.insert(2, "ai_variable", ai_var)
            topg.insert(3, "ai_label", label)
            firm_rows.append(topg)
    filing_top = pd.concat(rows, ignore_index=True)
    firm_top = pd.concat(firm_rows, ignore_index=True)
    return filing_top, firm_top


def top_latex(firm_top):
    # Main compact table: top 10 ITEM 1 firms by mean AI ratio and mean AI expertise ratio.
    lines=[]
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Top ITEM 1 AI disclosure firms}")
    lines.append("\\label{tab:top_item1_ai_firms}")
    lines.append("\\scriptsize")
    lines.append("\\resizebox{0.92\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("Panel & Company & Filings & Years & Mean AI ratio & Mean AI expertise ratio \\\\")
    lines.append("\\midrule")
    for ai_var, panel in [("ai_ratio", "Top by AI ratio"), ("ai_expertise_ratio", "Top by AI expertise")]:
        sub=firm_top[(firm_top.scope.eq("ITEM_1")) & (firm_top.ai_variable.eq(ai_var))].head(10)
        first=True
        for _, r in sub.iterrows():
            p = panel if first else ""
            first=False
            years = f"{int(r.first_year)}--{int(r.last_year)}"
            company = str(r.company_name).replace('&', '\\&')[:55]
            ticker = '' if pd.isna(r.ticker) else f" ({str(r.ticker)})"
            lines.append(f"{p} & {company}{ticker} & {int(r.filings)} & {years} & {100*r.mean_ai_ratio:.3f} & {100*r.mean_ai_expertise_ratio:.3f} \\\\")
        lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{0.25em}")
    lines.append("\\begin{minipage}{0.92\\textwidth}")
    lines.append("\\footnotesize")
    lines.append("\\textit{Notes:} This table lists high-AI-disclosure firms as a case-study style validity check. The table uses firm-level averages across 10-K filings from 2015--2025 and requires at least two filings for the firm to be included. Ratios are expressed as percentages of ITEM 1 sentences. The purpose is not statistical testing but face validity: firms at the top of the distribution should be firms for which high AI disclosure is substantively plausible. Privately held AI developers such as OpenAI or Anthropic are generally not expected to appear because the sample is based on public firms filing 10-K reports.")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    set_style()
    df = load_all()
    dist = distribution_summary(df)
    dist.to_csv(TAB_DIR / "ai_measure_distribution_summary.csv", index=False)
    for scope in INPUTS:
        draw_distribution(df, scope)

    reg = fit_patent_regs(df)
    reg.to_csv(TAB_DIR / "patent_validation_regressions.csv", index=False)
    (TAB_DIR / "patent_validation_regressions_overleaf.tex").write_text(regression_latex(reg), encoding="utf-8")

    filing_top, firm_top = top_companies(df)
    filing_top.to_csv(TAB_DIR / "top20_ai_filing_case_studies.csv", index=False)
    firm_top.to_csv(TAB_DIR / "top20_ai_firm_case_studies.csv", index=False)
    with pd.ExcelWriter(TAB_DIR / "top_ai_case_studies.xlsx") as writer:
        filing_top.to_excel(writer, sheet_name="top20_filing_level", index=False)
        firm_top.to_excel(writer, sheet_name="top20_firm_level", index=False)
    (TAB_DIR / "top_item1_ai_firms_overleaf.tex").write_text(top_latex(firm_top), encoding="utf-8")

    print(FIG_DIR)
    print(TAB_DIR)

if __name__ == "__main__":
    main()
