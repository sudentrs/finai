from pathlib import Path
import pandas as pd

BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
OUT = BASE / "paper_figure_tables" / "additional_validation"
OUT.mkdir(parents=True, exist_ok=True)
INPUTS = {
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
}
SCOPE_LABEL = {"ITEM_1": "ITEM 1", "DOCUMENT": "Full document"}
AI_VARS = {"ai_ratio": "AI ratio", "ai_expertise_ratio": "AI expertise ratio"}
USECOLS = [
    "company_name", "ticker", "cik", "accession", "year", "form_type", "total_sentences",
    "ai_sentences_count", "ai_expertise", "ai_ratio", "ai_expertise_ratio"
]

def read_scope(scope):
    df = pd.read_csv(INPUTS[scope], usecols=lambda c: c in USECOLS)
    df = df[df["form_type"].eq("10-K")].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(2015, 2025)].copy()
    df["scope"] = scope
    return df

def company_average_tables(min_filings=2):
    df = pd.concat([read_scope(s) for s in INPUTS], ignore_index=True)
    all_rows = []
    for scope in INPUTS:
        sdf = df[df["scope"].eq(scope)].copy()
        # Avoid tiny broken sections/filings dominating ratios.
        sdf = sdf[sdf["total_sentences"].fillna(0).ge(25)].copy()
        grouped = sdf.groupby(["scope", "company_name", "ticker", "cik"], dropna=False).agg(
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
        grouped = grouped[grouped["filings"].ge(min_filings)].copy()
        grouped["scope_label"] = SCOPE_LABEL[scope]
        for ai_var, ai_label in AI_VARS.items():
            rank_col = "mean_ai_ratio" if ai_var == "ai_ratio" else "mean_ai_expertise_ratio"
            ranked = grouped.sort_values([rank_col, "filings", "total_sentences"], ascending=[False, False, False]).copy()
            ranked.insert(0, "rank", range(1, len(ranked) + 1))
            ranked.insert(2, "ai_variable", ai_var)
            ranked.insert(3, "ai_label", ai_label)
            all_rows.append(ranked.head(20))
    out = pd.concat(all_rows, ignore_index=True)
    return out

def pct(x):
    return "" if pd.isna(x) else f"{100*x:.3f}"

def esc(s):
    if pd.isna(s):
        return ""
    return str(s).replace("&", "\\&")

def make_latex(df, topn=10):
    lines=[]
    lines.append("\\begin{table}[!htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Top {topn} companies by average AI disclosure}}");
    lines.append(f"\\label{{tab:top{topn}_average_ai_companies}}");
    lines.append("\\scriptsize")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lllrrrr}")
    lines.append("\\toprule")
    lines.append("Scope & Ranking variable & Company & Filings & Years & Mean AI ratio & Mean AI expertise ratio \\\\")
    lines.append("\\midrule")
    order = [("ITEM_1","ai_ratio"),("ITEM_1","ai_expertise_ratio"),("DOCUMENT","ai_ratio"),("DOCUMENT","ai_expertise_ratio")]
    for scope, ai_var in order:
        sub = df[(df.scope.eq(scope)) & (df.ai_variable.eq(ai_var))].sort_values("rank").head(topn)
        first=True
        for _, r in sub.iterrows():
            scope_txt = SCOPE_LABEL[scope] if first else ""
            rank_txt = r.ai_label if first else ""
            first=False
            company = esc(r.company_name)
            ticker = "" if pd.isna(r.ticker) else f" ({esc(r.ticker)})"
            years = f"{int(r.first_year)}--{int(r.last_year)}"
            lines.append(f"{scope_txt} & {rank_txt} & {company}{ticker} & {int(r.filings)} & {years} & {pct(r.mean_ai_ratio)} & {pct(r.mean_ai_expertise_ratio)} \\\\")
        lines.append("\\addlinespace")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\vspace{0.25em}")
    lines.append("\\begin{minipage}{0.94\\textwidth}")
    lines.append("\\footnotesize")
    lines.append(f"\\textit{{Notes:}} This table reports the top {topn} public companies by average AI disclosure intensity across 10-K filings from 2015--2025. Companies are ranked separately by scope and measure: ITEM 1 AI ratio, ITEM 1 AI expertise ratio, full-document AI ratio, and full-document AI expertise ratio. The averages are company-level means across filings, not single-filing maxima. Firms must have at least two 10-K filings and at least 25 sentences in the relevant text unit to be included. Ratios are expressed as percentages of sentences. This table is intended as a face-validity case study: firms ranked at the top should be substantively plausible high-AI-disclosure firms. Private firms such as OpenAI or Anthropic are not expected to appear because they do not file public 10-K reports.")
    lines.append("\\end{minipage}")
    lines.append("\\end{table}")
    return "\n".join(lines)

out = company_average_tables(min_filings=2)
out.to_csv(OUT / "top20_company_average_ai_disclosure.csv", index=False)
with pd.ExcelWriter(OUT / "top10_top20_company_average_ai_disclosure.xlsx") as writer:
    out[out["rank"].le(10)].to_excel(writer, sheet_name="top10_company_averages", index=False)
    out.to_excel(writer, sheet_name="top20_company_averages", index=False)
for n in [10, 20]:
    (OUT / f"top{n}_company_average_ai_disclosure_overleaf.tex").write_text(make_latex(out, topn=n), encoding="utf-8")
print(OUT / "top10_top20_company_average_ai_disclosure.xlsx")
print(OUT / "top20_company_average_ai_disclosure.csv")
print(OUT / "top10_company_average_ai_disclosure_overleaf.tex")
print(OUT / "top20_company_average_ai_disclosure_overleaf.tex")
