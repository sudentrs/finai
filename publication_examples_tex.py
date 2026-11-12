"""Write source-faithful longtables from the audited example CSVs."""
import pandas as pd

from publication_tables import OUT, latex_escape


def export():
    for sample in ("broad", "professor_price_ge5"):
        folder = OUT / sample
        data = pd.read_csv(folder / "table_311.csv").fillna("")
        lines = [
            r"\begingroup\small\setlength{\tabcolsep}{3pt}",
            r"\begin{longtable}{p{0.19\textwidth}rrrp{0.44\textwidth}}",
            r"\caption{Draft examples of Item 1 AI disclosures: " +
            ("broad filing sample" if sample == "broad" else "professor price-screened sample") +
            r"}\\\toprule",
            r"Company & Period year & Expertise (\%) & AI (\%) & Verbatim excerpt \\\midrule",
            r"\endfirsthead",
            r"\toprule Company & Period year & Expertise (\%) & AI (\%) & Verbatim excerpt \\\midrule",
            r"\endhead",
        ]
        previous = None
        for _, row in data.iterrows():
            if row.panel != previous:
                label = ("Panel A: highest average expertise" if row.panel == "A"
                         else "Panel B: AI-positive with zero expertise in all included filings")
                lines.append(r"\multicolumn{5}{l}{\textit{" + label + r"}}\\")
                previous = row.panel
            excerpt = row.excerpt if row.excerpt else (
                "No eligible non-noise sentence of at least ten words in the selected filing.")
            cells = [latex_escape(row.company), str(int(row.fiscal_year_label)),
                     f"{row.average_expertise_pct:.3f}", f"{row.average_ai_pct:.3f}",
                     latex_escape(excerpt)]
            lines.append(" & ".join(cells) + r"\\[5pt]")
        lines += [r"\bottomrule\end{longtable}",
                  r"\noindent\footnotesize\textit{Notes:} Ratios are raw, equal-weighted means across distinct "
                  r"reporting-period filings for firms with at least two included filings. Filing dates and "
                  r"reporting-period end years are restricted to 2015--2025. The displayed year is the "
                  r"reporting-period end year, not necessarily the Compustat fiscal-year label. "
                  r"The excerpt filing maximizes the relevant ratio; the eligible sentence maximizes "
                  r"topic weight within that filing. Panel A requires an expertise sentence. Panel B "
                  r"excludes noise. Missing eligible quotes are retained and flagged, not replaced. "
                  r"Main rankings do not impose the optional 25-sentence screen; a separate CSV reports "
                  r"that sensitivity. These are review drafts because ratio transformations and selection "
                  r"rules still require supervisor confirmation.",
                  r"\par\endgroup"]
        (folder / "table_311.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    export()
