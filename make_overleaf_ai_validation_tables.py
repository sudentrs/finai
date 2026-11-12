from pathlib import Path

import pandas as pd


BASE = Path(
    "analysis/patent_validation_improved/extended_validation_analysis/"
    "company_characteristics_ai_analysis/paper_figure_tables"
)

PROFILE_PATH = BASE / "ai_positive_quintile_profiles_display.csv"
UNIV_PATH = BASE / "ai_disclosure_validation_univariate_year_industry_fe.csv"
MULTI_PATH = BASE / "ai_disclosure_validation_multivariate_year_industry_fe.csv"
OUT_PATH = BASE / "overleaf_tables_ai_validation.tex"
DETAILED_OUT_PATH = BASE / "overleaf_tables_ai_validation_detailed_n.tex"


PROFILE_ORDER = [
    "Log market cap",
    "Log assets",
    "Log sales",
    "Log employees",
    "Market-to-book",
    "Market value / assets",
    "Profit margin",
    "ROA",
    "Sales growth",
    "Cash / assets",
    "R&D / assets",
    "R&D / sales",
    "CapEx / assets",
    "CapEx / sales",
    "Advertising / sales",
    "Leverage",
]

REG_ORDER = [
    "log_market_cap",
    "market_to_book",
    "rd_intensity_assets",
    "capex_intensity_assets",
    "cash_ratio",
    "log_sales",
    "log_assets",
    "employees",
]

DEP_ORDER = [
    ("ITEM_1", "ai_ratio", "ITEM 1 AI ratio"),
    ("ITEM_1", "ai_expertise_ratio", "ITEM 1 AI expertise"),
    ("DOCUMENT", "ai_ratio", "Document AI ratio"),
    ("DOCUMENT", "ai_expertise_ratio", "Document AI expertise"),
]


def esc(value) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_num(value, decimals=3):
    if pd.isna(value):
        return ""
    value = float(value)
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.{decimals}f}"


def fmt_count(value):
    if pd.isna(value):
        return ""
    return f"{int(float(value)):,}"


def add_stars(value, stars):
    if pd.isna(stars) or str(stars).lower() == "nan":
        stars = ""
    value = fmt_num(value)
    return f"{value}{stars}" if stars else value


def table_env(label, caption, note, body, colspec, size="\\scriptsize"):
    return rf"""
\begin{{table}}[!htbp]
\centering
\begin{{threeparttable}}
\caption{{{caption}}}
\label{{{label}}}
{size}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{{colspec}}}
\toprule
{body}
\bottomrule
\end{{tabular}}%
}}
\begin{{tablenotes}}
\footnotesize
\item \textit{{Notes:}} {note}
\end{{tablenotes}}
\end{{threeparttable}}
\end{{table}}
"""


def make_profile_table(profile, scope, ai_variable, label, caption, detailed_n=False):
    df = profile[(profile["scope"] == scope) & (profile["ai_variable"] == ai_variable)].copy()
    order = {name: idx for idx, name in enumerate(PROFILE_ORDER)}
    df["order"] = df["label"].map(order).fillna(999)
    df = df.sort_values(["order", "label"])

    lines = [
        (
            "Characteristic & No AI & Q1 & Q2 & Q3 & Q4 & Q5 & "
            "Q5--No AI & Q5--Q1 \\\\"
        ),
        "\\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    esc(row["label"]),
                    fmt_num(row["no_ai"]),
                    fmt_num(row["q1"]),
                    fmt_num(row["q2"]),
                    fmt_num(row["q3"]),
                    fmt_num(row["q4"]),
                    fmt_num(row["q5"]),
                    add_stars(row["q5_minus_no_ai"], row.get("sig_q5_vs_no_ai", "")),
                    add_stars(row["q5_minus_q1"], row.get("sig_q5_vs_q1", "")),
                ]
            )
            + r" \\"
        )
        if detailed_n:
            lines.append(
                " & ".join(
                    [
                        r"\textit{N}",
                        fmt_count(row["n_no_ai"]),
                        fmt_count(row["n_q1"]),
                        fmt_count(row["n_q2"]),
                        fmt_count(row["n_q3"]),
                        fmt_count(row["n_q4"]),
                        fmt_count(row["n_q5"]),
                        "",
                        "",
                    ]
                )
                + r" \\"
            )

    note = (
        "The table separates filings with zero AI disclosure from filings with positive "
        "AI disclosure. Q1--Q5 are quintiles formed only within the AI-positive sample. "
        "Entries are group means. Q5--No AI and Q5--Q1 report differences in means; "
        "stars are from Welch two-sample t-tests. Variables are measured at the filing "
        "level over 2015--2025. Size variables are in logs; ratio variables are winsorized "
        "at the 1st and 99th percentiles."
    )
    if detailed_n:
        note = (
            "The table separates filings with zero AI disclosure from filings with positive "
            "AI disclosure. Q1--Q5 are quintiles formed only within the AI-positive sample. "
            "For each characteristic, the first row reports group means and the italicized "
            "second row reports the number of non-missing observations in each group. "
            "Q5--No AI and Q5--Q1 report differences in means; stars are from Welch "
            "two-sample t-tests. Variables are measured at the filing level over 2015--2025. "
            "Size variables are in logs; ratio variables are winsorized at the 1st and 99th percentiles."
        )
    return table_env(label, caption, note, "\n".join(lines), "lrrrrrrrr")


def make_profile_n_summary(profile):
    rows = []
    for scope, ai_variable, name in [
        ("ITEM_1", "ai_ratio", "ITEM 1 AI ratio"),
        ("ITEM_1", "ai_expertise_ratio", "ITEM 1 AI expertise ratio"),
        ("DOCUMENT", "ai_ratio", "Document AI ratio"),
        ("DOCUMENT", "ai_expertise_ratio", "Document AI expertise ratio"),
    ]:
        df = profile[(profile["scope"] == scope) & (profile["ai_variable"] == ai_variable)].copy()
        order = {label: idx for idx, label in enumerate(PROFILE_ORDER)}
        df["order"] = df["label"].map(order).fillna(999)
        df = df.sort_values(["order", "label"])
        rows.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{esc(name)}}}}} \\")
        for _, row in df.iterrows():
            cells = [""]
            cells.append(esc(row["label"]))
            for col in ["n_no_ai", "n_q1", "n_q2", "n_q3", "n_q4", "n_q5"]:
                cells.append(fmt_count(row[col]))
            rows.append(" & ".join(cells) + r" \\")
        rows.append(r"\addlinespace")

    body = "\n".join(
        [
            "Panel & Characteristic & No AI & Q1 & Q2 & Q3 & Q4 & Q5 \\\\",
            "\\midrule",
            *rows,
        ]
    )
    note = (
        "This appendix table reports the exact number of non-missing observations for each "
        "firm characteristic and AI disclosure group. Exact row-level counts are shown because "
        "Compustat coverage differs across variables. The main text can report the cleaner "
        "profile tables without repeating sample sizes in every row."
    )
    return table_env(
        "tab:ai_profile_sample_size_ranges",
        "Exact sample sizes for AI disclosure profile tables",
        note,
        body,
        "llrrrrrr",
        size="\\scriptsize",
    )


def clean_regression_df(df, spec=None):
    out = df.copy()
    if spec is not None:
        out = out[out["specification"] == spec].copy()
    out = out[out["characteristic"].isin(REG_ORDER)].copy()
    out["dep_key"] = out["scope"] + "::" + out["dependent_variable"]
    out["order"] = out["characteristic"].map({v: i for i, v in enumerate(REG_ORDER)})
    return out.sort_values(["order", "characteristic"])


def regression_cell(row):
    if row is None:
        return "", ""
    coef = float(row["coefficient"]) * 10000
    se = float(row["std_error"]) * 10000
    stars = row.get("significance", "")
    if pd.isna(stars) or str(stars).lower() == "nan":
        stars = ""
    return f"{fmt_num(coef)}{stars}", f"({fmt_num(se)})"


def make_regression_table(df, label, caption, note, spec=None):
    reg = clean_regression_df(df, spec)
    labels = (
        reg[["characteristic", "label"]]
        .drop_duplicates()
        .assign(order=lambda x: x["characteristic"].map({v: i for i, v in enumerate(REG_ORDER)}))
        .sort_values("order")
    )

    lines = [
        "Characteristic & " + " & ".join(esc(dep_label) for _, _, dep_label in DEP_ORDER) + r" \\",
        "\\midrule",
    ]

    for _, label_row in labels.iterrows():
        coef_cells = [esc(label_row["label"])]
        se_cells = [""]
        for scope, dep, _ in DEP_ORDER:
            match = reg[
                (reg["scope"] == scope)
                & (reg["dependent_variable"] == dep)
                & (reg["characteristic"] == label_row["characteristic"])
            ]
            row = None if match.empty else match.iloc[0]
            coef, se = regression_cell(row)
            coef_cells.append(coef)
            se_cells.append(se)
        lines.append(" & ".join(coef_cells) + r" \\")
        lines.append(" & ".join(se_cells) + r" \\")

    lines.append("\\midrule")
    lines.append("Year fixed effects & Yes & Yes & Yes & Yes \\\\")
    lines.append("Industry fixed effects & SIC2 & SIC2 & SIC2 & SIC2 \\\\")
    lines.append("SE clustered by firm & Yes & Yes & Yes & Yes \\\\")

    n_line = ["Observations"]
    firm_line = ["Firms"]
    for scope, dep, _ in DEP_ORDER:
        sub = reg[(reg["scope"] == scope) & (reg["dependent_variable"] == dep)]
        n_line.append(fmt_count(sub["n"].max()) if not sub.empty else "")
        firm_line.append(fmt_count(sub["n_firms"].max()) if not sub.empty else "")
    lines.append(" & ".join(n_line) + r" \\")
    lines.append(" & ".join(firm_line) + r" \\")

    return table_env(label, caption, note, "\n".join(lines), "lrrrr")


def build_tex(profile, univ, multi, detailed_n=False):
    profile_note = (
        "% Tables \\ref{tab:item1_ai_ratio_profile}--\\ref{tab:document_ai_expertise_profile} "
        "describe which types of firms disclose more AI by comparing the zero-AI group with "
        "quintiles among AI-positive filings."
    )
    if not detailed_n:
        profile_note += (
            "\n% Table \\ref{tab:ai_profile_sample_size_ranges} gives the corresponding "
            "exact row-level sample sizes without crowding the main profile tables. We recommend "
            "placing this table in the appendix as an audit/support table."
        )
    parts = [
        "% Auto-generated from the paper_figure_tables CSV outputs.",
        "% Required LaTeX packages: booktabs, threeparttable, graphicx.",
        "% Significance levels: *** p<0.01, ** p<0.05, * p<0.10.",
        "% Coefficients in regression tables are scaled by 10,000 and therefore read as basis points.",
        "",
        "% Suggested text:",
        profile_note,
        "% Tables \\ref{tab:ai_disclosure_univariate_fe} and \\ref{tab:ai_disclosure_multivariate_fe} reverse the validation question and test whether firm characteristics predict AI disclosure after year and industry fixed effects.",
        make_profile_table(
            profile,
            "ITEM_1",
            "ai_ratio",
            "tab:item1_ai_ratio_profile",
            "Firm characteristics across ITEM 1 AI ratio groups",
            detailed_n=detailed_n,
        ),
        make_profile_table(
            profile,
            "ITEM_1",
            "ai_expertise_ratio",
            "tab:item1_ai_expertise_profile",
            "Firm characteristics across ITEM 1 AI expertise ratio groups",
            detailed_n=detailed_n,
        ),
        make_profile_table(
            profile,
            "DOCUMENT",
            "ai_ratio",
            "tab:document_ai_ratio_profile",
            "Firm characteristics across document-level AI ratio groups",
            detailed_n=detailed_n,
        ),
        make_profile_table(
            profile,
            "DOCUMENT",
            "ai_expertise_ratio",
            "tab:document_ai_expertise_profile",
            "Firm characteristics across document-level AI expertise ratio groups",
            detailed_n=detailed_n,
        ),
    ]
    if not detailed_n:
        parts.append(make_profile_n_summary(profile))
    parts.extend(
        [
            make_regression_table(
                univ,
                "tab:ai_disclosure_univariate_fe",
                "Firm characteristics predicting AI disclosure: univariate fixed-effect regressions",
                "Each column reports a separate regression of an AI disclosure measure on one standardized firm characteristic, year fixed effects, and two-digit SIC industry fixed effects. Standard errors are clustered by firm CIK and shown in parentheses. Coefficients are multiplied by 10,000, so they can be read as basis-point changes in the AI disclosure ratio for a one-standard-deviation increase in the firm characteristic.",
                spec="univariate_year_sic2_fe",
            ),
            make_regression_table(
                multi,
                "tab:ai_disclosure_multivariate_fe",
                "Firm characteristics predicting AI disclosure: multivariate fixed-effect regressions",
                "Each column reports a multivariate regression of an AI disclosure measure on standardized firm characteristics, year fixed effects, and two-digit SIC industry fixed effects. Standard errors are clustered by firm CIK and shown in parentheses. Coefficients are multiplied by 10,000, so they can be read as basis-point changes in the AI disclosure ratio for a one-standard-deviation increase in the firm characteristic. This specification is the stricter validation test because the firm characteristics enter jointly.",
                spec="core_growth_innovation",
            ),
        ]
    )
    return "\n".join(parts)


def main():
    profile = pd.read_csv(PROFILE_PATH)
    univ = pd.read_csv(UNIV_PATH)
    multi = pd.read_csv(MULTI_PATH)

    OUT_PATH.write_text(build_tex(profile, univ, multi, detailed_n=False), encoding="utf-8")
    DETAILED_OUT_PATH.write_text(build_tex(profile, univ, multi, detailed_n=True), encoding="utf-8")
    print(OUT_PATH)
    print(DETAILED_OUT_PATH)


if __name__ == "__main__":
    main()
