from pathlib import Path

import pandas as pd


BASE = Path("analysis/patent_validation_improved/extended_validation_analysis/company_characteristics_ai_analysis")
SUMMARY_PATH = BASE / "paper_figure_tables" / "ai_ratio_expertise_by_year_summary.csv"
SOURCE_FILES = {
    "ITEM_1": BASE / "ITEM_1_filing_with_company_characteristic_quintiles.csv",
    "DOCUMENT": BASE / "DOCUMENT_filing_with_company_characteristic_quintiles.csv",
}


def main():
    summary = pd.read_csv(SUMMARY_PATH)
    print(f"Summary rows: {len(summary)}")
    print(f"Summary years: {summary['year'].min()}-{summary['year'].max()}")

    all_recomputed = []
    for scope, path in SOURCE_FILES.items():
        df = pd.read_csv(
            path,
            usecols=[
                "year",
                "form_type",
                "scope",
                "ai_ratio",
                "ai_expertise_ratio",
                "ai_sentences_count",
                "ai_expertise",
                "total_sentences",
                "accession",
            ],
        )
        print("\n" + "=" * 72)
        print(scope)
        print(f"Rows: {len(df):,}")
        print(f"Year range in file: {df['year'].min()}-{df['year'].max()}")
        print(f"Form types: {df['form_type'].value_counts(dropna=False).to_dict()}")
        print(f"Scope values: {df['scope'].value_counts(dropna=False).to_dict()}")

        d = df[df["year"].between(2015, 2025)].copy()
        print(f"Rows in plotted window, 2015-2025: {len(d):,}")
        print(f"Unique accessions in plotted window: {d['accession'].nunique():,}")
        print(f"Duplicate accessions in plotted window: {d.duplicated('accession').sum():,}")
        print("Missing values:", d[["year", "ai_ratio", "ai_expertise_ratio", "total_sentences"]].isna().sum().to_dict())

        valid = d["total_sentences"].fillna(0).ne(0)
        ai_calc = d.loc[valid, "ai_sentences_count"] / d.loc[valid, "total_sentences"]
        expertise_calc = d.loc[valid, "ai_expertise"] / d.loc[valid, "total_sentences"]
        ai_diff = (d.loc[valid, "ai_ratio"] - ai_calc).abs()
        expertise_diff = (d.loc[valid, "ai_expertise_ratio"] - expertise_calc).abs()
        print(f"Rows with zero/missing total_sentences: {(~valid).sum():,}")
        print(f"Max abs ai_ratio minus ai_sentences_count/total_sentences: {ai_diff.max():.12g}")
        print(f"Max abs ai_expertise_ratio minus ai_expertise/total_sentences: {expertise_diff.max():.12g}")

        recomputed = (
            d.groupby("year", as_index=False)
            .agg(
                ai_ratio=("ai_ratio", "mean"),
                ai_expertise_ratio=("ai_expertise_ratio", "mean"),
                n_filings=("year", "size"),
            )
        )
        recomputed["scope"] = scope
        recomputed["ai_ratio_pct"] = recomputed["ai_ratio"] * 100
        recomputed["ai_expertise_ratio_pct"] = recomputed["ai_expertise_ratio"] * 100
        all_recomputed.append(recomputed)
        print("Year counts:")
        print(d.groupby("year").size().to_string())

    recomputed_all = pd.concat(all_recomputed, ignore_index=True)
    merged = summary.merge(
        recomputed_all,
        on=["scope", "year"],
        suffixes=("_summary", "_recomputed"),
        how="outer",
        indicator=True,
    )
    print("\n" + "=" * 72)
    print("Summary-vs-source recomputation check")
    print(f"Merge status: {merged['_merge'].value_counts().to_dict()}")
    for col in ["ai_ratio", "ai_expertise_ratio", "n_filings", "ai_ratio_pct", "ai_expertise_ratio_pct"]:
        left = f"{col}_summary"
        right = f"{col}_recomputed"
        if left in merged and right in merged:
            diff = (merged[left] - merged[right]).abs().max()
            print(f"Max abs diff for {col}: {diff}")


if __name__ == "__main__":
    main()
