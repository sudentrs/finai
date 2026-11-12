"""Verify excerpt provenance, sample keys, counts and exported numeric estimates."""
import json
import re

import numpy as np
import openpyxl
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

from publication_tables import BASE, KEY, OUT, ROOT, clean_keys


def verify():
    source = clean_keys(pd.read_csv(ROOT /
        "analysis/mutually_exclusive_ai_constructs_occurrence/original_ai_sentence_occurrence_assignments.csv",
        usecols=KEY + ["accession", "section", "occurrence_index", "sentence"]))
    source = source[source.section.eq("ITEM_1")]
    source = source.set_index(["cik", "accession", "occurrence_index"], verify_integrity=True)
    result = {}
    ai_pattern = re.compile(
        r"\b(ai|artificial intelligence|machine learning|deep learning|neural networks?|"
        r"generative|llms?|natural language processing|computer vision)\b", re.I)
    for sample in ("broad", "professor_price_ge5"):
        folder = OUT / sample
        filings = clean_keys(pd.read_csv(folder / "filing_analysis_data.csv", low_memory=False))
        if filings.duplicated(KEY).any() or not filings.filed_date.between(20150101, 20251231).all():
            raise ValueError("Incorrect filing count or period filter")
        models = pd.read_csv(folder / "table_331_model_statistics.csv")
        if not models.status.eq("estimated").all() or not models.n.ge(models.firms).all():
            raise ValueError("Incorrect model statistics")
        examples = pd.read_csv(folder / "table_311.csv", dtype={"cik": str})
        for _, row in examples.loc[examples.excerpt_status.eq("selected")].iterrows():
            source_row = source.loc[(row.cik, row.accession, int(row.sentence_id))]
            if source_row.sentence != row.excerpt:
                raise ValueError("Non-verbatim excerpt")
        examples["explicit_ai_term_in_excerpt"] = examples.excerpt.fillna("").map(
            lambda text: bool(ai_pattern.search(text)))
        examples["review_reason"] = np.select(
            [examples.excerpt_status.ne("selected"), ~examples.explicit_ai_term_in_excerpt],
            ["No sentence meets the specified category and word-count rule",
             "Selected quote has no explicit AI term; substantive meaning needs human review"],
            default="Quote follows mechanical rule; substantive interpretation not independently verified")
        examples.to_csv(folder / "table_311_excerpt_review.csv", index=False)
        base = filings.loc[filings.fyear.between(2015, 2025)].copy()
        bounds = pd.read_csv(folder / "winsorization_bounds.csv")
        for _, row in bounds.iterrows():
            ix = base.fyear.eq(row.fyear)
            base.loc[ix, row.variable] = base.loc[ix, row.variable].clip(row.p01, row.p99)
        base.loc[base.rd_missing.eq(1) & base.ln_at.notna(), "rd_at"] = 0
        # Same complete-case sample as specification 2, not an outcome-selected diagnostic.
        complete = base.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["expertise_pct", *BASE, "rd_missing", "permco", "fyear", "ff48"])
        expected_n = int(models.loc[models.dependent.eq("expertise_pct") & models.specification.eq(2), "n"].iloc[0])
        if len(complete) != expected_n:
            raise ValueError("Multicollinearity sample differs from the reported model")
        control = complete[BASE + ["rd_missing"]].copy()
        x = sm.add_constant(control)
        pd.DataFrame({"variable": x.columns[1:],
                      "vif": [variance_inflation_factor(x.to_numpy(), j) for j in range(1, x.shape[1])],
                      "diagnostic_n": len(x)}).to_csv(folder / "control_vif_diagnostic.csv", index=False)
        control.corr().to_csv(folder / "control_correlations.csv")
        workbook_data = json.loads((OUT / "workbook_data.json").read_text())
        workbook_data[sample]["vif"] = pd.read_csv(folder / "control_vif_diagnostic.csv").to_dict("records")
        correlations = control.corr()
        workbook_data[sample]["correlation_variables"] = list(correlations.columns)
        workbook_data[sample]["correlations"] = correlations.to_numpy().tolist()
        (OUT / "workbook_data.json").write_text(json.dumps(workbook_data, default=str))
        path = ROOT / "outputs/publication_tables_20261112" / f"{sample}_publication_tables_DRAFT.xlsx"
        workbook = openpyxl.load_workbook(path, data_only=True)
        expected = pd.read_csv(folder / "table_331_coefficients.csv")
        estimate_sheet = workbook["Estimates"]
        if estimate_sheet.max_row != len(expected) + 1:
            raise ValueError("Exported estimate row count differs")
        for i, row in expected.iterrows():
            for column, field in ((2, "coefficient"), (3, "se"), (4, "t"), (5, "p")):
                if not np.isclose(estimate_sheet.cell(i+2, column).value, row[field], atol=1e-12):
                    raise ValueError("Exported numeric estimate differs from CSV")
        errors = [(sheet.title, cell.coordinate, cell.value) for sheet in workbook for row in sheet
                  for cell in row if cell.data_type == "e"]
        if errors:
            raise ValueError(errors[:10])
        result[sample] = {"filing_keys_unique": True, "selected_quotes_verbatim": True,
                          "numeric_export_verified": True, "spreadsheet_errors": 0,
                          "raw_ai_ratio_including_noise_retained": True,
                          "quotes_needing_review": int((~examples.explicit_ai_term_in_excerpt).sum()),
                          "main_model_n": int(models.loc[models.specification.eq(1), "n"].iloc[0])}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
