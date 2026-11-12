# Pipeline map

## Disclosure processing

- process_forms_no_setion.py: historical processing baseline; preserve as legacy, not the new publication entry point.
- export_results_sections_sample_style.py, export_results_sections_document_level.py, export_section_sample_style.py: filing and document exports from the processed corpus.
- item1_ai_topic_exposure.py, document_ai_topic_exposure.py: exploratory topic measures.
- mutually_exclusive_ai_constructs.py: initial deduplicated analysis.
- mutually_exclusive_ai_constructs_occurrence.py: occurrence-based revision; training deduplication and final occurrence counting are distinct decisions.
- create_professor_filing_package.py: standardized filing exports retaining identifiers.

## Tables and figures

- publication_tables.py: new two-sample draft of Tables 3.1.1 and 3.3.1. Consult the release checklist before use.
- publication_examples_tex.py: source-faithful excerpt longtables.
- publication_tables_workbook.mjs: two draft workbooks, including numeric estimates, counts, definitions, correlations and VIF.
- verify_publication_tables.py and test_publication_tables.py: quotation/export checks and synthetic fixed-effect checks.
- company_characteristics_ai_analysis.py and company_characteristics_stat_tests.py: earlier characteristic-based analysis.
- ai_disclosure_validation_regressions.py, ai_positive_quintile_profile_tables.py, ai_positive_quintile_profile_regressions.py: earlier regression and group-profile designs, not interchangeable with the new templates.
- make_ai_ratio_time_series_figures.py, make_ff12_sector_ai_figures.py, industry_market_extensions.py: earlier figure builders; the new monthly FF48 layouts are still pending.

## Earnings calls

earnings_call_ai_indexes.py and earnings_call_ai_indexes_using_10k_topics.py are preserved research scripts. The projected-topic script refits a 10-K vectorizer/NMF and uses a regex candidate filter; it should not be described as applying a serialized, identical supervised sentence classifier without verification. The new filing-window table requires call-level dates and numerators/denominators, which are absent from the present workspace snapshot.

## Patent extension

- run_integrated.py and abstract_model.py: initial exploratory AIPat-based prototype, not the later AIPD surrogate.
- aipd_extension.py: v1 AIPD-trained abstract baseline.
- aipd_v2_prepare.py, aipd_v2_model.py, aipd_v2_audit.py: recency/component comparison, recent scores and blank recent audit packs.
- aipd_v3_features.py: title/character challengers; no promotion after insufficient gain.
- aipd_claims_benchmark.py: matched small claims-versus-abstract comparison.
- aipd_claims_fresh_sample.py, aipd_claims_expanded.py: larger claims ensemble and later test sample.
- aipd_soft_teacher.py: continuous-score challenger; calibration selected zero contribution.
- aipd_continuity_audit.py: partial U.S. recorded-link screen, not global family clearance.
- aipd_patent_bert_pilot.py: small frozen-encoder feasibility run.
- aipd_bert_next_prepare.py, aipd_bert_next_run.py, aipd_bert_next_evaluate.py: larger sealed-test preparation, remote/local embedding job and later evaluation. No test labels in the GPU package.
- aipd_recent_claims_coverage.py: source coverage audit, not scoring.

Multiple experiments already examined older test sets. Those are diagnostics, not reusable untouched tests. The larger sealed test remains separate and must not be used for candidate tuning.
