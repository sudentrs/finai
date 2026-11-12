# AIPD-style extension to 2024-2025 pre-grant publications

## What this model does

This model learns to predict the USPTO AIPD 2023 release's `predict93_any_ai` label from the text of a published U.S. patent application's abstract. The target equals one when any of the eight official component flags at the 0.93 operating point is positive. AIPD labels are model outputs, not human ground truth. This surrogate is not the USPTO model: it uses abstract-only word unigram/bigram TF-IDF and logistic regression, predicts one combined binary target, and does not produce separate component scores.

## Inputs

- `ai_model_predictions.csv/ai_model_predictions.csv`: AIPD teacher labels. The source audit checks the any-AI flag against the eight component flags and tests component thresholds against recorded scores. That is a corrected-release content check, not an official file signature.
- `pg_published_application.tsv/pg_published_application.tsv`: PatentsView publication dates, application IDs, and utility status. The existing SQLite frame is reused read-only after checking its source fingerprint. It keeps the first observed pre-grant publication per utility application to avoid counting repeat publications of the same application as distinct applications. This is not patent-family deduplication.
- `publication pack/pg_published_application_abstract.tsv.zip`: PatentsView text, joined by publication ID. The AIPD join is restricted to pre-grant records and must agree on application ID and publication date. Missing labels are not negatives.
- Prior pilot sample manifest: only application IDs are read to keep those documents out of the historical sample. No human or AI pilot labels are read. No professor data, 10-K measures, returns, issuer identifiers, or company matching lists are used.

## Evaluation and deployment

A seeded, label-blind sample is drawn from publication years 2015-2023. The drawing rates are 3% for each 2015-2021 cohort, 6% for 2022, and 8% for 2023. Model fitting uses 2015-2021 only. The 2022 sample sets the surrogate's threshold by maximizing F1; the surrogate threshold is distinct from AIPD's 0.93 target definition. The model is saved and fingerprinted before evaluating agreement on 2023. The same model and operating threshold are applied unchanged to all eligible first utility publications in the supplied 2024-2025 frame. The model is not refit after the benchmark.

The temporal benchmark is application-disjoint and removes exact normalized abstract matches across splits, giving priority to the later split. Related applications with nonidentical abstracts may remain. A documented family mapping is required before claiming family-disjoint evaluation. The benchmark measures agreement with AIPD, not independent patent-level accuracy. There is no reference label for 2024-2025 in this source; recent predictions require a separate audit before being treated as equivalent to official AIPD labels. Aggregate count correlation alone cannot certify individual predictions.

Publication year defines the scoring window, not filing year. This extends classification to observable records, not coverage of unpublished applications. Blanks indicate unavailable scores, not negative classifications. Do not replace these with zeros when aggregating. The input metadata ends December 25, 2025; no assumptions are made about releases after that date.

## Run and outputs

```powershell
python patent_extension/aipd_extension.py
python patent_extension/test_aipd_extension.py
```

The program writes to a fresh `patent_extension/aipd_surrogate_v1` directory by default. Key files: `source_audit.json`, `historical_split_manifest.csv`, `freeze.json`, `frozen_surrogate.joblib`, `benchmark.json`, `heldout_2023_predictions.csv`, `recent_coverage.csv`, and `publications_2024_2025.csv`. The staged abstract file is an intermediate, not an analysis output. Use `--resume-score` to finish scoring from the staged abstracts and verified frozen model after an interrupted benchmarked run.

Sources: [USPTO AIPD](https://www.uspto.gov/ip-policy/economic-research/research-datasets/artificial-intelligence-patent-dataset) and [PatentsView bulk data](https://patentsview.org/download/data-download-tables).
