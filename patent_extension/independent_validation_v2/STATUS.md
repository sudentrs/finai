# Validation status: 7 November 2026

## Completed

The version 2.1 review sample was drawn from patent records without reading the professor's financial panel, disclosure measures, company matching list or model predictions. The sampling unit is the first observed utility pre-grant publication per application, with publication years 2015-2025.

| Publication cohort | Eligible applications after prior-benchmark exclusion | Pilot | Development | Reserved test | Challenge available for review |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015-2018 | 1,506,582 | 20 | 200 | 600 | 200 |
| 2019-2022 | 1,632,565 | 20 | 200 | 600 | 200 |
| 2023-2025 | 1,235,067 | 20 | 200 | 600 | 199 |
| Total | 4,374,214 | 60 | 600 | 1,800 | 599 |

There were 3,060 sampled applications, all with nonempty abstracts. One challenge abstract duplicated text in another role and was quarantined without replacement, leaving 3,059 review documents. The representative reserved test remains 1,800 documents. All reviewer labels remain blank. Previously inspected benchmark applications were excluded before selection; 1,310 applications were excluded from the earliest cohort. No reserved test application appears in that benchmark.

Four automated design tests passed. The clearance checker correctly returns a non-ready result rather than treating missing evidence as successful validation. Original scripts, fitted models and the superseded initial draw were preserved.

## Still required before confirmatory evaluation

1. Approve the abstract-only annotation rubric, candidate plan and operating criteria before examining reserved-test labels.
2. Obtain a documented application-family crosswalk and the proposed training-family manifest. Clear training/evaluation and development/test family overlap, including appropriate near-duplicate checks. USPTO application IDs cannot simply be equated with PATSTAT or DOCDB identifiers.
3. Conduct the pilot, freeze the rubric, and obtain independent dual annotations and adjudication under the protocol. Uncertain labels must not silently become negatives.
4. Train/select candidates using training and development data only, freeze the model and threshold, then have an independent evaluator assess the reserved test. Report cohort-weighted performance and uncertainty; report the enriched challenge set separately.
5. Only after the measurement procedure is frozen, merge patent predictions with the economic samples and assess the disclosure-patenting relationship. Preserve ownership-link coverage and publication-lag limitations separately from classification accuracy.

## Interpretation

This is a prospective internal validation plan following exploratory work, not a retrospective preregistration. The old classifier has not become publication-ready merely because this review sample exists. Agreement with AIPD is teacher agreement, not independent human accuracy; high firm-level count correlations alone cannot establish patent-level validity. The professor's sample belongs in the later economic analysis, not classifier or threshold selection.
