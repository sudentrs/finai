# AIPD extension v2: forward evaluation and recent audit

This is a separate comparison and extension. The v1 model, predictions and benchmark remain unchanged. All model inputs are U.S. utility pre-grant publication abstracts; all targets come from the corrected AIPD 2023 release. There are no issuer, disclosure, financial or pilot labels in the model.

## Tests and model selection

- Compare four recipes: direct binary versus eight AIPD component classifiers, each with equal document weights or recency weights with a two-year half-life. The historical sampling rates already differ by year; the recency weights are additional weights on that fixed sample.
- In each forward test (2021 and 2022), fit on prior publication years and reserve a seeded 25% of the immediately preceding year for threshold selection. Fit text features on fitting data only. A component model uses the maximum of eight component scores to predict the overall label, with a separate overall threshold. The maximum is a ranking score, not a calibrated probability of any AI. Each head's own threshold is evaluated separately.
- Choose by mean F1 across the two forward tests with equal weight on each test year. Evaluate the chosen recipe on a newly drawn 2023 sample, not the v1 holdout already examined. Separately compare v1 and v2 on the same subset, excluding exact-text matches to v1 fitting and calibration records. Report precision, recall, F1, average precision, and confusion matrices, including the component heads and combined recall within each official component. Components can overlap; their counts do not sum to the overall AI count.
- Application IDs are unique and exact normalized abstract matches are excluded from earlier splits. Patent-family overlap involving different abstracts remains unresolved. The fresh benchmark is newly drawn for this comparison, not a guarantee that its documents were never encountered in other projects.
- Refit the chosen recipe through 2023, holding back a separate 2023 calibration sample and excluding the fresh benchmark and exact matches. Freeze the model and operating threshold, then apply unchanged to both 2024 and 2025. Benchmark and deployment are different fitted instances of the same recipe; the prospective benchmark metrics do not directly test the refitted deployment instance.

## Claims and family data

No comparable pre-grant claims archive or family crosswalk was located in the workspace. USPTO's [pre-grant XML documentation](https://www.uspto.gov/learning-and-resources/xml-resources) describes the relevant text format, and the [AIPD methodology](https://www.uspto.gov/sites/default/files/documents/oce-aipd-2023.pdf) uses abstracts and claims. The claims experiment requires actual publication-keyed claims in both periods. Do not substitute later granted claims for pre-grant claims; that would change the target text and availability. When acquired, report join and usable-text rates by year and compare abstract-only and abstract-plus-claims models on the same eligible records and splits.

## Recent blind audit

The audit draws 50 records per stratum: publication year (2024 or 2025), predicted class (positive or negative), and nearer or farther half of the score distances within that year/class. This yields 400 abstracts, not an unweighted estimate of model accuracy. The curator manifest stores inclusion probabilities and inverse-probability weights; previous review-set applications are excluded. The review files show abstract text and a masked ID, but not predictions, years, or scores. The review should be independent, with adjudication only afterward. Use the [abstract rubric](independent_validation_v2/ANNOTATOR_INSTRUCTIONS.md); assess U and its effect separately from negatives. If the target is full-document AI rather than abstract-visible AI, evaluate claims in a separate pass.

Use sampling weights to estimate precision and recall from adjudicated labels, reporting uncertainty and unresolved cases. The historical benchmark measures AIPD agreement; the recent human audit, once complete, measures agreement with human judgments. Neither should be represented as the other. Unpublished applications remain unobservable, so publication years are not complete filing cohorts.

## Reproduction

```powershell
python patent_extension/aipd_v2_prepare.py
python patent_extension/aipd_v2_model.py
python patent_extension/aipd_v2_audit.py
python patent_extension/test_aipd_v2.py
```

Outputs are under `aipd_surrogate_v2`. Send only the separate reviewer files and the rubric to reviewers; keep the curator file and predictions separate.
