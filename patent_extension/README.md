# Patent extension progress

A prospective, finance-independent evaluation redesign is now documented separately in [independent_validation_v2/PROTOCOL.md](independent_validation_v2/PROTOCOL.md). Its patent-only review draw supersedes the earlier company-linked queue as the primary evaluation design. Human annotation and family/training-overlap clearance remain pending; it is not a completed validation result.

The abstract-based computational run is now complete. Current results and limitations are in [FULL_RUN_REPORT.md](FULL_RUN_REPORT.md). The model has not passed publication validation; the earlier title pilot below is retained as a historical engineering checkpoint, not the final classifier.

Started 7 November 2026. All files are separate from the earlier patent pipeline and the professor's publication pack. This folder contains preliminary research code and audits, not replacement paper results.

## Completed in the first run

1. Inspected the full delivered monthly panel for sample keys, ratio consistency and candidate price-screen counts.
2. Created a candidate filing roster and a stock-month-to-filing crosswalk without changing source values.
3. Pinned and downloaded bounded AIPat reference files and its external labelled benchmark. No downloaded notebooks, pickle files or model weights were executed.
4. Audited all 15,437,542 local AIPD rows for score/93%-indicator consistency.
5. Audited publication and filing-year coverage in 8,402,596 local pre-grant publication rows.
6. Kept the external benchmark out of training and tuning. Its application identifiers were excluded from the diagnostic training data.
7. Ran a diagnostic TF-IDF/logistic model on publication titles only, using official AIPD labels as targets and temporal train/validation/test partitions.

## Main results

### Professor sample

- Delivered panel: 378,726 stock-months; candidate price >= $5 sample: 299,126.
- Candidate screened sample averages 2,266.106 stocks per month, consistent with the manuscript's rounded count.
- There are 32,943 distinct CIK/filing-date/period-end keys in the delivered panel. These are candidate filing keys; accession-level reconciliation remains pending.
- Ratios do not vary within those filing keys in this file. No filing key has multiple PERMCO values.
- `filed_year` and `filed_month` coincide with the panel year/month throughout; original numeric `filed_date` and `period_end` are retained in the crosswalk.
- Ratios reproduce their respective count/total-sentence calculations within the numerical tolerance checked.
- Two filings nevertheless have expertise counts above AI counts: Corteva, 14 February 2025 (2 versus 1), and Sempra Energy, 27 February 2020 (2 versus 1). These repeat across 23 stock-month observations, all passing the candidate price screen. Neither source observation was corrected or removed.
- Final winsorization/other transformations remain unconfirmed. Candidate screened ratio maxima do not reproduce the manuscript maxima.

Detailed outputs: `outputs/sample_audit/`. The delivered panel is not the earlier broad filing sample. No broad-sample monthly carry-forward panel has been constructed in this phase.

### Local patent data

- Local AIPD rows: 15,437,542.
- Disagreements between `predict93_any_ai` and any component score >= 0.93: **zero**.
- Invalid/missing/out-of-range component-score rows: **zero**.
- This verifies score consistency in the content; it does not establish the original download's official checksum or release provenance.
- Local pre-grant publication rows: 8,402,596.
- Latest local publication: **25 December 2025**.
- One metadata row has a missing/invalid filing or publication date.
- The local publication metadata file has titles but no abstracts or claims.

Detailed outputs: `outputs/local_patent_audit/`. Counts here are publication rows, not deduplicated applications or patent families.

### External benchmark audit

The pinned AIPat repository describes its benchmark as human-labelled. We verified the file has 4,000 abstracts, including 802 positives and 3,198 negatives. All are granted patents, published between 1981 and 2018. This does not substitute for a new-period pre-grant application benchmark. Annotation protocol and sampling weights remain to be inspected.

The benchmark's embedded `predict93_any_ai` indicator differs in 51 cases from a literal >= 0.93 rule using its embedded component scores. The two constructions are therefore reported separately. These findings do not establish the precise cause of the discrepancies. Indicators in this downloaded benchmark have not yet been matched row by row to the current local official file.

For the separately recomputed >= 0.93 benchmark flag, sample precision is 60.6%, recall 79.3%, and F1 68.7%. These figures describe this benchmark and its `actual` labels, not population-level accuracy. They are not results for our classifier.

Detailed outputs: `outputs/external_benchmark_audit/`.

### Diagnostic title-only model

The model is a low-cost reference point, not the planned final patent classifier. It uses no firm disclosure or financial outcomes as features.

- Deterministic approximately 1% publication-ID sample: 35,826 publication records from 2015-2023.
- Unmatched official labels: 131 records, retained in the label-availability audit and not treated as negative.
- Applications and normalized identical titles are separated globally before partitioning.
- Ten sampled records overlapping reserved benchmark application IDs were removed before fitting.
- Training: 2015-2020, 22,372 observations.
- Validation: 2021, 3,942 observations.
- Test: 2022-2023, 7,995 observations.
- TF-IDF title unigrams/bigrams with logistic regression. Hyperparameters fixed in code; threshold selected by validation F1, not test performance.
- Selected diagnostic threshold: 0.27. This is model-specific and is not an AIPD threshold or a new threshold for the 10-K classifier.
- Test precision against AIPD labels: **61.4%**.
- Test recall against AIPD labels: **71.4%**.
- Test F1 against AIPD labels: **66.0%**.
- Logistic fit completed in nine iterations, below its 1,000-iteration limit.

These are agreement metrics against a machine-generated label target, not independently established AI accuracy. Broader patent-family overlap is not yet controlled. No final-model promotion is justified by these results. The external human-labelled benchmark was not used to choose this model or its threshold. No full-universe predictions or patent regressions were generated.

Detailed outputs: `outputs/title_diagnostic_pilot/`. `heldout_predictions.csv` records test predictions, and `split_manifest.csv` records the partition membership.

## External resource inventory

Source: https://github.com/yusu2022/AIPat-Data-and-Methods-for-Identifying-Artificial-Intelligence-Related-Patents

Pinned commit: `24621519e2d7bf63bb6c609b0e632ab657629124`.

The downloaded source manifest includes URLs, SHA-256 hashes and Git LFS pointer metadata. The benchmark content hash was checked against its LFS pointer.

Files not downloaded in this phase include approximately 1.16 GB of training data, 416 MB of CNN model weights, an 11 MB dictionary, and 285 MB of pre-grant predictions. The repository inventory contains no explicit license file. External code/model reuse and redistribution permissions are not established by this audit. No environment packages were installed; PyTorch is absent from the current system Python environment.

Downloaded reference files are stored under `references/aipat/`, not mixed with project training data.

## Remaining work

- Obtain abstract text, preferably claims, for the relevant patent universe and verify its publication cutoff.
- Establish family identifiers/continuity links before final training and evaluation splits.
- Inspect the external benchmark's annotation and sampling methodology; add blinded recent-period application labels.
- Build and evaluate an abstract/claims baseline, with separate model-selection and test sets.
- Reconcile the two filing count inconsistencies and the manuscript's ratio transformations with the source pipeline/professor's implementation.
- Reconcile exact filing/fiscal-year control definitions, patent-assignee-to-parent ownership links, and final PPML table specification.
- Construct broad and screened sample outputs separately, and preserve official AIPD robustness results as a distinct measure.

## Reproducing this phase

Run from `C:\Users\sudet\Downloads\finai` with the existing system Python environment:

```powershell
python patent_extension/audit_professor_sample.py
python patent_extension/bootstrap_sources.py
python patent_extension/audit_external_benchmark.py
python patent_extension/audit_local_patent_sources.py
python patent_extension/pilot_title_baseline.py
```

The bootstrap requires network access. Other scripts read the already available local sources. The title pilot uses its saved sample on later runs; changing source versions requires removing or renaming that derived sample explicitly, not silently reusing it for a new source vintage.
