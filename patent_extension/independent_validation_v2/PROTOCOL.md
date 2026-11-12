# Finance-independent patent classifier validation protocol

Version 2.1, 7 November 2026. Prospective protocol for subsequent work, written after the prototype and its historical benchmark results were inspected. This is not a retrospective claim of preregistration, and it is not a completed validation study.

Version 2.0's initial unlabelled draw is preserved in `outputs/`. Before any new labels were collected, its exposure check found one reserved-test application in the already inspected historical benchmark. Version 2.1 excludes the historical benchmark's application IDs from the eligible review frame before drawing. It also quarantines cross-role exact-text duplicates, preserving the reserved test ahead of development, pilot and challenge roles. The current reviewer files are in `outputs_v2_1/`; the older draw is not the primary test.

## Research purpose and independence

The measurement question is whether a patent document contains substantive AI-related invention. The economic question, considered only after the measure is frozen, is whether independently measured patenting is associated with firms' AI disclosure. Neither the expected economic relationship nor its statistical significance determines the patent definition, model, threshold or evaluation sample.

The professor's financial panel belongs only to the later economic-analysis layer. Restricting that analysis to the paper's firms is necessary and is not itself training leakage. The earlier prototype's training did not read that panel, but deployment and the proposed recent review sample were restricted to its existing matched-patent universe. This limits evaluation generalizability and makes matching selection a distinct concern. The new primary review frame is constructed without that restriction.

`build_blind_review.py` reads only the supplied USPTO publication metadata, supplied abstract archive and patent-only sampling configuration. It does not read CIKs, CRSP/Compustat identifiers, 10-K classifications, stock returns, model predictions or previous company-patent links. Input lineage is recorded in `outputs/independence_audit.json`.

This design targets utility applications with an observed pre-grant publication in the supplied U.S. patent data. It does not target only U.S.-headquartered assignees, only listed firms, all inventions worldwide, or confidential applications that never appear as pre-grant publications. A later firm-sample analysis must identify its narrower target explicitly.

## What the prototype establishes and does not establish

- Its code and weights were fitted independently of the 10-K classifier. Its labels came from the public AIPat research training corpus; the target was not the official AIPD 93% flag.
- Very high internal family/time agreement coexists with 53.9% precision on an already inspected historical grant benchmark. This is a warning about label definition, corpus selection and domain transfer, not evidence that a more elaborate encoder alone will solve the problem.
- AIPD is itself a model-based measurement source, not ground truth. Agreement with AIPD is benchmark agreement. Learning its labels and then agreeing with it is not independent validation.
- Approximately 0.99 correlation in company-level counts does not establish patent-level validity. Total patenting scale, aggregation, common ownership links and many zero observations can sustain high count correlation despite disagreements about individual patents.
- The old historical benchmark and internal test have already influenced the research team's understanding of weaknesses. They remain reference diagnostics, not untouched confirmatory tests for future model variants.
- The old 600-document review queue was drawn from linked project firms. It is superseded as the primary general-population validation design, but can remain a separate, explicitly conditional financial-sample audit.

The precision/recall gap does not, on its own, identify its cause. Label quality, definitions, selection, preprocessing mismatch, historical-versus-modern language, and abstract-versus-claims information are separate hypotheses to investigate on development data.

## Construct and information set

Primary target: substantive AI methods or substantive use of AI in the invention described by its abstract. Merely using a computer, database, conventional equation, image sensor, optimization routine or automated control does not by itself establish AI content. Positive definitions cover learning methods, neural networks, language processing, AI-related perception/speech, knowledge-based inference and AI-specific planning/control or hardware. Conventional control and general-purpose hardware do not automatically qualify. Generative AI is a subset, not the whole construct.

These are proposed operational guidelines. They need to be reconciled with the primary AIPD paper's component definitions during the pilot and frozen before main-test annotation. The eight component labels may overlap; any-AI is a separate binary summary. The classifier's scores must not be presented as calibrated probabilities unless calibration has been assessed.

The primary assessment is abstract-only. Annotators record 0, 1 or U (uncertain/insufficient information), a brief reason and an evidence span. If claims or descriptions are consulted, that produces a separately documented full-document assessment; it is not silently substituted for an abstract-only label. Claims-only AI can explain why an abstract classifier disagrees with a claims-and-abstracts benchmark without necessarily being a model error under the narrower abstract target.

## Patent-only sampling frame

1. Read the full metadata extract, including years before 2015, so republications of an old application are not mistaken for a newly published application.
2. Keep utility publications with valid application/publication identifiers and a parseable publication date.
3. Retain the earliest observed publication per application, breaking same-date ties by publication number. Application-level counts do not count republications twice.
4. Restrict first observed publications to 2015-2025. Use three explicit publication cohorts: 2015-2018, 2019-2022 and 2023-2025.
5. Exclude previously inspected historical-benchmark application IDs, using identifier columns only, not their labels. The primary review population is therefore the eligible first-publication frame minus that documented reference set. Counts before/after exclusion are recorded. Generalization to the small excluded reference set is not established by the new test.
6. Draw using deterministic pseudorandom SHA-256 ordering of patent identifiers. The seed, exact identifiers, source-file hash, cohort population counts and manifest hash are retained.

"First observed" means first in the supplied extract, not an externally established earliest worldwide priority or first-ever publication. Metadata gaps and identifier errors can affect that distinction. The abstract archive is joined after sampling; missing or unusable abstracts are recorded rather than quietly replaced with easier cases.

### Proposed review allocation

| Role | Per cohort | Total | Purpose |
|---|---:|---:|---|
| Pilot | 20 | 60 | Refine the rubric and train annotators; not performance evidence. |
| Development | 200 | 600 | Diagnose errors, compare candidate methods and select the operating rule. |
| Reserved representative test | 600 | 1,800 | Final evaluation after all clearance and freezing steps. |
| Challenge set | 200 | 600 | Separate stress test of ambiguous technical language. |

Total: 3,060 documents, with two independent reviewers proposed. These are review-budget allocations, not a claim of sufficient power in every subgroup. Precision depends on the number of predicted positives and recall on the number of actual positives, not simply on total N. The pilot can inform a prospective sample-size calculation before main-test labels or predictions are inspected. Any enlargement must be documented, drawn without looking at test performance, and carry updated inclusion probabilities. No guarantee of a publication-quality confidence interval follows from this initial allocation.

The table gives draw counts. Non-test cross-role duplicate abstracts are quarantined before annotation rather than quietly replaced. `sampling_counts.csv` reports the resulting review counts. Test membership and its weights are preserved. Pilot/development results after any such quarantine describe those development cases; they are not used as weighted population-performance claims.

The representative roles are simple random samples without replacement within publication cohort. Each test observation has inclusion probability n_h/N_h and analysis weight N_h/n_h. Equal cohort allocations do not imply equal cohort population sizes. Report both cohort results and the appropriately weighted combined result. The sampling manifests contain the weights.

The challenge set uses only a fixed title pattern involving potentially ambiguous words such as model, training, image and automation, excluding already selected applications. It is enriched intentionally. Its unweighted performance describes that challenge set only; it must not be combined unweighted with the representative test. It is not a population prevalence estimate.

## Family and training overlap clearance

Application deduplication and exact-text checks are not a complete patent-family safeguard. The current supplied files lack a global mapping between USPTO applications, continuity/priority relationships and the DOCDB/PATSTAT family identifiers used in the external training corpus. Therefore the newly reserved sample is **provisional, not yet cleared for confirmatory use**.

Before final training:

- Obtain the relevant global application-family or continuity/priority mapping, document its vintage and define which relationships constitute a family. If only U.S. continuity links are available, describe that narrower grouping rather than calling it complete worldwide DOCDB grouping.
- Preserve representative test selection. Remove all test families from development and candidate training corpora before labels are used. Do not discard difficult test cases to make overlaps disappear. Document any replacement draws needed in development, not an outcome-dependent reshuffle of the test.
- Check raw/normalized text duplicates and a prespecified near-duplicate rule. Near-duplicate thresholds are chosen on development data; test labels are not needed for overlap detection.
- Explicitly reconcile the external training corpus's PATSTAT application IDs and DOCDB family IDs to the USPTO identifiers. Exact text matching alone did not establish that no family overlaps existed in the prototype.
- If reliable overlap clearance cannot be obtained, use a genuinely later independent evaluation source or change the strength and scope of claims; do not state that family independence has been demonstrated.

Test membership may be known to the data curator for decontamination. This is different from allowing test outcomes to influence model development. The folder name does not provide encryption or access control; the independent evaluator and research team need an actual procedural division of responsibilities.

## Annotation and adjudication

Two reviewers label independently before discussion. The blind files contain randomized document IDs, titles and abstracts only: no firm identifiers, disclosure scores, returns, model scores, official flags or known labels. Patent text itself can contain identifying information, so metadata masking does not imply perfect anonymity.

Record reviewer labels, reasons, uncertain cases and missing abstracts. Assess raw agreement and an imbalance-aware description of class-specific disagreement; Cohen's kappa can be reported with its prevalence limitations, not as a sole quality criterion. Resolve disagreements using an adjudicator and retain the original independent labels. The developer should not adjudicate the final test using model predictions.

Uncertain and missing-information cases remain in the flow diagram and coverage statistics. Report a determinate-case analysis alongside sensitivity bounds or explicit alternative treatments of uncertain labels. Do not recode U or blank as non-AI. If a material fraction cannot be assessed from abstracts, that is a limitation of the measurement target and source, not simply an annotation inconvenience.

Pilot examples, development examples and reserved-test examples are separate. Training annotators on inspected test mistakes, revising the rubric after seeing test results, or using test labels to set a new threshold consumes that test; another independent test is then required.

## Candidate models and threshold selection

The candidate list is recorded before new test outcomes are examined. A reasonable limited comparison is an interpretable raw-abstract TF-IDF model, an AIPD-targeted abstract model, and a patent-language embedding model if its resources and licensing allow reproducible use. More complexity is not assumed to improve validity. Normalization ablations must address the existing preprocessed-training/raw-deployment mismatch.

A practical next candidate is trained from raw USPTO abstracts with stable USPTO application identifiers and official AIPD targets, with gold development labels used separately for operating-rule selection. This avoids assuming that a PATSTAT training identifier is the same as a USPTO application number. AIPD remains a teacher/comparator, not human truth. The existing AIPat-trained prototype remains a separately named alternative; it is not silently presented as an AIPD reproduction.

Official AIPD labels and third-party research labels are tracked as distinct targets. Neither is silently relabelled as manual ground truth. Training versions, class prevalence, exclusions, weights, family membership and overlap clearance are recorded. Vocabulary, IDF, normalization parameters, learned calibration and feature selection are fitted only on the appropriate training/development data.

Operating thresholds are chosen from development evidence using an agreed error-cost criterion. For counting AI invention, false positives can strongly affect prevalence and company counts, but high precision obtained by near-zero recall is not adequate either. The supervisors must approve the intended use, precision/recall objectives and uncertainty criterion before the reserved test is opened. There is no universal numerical score that guarantees acceptance by a high-end journal.

Do not reuse 0.60 from the SEC sentence model, 0.19 from the prototype, or 0.93 from AIPD as automatically calibrated cutoffs for a new classifier. Do not select hyperparameters, thresholds, labels or sample exclusions based on disclosure-patent correlations, return coefficients, p-values or the presence of post-ChatGPT patterns.

## Final evaluation and publication reporting

Freeze code, preprocessing, training sources, model weights, threshold and dependency versions before the independent evaluator opens final labels. Preserve a hash-based registry of attempted variants; report unsuccessful candidates rather than publishing only the most favorable result.

Report on the representative test:

- Weighted and cohort-specific confusion counts, precision, recall, F1 and false-positive/false-negative rates, with denominators and uncertainty.
- Precision-recall curves and average precision for the chosen information set; ROC-AUC is supplementary, not a substitute for rare-class performance.
- Calibration if scores will be interpreted probabilistically or summed as expected counts.
- Missing text, short text, U labels, sample exclusions, and exact test N by cohort.
- Family-clustered, cohort-stratified resampling where family data permit it. Report the sampling design and weighting; do not assume independent identical Bernoulli observations after family sampling.
- Agreement with official AIPD on overlapping documents as a secondary comparator, with corrected release provenance and explicit score thresholds. Never compare an abstract-only model against an abstract-plus-claims label without acknowledging the information-set difference.
- Separate challenge-set results and separate target-firm conditional diagnostics after the primary evaluation. A failed or imprecise recent-period evaluation cannot be replaced by a good historical overall score.

The untouched final test evaluates the locked procedure once. If it motivates another model revision, its results become development evidence and confirmation requires another reserved set. Financial associations are not a replacement stopping rule.

## Subsequent company and financial analysis

Only after acceptable measurement validation is documented should a frozen classifier be applied consistently across all included years. Keep the broad patent classification layer separate from a versioned applicant/assignee-to-parent linkage table. Ownership dates, subsidiary changes, joint ownership, duplicate applications and unmatched firms require separate audits.

Merge the frozen company-level measures with the professor's sample only at the final analysis layer. Use the same classifier and matching rules for broad and screened samples; disclose their different coverage. A selected financial sample is not inherently circular, but its restricted population, matching loss and conditioning rules limit generalization. Zero observed applications under a verified link, unknown links, absent labels and incomplete publication windows are different states.

Preserve the official AIPD conservative-window analysis as a distinct benchmark. A classifier cannot remove the publication lag or observe secret applications. Publication year, application date, priority date and fiscal reporting period answer different timing questions. A retrospective classifier and current revised patent metadata do not automatically constitute information available to investors at the time of an earlier return.

Use raw count PPML for the requested primary count specification if confirmed with the professors, and the binary outcome as a separate extensive margin. Estimate consistent samples, year/industry effects and firm-clustered inference as specified in the final template. Control definitions, sample screens and source transformations must be reconciled before numerical table replacement. Report associations, not causal ChatGPT effects, unless a separate identification design justifies that claim.

If the classifier cannot meet the agreed validation requirements, retain the official-label validation and describe its time coverage honestly. Incomplete recent patent evidence is preferable to an unvalidated extension presented as established evidence.

## Appendix deliverables

The measurement appendix should include a source/version table; sample and annotation flow diagram; operational definition and difficult examples; family/decontamination procedure; training and test class distributions; candidate/threshold selection rule; confusion tables and confidence intervals; recent-period results; calibration where relevant; official-label comparisons; and full limitations. The matching appendix separately reports link provenance and unmatched coverage. Provide code and permitted artifacts, with access restrictions and third-party licensing stated accurately.

## Sources

- USPTO AIPD 2023 methodology, especially its component models, information set and benchmark discussion: https://www.uspto.gov/sites/default/files/documents/oce-aipd-2023.pdf
- scikit-learn threshold-selection documentation, including separation of fitting and threshold tuning: https://scikit-learn.org/stable/modules/classification_threshold.html
- Kapoor and Narayanan, Leakage and the Reproducibility Crisis in ML-based Science: https://arxiv.org/abs/2207.07048
- Saito and Rehmsmeier (2015), precision-recall evaluation under imbalance: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

This protocol proposes safeguards for this study. It does not claim that any cited source mandates the exact sample sizes or acceptance criteria above.
