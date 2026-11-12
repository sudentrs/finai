# Independent validation redesign

This directory replaces the **evaluation design**, not the original source data or historical scripts. The previous prototype remains exploratory. No claim is made that the new review files already contain expert labels or that the new test has passed family/training-overlap clearance.

Current version: **2.1**. Current files are under `outputs_v2_1/`. The initial `outputs/` draw is retained as an audit trail and superseded because one of its proposed test applications appeared in the previously inspected historical benchmark. Version 2.1 removes prior benchmark application IDs before selection and quarantines duplicate text across roles.

## Contents

- `PROTOCOL.md`: detailed measurement, sampling, annotation, model-selection and publication-reporting protocol, including limits and unresolved requirements.
- `design.json`: versioned publication cohorts, sample allocations, seed and independence rules.
- `build_blind_review.py`: patent-only sampling and abstract extraction. It reads no financial panel or company matching list.
- `ANNOTATOR_INSTRUCTIONS.md`: proposed abstract-only 0/1/U rubric and reviewer procedure.
- `check_clearance.py`: records unresolved family overlap, prior benchmark exposure and confirmatory-use requirements. It does not turn missing evidence into a pass.
- `PROTOTYPE_AUDIT.md`: what the old run did and did not establish, including its finance restriction and already-consumed benchmark.
- `candidate_registry.json`: separates the old exploratory model from proposed new candidates and prohibits finance-outcome-based selection.

## Generated outputs

- `outputs_v2_1/sampling_counts.csv`: exact patent-universe and sample counts by publication cohort and role, including prior-benchmark exclusions and post-quarantine counts.
- `outputs_v2_1/CURATOR_sample_manifest.csv`: patent identifiers, roles, source linkage, inclusion probabilities, weights and pending-clearance flags. This is not the blind reviewer copy.
- `outputs_v2_1/{pilot,development,reserved_test,challenge}/reviewer_{1,2}_BLIND_UNLABELLED.csv`: randomized reviewer files containing masked review IDs, titles, abstracts and blank labels/reasons/evidence fields. No model scores or company identifiers are supplied.
- `outputs_v2_1/independence_audit.json`: actual inputs and finance/model-score exclusions.
- `outputs_v2_1/sample_manifest.sha256`: fingerprint of the saved sampling manifest.
- `outputs_v2_1/confirmatory_clearance.json`: outstanding requirements; the current result is not a validation pass.

The planned allocation is 60 pilot documents, 600 development documents, 1,800 representative test documents and 600 challenge documents. Cohort weighting applies to the representative test; the enriched challenge set stays separate. A missing abstract remains a sampled case, not an automatic negative label.

The test folder is reserved by procedure, not secured cryptographically. Its labels must be handled by an independent evaluator under the protocol. The actual team division and operating criteria need approval. The global patent-family mapping needed to clear training/evaluation overlaps has not been supplied.

The old `patent_extension/run_integrated.py` is the exploratory prototype runner. It is not this prospective protocol, does not certify independence under this protocol, and should not be described as having run the new validation study.
