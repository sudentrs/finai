# Pilot annotation handoff

## Send separately

Reviewer 1 receives `outputs_v2_1/pilot/reviewer_1_BLIND_UNLABELLED.csv`.
Reviewer 2 receives `outputs_v2_1/pilot/reviewer_2_BLIND_UNLABELLED.csv`.
Both receive `ANNOTATOR_INSTRUCTIONS.md`. Do not send the curator manifest, model predictions or the other reviewer's completed copy.

Each file contains the same 60 abstracts. Fill only `abstract_ai_label`, `reason` and `evidence_span`; retain the review IDs and original text. Use 1 for substantive AI content, 0 for no substantive AI content, and U for uncertain or insufficient evidence. A blank is unfinished, not a negative. Save completed copies under different names so the original blank files remain intact.

## Message for reviewers

Please independently review these 60 patent abstracts using the attached instructions. We are checking whether the abstract describes substantive AI content, not whether its company is associated with AI. Ordinary uses of words such as model, training, network or prediction are not sufficient on their own. Record a short reason and supporting phrase, and use U when the abstract does not support a clear decision. Please do not consult model predictions or discuss labels with the other reviewer until both copies have been returned.

## After both copies return

Run the checker with the two completed file paths:

```powershell
python patent_extension/independent_validation_v2/check_pilot.py --reviewer-1 "PATH_TO_COMPLETED_REVIEWER_1.csv" --reviewer-2 "PATH_TO_COMPLETED_REVIEWER_2.csv"
```

It reports completion, invalid entries, raw agreement, three-category Cohen's kappa, and IDs requiring discussion. Agreement is descriptive; it does not establish classifier accuracy or guarantee independent human review. U is not recoded to zero. Kappa is undefined when its expected-agreement denominator is zero.

Preserve each reviewer's original labels. Record adjudication separately with review ID, both initial labels, adjudicated label, rationale, adjudicator and date. Discuss disagreements and uncertain cases, revise the rubric if needed, and record the final rubric version and approval date before main annotation. Do not use reserved-test labels to refine the rubric.

## Outstanding safeguards

Reviewers have not been assigned by this handoff. All labels are currently blank. Two independent reviewers and a designated adjudicator must be named by the research team. Patent-family and training-overlap clearance is still pending; completing this pilot does not clear that requirement. The reserved test must remain outside model development.
