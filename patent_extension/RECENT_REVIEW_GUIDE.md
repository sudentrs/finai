# Recent patent abstract review

The review file is `outputs/integrated_panels/recent_abstract_review_queue_UNLABELLED.csv`. It is a random sample of recent publications linked to the project's existing company-matching universe. It is not a random sample of every USPTO application.

The review tests a research candidate, not an already validated patent measure. Model predictions and official flags are deliberately absent from the review file. The three label columns are blank; no automated label is represented as a human assessment.

## Labels

- **1: AI-related.** The abstract describes an AI method or a substantive use of AI in the claimed invention. Examples include machine learning, neural networks, language modelling, knowledge-based inference and explicitly described learning or reasoning systems.
- **0: Not AI-related.** The abstract describes a conventional device, algorithm, statistical calculation, database, control system or automation process without substantive AI content.
- **U: Uncertain.** The abstract is insufficient or ambiguous; further claim/description review is needed.

Generic terms such as intelligent, model, training, image, prediction or automatic are not by themselves sufficient for a positive label. Conversely, an abstract need not literally say artificial intelligence if it clearly describes an AI technique. The target is broad AI invention, not generative AI alone. This is the project's proposed annotation definition; it is not a claim to duplicate every USPTO examiner decision.

## Review record

`reviewer_1_ai` and `reviewer_2_ai` are separate assessments. `adjudicated_ai` records the resolved label after disagreements are discussed. Publication/application identifiers link each row to its original document. A blank label means no review, not a negative label.

All selected rows should remain in the audit, including uncertain cases and records with unavailable abstracts. Review coverage, disagreements and uncertain cases are part of the evaluation. Claims-based follow-up should record the additional source and rationale rather than silently changing the original abstract-only assessment.

These labels must not be used both to select a new threshold and to claim independent final performance on the same observations. If used for model development, another untouched evaluation sample is needed.
