# Blind abstract review instructions

The files in each role folder are separate reviewer copies of the same documents, with the row order randomized. Do not share labels before independent annotation is complete. There are no completed human labels in these files.

## Task

Decide whether the abstract describes substantive AI-related invention or a substantive use of AI in the invention. The target is not whether the firm is an AI company, whether the invention is commercially valuable, whether a patent was granted, or whether the patent will correlate with corporate disclosure.

Enter one of the following in `abstract_ai_label`:

- `1`: substantive AI content is described.
- `0`: no substantive AI content is described.
- `U`: insufficient information or a genuinely uncertain borderline case.

Enter a short explanation in `reason` and quote the decisive abstract phrase in `evidence_span`. If the abstract is absent, record U with the reason `missing abstract`. A blank label means not yet reviewed, not zero.

Ordinary computing, camera settings, image sensors, generic automation, databases, deterministic equations and conventional control systems are not automatically AI. Learning methods, neural networks, AI language/perception systems and explicitly described knowledge-based reasoning can qualify even without the literal phrase artificial intelligence. Mention of a model, training, prediction, intelligence or optimization alone is insufficient.

The pilot refines these proposed guidelines before the main rubric is frozen. Do not introduce new rules while reviewing the reserved test. Record difficult cases for adjudication instead. Do not consult model scores, official AI labels, stock returns or the company's 10-K scores.

The primary task uses the abstract only. If full-document follow-up is authorized, record that assessment separately with the additional source; do not silently replace the abstract-only judgement. Labels on the final test are handled by an independent evaluator rather than used by model developers to tune a threshold.

Identifiers are masked in the reviewer copies, but the text may itself reveal a company or technology. This is metadata blinding, not a guarantee of complete anonymity.
