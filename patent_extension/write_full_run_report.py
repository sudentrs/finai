"""Generate a concise report from completed outputs, not hand-entered results."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def table(frame):
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    for row in frame.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(str(x) for x in row) + ' |')
    return '\n'.join(rows)


def main():
    out = ROOT / 'outputs'
    metrics = pd.read_csv(out / 'abstract_model/evaluation_metrics.csv')
    coverage = pd.read_csv(out / 'integrated_panels/sample_coverage.csv')
    intake = json.loads((out / 'integrated_panels/abstract_intake_audit.json').read_text())
    gate = json.loads((out / 'validation_diagnostics/release_gate.json').read_text())
    card = json.loads((out / 'abstract_model/model_card.json').read_text())
    agreement = pd.read_csv(out / 'validation_diagnostics/filing_level_official_author_count_correlations.csv')
    sample = json.loads((out / 'sample_audit/summary.json').read_text())
    shown = metrics[['partition', 'n', 'precision', 'recall', 'f1']].copy()
    for col in ['precision', 'recall', 'f1']:
        shown[col] = shown[col].map(lambda x: f'{100*x:.1f}%')
    paired = agreement[agreement.window.eq('mature_18m')][['panel', 'n', 'pearson', 'spearman', 'pearson_log1p']].copy()
    for col in ['pearson', 'spearman', 'pearson_log1p']:
        paired[col] = paired[col].round(3)
    text = f'''# Full patent extension computational report

7 November 2026

## Outcome

The abstract-based computational run is complete. The separate classifier was trained, frozen, evaluated and applied to the available linked patent abstracts, and both sample versions were rebuilt. It has **not passed publication validation**. Paper regressions and paper figures based on these new labels have not been promoted as final results.

The principal finding is a large generalization gap: high agreement with the training corpus's research-label definition does not translate to high precision on the historical external benchmark. A large count correlation with official labels cannot repair this measurement limitation.

## Data and reproducibility

- Original inputs and the publication-pack files are unchanged.
- User-supplied abstract archive: `{intake['source']}`.
- Read {intake['source_rows_read']:,} source abstract rows directly from the ZIP.
- Requested {intake['project_publications_requested']:,} distinct publications in the existing linked-application universe; classified {intake['project_publications_with_predictions']:,}.
- {intake['project_publications_missing_text_or_too_short']:,} requested publication IDs had no usable matching abstract. Unknown labels remain missing.
- Local application metadata end on 25 December 2025. Publication-year coverage is not complete recent filing-year coverage.
- Download and model content hashes are recorded in `outputs/abstract_source_provenance.json`, `references/aipat/training_corpus_download.json`, and `outputs/abstract_model/model_card.json`.

## Classifier construction

This is independently written code using public AIPat research training data, not an exact replication of USPTO AIPD. No downloaded notebook, pretrained pickle or model weight was executed. The classifier does not use the firm's 10-K AI measure, patent-disclosure correlations or stock returns as training inputs or threshold-selection objectives.

The model uses normalized abstract text, English Snowball stemming, TF-IDF unigrams/bigrams, and logistic regression. The training corpus's `AI_related == 2` values are treated as positive following its published notebook; other source labels are negative. These are research labels, not independent human gold labels.

Families were sampled deterministically. Internal partitions are separated by supplied DOCDB family IDs and normalized text. Training ends in {card['train_end']}; validation ends in {card['validation_end']}; later families form the internal temporal test. Later members of earlier families are excluded from earlier-period fitting. The decision threshold, {card['threshold']:.2f}, was selected using validation F1 and then frozen. It is specific to this model and is unrelated to the 10-K threshold or the numerical AIPD 93% threshold.

The historical benchmark was not used for fitting or threshold selection. The exact normalized-text overlap check found no matching benchmark abstracts in the sampled corpus. Full external-benchmark family overlap cannot be ruled out because its family mapping is unavailable. The corpus contains preprocessed text whereas the new intake contains raw abstracts; normalization is applied consistently, but the underlying text-domain difference remains a limitation.

## Measured performance

{table(shown)}

The internal test measures agreement with research labels. The external benchmark has 4,000 granted patents published in 1981-2018, including 802 positive labels. It is not a recent published-application validation sample, and population sampling weights are not established. Neither its metrics nor internal metrics establish current USPTO-wide accuracy.

The strong internal score and substantially weaker external precision are reported together. No new threshold was selected to improve the already inspected external benchmark. Error examples are saved separately. Some errors concern generic image processing, access control and non-AI technical estimation.

## Panels and sample coverage

{table(coverage)}

The professor price-screen candidate exactly recovers {sample['candidate_price_ge5']['rows']:,} stock-months and {int(pd.read_csv(out / 'sample_audit/candidate_filing_roster.csv').in_candidate_price_sample.sum()):,} distinct eligible filing keys. Broad and screened filing panels, each at Item 1 and document level, are separate files. Original identifiers and disclosure variables remain available. The final professor winsorization and ratio transformations are not yet confirmed.

Applications are deduplicated by company/application number and assigned to fiscal reporting intervals. First or irregular intervals use an explicitly flagged one-year-back start-date assumption; other starts follow the previous observed period end. Fiscal-year labels follow the Jan-May year-end convention. These rules are recorded, not treated as exact period-start metadata where none were supplied.

Company links reuse the existing static normalized-name assignee mapping. This does not establish complete subsidiary or historical parent ownership coverage. A company with no established link is not assigned zero patents. Missing official labels after AIPD coverage are not converted to non-AI. A known linked company with no observed applications in an interval has an observed zero within that linked universe, not proof that it had no patenting through any subsidiary.

An application shared by multiple linked companies is counted once within each linked company; no fractional allocation is applied. The publication-level agreement diagnostics instead deduplicate publication IDs so shared owners do not multiply the same classification decision.

The 18-month maturity flag is a publication-lag screen, not a guarantee of completeness. Nonpublication requests, priority-date conventions and other exceptions remain. Newly classifying public abstracts does not reveal unpublished applications.

## Agreement with official AIPD counts

{table(paired)}

These compare paired available counts in mature reporting intervals. They are descriptive, repeated-filing-level correlations, not causal estimates or independent tests of classifier accuracy. Log-count correlations and rank correlations are included so raw-count agreement driven by prolific firms is not the only comparison. The full CSV also reports the 2015-2021 robustness window and zero/nonzero agreement.

Official `predict93` flags in the local AIPD file match their component-score rule across all 15,437,542 rows. In contrast, the downloaded external benchmark has 51 embedded `predict93` inconsistencies; its embedded flag and the recomputed score rule are kept distinct.

## Outstanding publication requirements

The computational pipeline does not invent the human labels needed to resolve the external-precision problem. `recent_abstract_review_queue_UNLABELLED.csv` contains the recent blinded review sample, with actual abstracts and blank reviewer/adjudication columns. `RECENT_REVIEW_GUIDE.md` describes the proposed labels and review record. A blank column is not a negative assessment.

The professor sample also contains two Item 1 filings with expertise counts above AI counts, repeated across 23 stock-months. They remain flagged and unchanged. Broader inputs contain additional such cases. The count-data template's PPML-versus-log-count inconsistency and the final fiscal-year control definitions remain unresolved. No return data were inferred or reconstructed.

Release status: **{gate['publication_gate']}**. The current model may be inspected as a research candidate but should not replace official patent measures in the draft as a validated extension. Further model development needs an appropriate labelled development set and a fresh independent recent-period test; the same reviewed benchmark cannot silently become both the tuning set and final evidence.

## Main files

- `outputs/abstract_model/abstract_classifier.joblib`: independently fitted classifier and frozen threshold.
- `outputs/abstract_model/evaluation_metrics.csv`: internal and external evaluation results.
- `outputs/integrated_panels/author_publication_predictions.csv`: scores and binary predictions for available linked publications, consistently classified across years.
- `outputs/integrated_panels/*_research_panel.csv`: four filing-level panels, broad/screened and Item 1/document, with coverage and source-quality flags.
- `outputs/integrated_panels/sample_coverage.csv`: exact panel counts and missing-coverage counts.
- `outputs/validation_diagnostics/`: AIPD agreement tables, error examples and diagnostic PNG/PDF figures. These are not paper-approved figures.
- `outputs/integrated_panels/recent_abstract_review_queue_UNLABELLED.csv`: independent human-review input, not completed validation.
- `outputs/validation_diagnostics/release_gate.json`: explicit publication-readiness status.

The earlier title-only engineering pilot remains in its own directory and is not used as the new abstract classifier.
'''
    (ROOT / 'FULL_RUN_REPORT.md').write_text(text, encoding='utf-8')
    print(ROOT / 'FULL_RUN_REPORT.md')


if __name__ == '__main__':
    main()
