"""Audit a reserved external benchmark. This script does not train a model."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / 'references' / 'aipat' / 'df_smj_4k_final.csv'
    out = ROOT / 'outputs' / 'external_benchmark_audit'
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(path, dtype={'doc_id': str, 'application_id': str, 'appln_nr_8': str})
    labels = pd.to_numeric(d.actual, errors='raise')
    assert labels.isin([0, 1]).all()
    summary = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'rows': len(d), 'positives': int(labels.sum()), 'negative': int(labels.eq(0).sum()),
               'label_column': 'actual', 'label_provenance': 'Repository describes human-labelled benchmark; annotation protocol not independently verified here',
               'training_use': 'RESERVED: no model training or threshold selection on this file',
               'document_types': d.flag_patent.value_counts().to_dict(),
               'publication_min': d.pub_dt.min(), 'publication_max': d.pub_dt.max(),
               'duplicate_doc_ids': int(d.duplicated(['doc_id', 'flag_patent']).sum()),
               'duplicate_abstracts': int(d.abstract.str.lower().str.replace(r'\s+', ' ', regex=True).duplicated().sum()),
               'application_duplicates': int(d.application_id.duplicated().sum()),
               'abstract_missing': int(d.abstract.isna().sum())}
    rows = []
    components = ['ml', 'evo', 'nlp', 'speech', 'vision', 'planning', 'kr', 'hardware']
    score = d[['ai_score_' + c for c in components]].max(axis=1)
    for threshold in [50, 86, 93]:
        flag = d[f'predict{threshold}_any_ai'].astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, flag, labels=[0, 1]).ravel()
        rows.append({'indicator': f'predict{threshold}_any_ai', 'n': len(d), 'tp': int(tp), 'fp': int(fp),
                     'tn': int(tn), 'fn': int(fn), 'precision': precision_score(labels, flag, zero_division=0),
                     'recall': recall_score(labels, flag, zero_division=0), 'f1': f1_score(labels, flag),
                     'accuracy': accuracy_score(labels, flag),
                     'flag_score_threshold_disagreements': int(flag.ne(score.ge(threshold/100).astype(int)).sum())})
    # The embedded predict93 flag may predate USPTO's corrected release.
    flag = score.ge(.93).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, flag, labels=[0, 1]).ravel()
    rows.append({'indicator': 'recomputed_ge093_from_embedded_component_scores', 'n': len(d),
                 'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
                 'precision': precision_score(labels, flag, zero_division=0),
                 'recall': recall_score(labels, flag, zero_division=0), 'f1': f1_score(labels, flag),
                 'accuracy': accuracy_score(labels, flag), 'flag_score_threshold_disagreements': 0})
    summary['embedded_predict93_score_disagreements'] = int(d.predict93_any_ai.ne(flag).sum())
    summary['interpretation_limits'] = ['All documents are grants published no later than 2018, not recent pre-grant applications.',
        'Benchmark sampling weights and annotation protocol not yet established; metrics describe this sample only.',
        'Embedded indicators have not been merged against the current local official file; recomputed score threshold is separate.']
    pd.DataFrame(rows).to_csv(out / 'official_indicators_on_external_benchmark.csv', index=False)
    d.groupby(d.pub_dt.str[:4]).agg(n=('actual', 'size'), positive=('actual', 'sum')).to_csv(out / 'benchmark_by_publication_year.csv')
    # Row identifiers only, so later training data can be explicitly purged of overlap.
    d[['doc_id', 'flag_patent', 'application_id', 'appln_nr_8']].to_csv(out / 'reserved_benchmark_identifiers.csv', index=False)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
