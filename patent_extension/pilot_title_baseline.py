"""Diagnostic title-only baseline against official labels; NOT a paper-ready patent model.

Sampling and thresholds use no disclosure scores or stock returns. The historical
external human benchmark is not used for training, tuning or model selection.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs' / 'title_diagnostic_pilot'


def prepare():
    path = ROOT.parent / 'pg_published_application.tsv' / 'pg_published_application.tsv'
    parts = []
    for d in pd.read_csv(path, sep='\t', usecols=['pgpub_id', 'application_id', 'published_date', 'application_title'],
                         dtype=str, chunksize=400_000):
        year = pd.to_datetime(d.published_date, errors='coerce').dt.year
        # Fixed deterministic ~1% sample; no balancing of test-set class prevalence.
        keep = year.between(2015, 2023) & pd.util.hash_pandas_object(d.pgpub_id, index=False).mod(100).eq(0)
        parts.append(d[keep].assign(publication_year=year[keep]))
    sample = pd.concat(parts, ignore_index=True).drop_duplicates('pgpub_id')
    ids = set(sample.pgpub_id)
    labels = []
    for d in pd.read_csv(ROOT.parent / 'ai_model_predictions.csv' / 'ai_model_predictions.csv',
                         usecols=['doc_id', 'flag_patent', 'predict93_any_ai'], dtype={'doc_id': str}, chunksize=500_000):
        labels.append(d[d.flag_patent.eq(0) & d.doc_id.isin(ids)])
    labels = pd.concat(labels, ignore_index=True)
    if labels.doc_id.duplicated().any():
        raise ValueError('Official publication identifiers not unique')
    merged = sample.merge(labels[['doc_id', 'predict93_any_ai']], left_on='pgpub_id', right_on='doc_id', how='left', validate='one_to_one')
    merged.to_csv(OUT / 'sample_with_label_availability.csv', index=False)
    return merged


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'sample_with_label_availability.csv'
    d = pd.read_csv(path, dtype={'pgpub_id': str, 'application_id': str}) if path.exists() else prepare()
    audit = {'purpose': 'exploratory title-only engineering baseline, not final model or independent validation',
             'sample_rows': len(d), 'unmatched_official_label': int(d.predict93_any_ai.isna().sum()),
             'family_leakage_control': 'application and normalized-title separation only; broader patent families not yet available'}
    d = d.dropna(subset=['predict93_any_ai', 'application_title', 'application_id']).copy()
    reserved = pd.read_csv(ROOT / 'outputs' / 'external_benchmark_audit' / 'reserved_benchmark_identifiers.csv',
                           dtype=str)
    reserved_apps = set(reserved.application_id.dropna()) | set(reserved.appln_nr_8.dropna())
    audit['reserved_benchmark_application_rows_removed'] = int(d.application_id.isin(reserved_apps).sum())
    d = d[~d.application_id.isin(reserved_apps)].copy()
    d['normalized_title'] = d.application_title.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    d = d[d.normalized_title.str.len().gt(0)]
    # Retain earliest publication per application/title across temporal partitions.
    d = d.sort_values(['published_date', 'pgpub_id']).drop_duplicates('application_id').drop_duplicates('normalized_title')
    d['partition'] = np.select([d.publication_year.le(2020), d.publication_year.eq(2021)], ['train', 'validation'], default='test')
    d[['pgpub_id', 'application_id', 'publication_year', 'partition']].to_csv(OUT / 'split_manifest.csv', index=False)
    train, val, test = [d[d.partition.eq(x)] for x in ['train', 'validation', 'test']]
    assert set(train.application_id).isdisjoint(test.application_id)
    assert set(train.normalized_title).isdisjoint(test.normalized_title)
    model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True)),
                      ('logit', LogisticRegression(solver='liblinear', C=1, max_iter=1000, random_state=42))])
    model.fit(train.application_title, train.predict93_any_ai.astype(int))
    pv = model.predict_proba(val.application_title)[:, 1]
    grid = pd.DataFrame([{'threshold': float(t), 'validation_f1': f1_score(val.predict93_any_ai, pv >= t, zero_division=0)}
                         for t in np.arange(.05, .951, .01)])
    chosen = float(grid.sort_values(['validation_f1', 'threshold'], ascending=[False, False]).iloc[0].threshold)
    grid.to_csv(OUT / 'validation_thresholds.csv', index=False)
    results = []
    for name, frame in [('train', train), ('validation', val), ('test', test)]:
        prob = model.predict_proba(frame.application_title)[:, 1]
        pred = prob >= chosen
        tn, fp, fn, tp = confusion_matrix(frame.predict93_any_ai, pred, labels=[0, 1]).ravel()
        results.append({'partition': name, 'n': len(frame), 'positive': int(frame.predict93_any_ai.sum()),
                        'threshold': chosen, 'precision': precision_score(frame.predict93_any_ai, pred, zero_division=0),
                        'recall': recall_score(frame.predict93_any_ai, pred, zero_division=0),
                        'f1': f1_score(frame.predict93_any_ai, pred, zero_division=0),
                        'average_precision': average_precision_score(frame.predict93_any_ai, prob),
                        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)})
        if name == 'test':
            frame = frame.copy()
            frame['predicted_score'], frame['predicted_ai'] = prob, pred.astype(int)
            frame.to_csv(OUT / 'heldout_predictions.csv', index=False)
    pd.DataFrame(results).to_csv(OUT / 'metrics.csv', index=False)
    audit['logistic_iterations'] = model.named_steps['logit'].n_iter_.tolist()
    audit['selected_threshold'] = chosen
    audit['training_years'] = '2015-2020'
    audit['validation_years'] = '2021'
    audit['test_years'] = '2022-2023'
    audit['model_promotion'] = 'NOT APPROVED: titles only, pseudo-label target, no family-level split, no recent manual validation'
    (OUT / 'summary.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
