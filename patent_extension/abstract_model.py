"""Independent abstract classifier with family-separated temporal evaluation.

Public research labels are training targets, not hand-verified truth. No source
notebook/model is executed, and no financial variable is used in model fitting.
"""
import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
STEMMER = SnowballStemmer('english')


@lru_cache(maxsize=200_000)
def stem_word(word):
    return STEMMER.stem(word)


def normalize(text):
    words = re.findall(r'[a-z]{2,}', str(text).lower())
    return ' '.join(stem_word(word) for word in words)


def metrics(y, probability, threshold):
    prediction = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {'n': len(y), 'positive': int(np.sum(y)), 'threshold': float(threshold),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'f1': float(f1_score(y, prediction, zero_division=0)),
            'average_precision': float(average_precision_score(y, probability)),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}


def train():
    out = ROOT / 'outputs' / 'abstract_model'
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'references/aipat/DATA/df_train_v6_af_pu_af_cpc_select_af_family_af_word_select.csv'
    if not source.exists():
        raise FileNotFoundError('Verified abstract-training corpus is not available')
    sample_path = out / 'prepared_training_sample.csv'
    if sample_path.exists():
        d = pd.read_csv(sample_path, dtype={'family_id': str, 'appln_id': str})
    else:
        parts, source_rows = [], 0
        for frame in pd.read_csv(source, chunksize=100_000, dtype={'docdb_family_id': str, 'appln_id': str}):
            source_rows += len(frame)
            family = frame.docdb_family_id.str.replace(r'\.0$', '', regex=True)
            eligible = family.notna() & frame.appln_abstract.notna()
            h = pd.util.hash_pandas_object(family, index=False)
            keep = eligible & h.mod(100).lt(12)
            sub = frame.loc[keep, ['appln_id', 'appln_filing_year', 'appln_abstract', 'AI_related']].copy()
            sub['family_id'] = family[keep]
            parts.append(sub)
        d = pd.concat(parts, ignore_index=True)
        if not d.AI_related.isin([0, 1, 2]).all():
            raise ValueError('Unrecognized source labels')
        d['label'] = d.AI_related.eq(2).astype(int)
        d['year'] = pd.to_numeric(d.appln_filing_year, errors='coerce')
        d = d.dropna(subset=['year']).drop_duplicates('appln_id')
        print('Normalizing abstract sample', len(d), flush=True)
        d['text'] = d.appln_abstract.map(normalize)
        d = d[d.text.str.split().str.len().ge(15)].copy()
        # Exact text overlap removal uses benchmark text only, never its labels.
        benchmark = pd.read_csv(ROOT / 'references/aipat/df_smj_4k_final.csv')
        reserved_text = set(benchmark.abstract.map(normalize))
        contaminated_families = set(d.loc[d.text.isin(reserved_text), 'family_id'])
        d = d[~d.family_id.isin(contaminated_families)].copy()
        d = d.sort_values(['year', 'appln_id']).drop_duplicates('text')
        d.to_csv(sample_path, index=False)
        (out / 'preparation.json').write_text(json.dumps({'source_rows': source_rows,
            'source_family_hash_sample_percent': 12, 'sample_rows': len(d),
            'exact_benchmark_overlap_families_removed': len(contaminated_families),
            'benchmark_family_overlap': 'Broader external benchmark family membership is unavailable; not proven absent.',
            'source_label_mapping': 'AI_related==2 positive; other source labels negative, as documented in the source train notebook.'}, indent=2), encoding='utf-8')
    first = d.groupby('family_id').year.min()
    train_end, val_end = int(first.quantile(.70)), int(first.quantile(.85))
    if train_end >= val_end:
        raise ValueError('Cannot construct distinct temporal partition cutoffs')
    d['family_first_year'] = d.family_id.map(first)
    d['partition'] = np.select([d.family_first_year.le(train_end), d.family_first_year.le(val_end)],
                               ['train', 'validation'], default='test')
    # Do not admit later publications of an old family into an earlier period.
    future = (d.partition.eq('train') & d.year.gt(train_end)) | (d.partition.eq('validation') & d.year.gt(val_end))
    d.loc[future, 'partition'] = 'excluded_later_family_member'
    d[['appln_id', 'family_id', 'year', 'partition']].to_csv(out / 'split_manifest.csv', index=False)
    train_d, val, test = [d[d.partition.eq(name)] for name in ['train', 'validation', 'test']]
    for a, b in [(train_d, val), (train_d, test), (val, test)]:
        assert set(a.family_id).isdisjoint(b.family_id)
        assert set(a.text).isdisjoint(b.text)
    for frame in [train_d, val, test]:
        if frame.label.nunique() != 2:
            raise ValueError('A temporal partition has only one class')
    model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=120_000,
                                              sublinear_tf=True, dtype=np.float32)),
                      ('logit', LogisticRegression(C=1., solver='liblinear', max_iter=2000, random_state=42))])
    print('Fitting family-separated abstract classifier', len(train_d), flush=True)
    model.fit(train_d.text, train_d.label)
    probability = model.predict_proba(val.text)[:, 1]
    candidates = pd.DataFrame([metrics(val.label, probability, t) for t in np.arange(.05, .951, .01)])
    candidates.to_csv(out / 'validation_threshold_candidates.csv', index=False)
    threshold = float(candidates.sort_values(['f1', 'threshold'], ascending=[False, False]).iloc[0].threshold)
    result = []
    for name, frame in [('validation', val), ('temporal_family_test', test)]:
        p = model.predict_proba(frame.text)[:, 1]
        result.append(dict(partition=name, **metrics(frame.label, p, threshold)))
        frame = frame.copy()
        frame['model_probability'] = p
        frame.to_csv(out / (name + '_predictions.csv'), index=False)
    # Evaluate once after threshold freezing. Labels are not inputs to model selection.
    external = pd.read_csv(ROOT / 'references/aipat/df_smj_4k_final.csv', dtype={'doc_id': str})
    external['model_probability'] = model.predict_proba(external.abstract.map(normalize))[:, 1]
    result.append(dict(partition='external_historical_grants', **metrics(external.actual, external.model_probability, threshold)))
    external[['doc_id', 'actual', 'model_probability']].to_csv(out / 'external_benchmark_predictions.csv', index=False)
    pd.DataFrame(result).to_csv(out / 'evaluation_metrics.csv', index=False)
    artifact = {'model': model, 'threshold': threshold, 'normalization': 'regex ascii words plus Snowball English stemming',
                'status': 'research candidate, not approved for paper conclusions',
                'train_end': train_end, 'validation_end': val_end}
    joblib.dump(artifact, out / 'abstract_classifier.joblib')
    summary = {k: v for k, v in artifact.items() if k != 'model'}
    summary.update(iterations=model.named_steps['logit'].n_iter_.tolist(),
                   recent_manual_validation=False, training_target='AIPat research labels; not AIPD or human gold labels',
                   model_sha256=hashlib.file_digest((out / 'abstract_classifier.joblib').open('rb'), 'sha256').hexdigest())
    (out / 'model_card.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(pd.DataFrame(result).to_string(index=False), flush=True)


if __name__ == '__main__':
    train()
