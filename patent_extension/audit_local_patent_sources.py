"""Stream local patent sources for coverage and official flag consistency."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent


def main():
    out = ROOT / 'outputs' / 'local_patent_audit'
    out.mkdir(parents=True, exist_ok=True)
    fields = ['ml', 'evo', 'nlp', 'speech', 'vision', 'planning', 'kr', 'hardware']
    scores = ['ai_score_' + x for x in fields]
    cols = ['doc_id', 'flag_patent', 'pub_dt', 'predict93_any_ai'] + scores
    summaries, mismatches, n, invalid_scores = [], 0, 0, 0
    source = WORKSPACE / 'ai_model_predictions.csv' / 'ai_model_predictions.csv'
    for i, d in enumerate(pd.read_csv(source, usecols=cols, chunksize=400_000, dtype={'doc_id': str}), start=1):
        n += len(d)
        values = d[scores]
        invalid_scores += int((values.isna().any(axis=1) | values.lt(0).any(axis=1) | values.gt(1).any(axis=1)).sum())
        implied = values.ge(.93).any(axis=1).astype(int)
        bad = d.predict93_any_ai.ne(implied)
        mismatches += int(bad.sum())
        if bad.any():
            example_path = out / 'predict93_score_mismatch_examples.csv'
            if not example_path.exists():
                d.loc[bad].head(30).to_csv(example_path, index=False)
        d['publication_year'] = pd.to_datetime(d.pub_dt, errors='coerce').dt.year
        summaries.append(d.groupby(['flag_patent', 'publication_year'], dropna=False).agg(
            rows=('doc_id', 'size'), ai93=('predict93_any_ai', 'sum')).reset_index())
        if i % 10 == 0:
            print('AIPD scanned', n, flush=True)
    pd.concat(summaries).groupby(['flag_patent', 'publication_year'], dropna=False).sum(numeric_only=True).to_csv(out / 'aipd_publication_year_coverage.csv')
    summary = {'aipd_rows': n, 'predict93_vs_component_score_ge093_disagreements': mismatches,
               'invalid_component_score_rows': invalid_scores,
               'release_note': 'Score consistency is a content check, not proof of an official file checksum or download provenance.'}
    source = WORKSPACE / 'pg_published_application.tsv' / 'pg_published_application.tsv'
    yearly, latest, app_rows, bad_dates = [], None, 0, 0
    for d in pd.read_csv(source, sep='\t', usecols=['pgpub_id', 'application_id', 'filing_date', 'published_date', 'application_title'],
                         chunksize=400_000, dtype=str):
        app_rows += len(d)
        pub = pd.to_datetime(d.published_date, errors='coerce')
        filed = pd.to_datetime(d.filing_date, errors='coerce')
        bad_dates += int((pub.isna() | filed.isna()).sum())
        if pub.notna().any():
            latest = max(latest or pub.max(), pub.max())
        d['publication_year'], d['filing_year'] = pub.dt.year, filed.dt.year
        yearly.append(d.groupby(['filing_year', 'publication_year'], dropna=False).size().rename('rows').reset_index())
    pd.concat(yearly).groupby(['filing_year', 'publication_year'], dropna=False).rows.sum().to_csv(out / 'pgpub_filing_publication_year_coverage.csv')
    summary.update(pgpub_rows=app_rows, pgpub_latest_publication=str(latest), pgpub_invalid_date_rows=bad_dates,
                   text_available='application_title only; abstract and claims absent from this metadata file')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
