"""Join one classifier consistently across years; never replace missing labels with zero."""
import json
import re
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from abstract_model import normalize

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUT = ROOT / 'outputs' / 'integrated_panels'


def identifier(series):
    return series.astype('string').str.replace(r'\.0$', '', regex=True).str.lstrip('0')


def dates(series):
    return pd.to_datetime(series.astype('string').str.replace(r'\.0$', '', regex=True), format='%Y%m%d', errors='coerce')


def load_applications():
    path = OUT / 'linked_applications.csv'
    if path.exists():
        d = pd.read_csv(path, dtype={'cik': str, 'application_id': str, 'pgpub_id': str})
        d['filing_date'] = pd.to_datetime(d.filing_date)
        d['published_date'] = pd.to_datetime(d.published_date)
        return d
    source = WORKSPACE / 'analysis/patent_validation_improved/matched_patent_applications_detail.csv'
    cols = ['cik', 'pgpub_id', 'application_id', 'filing_date', 'published_date', 'application_title',
            'firm_match_method', 'has_aipd_prediction', 'predict93_any_ai']
    d = pd.read_csv(source, usecols=cols, dtype={'cik': str, 'pgpub_id': str, 'application_id': str})
    for name in ['cik', 'pgpub_id', 'application_id']:
        d[name] = identifier(d[name])
    for name in ['filing_date', 'published_date']:
        d[name] = pd.to_datetime(d[name], errors='coerce')
    known = d.has_aipd_prediction.astype(str).str.lower().isin(['true', '1'])
    d['official_ai93'] = pd.to_numeric(d.predict93_any_ai, errors='coerce').where(known)
    d['author_ai'], d['author_probability'] = np.nan, np.nan
    d = d.dropna(subset=['cik', 'application_id', 'pgpub_id', 'filing_date'])
    # This preserves the existing match but counts distinct applications, not publications.
    d = d.sort_values(['published_date', 'pgpub_id']).drop_duplicates(['cik', 'application_id'])
    d.drop(columns=['predict93_any_ai']).to_csv(path, index=False)
    return d.drop(columns=['predict93_any_ai'])


def predict_abstracts(applications, abstract_path):
    artifact = joblib.load(ROOT / 'outputs/abstract_model/abstract_classifier.joblib')
    wanted = set(applications.pgpub_id)
    review = applications[applications.published_date.dt.year.ge(2023)].sample(
        n=min(600, int(applications.published_date.dt.year.ge(2023).sum())), random_state=42)
    review_ids = set(review.pgpub_id)
    review_texts = []
    archive = zipfile.ZipFile(abstract_path) if abstract_path.suffix == '.zip' else None
    if archive:
        names = [n for n in archive.namelist() if n.endswith(('.tsv', '.csv')) and not n.startswith('__MACOSX/')]
        if len(names) != 1:
            raise ValueError('Abstract archive must contain exactly one data table')
        source = archive.open(names[0])
    else:
        source = abstract_path
    output = []
    source_rows = 0
    source_year_counts = {}
    for d in pd.read_csv(source, sep='\t', dtype=str, chunksize=50_000):
        source_rows += len(d)
        if 'pgpub_id' not in d.columns:
            raise ValueError(f'Abstract table lacks pgpub_id: {d.columns.tolist()}')
        text_cols = [c for c in ['abstract', 'application_abstract', 'pgpub_abstract', 'abstract_text'] if c in d.columns]
        if len(text_cols) != 1:
            raise ValueError(f'Cannot identify one abstract text field: {d.columns.tolist()}')
        d['pgpub_id'] = identifier(d.pgpub_id)
        for year, count in d.pgpub_id.str[:4].value_counts().items():
            source_year_counts[str(year)] = source_year_counts.get(str(year), 0) + int(count)
        d = d[d.pgpub_id.isin(wanted) & d[text_cols[0]].notna()].copy()
        if d.empty:
            continue
        review_texts.append(d.loc[d.pgpub_id.isin(review_ids), ['pgpub_id', text_cols[0]]].rename(columns={text_cols[0]: 'abstract'}))
        text = d[text_cols[0]].map(normalize)
        valid = text.str.split().str.len().ge(15)
        d = d[valid].copy()
        if d.empty:
            continue
        d['author_probability'] = artifact['model'].predict_proba(text[valid])[:, 1]
        d['author_ai'] = d.author_probability.ge(artifact['threshold']).astype(int)
        output.append(d[['pgpub_id', 'author_probability', 'author_ai']])
        if len(output) % 20 == 0:
            print('Abstract source rows', source_rows, 'classified project rows', sum(len(p) for p in output), flush=True)
    if archive:
        source.close()
        archive.close()
    if not output:
        raise ValueError('No project applications have usable abstracts in this file')
    prediction = pd.concat(output, ignore_index=True)
    if prediction.pgpub_id.duplicated().any():
        raise ValueError('Duplicate publication identifiers in abstract source')
    prediction.to_csv(OUT / 'author_publication_predictions.csv', index=False)
    if review_texts:
        queue = review[['pgpub_id', 'application_id', 'filing_date', 'published_date', 'application_title']].merge(
            pd.concat(review_texts, ignore_index=True).drop_duplicates('pgpub_id'), on='pgpub_id', how='left', validate='many_to_one')
        queue['reviewer_1_ai'], queue['reviewer_2_ai'], queue['adjudicated_ai'] = '', '', ''
        queue.to_csv(OUT / 'recent_abstract_review_queue_UNLABELLED.csv', index=False)
    (OUT / 'abstract_intake_audit.json').write_text(json.dumps({'source': str(abstract_path),
        'source_rows_read': source_rows, 'project_publications_with_predictions': len(prediction),
        'abstract_publication_year_prefix_counts': source_year_counts,
        'project_publications_requested': len(wanted), 'project_publications_missing_text_or_too_short': len(wanted-set(prediction.pgpub_id))}, indent=2), encoding='utf-8')
    return applications.drop(columns=['author_probability', 'author_ai']).merge(prediction, on='pgpub_id', how='left', validate='many_to_one')


def ff48(sic):
    lookup = {}
    industry = None
    for line in (WORKSPACE / 'Siccodes48.txt').read_text().splitlines():
        header = re.match(r'^\s*(\d+)\s+[A-Za-z]', line)
        interval = re.match(r'^\s*(\d{4})-(\d{4})', line)
        if header:
            industry = int(header.group(1))
        if interval:
            lo, hi = map(int, interval.groups())
            lookup.update({i: industry for i in range(lo, hi + 1)})
    return pd.to_numeric(sic, errors='coerce').map(lookup)


def prepare_filings():
    base = WORKSPACE / 'analysis/professor_filing_level_package_20260804'
    roster = pd.read_csv(ROOT / 'outputs/sample_audit/candidate_filing_roster.csv', dtype={'cik': str})
    keys = ['cik', 'filed_date', 'period_end']
    roster.cik = identifier(roster.cik)
    for name in ['filed_date', 'period_end']:
        roster[name] = pd.to_numeric(roster[name], errors='coerce').astype('Int64')
    professor = pd.read_stata(Path(r'C:\Users\sudet\Desktop\finai\publication pack\ITEM1_reg.dta'),
        columns=keys + ['permco', 'crsp_permno', 'expertiseratio', 'airatio', 'prc', 'mdate', 'ff48'],
        convert_categoricals=False)
    professor.cik = identifier(professor.cik)
    professor = professor[professor.prc.ge(5)].sort_values('mdate').drop_duplicates(keys)
    outputs = []
    for scope in ['ITEM_1', 'DOCUMENT']:
        d = pd.read_csv(base / f'{scope}_filing_standardized.csv', dtype={'cik': str}, low_memory=False)
        d.cik = identifier(d.cik)
        for name in ['filed_date', 'period_end']:
            d[name] = pd.to_numeric(d[name], errors='coerce').astype('Int64')
        d['filing_date_actual'], d['period_end_actual'] = dates(d.filed_date), dates(d.period_end)
        d = d[d.filing_date_actual.dt.year.between(2015, 2025)].copy()
        if d.duplicated(keys).any():
            d[d.duplicated(keys, keep=False)].to_csv(OUT / f'{scope}_ambiguous_filing_keys.csv', index=False)
            raise ValueError(f'Ambiguous filing keys in {scope}; accession-level resolution required')
        d['scope'] = scope
        d['ff48'] = ff48(d.sic)
        # Bounds are based on actual reporting ends, never the overwritten panel year.
        period = d[['cik', 'period_end_actual']].drop_duplicates().sort_values(['cik', 'period_end_actual'])
        period['previous_period_end'] = period.groupby('cik').period_end_actual.shift()
        gap = (period.period_end_actual - period.previous_period_end).dt.days
        period['period_start'] = period.previous_period_end + pd.Timedelta(days=1)
        fallback = ~gap.between(330, 400)
        period.loc[fallback, 'period_start'] = period.loc[fallback, 'period_end_actual'] - pd.DateOffset(years=1) + pd.Timedelta(days=1)
        period['period_start_method'] = np.where(fallback, 'one_year_back_assumption', 'previous_observed_period_end_plus_one')
        d = d.merge(period, on=['cik', 'period_end_actual'], validate='many_to_one')
        d['fiscal_year_label'] = d.period_end_actual.dt.year - d.period_end_actual.dt.month.le(5).astype(int)
        d['sample'] = 'broad_filing_sample'
        outputs.append(d)
        screened = d.merge(professor[keys + ['permco', 'crsp_permno', 'expertiseratio', 'airatio', 'ff48']],
                           on=keys, how='inner', suffixes=('', '_professor'), validate='one_to_one')
        screened['sample'] = 'professor_price_ge5_candidate'
        if scope == 'ITEM_1':
            screened['source_ratio_difference'] = (screened.ai_ratio - screened.airatio).abs() + (screened.ai_expertise_ratio - screened.expertiseratio).abs()
            # Preserve delivered professor ratios; originals remain separately available.
            screened['ai_ratio'] = screened.airatio
            screened['ai_expertise_ratio'] = screened.expertiseratio
        screened['ff48'] = screened.ff48_professor
        outputs.append(screened)
    return pd.concat(outputs, ignore_index=True, sort=False)


def aggregate(filings, applications):
    by_firm = {key: group for key, group in applications.groupby('cik')}
    cutoff = pd.Timestamp(json.loads((ROOT / 'outputs/local_patent_audit/summary.json').read_text())['pgpub_latest_publication'])
    unique_periods = filings[['cik', 'period_start', 'period_end_actual']].drop_duplicates()
    author_source_available = bool(applications.author_ai.notna().any())
    rows = []
    for row in unique_periods.itertuples(index=False):
        candidates = by_firm.get(row.cik)
        known_link = candidates is not None
        data = candidates[candidates.filing_date.between(row.period_start, row.period_end_actual)] if known_link else applications.iloc[:0]
        result = {'cik': row.cik, 'period_start': row.period_start, 'period_end_actual': row.period_end_actual,
                  'assignee_link_status': 'has_existing_static_name_link' if known_link else 'unverified_no_existing_link',
                  'observed_total_applications': len(data) if known_link else np.nan,
                  'publication_maturity_18m': bool(row.period_end_actual + pd.DateOffset(months=18) <= cutoff),
                  'publication_cutoff': cutoff}
        for label, column in [('author', 'author_ai'), ('official93', 'official_ai93')]:
            n_known = int(data[column].notna().sum())
            count = float(data[column].sum()) if known_link and n_known == len(data) else np.nan
            if label == 'author' and not author_source_available:
                count = np.nan
            result.update({f'{label}_classified_applications': n_known if known_link else np.nan,
                           f'{label}_unclassified_applications': len(data) - n_known if known_link else np.nan,
                           f'{label}_ai_application_count': count,
                           f'{label}_has_ai_application': float(count > 0) if pd.notna(count) else np.nan})
        rows.append(result)
    return filings.merge(pd.DataFrame(rows), on=['cik', 'period_start', 'period_end_actual'], validate='many_to_one')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    applications = load_applications()
    paths = list((ROOT / 'data').glob('pg_published_application_abstract.tsv*')) if (ROOT / 'data').exists() else []
    paths += list(WORKSPACE.glob('pg_published_application_abstract.tsv*'))
    paths += list(Path(r'C:\Users\sudet\Desktop\finai\publication pack').glob('pg_published_application_abstract.tsv*'))
    paths = [p for p in paths if p.is_file()]
    if len(paths) > 1:
        raise ValueError('Multiple abstract files; select one explicit source vintage')
    text_status = 'missing_abstract_source'
    if paths and (ROOT / 'outputs/abstract_model/abstract_classifier.joblib').exists():
        applications = predict_abstracts(applications, paths[0])
        text_status = 'predictions_available_research_candidate'
    filings = prepare_filings()
    panel = aggregate(filings, applications)
    panel['nonexpertise_ai_ratio'] = panel.ai_ratio - panel.ai_expertise_ratio
    panel['invalid_expertise_subset'] = panel.nonexpertise_ai_ratio.lt(-1e-7)
    # No paper-eligible author outcome without both labels and demonstrated recent validation.
    panel['author_recent_manual_validation_complete'] = False
    panel['author_paper_ready'] = False
    counts = []
    for (sample, scope), d in panel.groupby(['sample', 'scope']):
        name = sample + '_' + scope
        d.to_csv(OUT / f'{name}_research_panel.csv', index=False)
        counts.append({'sample': sample, 'scope': scope, 'filings': len(d),
                       'mature_18m_filings': int(d.publication_maturity_18m.sum()),
                       'unknown_firm_link': int(d.assignee_link_status.eq('unverified_no_existing_link').sum()),
                       'author_outcome_available': int(d.author_ai_application_count.notna().sum()),
                       'official_outcome_available': int(d.official93_ai_application_count.notna().sum()),
                       'invalid_expertise_subset': int(d.invalid_expertise_subset.sum())})
    pd.DataFrame(counts).to_csv(OUT / 'sample_coverage.csv', index=False)
    # No labels are included in this queue; it is not a completed manual validation.
    queue = applications[applications.published_date.dt.year.ge(2023)].sample(
        n=min(300, int(applications.published_date.dt.year.ge(2023).sum())), random_state=42)
    queue = queue[['pgpub_id', 'application_id', 'filing_date', 'published_date', 'application_title']].copy()
    queue['reviewer_1_ai'], queue['reviewer_2_ai'], queue['adjudicated_ai'] = '', '', ''
    queue.to_csv(OUT / 'recent_manual_review_queue_unlabelled.csv', index=False)
    status = {'text_status': text_status, 'linked_distinct_firm_applications': len(applications),
              'sample_status': 'professor price-screen candidate; final winsorization/control definitions unresolved',
              'match_status': 'reused static normalized-name links; ownership/subsidiary coverage not independently established',
              'paper_ready': False, 'missing_outcome_policy': 'unknown labels and unknown firm links remain missing, not zero'}
    (OUT / 'integration_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(pd.DataFrame(counts).to_string(index=False), flush=True)
    print(json.dumps(status, indent=2), flush=True)


if __name__ == '__main__':
    main()
