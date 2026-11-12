"""Patent-only probability sampling, independent of the finance project panel.

This program never reads a financial/disclosure file or a model prediction. Test
membership is reserved prospectively but is NOT cleared for confirmatory use
until family and historical training overlap checks are complete.
"""
import hashlib
import json
import re
import sqlite3
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
META = WORKSPACE / 'pg_published_application.tsv/pg_published_application.tsv'
ABSTRACT = Path(r'C:\Users\sudet\Desktop\finai\publication pack\pg_published_application_abstract.tsv.zip')
OUT = ROOT / 'outputs'


def sample_group(frame, period, role, seed, n):
    frame = frame.copy()
    frame['draw_key'] = frame.application_id.map(
        lambda x: hashlib.sha256(f'{seed}|{period}|{role}|{x}'.encode()).hexdigest())
    return frame.sort_values(['draw_key', 'application_id']).head(n).drop(columns='draw_key')


def main():
    global OUT
    design = json.loads((ROOT / 'design.json').read_text())
    OUT = ROOT / design.get('output_subdir', 'outputs')
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'outputs'
    cache.mkdir(exist_ok=True)
    db = sqlite3.connect(cache / 'patent_only_sampling_frame.sqlite')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS applications (application_id TEXT PRIMARY KEY, pgpub_id TEXT, published_date TEXT, filing_date TEXT, title TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS provenance (key TEXT PRIMARY KEY, value TEXT)')
    fingerprint = hashlib.file_digest(META.open('rb'), 'sha256').hexdigest()
    existing = db.execute('SELECT value FROM provenance WHERE key=?', ('metadata_sha256',)).fetchone()
    if existing and existing[0] != fingerprint:
        raise ValueError('Metadata vintage changed; use a new versioned output directory')
    if not existing:
        if db.execute('SELECT COUNT(*) FROM applications').fetchone()[0]:
            # A partial prior build is not a completed frame.
            db.execute('DELETE FROM applications')
            db.commit()
        n_source, skipped = 0, 0
        cols = ['application_id', 'pgpub_id', 'published_date', 'filing_date', 'patent_type', 'application_title']
        statement = '''INSERT INTO applications VALUES (?,?,?,?,?)
            ON CONFLICT(application_id) DO UPDATE SET pgpub_id=excluded.pgpub_id,
            published_date=excluded.published_date, filing_date=excluded.filing_date,
            title=excluded.title WHERE excluded.published_date < applications.published_date
            OR (excluded.published_date = applications.published_date AND excluded.pgpub_id < applications.pgpub_id)'''
        for chunk in pd.read_csv(META, sep='\t', usecols=cols, chunksize=200_000, dtype=str):
            n_source += len(chunk)
            chunk = chunk[chunk.patent_type.str.lower().eq('utility')].copy()
            valid = chunk.application_id.str.fullmatch(r'\d+', na=False) & chunk.pgpub_id.str.fullmatch(r'\d{11}', na=False)
            valid &= pd.to_datetime(chunk.published_date, errors='coerce').notna()
            skipped += int((~valid).sum())
            chunk = chunk[valid]
            records = chunk[['application_id', 'pgpub_id', 'published_date', 'filing_date', 'application_title']].fillna('').itertuples(index=False, name=None)
            db.executemany(statement, records)
            db.commit()
            if n_source % 1_000_000 == 0:
                print('Patent-only source rows scanned', n_source, flush=True)
        db.executemany('INSERT INTO provenance VALUES (?,?)', [('metadata_sha256', fingerprint),
            ('source_rows', str(n_source)), ('invalid_utility_rows_skipped', str(skipped))])
        db.commit()
    benchmark_path = ROOT.parent / 'references/aipat/df_smj_4k_final.csv'
    prior = pd.read_csv(benchmark_path, usecols=['application_id', 'appln_nr_8'], dtype=str)
    exposed = set(prior.application_id.dropna()) | set(prior.appln_nr_8.dropna())
    pieces, counts = [], []
    for period in design['periods']:
        frame = pd.read_sql_query('SELECT * FROM applications WHERE published_date BETWEEN ? AND ?',
                                  db, params=[period['start'], period['end']])
        before_exposure_exclusion = len(frame)
        frame = frame[~frame.application_id.isin(exposed)].copy()
        total = len(frame)
        used = set()
        for role in ['pilot', 'development', 'reserved_test', 'challenge']:
            candidates = frame[~frame.application_id.isin(used)]
            if role == 'challenge':
                candidates = candidates[candidates.title.str.contains(design['challenge_title_regex'], case=False, regex=True, na=False)]
            take = design['per_period'][role]
            selected = sample_group(candidates, period['name'], 'challenge' if role == 'challenge' else 'primary', design['seed'], take)
            if len(selected) != take:
                raise ValueError(f'Insufficient cases for {period["name"]}/{role}')
            selected['period'], selected['role'] = period['name'], role
            selected['stratum_population_n'] = total
            selected['inclusion_probability'] = take / total if role != 'challenge' else float('nan')
            selected['population_weight'] = total / take if role != 'challenge' else float('nan')
            selected['family_clearance'] = 'PENDING'
            selected['training_overlap_clearance'] = 'PENDING'
            used.update(selected.application_id)
            pieces.append(selected)
            counts.append({'period': period['name'], 'role': role, 'population_n': total,
                           'previously_exposed_applications_excluded': before_exposure_exclusion-total, 'selected_n': take})
    db.close()
    manifest = pd.concat(pieces, ignore_index=True)
    if manifest.application_id.duplicated().any() or manifest.pgpub_id.duplicated().any():
        raise ValueError('Review sample contains repeated applications/publications')
    wanted = set(manifest.pgpub_id)
    texts = []
    with zipfile.ZipFile(ABSTRACT) as archive:
        members = [x for x in archive.namelist() if x.endswith('.tsv')]
        if len(members) != 1:
            raise ValueError('Unexpected abstract ZIP structure')
        with archive.open(members[0]) as stream:
            for chunk in pd.read_csv(stream, sep='\t', dtype=str, chunksize=100_000):
                texts.append(chunk.loc[chunk.pgpub_id.isin(wanted), ['pgpub_id', 'application_abstract']])
    text = pd.concat(texts, ignore_index=True)
    if text.pgpub_id.duplicated().any():
        raise ValueError('Duplicate abstract publication keys')
    manifest = manifest.merge(text, on='pgpub_id', how='left', validate='one_to_one')
    manifest['review_id'] = manifest.pgpub_id.map(lambda x: 'PV2-' + hashlib.sha256((design['seed']+'|'+x).encode()).hexdigest()[:12])
    normalized = manifest.application_abstract.fillna('').str.lower().str.replace(r'\W+', ' ', regex=True).str.strip()
    manifest['abstract_sha256'] = normalized.map(lambda x: hashlib.sha256(x.encode()).hexdigest() if x else '')
    cross = manifest[manifest.abstract_sha256.ne('')].groupby('abstract_sha256').role.nunique()
    collision_hashes = set(cross[cross.gt(1)].index)
    manifest['cross_role_identical_text'] = manifest.abstract_sha256.isin(collision_hashes)
    manifest['drawn_role'] = manifest.role
    manifest['quarantine_reason'] = ''
    priority = {'reserved_test': 0, 'development': 1, 'pilot': 2, 'challenge': 3}
    for text_hash in collision_hashes:
        indices = manifest.index[manifest.abstract_sha256.eq(text_hash)]
        keep_role = min(manifest.loc[indices, 'role'], key=lambda x: priority[x])
        quarantine = indices[manifest.loc[indices, 'role'].ne(keep_role)]
        manifest.loc[quarantine, 'role'] = 'quarantined_duplicate'
        manifest.loc[quarantine, 'quarantine_reason'] = 'Exact abstract duplicate of a higher-priority role; no labels inspected'
        manifest.loc[quarantine, ['population_weight', 'inclusion_probability']] = float('nan')
    manifest.drop(columns=['application_abstract']).to_csv(OUT / 'CURATOR_sample_manifest.csv', index=False)
    counts = pd.DataFrame(counts)
    final_counts = manifest.groupby(['period', 'role']).size().rename('review_n_after_quarantine').reset_index()
    counts = counts.merge(final_counts, on=['period', 'role'], how='left', validate='one_to_one')
    counts['review_n_after_quarantine'] = counts.review_n_after_quarantine.fillna(0).astype(int)
    counts.to_csv(OUT / 'sampling_counts.csv', index=False)
    for role in design['per_period']:
        folder = OUT / role
        folder.mkdir(exist_ok=True)
        pack = manifest[manifest.role.eq(role)][['review_id', 'title', 'application_abstract']].copy()
        pack = pack.sample(frac=1, random_state=42)
        for reviewer in ['reviewer_1', 'reviewer_2']:
            template = pack.copy()
            template['abstract_ai_label'], template['reason'], template['evidence_span'] = '', '', ''
            template.to_csv(folder / f'{reviewer}_BLIND_UNLABELLED.csv', index=False)
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'protocol_version': design['protocol_version'],
        'finance_or_disclosure_inputs_read': [], 'model_prediction_inputs_read': [],
        'inputs': {'metadata': str(META), 'metadata_sha256': fingerprint, 'abstracts': str(ABSTRACT),
                   'previously_exposed_benchmark_identifiers_only': str(benchmark_path),
                   'abstract_zip_sha256': hashlib.file_digest(ABSTRACT.open('rb'), 'sha256').hexdigest()},
        'sampling_unit': 'first observed utility publication per application',
        'selected_applications': len(manifest), 'missing_or_empty_abstract': int(normalized.eq('').sum()),
        'cross_role_identical_text_groups': len(collision_hashes),
        'quarantined_duplicate_documents': int(manifest.role.eq('quarantined_duplicate').sum()),
        'reserved_test_previously_exposed_applications': int(manifest.loc[manifest.role.eq('reserved_test'), 'application_id'].isin(exposed).sum()),
        'family_clearance': 'PENDING: no global continuity/family mapping supplied',
        'training_overlap_clearance': 'PENDING: external training uses PATSTAT and DOCDB identifiers; global USPTO-to-family linkage absent',
        'confirmatory_test_ready': False, 'test_labels_examined': False,
        'reserved_test_note': 'Reserved by procedure, not encrypted or access-controlled. All labels remain blank.',
        'weights_note': 'N_h/n_h for each representative role; challenge weights not defined and challenge results must stay separate.'}
    (OUT / 'independence_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    digest = hashlib.file_digest((OUT / 'CURATOR_sample_manifest.csv').open('rb'), 'sha256').hexdigest()
    (OUT / 'sample_manifest.sha256').write_text(digest+'\n', encoding='ascii')
    print(json.dumps(audit, indent=2), flush=True)
    print(counts.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
