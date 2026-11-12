"""Fail-closed audit of a reserved test before any new confirmatory claim."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family-map', type=Path, help='Global USPTO application_id to family_id mapping in a documented namespace')
    parser.add_argument('--training-families', type=Path, help='Actual proposed training family_id manifest in the same namespace')
    parser.add_argument('--audit-only', action='store_true', help='Write the audit without a failing process exit status')
    args = parser.parse_args()
    design = json.loads((ROOT / 'design.json').read_text())
    out = ROOT / design.get('output_subdir', 'outputs')
    sample = pd.read_csv(out / 'CURATOR_sample_manifest.csv', dtype={'application_id': str, 'family_id': str})
    audit = {'confirmatory_test_ready': False, 'finance_inputs_used': False, 'issues': []}
    test = sample[sample.role.eq('reserved_test')]
    other = sample[~sample.role.isin(['reserved_test', 'quarantined_duplicate'])]
    assert set(test.application_id).isdisjoint(other.application_id)
    benchmark = pd.read_csv(ROOT.parent / 'references/aipat/df_smj_4k_final.csv',
                            usecols=['application_id', 'appln_nr_8'], dtype=str)
    exposed = set(benchmark.application_id.dropna()) | set(benchmark.appln_nr_8.dropna())
    seen = sorted(set(test.application_id) & exposed)
    audit['reserved_applications_in_previously_inspected_historical_benchmark'] = len(seen)
    if seen:
        pd.DataFrame({'application_id': seen}).to_csv(out / 'previously_exposed_test_ids.csv', index=False)
        audit['issues'].append('Some reserved applications have previously inspected benchmark labels; replace via documented patent-only exposure rule before test use.')
    test_hashes = set(test.abstract_sha256.dropna()) - {''}
    if test_hashes & set(other.abstract_sha256.dropna()):
        audit['issues'].append('Identical abstracts cross a test/non-test role boundary.')
    if args.family_map is None or args.training_families is None:
        audit['issues'].append('Global application-family map and actual proposed training-family manifest are required.')
    else:
        mapping = pd.read_csv(args.family_map, usecols=['application_id', 'family_id'], dtype=str)
        if mapping.application_id.duplicated().any() or mapping.family_id.isna().any():
            raise ValueError('Family mapping must have one nonmissing family ID per application')
        joined = sample.merge(mapping, on='application_id', how='left', validate='many_to_one')
        if joined.family_id.isna().any():
            audit['issues'].append('Some sampled applications lack family mappings.')
        test_families = set(joined.loc[joined.role.eq('reserved_test'), 'family_id'].dropna())
        development_families = set(joined.loc[joined.role.ne('reserved_test'), 'family_id'].dropna())
        training = set(pd.read_csv(args.training_families, usecols=['family_id'], dtype=str).family_id.dropna())
        audit['test_non_test_family_overlaps'] = len(test_families & development_families)
        audit['test_training_family_overlaps'] = len(test_families & training)
        if test_families & (development_families | training):
            audit['issues'].append('Reserved-test families overlap candidate development or training data.')
        audit['issues'].append('Mapping provenance/namespace and near-duplicate clearance still require documented curator verification.')
    audit['issues'].extend(['Final operating criteria require prospective supervisor approval.',
                            'Human annotation and independent evaluator sign-off are not yet complete.'])
    (out / 'confirmatory_clearance.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    if not audit['confirmatory_test_ready'] and not args.audit_only:
        sys.exit(2)


if __name__ == '__main__':
    main()
