import json
import unittest
from pathlib import Path

import pandas as pd

from build_blind_review import sample_group

ROOT = Path(__file__).resolve().parent
OUT = ROOT / json.loads((ROOT / 'design.json').read_text()).get('output_subdir', 'outputs')


class DesignTests(unittest.TestCase):
    def test_draw_is_deterministic_and_order_independent(self):
        d = pd.DataFrame({'application_id': [str(i) for i in range(100)]})
        a = sample_group(d, 'cohort', 'primary', 'seed', 10)
        b = sample_group(d.sample(frac=1, random_state=5), 'cohort', 'primary', 'seed', 10)
        self.assertEqual(a.application_id.tolist(), b.application_id.tolist())

    def test_role_draws_are_disjoint_when_taken_from_remainder(self):
        d = pd.DataFrame({'application_id': [str(i) for i in range(100)]})
        a = sample_group(d, 'cohort', 'primary', 'seed', 10)
        b = sample_group(d[~d.application_id.isin(a.application_id)], 'cohort', 'primary', 'seed', 10)
        self.assertFalse(set(a.application_id) & set(b.application_id))

    def test_saved_review_keys_and_weights(self):
        d = pd.read_csv(OUT / 'CURATOR_sample_manifest.csv', dtype={'application_id': str})
        self.assertEqual(len(d), 3060)
        self.assertFalse(d.application_id.duplicated().any())
        self.assertFalse(d.pgpub_id.duplicated().any())
        for _, frame in d[d.role.eq('reserved_test')].groupby('period'):
            self.assertEqual(len(frame), 600)
            self.assertAlmostEqual(frame.population_weight.sum(), float(frame.stratum_population_n.iloc[0]))
            self.assertAlmostEqual(float(frame.inclusion_probability.iloc[0]), 600/float(frame.stratum_population_n.iloc[0]))

    def test_reviewer_files_are_unlabelled_and_finance_free(self):
        audit = json.loads((OUT / 'independence_audit.json').read_text())
        self.assertEqual(audit['finance_or_disclosure_inputs_read'], [])
        self.assertEqual(audit['model_prediction_inputs_read'], [])
        self.assertFalse(audit['confirmatory_test_ready'])
        self.assertEqual(audit['reserved_test_previously_exposed_applications'], 0)
        expected = {'review_id', 'title', 'application_abstract', 'abstract_ai_label', 'reason', 'evidence_span'}
        for path in OUT.glob('*/*_BLIND_UNLABELLED.csv'):
            frame = pd.read_csv(path)
            self.assertEqual(set(frame.columns), expected)
            self.assertTrue(frame.abstract_ai_label.isna().all())


if __name__ == '__main__':
    unittest.main()
