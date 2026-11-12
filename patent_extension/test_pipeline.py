import unittest

import numpy as np
import pandas as pd

from abstract_model import metrics, normalize
from build_patent_panels import aggregate, dates, identifier


class PipelineTests(unittest.TestCase):
    def applications(self):
        return pd.DataFrame({'cik': ['1'], 'filing_date': [pd.Timestamp('2020-06-01')],
                             'official_ai93': [1.], 'author_ai': [np.nan]})

    def filings(self):
        return pd.DataFrame({'cik': ['1', '2'],
                             'period_start': [pd.Timestamp('2020-01-01')] * 2,
                             'period_end_actual': [pd.Timestamp('2020-12-31')] * 2})

    def test_missing_author_labels_are_not_zero(self):
        result = aggregate(self.filings(), self.applications())
        self.assertTrue(result.author_ai_application_count.isna().all())
        self.assertEqual(result.loc[0, 'official93_ai_application_count'], 1.)

    def test_missing_firm_link_is_not_zero(self):
        result = aggregate(self.filings(), self.applications())
        self.assertTrue(pd.isna(result.loc[1, 'observed_total_applications']))
        self.assertTrue(pd.isna(result.loc[1, 'official93_ai_application_count']))

    def test_unknown_official_label_is_not_zero(self):
        apps = self.applications()
        apps['official_ai93'] = np.nan
        result = aggregate(self.filings(), apps)
        self.assertTrue(result.official93_ai_application_count.isna().all())

    def test_counts_use_fiscal_dates(self):
        apps = self.applications()
        apps['filing_date'] = pd.Timestamp('2019-12-31')
        result = aggregate(self.filings(), apps)
        self.assertEqual(result.loc[0, 'observed_total_applications'], 0)

    def test_identifiers_preserve_digits(self):
        self.assertEqual(identifier(pd.Series(['000123.0'])).iloc[0], '123')

    def test_original_numeric_dates(self):
        self.assertEqual(dates(pd.Series([20250214])).iloc[0], pd.Timestamp('2025-02-14'))

    def test_normalization_is_deterministic(self):
        self.assertEqual(normalize('Machine LEARNING systems.'), normalize('machine learning systems'))

    def test_confusion_counts(self):
        result = metrics([0, 1, 1, 0], [.1, .8, .2, .9], .5)
        self.assertEqual([result[k] for k in ['tp', 'fp', 'tn', 'fn']], [1, 1, 1, 1])


if __name__ == '__main__':
    unittest.main()
