import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aipd_extension import metrics, operating_threshold, score_recent, separate_historical


class FixedModel:
    def predict_proba(self, texts):
        p = np.where(texts.str.contains("neural"), .8, .1)
        return np.column_stack([1 - p, p])


class ExtensionTests(unittest.TestCase):
    def test_exact_duplicates_keep_latest_split(self):
        df = pd.DataFrame({"pgpub_id": list("abcde"), "year": [2020, 2022, 2023, 2020, 2022],
                           "application_abstract": ["A Neural-network", "A neural network",
                                                    "A NEURAL network!", "geometry", ""]})
        train, dev, test, audit = separate_historical(df)
        self.assertEqual(list(train.pgpub_id), ["d"])
        self.assertTrue(dev.empty)
        self.assertEqual(list(test.pgpub_id), ["c"])
        self.assertEqual(int(audit.included.sum()), 2)

    def test_threshold_and_metrics(self):
        threshold = operating_threshold([0, 1, 0, 1], [.1, .9, .2, .7])
        self.assertEqual(threshold, .7)
        self.assertEqual(metrics([0, 1, 0, 1], np.array([.1, .9, .2, .7]), threshold)["f1"], 1)

    def test_scoring_multiple_chunks_and_missing(self):
        ids = list(map(str, range(10003)))
        recent = pd.DataFrame({"pgpub_id": ids, "application_id": ids, "published_date": "2025-01-01"})
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            stage = out / "staged.csv"
            pd.DataFrame({"pgpub_id": ids[:-1], "application_abstract":
                          ["ordinary"] * 10000 + ["neural", ""]}).to_csv(stage, index=False)
            score_recent(FixedModel(), .7, stage, recent, out)
            scored = pd.read_csv(out / "publications_2024_2025.csv", dtype={"pgpub_id": str})
            self.assertEqual(len(scored), len(ids))
            self.assertEqual(scored.aipd_surrogate_label.sum(), 1)
            self.assertEqual(list(scored.tail(3).score_status),
                             ["scored", "empty_abstract", "abstract_not_joined"])


if __name__ == "__main__":
    unittest.main()
