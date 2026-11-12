import unittest

import numpy as np
import pandas as pd

from aipd_v2_model import combined, split


class Head:
    def __init__(self, p):
        self.p = np.asarray(p)

    def predict_proba(self, x):
        return np.column_stack((1 - self.p, self.p))


class V2Tests(unittest.TestCase):
    def test_forward_split_protects_benchmark_and_calibration(self):
        frame = pd.DataFrame({
            "year": [2020, 2021, *([2022] * 12), 2023, 2023],
            "hash": ["a", "benchmark", *list("bcdefghijklm"), "benchmark", "old"],
            "fresh_benchmark": [False] * 14 + [True, False],
        })
        fit, cal, test = split(frame, 2023, fresh=True)
        self.assertEqual(list(test.hash), ["benchmark"])
        self.assertTrue(set(fit.hash).isdisjoint(test.hash))
        self.assertTrue(set(fit.hash).isdisjoint(cal.hash))
        self.assertTrue(set(cal.hash).isdisjoint(test.hash))
        self.assertTrue(fit.year.le(2022).all())
        self.assertTrue(cal.year.eq(2022).all())

    def test_component_score_uses_largest_head(self):
        p = combined({"a": Head([.2, .9]), "b": Head([.7, .1])}, None)
        np.testing.assert_allclose(p, [.7, .9])


if __name__ == "__main__":
    unittest.main()
