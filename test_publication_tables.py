import unittest

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from publication_tables import regression


class FixedEffectTests(unittest.TestCase):
    def test_absorbed_estimates_match_explicit_dummies(self):
        rng = np.random.default_rng(14)
        frame = pd.DataFrame({"permco": np.repeat(np.arange(12), 10),
                              "fyear": np.tile(np.arange(2015, 2025), 12),
                              "ff48": np.repeat(np.arange(3), 40), "x": rng.normal(size=120)})
        frame["y"] = 2 * frame.x + frame.ff48 + .3 * frame.fyear + rng.normal(size=120)
        for firm in (False, True):
            coef, info = regression(frame, "y", ["x"], firm)
            effects = "C(permco)" if firm else "C(ff48)"
            model = smf.ols(f"y ~ x + {effects} + C(fyear)", frame).fit()
            self.assertAlmostEqual(coef[0]["coefficient"], model.params.x, places=8)
            self.assertEqual(info["fe_rank"], np.linalg.matrix_rank(model.model.exog) - 1)
            self.assertAlmostEqual(info["adjusted_r2"], model.rsquared_adj, places=8)
            clustered = model.get_robustcov_results(cov_type="cluster", groups=frame.permco)
            index = model.model.exog_names.index("x")
            self.assertAlmostEqual(coef[0]["se"], clustered.bse[index], places=8)


if __name__ == "__main__":
    unittest.main()
