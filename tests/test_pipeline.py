"""Tests for leakage-safety, price conversion, metrics and the 2023 audit.

Run with:  python -m pytest   (or: python -m unittest discover tests)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from btcforecast.audit import audit  # noqa: E402
from btcforecast.data import load_prices  # noqa: E402
from btcforecast.evaluate import diebold_mariano, price_metrics, walk_forward_folds  # noqa: E402
from btcforecast.features import (add_features, make_windows, returns_to_prices,  # noqa: E402
                                  scaled_return_windows, window_positions)


def toy_features(n=400):
    idx = pd.date_range("2019-06-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    volume = np.exp(rng.normal(20, 0.3, n))
    return add_features(pd.DataFrame({"price_usd": price, "volume_usd": volume}, index=idx))


class TestWindows(unittest.TestCase):
    def test_window_ends_the_day_before_the_target(self):
        values = np.arange(10.0)
        pos = window_positions(10, 3)
        X = make_windows(values, pos, 3)
        self.assertEqual(pos[0], 3)
        np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2])  # forecasting day 3
        np.testing.assert_array_equal(X[-1].ravel(), [6, 7, 8])  # forecasting day 9
        self.assertTrue(all(X[k].max() < pos[k] for k in range(len(pos))))

    def test_scaler_ignores_test_period(self):
        feat = toy_features()
        pos = window_positions(len(feat), 30)
        train, test = pos[:250], pos[250:]
        X_tr, _, _ = scaled_return_windows(feat, 30, train, test)
        # Changing the test-period data must not change the training inputs
        shocked = feat.copy()
        shocked.iloc[test[0]:, shocked.columns.get_loc("ret")] *= 50
        X_tr2, _, _ = scaled_return_windows(shocked, 30, train, test)
        np.testing.assert_allclose(X_tr, X_tr2)

    def test_folds_never_train_on_the_test_year(self):
        feat = toy_features(900)
        for year, train, test in walk_forward_folds(feat, 30, [2020, 2021]):
            self.assertLess(feat.index[train].max(), pd.Timestamp(f"{year}-01-01"))
            self.assertTrue((feat.index[test].year == year).all())
            self.assertLess(train.max(), test.min())


class TestConversionAndMetrics(unittest.TestCase):
    def test_true_returns_reproduce_prices(self):
        feat = toy_features()
        pos = np.arange(1, len(feat))
        prices = returns_to_prices(feat["price"].values[pos - 1], feat["ret"].values[pos])
        np.testing.assert_allclose(prices, feat["price"].values[pos])

    def test_metrics_known_values(self):
        m = price_metrics(actual=[110, 90], pred=[100, 100], prev=[100, 100])
        self.assertAlmostEqual(m["MAE"], 10)
        self.assertAlmostEqual(m["RMSE"], 10)
        self.assertTrue(np.isnan(m["Direction"]))  # zero-change forecast has no direction
        m = price_metrics(actual=[110, 90], pred=[105, 105], prev=[100, 100])
        self.assertAlmostEqual(m["Direction"], 50)

    def test_dm_sign(self):
        rng = np.random.default_rng(1)
        bench = rng.normal(0, 2, 500)
        stat, p = diebold_mariano(bench * 0.5, bench)
        self.assertLess(stat, 0)
        self.assertLess(p, 0.01)


class TestAudit(unittest.TestCase):
    def test_diagnosis_reconstructs_actual_price(self):
        a = audit()
        self.assertLess(a["reconstruction_error_usd"], 1.0)
        self.assertGreater(a["true_mae_usd"], a["printed_mae_usd"])


class TestData(unittest.TestCase):
    def test_dataset_is_daily_and_positive(self):
        df = load_prices()
        self.assertGreater(len(df), 4000)
        self.assertTrue((df > 0).all().all())


if __name__ == "__main__":
    unittest.main()
