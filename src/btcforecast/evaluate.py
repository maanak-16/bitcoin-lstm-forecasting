"""Walk-forward evaluation, error metrics and a Diebold-Mariano test."""
import time

import numpy as np
import pandas as pd
from scipy import stats

from .features import returns_to_prices, window_positions


def walk_forward_folds(feat: pd.DataFrame, window: int, test_years):
    """Expanding-window folds: train on every day before the test year, test on that year."""
    positions = window_positions(len(feat), window)
    dates = feat.index[positions]
    folds = []
    for year in test_years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        train_pos = positions[dates < start]
        test_pos = positions[(dates >= start) & (dates <= end)]
        if len(test_pos):
            folds.append((year, train_pos, test_pos))
    return folds


def price_metrics(actual, pred, prev):
    """MAE / RMSE in USD, MAPE in %, and directional accuracy in %.

    Directional accuracy counts days where the forecast moved in the same
    direction as the actual price; it is undefined for a zero-change forecast.
    """
    actual, pred, prev = map(np.asarray, (actual, pred, prev))
    err = pred - actual
    moved = np.sign(pred - prev)
    direction = (np.nan if np.all(moved == 0)
                 else 100 * np.mean(moved == np.sign(actual - prev)))
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE": 100 * np.mean(np.abs(err) / actual),
        "Direction": direction,
    }


def diebold_mariano(err_model, err_bench):
    """Two-sided DM test on squared errors for 1-step forecasts.

    Negative statistic = the model has lower squared error than the benchmark.
    """
    d = np.asarray(err_model) ** 2 - np.asarray(err_bench) ** 2
    stat = d.mean() / np.sqrt(d.var(ddof=1) / len(d))
    return stat, 2 * stats.norm.sf(abs(stat))


def run_walk_forward(feat, models: dict, windows: dict, test_years, seed=0, log=print):
    """Run every model on every fold.

    ``models`` maps a name to a function(feat, window, train_pos, test_pos, seed)
    returning predicted next-day log returns. ``windows`` maps each name to its
    input window length. Returns one row per test day with each model's price forecast.
    """
    rows = []
    for name, fn in models.items():
        for year, train_pos, test_pos in walk_forward_folds(feat, windows[name], test_years):
            t0 = time.time()
            pred_ret = fn(feat, windows[name], train_pos, test_pos, seed=seed)
            prev = feat["price"].values[test_pos - 1]
            rows.append(pd.DataFrame({
                "date": feat.index[test_pos], "fold": year, "model": name,
                "actual": feat["price"].values[test_pos], "prev": prev,
                "pred": returns_to_prices(prev, pred_ret),
            }))
            log(f"{name:<14} {year}: {len(train_pos):>5} train / {len(test_pos):>3} test days "
                f"({time.time() - t0:.0f}s)")
    return pd.concat(rows, ignore_index=True)


def summarise(preds: pd.DataFrame, benchmark="Naive"):
    """Metrics by fold and pooled over all folds, with a DM test against the benchmark."""
    by_fold = (preds.groupby(["model", "fold"])
               .apply(lambda g: pd.Series(price_metrics(g.actual, g.pred, g.prev)),
                      include_groups=False)
               .reset_index())
    overall = []
    bench = preds[preds.model == benchmark].set_index("date")
    for name, g in preds.groupby("model", sort=False):
        g = g.set_index("date")
        row = {"model": name, "days": len(g), **price_metrics(g.actual, g.pred, g.prev)}
        if name != benchmark:
            common = g.index.intersection(bench.index)
            stat, p = diebold_mariano((g.pred - g.actual)[common],
                                      (bench.pred - bench.actual)[common])
            row.update({"DM_stat": stat, "DM_p": p})
        overall.append(row)
    return by_fold, pd.DataFrame(overall)
