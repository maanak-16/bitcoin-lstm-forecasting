"""Feature engineering and leakage-safe sliding windows."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["ret", "vol_chg"]


def add_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log return of price and daily log change in traded volume.

    Row t holds information known at the end of day t only.
    """
    out = pd.DataFrame(index=prices.index)
    out["price"] = prices["price_usd"]
    out["ret"] = np.log(prices["price_usd"]).diff()
    out["vol_chg"] = np.log(prices["volume_usd"]).diff()
    return out.dropna()


def window_positions(n_rows: int, window: int) -> np.ndarray:
    """Positions j that can be forecast: each needs rows j-window .. j-1 as input."""
    return np.arange(window, n_rows)


def make_windows(values: np.ndarray, positions: np.ndarray, window: int) -> np.ndarray:
    """X[k] = values[j-window : j] for j = positions[k].

    The input for forecasting day j ends at day j-1, so the target day is
    never part of its own input.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return np.stack([values[j - window:j] for j in positions])


def scaled_return_windows(feat: pd.DataFrame, window: int, train_pos: np.ndarray,
                          test_pos: np.ndarray):
    """Standardised feature windows with the scaler fitted on pre-test rows only.

    Returns X_train, y_train, X_test, with y as raw next-day log returns.
    """
    cutoff = test_pos[0]  # first test day; nothing from it onward is used to fit
    scaler = StandardScaler().fit(feat[FEATURES].values[:cutoff])
    values = scaler.transform(feat[FEATURES].values)
    ret = feat["ret"].values
    return (make_windows(values, train_pos, window), ret[train_pos],
            make_windows(values, test_pos, window))


def returns_to_prices(prev_prices: np.ndarray, pred_returns: np.ndarray) -> np.ndarray:
    """Convert predicted log returns for day j into price forecasts: P_{j-1} * exp(r_j)."""
    return np.asarray(prev_prices) * np.exp(np.asarray(pred_returns))
