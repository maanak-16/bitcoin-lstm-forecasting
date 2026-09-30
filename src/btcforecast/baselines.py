"""Baseline forecasters. Each returns predicted next-day log returns for the test days."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

from .features import scaled_return_windows


def naive(feat, window, train_pos, test_pos, seed=None):
    """Random walk: tomorrow's price equals today's (predicted return 0)."""
    return np.zeros(len(test_pos))


def drift(feat, window, train_pos, test_pos, seed=None):
    """Random walk with drift: predict the average training-period daily return."""
    return np.full(len(test_pos), feat["ret"].values[train_pos].mean())


def ridge(feat, window, train_pos, test_pos, seed=None):
    """Linear model on the same lagged features the LSTM sees.

    The regularisation strength is chosen by time-series cross-validation
    inside the training period only.
    """
    X_tr, y_tr, X_te = scaled_return_windows(feat, window, train_pos, test_pos)
    model = RidgeCV(alphas=np.logspace(-1, 5, 25), cv=TimeSeriesSplit(n_splits=5))
    model.fit(X_tr.reshape(len(X_tr), -1), y_tr)
    return model.predict(X_te.reshape(len(X_te), -1))
