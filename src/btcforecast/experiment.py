"""Experiment configuration shared by the script and the notebook."""
from . import baselines

TEST_YEARS = [2020, 2021, 2022, 2023, 2024, 2025, 2026]


def build_models(skip_lstm=False, quick=False, seeds=3):
    """Model functions and their input window lengths (days)."""
    models = {"Naive": baselines.naive, "Drift": baselines.drift, "Ridge": baselines.ridge}
    windows = {"Naive": 30, "Drift": 30, "Ridge": 30}
    if not skip_lstm:
        from . import lstm
        n_seeds, epochs, orig_epochs = (1, 20, 10) if quick else (seeds, 100, 30)
        models["LSTM (returns)"] = lambda *a, seed=0: lstm.return_lstm(
            *a, seed=seed, n_seeds=n_seeds, epochs=epochs)
        models["LSTM (2023 design)"] = lambda *a, seed=0: lstm.original_lstm(
            *a, seed=seed, epochs=orig_epochs)
        windows.update({"LSTM (returns)": 30, "LSTM (2023 design)": 60})
    return models, windows
