"""Loading the daily Bitcoin price and volume data."""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "btc_daily_coinmetrics.csv"


def load_prices(path=DATA_PATH, start="2015-01-01", end=None):
    """Load daily BTC price (USD) and reported spot volume (USD).

    Returns a DataFrame indexed by date with columns ``price_usd`` and
    ``volume_usd``. Raises if any calendar day is missing, because the
    sliding windows assume one row per day.
    """
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df = df.loc[start:end].dropna()
    gaps = df.index.to_series().diff().dropna()
    if not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError("Price series has missing days; windows would be misaligned.")
    return df
