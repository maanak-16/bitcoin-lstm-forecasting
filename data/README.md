# Data

`btc_daily_coinmetrics.csv` contains daily Bitcoin data from 2014-01-01 to 2026-05-23:

| Column | Coin Metrics field | Meaning |
|---|---|---|
| `date` | `time` | UTC calendar day |
| `price_usd` | `PriceUSD` | Coin Metrics reference price, USD |
| `volume_usd` | `volume_reported_spot_usd_1d` | Reported spot trading volume, USD |

**Source:** [Coin Metrics Community Data](https://github.com/coinmetrics/data), file `csv/btc.csv`, commit `f1a36af`. © Coin Metrics, Inc., licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). These three columns were extracted and renamed; no values were changed.

To refresh the data, download the latest `csv/btc.csv` from the repository above and extract the same three columns.
