"""Run the full walk-forward experiment and write results + figures.

Usage (from the repo root):
    python scripts/run_experiments.py              # everything (needs TensorFlow)
    python scripts/run_experiments.py --skip-lstm  # baselines and audit only
    python scripts/run_experiments.py --quick      # 1 seed, fewer epochs (smoke test)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btcforecast.audit import audit  # noqa: E402
from btcforecast.data import load_prices  # noqa: E402
from btcforecast.experiment import TEST_YEARS, build_models  # noqa: E402
from btcforecast.evaluate import price_metrics, run_walk_forward, summarise  # noqa: E402
from btcforecast.features import add_features  # noqa: E402
from btcforecast.plots import (plot_folds, plot_relative_mae, plot_zoom,  # noqa: E402
                               to_markdown, update_readme)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"


def audit_table(feat):
    a = audit()
    window = feat.loc["2023-01-02":"2023-04-25"]
    prev = feat["price"].shift(1).loc[window.index]
    naive_mae = price_metrics(window["price"], prev, prev)["MAE"]
    return pd.DataFrame([
        {"item": "MAE printed in 2023 notebook", "usd": a["printed_mae_usd"]},
        {"item": "MAE after correcting the inverse scaling", "usd": a["true_mae_usd"]},
        {"item": "Naive 'tomorrow = today' MAE, same test days", "usd": naive_mae},
    ]), a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    feat = add_features(load_prices())
    print(f"Data: {feat.index[0].date()} to {feat.index[-1].date()} ({len(feat)} days)")

    audit_df, a = audit_table(feat)
    audit_df.to_csv(RESULTS / "original_2023_audit.csv", index=False)
    print(f"\n2023 audit: recovered training min ${a['recovered_training_min_open']:,.2f}; "
          f"last-day reconstruction error ${a['reconstruction_error_usd']:.2f}")
    print(audit_df.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))

    models, windows = build_models(args.skip_lstm, args.quick, args.seeds)
    preds = run_walk_forward(feat, models, windows, TEST_YEARS)
    by_fold, overall = summarise(preds)

    preds.to_csv(RESULTS / "predictions.csv", index=False)
    by_fold.to_csv(RESULTS / "metrics_by_fold.csv", index=False)
    overall.to_csv(RESULTS / "metrics_overall.csv", index=False)
    table = to_markdown(overall)
    (RESULTS / "metrics_overall.md").write_text(table + "\n")
    print("\n" + table)

    plot_folds(feat, FIGURES / "walk_forward_design.png", TEST_YEARS)
    plot_relative_mae(by_fold, FIGURES / "relative_mae_by_year.png")
    plot_zoom(preds, FIGURES / "forecasts_may_2021.png")
    update_readme(ROOT / "README.md", table, overall)
    print(f"\nWrote results to {RESULTS}")


if __name__ == "__main__":
    main()
