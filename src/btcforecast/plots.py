"""Figures and tables shared by the experiment script and the notebook."""
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"Naive": "#6b7280", "Drift": "#a3a3a3", "Ridge": "#2563eb",
          "LSTM (returns)": "#dc2626", "LSTM (2023 design)": "#d97706"}


def plot_folds(feat, path, test_years):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(feat.index, feat["price"], color="#111827", lw=1)
    for i, y in enumerate(test_years):
        ax.axvspan(pd.Timestamp(f"{y}-01-01"), min(pd.Timestamp(f"{y}-12-31"), feat.index[-1]),
                   color="#2563eb" if i % 2 == 0 else "#60a5fa", alpha=0.12, lw=0)
        ax.text(pd.Timestamp(f"{y}-07-01"), feat["price"].max() * 2.2, str(y),
                ha="center", fontsize=8, color="#1e3a8a")
    ax.set_yscale("log")
    ax.set_ylim(top=feat["price"].max() * 4)
    ax.set_ylabel("BTC price (USD, log scale)")
    ax.set_title("Walk-forward design: each shaded year is forecast by models trained only on earlier data")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_relative_mae(by_fold, path, cap=25):
    """Percentage change in MAE versus naive; bars beyond +/-cap% are clipped and labelled."""
    naive = by_fold[by_fold.model == "Naive"].set_index("fold")["MAE"]
    others = [m for m in COLORS if m in set(by_fold.model) and m != "Naive"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    width = 0.8 / len(others)
    for k, m in enumerate(others):
        s = 100 * (by_fold[by_fold.model == m].set_index("fold")["MAE"] / naive - 1)
        x = s.index + (k - (len(others) - 1) / 2) * width
        ax.bar(x, s.clip(-cap, cap).values, width, label=m, color=COLORS[m])
        for xi, v in zip(x, s.values):
            if abs(v) > cap:
                ax.text(xi, cap * 0.92 * (1 if v > 0 else -1), f"{v:+.0f}%", rotation=90,
                        ha="center", va="top" if v > 0 else "bottom", fontsize=7, color="white")
    ax.axhline(0, color="#111827", lw=1)
    ax.set_ylim(-cap, cap)
    ax.set_ylabel("MAE vs naive (%)  — below 0 = better")
    ax.set_xticks(sorted(naive.index))
    ax.set_title("Next-day forecast error relative to 'tomorrow = today', by test year")
    ax.legend(frameon=False, fontsize=8, ncol=len(others), loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def plot_zoom(preds, path, start="2021-04-15", end="2021-06-30"):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    window = preds[(preds.date >= start) & (preds.date <= end)]
    actual = window.drop_duplicates("date").set_index("date")["actual"]
    ax.plot(actual.index, actual.values, color="#111827", lw=2, label="Actual")
    for m, g in window.groupby("model"):
        if m in ("Naive", "Drift"):
            continue
        ax.plot(g.date, g.pred, color=COLORS.get(m), lw=1.2, label=m)
    ax.set_ylabel("USD")
    ax.set_title("Forecasts through the May 2021 crash")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def fmt_p(p):
    """p-value for display; tiny values read as < 0.001 rather than 0.000."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def to_markdown(overall):
    lines = ["| Model | MAE (USD) | RMSE (USD) | MAPE | Direction | DM vs naive (p) |",
             "|---|---:|---:|---:|---:|---:|"]
    for _, r in overall.iterrows():
        direction = "–" if pd.isna(r.Direction) else f"{r.Direction:.1f}%"
        dm = "–" if pd.isna(r.get("DM_p")) else f"{r.DM_stat:+.2f} ({fmt_p(r.DM_p)})"
        lines.append(f"| {r.model} | {r.MAE:,.0f} | {r.RMSE:,.0f} | {r.MAPE:.2f}% "
                     f"| {direction} | {dm} |")
    return "\n".join(lines)


def findings(overall, benchmark="Naive"):
    """One plain-English line per model comparing it with the naive benchmark."""
    naive_mae = overall.set_index("model").loc[benchmark, "MAE"]
    lines = []
    for _, r in overall.iterrows():
        if r.model == benchmark:
            continue
        change = 100 * (r.MAE / naive_mae - 1)
        verdict = ("significantly better than" if r.DM_p < 0.05 and r.DM_stat < 0 else
                   "significantly worse than" if r.DM_p < 0.05 else
                   "not significantly different from")
        lines.append(f"- **{r.model}**: MAE {change:+.1f}% vs naive; "
                     f"{verdict} naive (Diebold-Mariano p {'<' if r.DM_p < 0.001 else '='} "
                     f"{fmt_p(r.DM_p).lstrip('< ')}).")
    return "\n".join(lines)


def update_readme(readme_path, table, overall):
    """Replace the auto-generated results block in the README, if the markers exist."""
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    text = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""
    if start not in text or end not in text:
        return
    block = f"{start}\n{table}\n\n{findings(overall)}\n{end}"
    before, rest = text.split(start, 1)
    readme_path.write_text(before + block + rest.split(end, 1)[1], encoding="utf-8")
