import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def _annotate_mean_std(ax, bars, means, stds, fontsize=8):
    for bar, m, s in zip(bars, means, stds):
        ax.annotate(
            f"{m:.2f}±{s:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _smart_ylim(means, stds, floor_zero=False):
    lows = [m - s for m, s in zip(means, stds)]
    highs = [m + s for m, s in zip(means, stds)]
    lo = min(lows)
    hi = max(highs)
    rng = hi - lo
    margin = max(0.5, 0.2 * rng)
    lo2 = lo - margin
    hi2 = hi + margin
    if floor_zero:
        lo2 = max(0, lo2)
    return lo2, hi2


def metric_title(metric: str) -> str:
    if metric.lower().startswith("macro"):
        return "Robustness under added noise (σ=25): Macro-F1@BestF1"
    if metric.lower().startswith("accuracy"):
        return "Robustness under added noise (σ=25): Accuracy@BestF1"
    return f"Robustness: {metric}"


def metric_ylabel(metric: str) -> str:
    if metric.lower().startswith("macro"):
        return "Macro-F1 (%)"
    if metric.lower().startswith("accuracy"):
        return "Accuracy (%)"
    return metric


def plot_metric(dfm: pd.DataFrame, out_path: Path):
    if "plot_order" in dfm.columns:
        dfm = dfm.sort_values("plot_order")
    labels = dfm["model_label"].astype(str).tolist()
    means = dfm["mean"].astype(float).tolist()
    stds = dfm["std"].astype(float).tolist()
    metric = str(dfm["metric"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = range(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(metric_ylabel(metric), fontsize=10)
    ax.set_title(metric_title(metric), fontsize=11)
    ax.grid(axis="y", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    lo, hi = _smart_ylim(means, stds, floor_zero=False)
    # keep within valid percentage range if this is % metric
    lo = max(0, lo)
    hi = min(100, hi)
    ax.set_ylim(lo, hi)

    _annotate_mean_std(ax, bars, means, stds, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate robustness figures (mean±std with error bars).")
    parser.add_argument("--input", required=True, help="Path to plots_data_robustness.csv")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    required = {"model_label", "metric", "mean", "std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for metric in df["metric"].dropna().unique():
        dfm = df[df["metric"] == metric].copy()
        filename = f"robustness_{_slug(metric)}.png"
        plot_metric(dfm, out_dir / filename)
        print(f"Saved: {out_dir / filename}")

    print(f"All robustness figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
