import argparse
from pathlib import Path
import re

import pandas as pd
import matplotlib.pyplot as plt


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_").lower()


def _pick_label_column(df: pd.DataFrame) -> str:
    for col in ["pipeline_label_short", "pipeline_label_full", "pipeline"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found. Expected one of: pipeline_label_short, pipeline_label_full, pipeline")


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    if "group_order" in df.columns:
        return df.sort_values("group_order").reset_index(drop=True)
    return df.reset_index(drop=True)


def _annotate_bars(ax, bars, values, fmt="{:.2f}", fontsize=8):
    for bar, v in zip(bars, values):
        ax.annotate(
            fmt.format(v),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def _smart_ylim(values, margin=1.0):
    vmin = min(values)
    vmax = max(values)
    rng = vmax - vmin
    if rng < 3:
        lo = max(0, vmin - margin)
        hi = min(100, vmax + margin)
    elif rng < 10:
        lo = max(0, vmin - 1.5)
        hi = min(100, vmax + 1.0)
    else:
        lo = 0
        hi = min(100, vmax + 2.0)
    return lo, hi


def _draw_group_separators(ax, labels, groups):
    if groups is None:
        return
    # vertical separators where group changes
    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            ax.axvline(i - 0.5, linewidth=0.8, alpha=0.5)
    # top group labels (optional, concise)
    unique_spans = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[i - 1]:
            unique_spans.append((groups[start], start, i - 1))
            start = i
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.03 * (ymax - ymin)
    for g, s, e in unique_spans:
        x = (s + e) / 2
        ax.text(x, y_text, str(g), ha="center", va="top", fontsize=8)


def plot_metric(df: pd.DataFrame, metric_col: str, y_label: str, out_path: Path, title: str = None):
    if metric_col not in df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    label_col = _pick_label_column(df)
    labels = df[label_col].astype(str).tolist()
    values = df[metric_col].astype(float).tolist()
    groups = df["regime_group"].astype(str).tolist() if "regime_group" in df.columns else None

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    bars = ax.bar(range(len(labels)), values)

    ax.set_ylabel(y_label, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    lo, hi = _smart_ylim(values)
    ax.set_ylim(lo, hi)

    _annotate_bars(ax, bars, values, fmt="{:.2f}", fontsize=8)
    _draw_group_separators(ax, labels, groups)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    # also save vector format for Word (optional but useful)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate downstream V4/V5 thesis figures (Accuracy and Macro-F1).")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to plots_data_accuracy_macrof1.csv",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for figures",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = _sort_df(df)

    plot_metric(
        df,
        metric_col="accuracy_pct",
        y_label="Accuracy (%)",
        out_path=out_dir / "downstream_accuracy_v4v5.png",
        title="Downstream 5-class classification (ResNet-18): Accuracy",
    )

    plot_metric(
        df,
        metric_col="macro_f1_pct",
        y_label="Macro-F1 (%)",
        out_path=out_dir / "downstream_macrof1_v4v5.png",
        title="Downstream 5-class classification (ResNet-18): Macro-F1",
    )

    print(f"Saved figures to: {out_dir.resolve()}")
    print("- downstream_accuracy_v4v5.png / .svg")
    print("- downstream_macrof1_v4v5.png / .svg")


if __name__ == "__main__":
    main()
