import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _annotate(ax, bars, fmt="{:.3f}", fontsize=8):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_bar(labels, values, out_path, y_label, title, annotate_fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(range(len(labels)), values)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)

    vmin, vmax = min(values), max(values)
    margin = max(0.2, 0.08 * (vmax - vmin if vmax > vmin else 1))
    ax.set_ylim(max(0, vmin - margin), vmax + margin * 2)

    _annotate(ax, bars, fmt=annotate_fmt, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate latency (and optional params) figures.")
    parser.add_argument("--input", required=True, help="Path to latency_params_summary.csv")
    parser.add_argument("--out", required=True, help="Output directory for figures")
    parser.add_argument(
        "--with-params",
        action="store_true",
        help="Also generate a parameter-count chart if 'params' column exists.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if "model" not in df.columns or "latency_ms" not in df.columns:
        raise ValueError("CSV must contain columns: model, latency_ms")

    labels = df["model"].astype(str).tolist()
    lat = df["latency_ms"].astype(float).tolist()

    plot_bar(
        labels=labels,
        values=lat,
        out_path=out_dir / "latency_ms_v4v5.png",
        y_label="Latency (ms)",
        title="Inference latency (batch=1, 256×256, RTX 3060)",
        annotate_fmt="{:.3f}",
    )
    print(f"Saved: {out_dir / 'latency_ms_v4v5.png'}")

    if args.with_params and "params" in df.columns:
        params = df["params"].astype(float).tolist()
        plot_bar(
            labels=labels,
            values=params,
            out_path=out_dir / "params_v4v5.png",
            y_label="Parameters",
            title="Parameter count by model variant",
            annotate_fmt="{:.0f}",
        )
        print(f"Saved: {out_dir / 'params_v4v5.png'}")

    print(f"All latency figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
 
