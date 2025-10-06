#!/usr/bin/env python3
# modes_cluster_report_min.py
# Merge nodal_features.csv with modes_clusters.csv, compute per-cluster stats,
# and save simple plots (one chart per figure, matplotlib defaults).

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)     # nodal_features.csv
    ap.add_argument("--clusters", required=True)     # modes_clusters.csv
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    dfF = pd.read_csv(args.features)
    dfC = pd.read_csv(args.clusters)
    # Expect 'id' column in both
    if "id" not in dfF.columns or "id" not in dfC.columns:
        raise SystemExit("Both CSVs must contain an 'id' column.")

    df = pd.merge(dfF, dfC[["id","cluster"]], on="id", how="inner")
    merged_csv = outdir / "merged_with_clusters.csv"
    df.to_csv(merged_csv, index=False, encoding="utf-8")

    # Per-cluster stats
    clusters = sorted(df["cluster"].unique())
    rows = []
    for c in clusters:
        d = df[df["cluster"]==c]
        freq_min = d["freq_hz"].min() if "freq_hz" in d else np.nan
        freq_max = d["freq_hz"].max() if "freq_hz" in d else np.nan
        row = {
            "cluster": c,
            "count": len(d),
            "freq_min": freq_min,
            "freq_max": freq_max
        }
        # Add numeric means
        for col in d.select_dtypes(include=[np.number]).columns:
            if col in ("cluster",): continue
            row[f"mean_{col}"] = d[col].mean()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_csv = outdir / "cluster_summary.csv"
    summary.to_csv(summary_csv, index=False, encoding="utf-8")

    # Plot frequency histogram per cluster (one chart per cluster)
    if "freq_hz" in df.columns:
        for c in clusters:
            d = df[df["cluster"]==c]
            plt.figure()
            plt.hist(d["freq_hz"].values, bins=30)
            plt.title(f"Cluster {c} - Frequency histogram")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(outdir / f"cluster_{c}_freq_hist.png")
            plt.close()

        # Scatter: freq vs zero_frac per cluster (one figure per cluster)
        if "zero_frac" in df.columns:
            for c in clusters:
                d = df[df["cluster"]==c]
                plt.figure()
                plt.scatter(d["freq_hz"].values, d["zero_frac"].astype(float).values, s=10)
                plt.title(f"Cluster {c} - freq vs zero_frac")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("zero_frac")
                plt.tight_layout()
                plt.savefig(outdir / f"cluster_{c}_freq_vs_zero_frac.png")
                plt.close()

    print("Wrote:")
    print(" -", merged_csv)
    print(" -", summary_csv)
    print(" - plots: cluster_#_freq_hist.png, cluster_#_freq_vs_zero_frac.png (if available)")

if __name__ == "__main__":
    main()
