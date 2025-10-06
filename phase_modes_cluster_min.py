#!/usr/bin/env python3
# ASCII-only
# Cluster nodal features CSV into modes (unsupervised). Picks best k by silhouette.
# Requires scikit-learn and pandas.

import argparse
from pathlib import Path
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-csv", required=True)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--out-csv", default="modes_clusters.csv")
    args = ap.parse_args()

    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    df = pd.read_csv(args.feat_csv)

    # Select numeric columns
    df_num = df.select_dtypes(include=[np.number]).copy()
    # Drop obvious ID-like scale columns that can dominate (keep sign/shape metrics etc.)
    drop_cols = [c for c in df_num.columns if c.lower() in ("freq_hz","valid_px")]
    X = df_num.drop(columns=drop_cols, errors="ignore").values

    # Remove rows with all zeros (invalid frames)
    good = np.any(X != 0.0, axis=1)
    X = X[good]
    df_good = df.loc[good].reset_index(drop=True)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    best = None
    for k in range(max(2, args.kmin), max(2, args.kmax)+1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        y = km.fit_predict(Xs)
        try:
            score = silhouette_score(Xs, y)
        except Exception:
            score = -1.0
        if (best is None) or (score > best[0]):
            best = (score, k, y, km)

    if best is None:
        print("[ERR] clustering failed"); return

    score, k, y, km = best
    print(f"[BEST] k={k} silhouette={score:.4f}")

    out = df_good.copy()
    out["cluster"] = y
    out_csv = Path(args.out_csv)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print("Clusters written to:", out_csv)

    # Text summary per cluster
    with open(out_csv.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write(f"BEST k={k} silhouette={score:.4f}\n\n")
        for c in sorted(np.unique(y)):
            f.write(f"[Cluster {c}]\n")
            ids = out["id"][out["cluster"]==c].tolist()
            for s in ids[:80]:
                f.write(f"  {s}\n")
            f.write("\n")

if __name__ == "__main__":
    main()
