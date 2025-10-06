#!/usr/bin/env python3
# modes_predict_min.py
# Apply trained pipeline to a new nodal_features.csv and output predicted labels.

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="modes_rf_pipeline.joblib from training")
    ap.add_argument("--meta",  required=True, help="meta.json from training (feature order)")
    ap.add_argument("--features", required=True, help="nodal_features.csv for new set")
    ap.add_argument("--outcsv", required=True)
    args = ap.parse_args()

    pipe = joblib.load(args.model)
    with open(args.meta, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feat_names = meta["feature_names"]

    df = pd.read_csv(args.features)
    Xdf = df.select_dtypes(include=[np.number]).copy()
    for col in ("freq_hz","valid_px","cluster"):
        if col in Xdf.columns: Xdf.drop(columns=[col], inplace=True)
    # Ensure same feature order
    Xdf = Xdf.reindex(columns=feat_names, fill_value=0.0)

    y_pred = pipe.predict(Xdf.values)
    out = pd.DataFrame({"id": df.get("id", pd.Series(range(len(y_pred)))), "pred_label": y_pred})
    out.to_csv(args.outcsv, index=False, encoding="utf-8")
    print("Wrote:", args.outcsv)

if __name__ == "__main__":
    main()
