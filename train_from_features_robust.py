import argparse, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import joblib

def read_auto(path):
    return pd.read_csv(path, sep=None, engine="python")

def pick_name_column(df):
    for cand in ["name","filename","file","id"]:
        if cand in df.columns: return cand
    return df.columns[0]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="features CSV (ideally features_numeric_only.csv)")
    ap.add_argument("--labels", required=True, help="labels CSV (name,label)")
    ap.add_argument("--out-model", required=True)
    ap.add_argument("--out-meta",  default=None)
    args=ap.parse_args()

    F = read_auto(args.feats)
    L = read_auto(args.labels)

    name_f = pick_name_column(F)
    name_l = pick_name_column(L)

    # cast labels: last column assumed label
    labels = dict(zip(L[name_l].astype(str), L.iloc[:, -1].astype(int)))
    # keep only numeric feature cols
    Xnum = F.drop(columns=[name_f], errors="ignore").apply(pd.to_numeric, errors="coerce")
    Xnum = Xnum.dropna(axis=1, how="all")
    if Xnum.shape[1] == 0:
        raise ValueError("No numeric feature columns found after coercion. Check features CSV.")

    names = F[name_f].astype(str).tolist()
    y = []
    keep_idx = []
    for i,n in enumerate(names):
        if n in labels:
            y.append(int(labels[n])); keep_idx.append(i)
    if not keep_idx:
        raise ValueError("No name intersection between features and labels.")

    X = Xnum.iloc[keep_idx].to_numpy(dtype=np.float32)
    y = np.array(y, dtype=int)

    # small CV with RandomForest (better for non-linear features)
    model = RandomForestClassifier(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )
    pipe = make_pipeline(model)  # RF doesn't need scaler
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)  # 4-fold for stability
    f1s=[]
    for k,(tr,va) in enumerate(skf.split(X,y),1):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[va])
        f1 = f1_score(y[va], yp, average="macro")
        f1s.append(f1)
        print(f"[F{k}] macro-F1={f1:.3f}")
    print(f"[CV] macro-F1 mean={np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # final fit + save
    pipe.fit(X, y)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out_model)
    meta = {
        "features_path": args.feats,
        "labels_path": args.labels,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "classes": [int(c) for c in sorted(list(set(y)))],
        "feature_names": [str(c) for c in Xnum.columns]
    }
    meta_path = args.out_meta or str(Path(args.out_model).with_suffix(".meta.json"))
    Path(meta_path).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] Saved model -> {args.out_model}")
    print(f"[OK] Saved meta  -> {meta_path}")

if __name__ == "__main__":
    main()
