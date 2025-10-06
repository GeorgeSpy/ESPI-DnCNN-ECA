import argparse, csv, numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import joblib

def load_feats(csv_path):
    names=[]; X=[]
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd=csv.reader(f); header=next(rd)
        for r in rd:
            names.append(r[0]); X.append([float(x) for x in r[1:]])
    return names, np.array(X, np.float32)

def load_assign(assign_path, names):
    y_map={}
    with open(assign_path, newline="", encoding="utf-8") as f:
        rd=csv.reader(f); _=next(rd)
        for name, c in rd: y_map[name]=int(c)
    y=np.array([y_map[n] for n in names], int)
    return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--assign", required=True)
    ap.add_argument("--out-model", required=True)
    args=ap.parse_args()

    names, X = load_feats(args.feats)
    y        = load_assign(args.assign, names)

    counts = Counter(y)
    min_class_count = min(counts.values())
    n_splits = max(2, min(5, min_class_count))  # προσαρμοστικό CV
    print(f"[INFO] class counts: {dict(counts)}  -> n_splits={n_splits}")

    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, class_weight="balanced", multi_class="auto")
    )

    # Stratified CV
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s=[]
    for fold,(tr,va) in enumerate(skf.split(X,y),1):
        pipe.fit(X[tr], y[tr])
        yp = pipe.predict(X[va])
        f1 = f1_score(y[va], yp, average="macro")
        f1s.append(f1)
        print(f"\n[F{fold}] macro-F1={f1:.3f}")
        print(classification_report(y[va], yp, digits=3))
        print("Confusion:\n", confusion_matrix(y[va], yp))

    print(f"\n[CV] macro-F1 mean={np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # train on ALL & save
    pipe.fit(X, y)
    Path(args.out_model).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model":pipe, "names":names}, args.out_model)
    print(f"[OK] Saved model to {args.out_model}")

if __name__=="__main__":
    main()
