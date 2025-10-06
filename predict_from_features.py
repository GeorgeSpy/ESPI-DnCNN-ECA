import argparse, csv
import numpy as np
from pathlib import Path
import joblib

def load_feats(csv_path):
    names=[]; X=[]
    with open(csv_path, newline="", encoding="utf-8") as f:
        rd=csv.reader(f); header=next(rd)
        for r in rd:
            names.append(r[0]); X.append([float(x) for x in r[1:]])
    return names, np.array(X, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True, help="CSV από espi_features_nodal.py (dedup αν υπάρχει).")
    ap.add_argument("--model", required=True, help="joblib από τον trainer.")
    ap.add_argument("--out-csv", required=True, help="πού θα σωθεί το predictions CSV.")
    args = ap.parse_args()

    names, X = load_feats(args.feats)
    bundle = joblib.load(args.model)
    clf = bundle["model"]

    preds = clf.predict(X)
    has_proba = hasattr(clf, "predict_proba")
    if has_proba:
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
    else:
        proba = None
        classes = []

    outp = Path(args.out_csv)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        header = ["name", "pred"]
        if has_proba:
            header += [f"p_{c}" for c in classes]
        wr.writerow(header)
        for i, n in enumerate(names):
            row = [n, int(preds[i])]
            if has_proba:
                row += [float(x) for x in proba[i]]
            wr.writerow(row)

    # μικρή σύνοψη
    _, counts = np.unique(preds, return_counts=True)
    print(f"[OK] Wrote predictions: {outp}  (N={len(names)})")
    print(f"[INFO] class counts: {dict(zip(sorted(set(preds)), counts))}")

if __name__ == "__main__":
    main()
