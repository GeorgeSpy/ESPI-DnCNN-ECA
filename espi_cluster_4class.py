import argparse, csv
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load_csv(path):
    names = []
    X = []
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            names.append(row[0])
            feats = [float(x) for x in row[1:]]  # όλα εκτός name
            X.append(feats)
    return header, names, np.array(X, np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv", required=True)
    ap.add_argument("--out-assign", required=True)
    ap.add_argument("--out-pca2", required=False, default="")
    args = ap.parse_args()

    header, names, X = load_csv(args.in_csv)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA για οπτικοποίηση (προαιρετικό)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    km = KMeans(n_clusters=4, n_init="auto", random_state=42)
    y = km.fit_predict(Xs)

    sil = silhouette_score(Xs, y)
    print(f"[INFO] silhouette={sil:.3f}")

    # γράψε αναθέσεις
    outp = Path(args.out_assign)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["name","cluster"])
        for n, c in zip(names, y):
            wr.writerow([n, int(c)])

    if args.out_pca2:
        with open(args.out_pca2, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f); wr.writerow(["name","pc1","pc2","cluster"])
            for n, (pc1,pc2), c in zip(names, Z, y):
                wr.writerow([n, float(pc1), float(pc2), int(c)])

    print(f"[OK] Assignments: {args.out_assign}")
    if args.out_pca2:
        print(f"[OK] PCA2: {args.out_pca2}")

if __name__ == "__main__":
    main()
