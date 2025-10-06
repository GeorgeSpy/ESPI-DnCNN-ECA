import csv, argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def load_feats(csv_path):
    names=[]; X=[]
    with open(csv_path, newline='', encoding='utf-8') as f:
        rd=csv.reader(f); header=next(rd)
        for r in rd:
            names.append(r[0]); X.append([float(x) for x in r[1:]])
    return names, np.array(X, np.float32)

def load_assign(csv_path):
    y={}
    with open(csv_path, newline='', encoding='utf-8') as f:
        rd=csv.reader(f); _=next(rd)
        for name, c in rd: y[name]=int(c)
    return y

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--feats', required=True)
    ap.add_argument('--assign', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--topk', type=int, default=12)
    args=ap.parse_args()

    names, X = load_feats(args.feats)
    y_map     = load_assign(args.assign)

    keep=[i for i,n in enumerate(names) if n in y_map]
    names=[names[i] for i in keep]; X=X[keep]
    import numpy as np
    y=np.array([y_map[n] for n in names], dtype=int)

    scaler=StandardScaler().fit(X)
    Xs=scaler.transform(X)

    K = int(y.max())+1
    counts=[int((y==k).sum()) for k in range(K)]

    centers=[]
    for k in range(K):
        centers.append(Xs[y==k].mean(axis=0) if counts[k]>0 else np.zeros(Xs.shape[1], np.float32))
    centers=np.vstack(centers)

    dist=np.linalg.norm(Xs[:,None,:]-centers[None,:,:], axis=2)

    top_lists=[]
    for k in range(K):
        idx=np.where(y==k)[0]
        order=idx[np.argsort(dist[idx, k])]
        top=order[:min(args.topk, len(order))]
        top_lists.append([names[i] for i in top])

    outp=Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, 'w', encoding='utf-8') as f:
        f.write(f'Counts per cluster: {counts}\n')
        for k, lst in enumerate(top_lists):
            f.write(f'\nCluster {k}  top{len(lst)} representatives:\n')
            for nm in lst: f.write(f'  {nm}\n')
    print(f'[OK] Wrote report to {outp}  (counts={counts})')

if __name__=='__main__':
    main()
