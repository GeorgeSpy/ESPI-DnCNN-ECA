import argparse, os
import numpy as np
from pathlib import Path
from PIL import Image

def load_any(p):
    p = Path(p)
    if p.suffix.lower()==".npy":
        a = np.load(p).astype(np.float32)
        return a
    im = Image.open(p)
    a = np.array(im)
    if a.dtype==np.uint16:
        a = a.astype(np.float32)/65535.0
    else:
        a = a.astype(np.float32)
        if a.max()>0: a/=a.max()
    return a

def load_mask(p):
    m = np.array(Image.open(p).convert("L"))
    return (m>0).astype(np.float32)

def zscore(x,m):
    v = x[m>0]
    if v.size<10: return x*0.0
    mu = float(v.mean())
    sd = float(v.std()) or 1.0
    return (x-mu)/sd

def cosine(a,b,m):
    av = a[m>0].ravel(); bv = b[m>0].ravel()
    na = float(np.linalg.norm(av)) or 1.0
    nb = float(np.linalg.norm(bv)) or 1.0
    return float(np.dot(av,bv)/(na*nb))

def pick(root,stem):
    root = Path(root)
    for ext in (".npy",".png"):
        p = root/f"{stem}{ext}"
        if p.exists(): return str(p)
    for p in root.glob(stem+".*"):
        if p.suffix.lower() in (".npy",".png"): return str(p)
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--qmin", type=float, default=0.20)
    args = ap.parse_args()

    out_files = [p for p in os.listdir(args.out_root) if p.lower().endswith((".png",".npy"))]
    ref_files = [p for p in os.listdir(args.ref_root) if p.lower().endswith((".png",".npy"))]
    out_stems = {Path(p).stem for p in out_files}
    ref_stems = {Path(p).stem for p in ref_files}
    stems = sorted(out_stems.intersection(ref_stems))

    m = load_mask(args.roi_mask)
    total = 0; passed = 0
    print("name,q_score,sign")
    for s in stems:
        pf = pick(args.out_root, s)
        rf = pick(args.ref_root, s)
        if not pf or not rf:
            continue
        a = load_any(pf); b = load_any(rf)
        H = min(a.shape[0], b.shape[0], m.shape[0])
        W = min(a.shape[1], b.shape[1], m.shape[1])
        a = a[:H,:W]; b = b[:H,:W]; mm = m[:H,:W]
        az = zscore(a, mm); bz = zscore(b, mm)
        c1 = cosine(az, bz, mm); c2 = cosine(az, -bz, mm)
        q = max(c1,c2); sign = "+" if c1>=c2 else "-"
        print(f"{s},{q:.3f},{sign}")
        total += 1; passed += (q>=args.qmin)
    print(f"SUMMARY: {passed}/{total} passed (qmin={args.qmin})")

if __name__ == "__main__":
    main()
