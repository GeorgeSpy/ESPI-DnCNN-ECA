import argparse, os, re
import numpy as np
from pathlib import Path
from PIL import Image

def load_any(p: Path) -> np.ndarray:
    p = Path(p)
    if p.suffix.lower() == ".npy":
        a = np.load(p).astype(np.float32)
        return a
    im = Image.open(p)
    a = np.array(im)
    if a.dtype == np.uint16:
        a = a.astype(np.float32)/65535.0
    else:
        a = a.astype(np.float32)
        if a.max() > 0: a /= a.max()
    return a

def load_mask(p: Path) -> np.ndarray:
    m = np.array(Image.open(p).convert("L"))
    return (m > 0).astype(np.float32)

KEY_RE = re.compile(r"(\d{4}Hz_\d+\.\d+db)", re.IGNORECASE)
STRIP_TAIL_RE = re.compile(r"(_?(phase(out)?|unwrap(ped)?|masked|grid|den|clean|v\d+))+$", re.IGNORECASE)

def norm_key(stem: str) -> str:
    m = KEY_RE.search(stem)
    if m: return m.group(1)
    return STRIP_TAIL_RE.sub("", stem)

def collect_files(root: Path):
    items = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".png",".npy")):
                items.append(Path(dirpath)/fn)
    return items

def build_map(root: Path):
    mp = {}
    for p in collect_files(root):
        k = norm_key(p.stem)
        if k in mp:
            if p.suffix.lower() == ".npy":
                mp[k] = p
        else:
            mp[k] = p
    return mp

def zscore(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = x[m > 0]
    if v.size < 10:
        return x * 0.0
    mu = float(v.mean())
    sd = float(v.std()) or 1.0
    return (x - mu) / sd

def cosine(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
    av = a[m > 0].ravel()
    bv = b[m > 0].ravel()
    na = float(np.linalg.norm(av)) or 1.0
    nb = float(np.linalg.norm(bv)) or 1.0
    return float(np.dot(av, bv) / (na * nb))

def best_q_with_shift(a: np.ndarray, b: np.ndarray, m: np.ndarray, maxshift: int = 8):
    H = min(a.shape[0], b.shape[0], m.shape[0])
    W = min(a.shape[1], b.shape[1], m.shape[1])
    a = a[:H,:W]; b = b[:H,:W]; m = m[:H,:W]
    az = zscore(a, m)
    best = (-1.0, 0, 0, "+")  # (q, dy, dx, sign)
    for dy in range(-maxshift, maxshift+1):
        for dx in range(-maxshift, maxshift+1):
            y0 = max(0, dy); y1 = H + min(0, dy)
            x0 = max(0, dx); x1 = W + min(0, dx)
            ay0 = max(0,-dy); ax0 = max(0,-dx)
            b_crop = b[y0:y1, x0:x1]
            a_crop = az[ay0:ay0+(y1-y0), ax0:ax0+(x1-x0)]
            m_crop = m[ay0:ay0+(y1-y0), ax0:ax0+(x1-x0)]
            if b_crop.size < 1000:  # μικρή επικάλυψη; αγνόησε
                continue
            bz = zscore(b_crop, m_crop)
            c1 = cosine(a_crop, bz, m_crop)
            c2 = cosine(a_crop, -bz, m_crop)
            q = c1; s = "+"
            if c2 > c1:
                q = c2; s = "-"
            if q > best[0]:
                best = (q, dy, dx, s)
    return best  # q, dy, dx, sign

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--qmin", type=float, default=0.20)
    ap.add_argument("--maxshift", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="evaluate first N items only (0 = all)")
    args = ap.parse_args()

    out_map = build_map(Path(args.out_root))
    ref_map = build_map(Path(args.ref_root))
    mask = load_mask(Path(args.roi_mask))

    keys = sorted(set(out_map.keys()) & set(ref_map.keys()))
    if args.limit > 0:
        keys = keys[:args.limit]

    print(f"[INFO] items: {len(keys)}   (out={len(out_map)}, ref={len(ref_map)})")
    print("name,q_score,sign,dy,dx")
    total = 0; passed = 0
    for k in keys:
        a = load_any(out_map[k]); b = load_any(ref_map[k])
        q, dy, dx, s = best_q_with_shift(a, b, mask, maxshift=args.maxshift)
        print(f"{k},{q:.3f},{s},{dy},{dx}")
        total += 1; passed += (q >= args.qmin)
    print(f"SUMMARY: {passed}/{total} passed (qmin={args.qmin}, maxshift={args.maxshift})")

if __name__ == "__main__":
    main()
