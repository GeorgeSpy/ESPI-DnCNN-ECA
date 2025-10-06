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

# Εξάγουμε "key" π.χ. 0050Hz_90.0db από οποιοδήποτε filename
KEY_RE = re.compile(r"(\d{4}Hz_\d+\.\d+db)", re.IGNORECASE)
STRIP_TAIL_RE = re.compile(r"(_?(phase(out)?|unwrap(ped)?|masked|grid|den|clean|v\d+))+$", re.IGNORECASE)

def norm_key(stem: str) -> str:
    m = KEY_RE.search(stem)
    if m:
        return m.group(1)
    # Αλλιώς, βγάλε κοινές ουρές
    s = STRIP_TAIL_RE.sub("", stem)
    return s

def collect_files(root: Path):
    items = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".png",".npy")):
                p = Path(dirpath)/fn
                items.append(p)
    return items

def build_map(root: Path):
    files = collect_files(root)
    mp = {}
    for p in files:
        k = norm_key(p.stem)
        # Προτίμησε .npy αν υπάρχουν διπλά
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--qmin", type=float, default=0.20)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ref_root = Path(args.ref_root)
    mask = load_mask(Path(args.roi_mask))

    out_map = build_map(out_root)
    ref_map = build_map(ref_root)

    out_keys = set(out_map.keys())
    ref_keys = set(ref_map.keys())
    common = sorted(out_keys & ref_keys)

    print(f"[INFO] out files: {len(out_map)}  ref files: {len(ref_map)}  common keys: {len(common)}")
    if len(common) == 0:
        # δείξε ενδεικτικά κλειδιά για διάγνωση
        print("[HINT] First 10 OUT keys:", sorted(list(out_keys))[:10])
        print("[HINT] First 10 REF keys:", sorted(list(ref_keys))[:10])
        return

    total = 0; passed = 0
    print("name,q_score,sign")
    for k in common:
        a = load_any(out_map[k]); b = load_any(ref_map[k])
        H = min(a.shape[0], b.shape[0], mask.shape[0])
        W = min(a.shape[1], b.shape[1], mask.shape[1])
        a = a[:H,:W]; b = b[:H,:W]; m = mask[:H,:W]
        az = zscore(a, m); bz = zscore(b, m)
        c1 = cosine(az, bz, m); c2 = cosine(az, -bz, m)
        q = max(c1, c2); sign = "+" if c1 >= c2 else "-"
        print(f"{k},{q:.3f},{sign}")
        total += 1; passed += (q >= args.qmin)
    print(f"SUMMARY: {passed}/{total} passed (qmin={args.qmin})")

if __name__ == "__main__":
    main()
