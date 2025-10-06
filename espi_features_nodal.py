import argparse, os, re, math
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, binary_opening, disk
from skimage.filters import threshold_otsu
from scipy.ndimage import convolve, label

# --------- I/O helpers ----------
def load_any(path: Path) -> np.ndarray:
    p = str(path).lower()
    if p.endswith(".npy"):
        arr = np.load(path).astype(np.float32)
        return arr
    im = Image.open(path)
    a = np.array(im)
    if a.dtype == np.uint16:
        a = a.astype(np.float32) / 65535.0
    else:
        a = a.astype(np.float32)
        if a.max() > 0: a /= a.max()
    return a

def load_mask(mask_path: Path) -> np.ndarray:
    m = np.array(Image.open(mask_path).convert("L"))
    m = (m > 0).astype(np.float32)
    return m

def collect_files(root: Path):
    files = []
    for dp,_,fnames in os.walk(root):
        for fn in fnames:
            if fn.lower().endswith((".png",".npy")):
                files.append(Path(dp)/fn)
    return sorted(files)

# --------- image ops ----------
Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
Ky = Kx.T

def sobel_mag_dir(x: np.ndarray):
    xpad = np.pad(x, ((1,1),(1,1)), mode="edge")
    gx = (
        Kx[0,0]*xpad[:-2,:-2] + Kx[0,1]*xpad[:-2,1:-1] + Kx[0,2]*xpad[:-2,2:] +
        Kx[1,0]*xpad[1:-1,:-2] + Kx[1,1]*xpad[1:-1,1:-1] + Kx[1,2]*xpad[1:-1,2:] +
        Kx[2,0]*xpad[2:,:-2] + Kx[2,1]*xpad[2:,1:-1] + Kx[2,2]*xpad[2:,2:]
    )
    gy = (
        Ky[0,0]*xpad[:-2,:-2] + Ky[0,1]*xpad[:-2,1:-1] + Ky[0,2]*xpad[:-2,2:] +
        Ky[1,0]*xpad[1:-1,:-2] + Ky[1,1]*xpad[1:-1,1:-1] + Ky[1,2]*xpad[1:-1,2:] +
        Ky[2,0]*xpad[2:,:-2] + Ky[2,1]*xpad[2:,1:-1] + Ky[2,2]*xpad[2:,2:]
    )
    mag = np.sqrt(gx*gx + gy*gy)
    ang = np.arctan2(gy, gx)  # (-π, π]
    return mag, ang

def zscore_in_mask(x: np.ndarray, m: np.ndarray):
    v = x[m>0]
    if v.size < 10:
        return x*0.0
    mu = float(v.mean())
    sd = float(v.std()) or 1.0
    return (x - mu) / sd

def skeleton_and_stats(edge_bin: np.ndarray):
    if edge_bin.sum() == 0:
        return edge_bin, 0, 0, 0
    sk = skeletonize(edge_bin)
    # components & junctions
    struct = np.array([[1,1,1],[1,1,1],[1,1,1]], bool)
    lbl, ncomp = label(sk, structure=struct)
    # degree counts
    from scipy.signal import convolve2d
    deg = convolve2d(sk.astype(np.uint8), np.ones((3,3),np.uint8), mode="same", boundary="symm")
    # pixel itself counts, so endpoints ~2, junctions >=4
    endpoints = int(((deg==2) & sk).sum())
    junctions = int(((deg>=4) & sk).sum())
    return sk, int(ncomp), endpoints, junctions

def radial_hist(skel: np.ndarray, m: np.ndarray, bins=8):
    ys, xs = np.nonzero(m>0)
    if ys.size == 0: return np.zeros(bins, np.float32)
    cy = ys.mean(); cx = xs.mean()
    sy, sx = np.nonzero(skel)
    if sy.size == 0: return np.zeros(bins, np.float32)
    r = np.sqrt((sy-cy)**2 + (sx-cx)**2)
    if r.size < 10: return np.zeros(bins, np.float32)
    h, _ = np.histogram(r, bins=bins, range=(0, r.max()))
    h = h.astype(np.float32); h /= max(h.sum(), 1)
    return h

def orient_hist(ang: np.ndarray, edge_mask: np.ndarray, bins=8):
    a = ang[edge_mask>0]
    if a.size < 10: return np.zeros(bins, np.float32)
    # map to [0, π) because θ και θ+π είναι ίδια κατεύθυνση
    a = (a % np.pi + np.pi) % np.pi
    h, _ = np.histogram(a, bins=bins, range=(0, math.pi))
    h = h.astype(np.float32); h /= max(h.sum(), 1)
    return h

# --------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-root", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--edge-quantile", type=float, default=0.85, help="quantile over Sobel mag inside ROI")
    ap.add_argument("--min-comp-size", type=int, default=32, help="filter very tiny edge blobs (pixels)")
    args = ap.parse_args()

    root = Path(args.phase_root)
    mask = load_mask(Path(args.roi_mask))
    files = collect_files(root)
    os.makedirs(Path(args.out_csv).parent, exist_ok=True)

    import csv
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        # header
        cols = ["name","pix_area","skel_len","n_comp","n_end","n_junc",
                "skel_len_norm","comp_per_area","junc_per_len"]
        cols += [f"rad_{i}" for i in range(8)]
        cols += [f"ori_{i}" for i in range(8)]
        wr.writerow(cols)

        for p in files:
            a = load_any(p)
            H = min(a.shape[0], mask.shape[0]); W = min(a.shape[1], mask.shape[1])
            a = a[:H,:W]; m = mask[:H,:W]

            z = zscore_in_mask(a, m)
            mag, ang = sobel_mag_dir(z)
            # threshold within ROI
            vm = mag[m>0]
            if vm.size < 10:
                continue
            thr = np.quantile(vm, args.edge_quantile)
            edge = (mag >= thr).astype(np.uint8) * m.astype(np.uint8)

            # remove tiny specks
            from skimage.morphology import remove_small_objects
            edge_bool = edge.astype(bool)
            edge_bool = remove_small_objects(edge_bool, min_size=args.min_comp_size)
            sk, ncomp, nend, njunc = skeleton_and_stats(edge_bool)

            pix_area = int(m.sum())
            sk_len = int(sk.sum())
            sk_len_norm = sk_len / max(pix_area, 1)
            comp_per_area = ncomp / max(pix_area, 1)
            junc_per_len = njunc / max(sk_len, 1)

            rh = radial_hist(sk, m, bins=8)
            oh = orient_hist(ang, edge_bool, bins=8)

            row = [p.stem, pix_area, sk_len, ncomp, nend, njunc,
                   sk_len_norm, comp_per_area, junc_per_len] + list(rh) + list(oh)
            wr.writerow(row)

    print(f"[OK] Features written to {args.out_csv} (rows={len(files)})")

if __name__ == "__main__":
    main()
