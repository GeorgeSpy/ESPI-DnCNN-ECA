# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image


def roi_mask(png_path: Path) -> np.ndarray:
    m = np.array(Image.open(png_path))
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0)


def unwrap2d(a: np.ndarray) -> np.ndarray:
    a1 = np.unwrap(a, axis=1)
    a2 = np.unwrap(a1, axis=0)
    return a2.astype(np.float32)


def fit_plane(res: np.ndarray, roi: np.ndarray) -> np.ndarray:
    H, W = res.shape
    yy, xx = np.mgrid[0:H, 0:W]
    X = np.stack([xx[roi], yy[roi], np.ones(np.count_nonzero(roi), dtype=np.float32)], axis=1)
    y = res[roi][:, None]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, c = coef.ravel().astype(np.float32)
    plane = (a * xx + b * yy + c).astype(np.float32)
    return plane


def load_npy(root: Path, sub: str, name: str) -> np.ndarray:
    return np.load(root / sub / f"{name}.npy").astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--roi-mask", required=True)
    ap.add_argument("--name", required=True, help="basename without extension, e.g., 0125Hz_90.0db")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test_root = Path(args.test_root)
    ref_root = Path(args.ref_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    roi = roi_mask(Path(args.roi_mask))
    pi = np.pi

    w_test = load_npy(test_root, "phase_wrapped_npy", args.name)
    w_ref = load_npy(ref_root, "phase_wrapped_npy", args.name)

    d_wrapped = np.angle(np.exp(1j * (w_test - w_ref))).astype(np.float32)
    d_unw = unwrap2d(d_wrapped)
    plane = fit_plane(d_unw, roi)
    d_unw2 = d_unw - plane

    dd = d_unw2[roi]
    rmse = float(np.sqrt(np.mean(dd * dd)))
    pct2 = float(100.0 * np.mean(np.abs(dd) > (pi / 2)))
    pct4 = float(100.0 * np.mean(np.abs(dd) > (pi / 4)))

    np.save(out_dir / f"{args.name}_diff.npy", d_unw2)
    np.save(out_dir / f"{args.name}_plane.npy", plane)

    with open(out_dir / f"{args.name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"rmse": rmse, "pct_gt_pi2": pct2, "pct_gt_pi4": pct4}, f, indent=2)

    print(f"[OK] {args.name}  rmse={rmse:.4f}  %>|pi/2|={pct2:.2f}  %>|pi/4|={pct4:.2f}")


if __name__ == "__main__":
    main()




