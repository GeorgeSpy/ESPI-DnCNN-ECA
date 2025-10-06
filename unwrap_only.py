#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
unwrap_only.py
--------------
Run phase unwrapping on an existing PhaseOut folder that already contains wrapped phase
(phase_wrapped_npy/ or phase_wrapped_png/). Saves to phase_unwrapped_npy/png.

Requires: scikit-image (skimage)
Install:  pip install --upgrade pip wheel && pip install scikit-image

Usage:
python unwrap_only.py --out-root "C:\...\W01_PhaseOut_masked"
"""
import argparse, numpy as np
from pathlib import Path
from PIL import Image

def load_wrapped(dir_root: Path):
    w_npy = dir_root / "phase_wrapped_npy"
    w_png = dir_root / "phase_wrapped_png"
    names = []
    if w_npy.exists():
        names = [p.stem for p in sorted(w_npy.glob("*.npy"))]
        mode = "npy"
    elif w_png.exists():
        names = [p.stem for p in sorted(w_png.glob("*.png"))]
        mode = "png"
    else:
        raise FileNotFoundError("No wrapped phase found.")
    return mode, names, w_npy, w_png

def read_phase(path_npy: Path, path_png: Path, is_phase: bool) -> np.ndarray:
    if path_npy.exists():
        return np.load(path_npy).astype(np.float32)
    if path_png.exists():
        arr01 = np.array(Image.open(path_png).convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
        if is_phase:
            return arr01 * 2*np.pi - np.pi
        return arr01
    raise FileNotFoundError

def write_phase(arr: np.ndarray, dst_npy: Path, dst_png: Path):
    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst_npy, arr.astype(np.float32))
    arr01 = (np.clip((arr + np.pi)/(2*np.pi), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr01, mode="L").save(dst_png)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()

    try:
        from skimage.restoration import unwrap_phase
    except Exception as e:
        print("[ERROR] scikit-image is required. Install it with:")
        print("  pip install --upgrade pip wheel")
        print("  pip install scikit-image")
        raise SystemExit(1)

    root = Path(args.out_root)
    mode, names, w_npy, w_png = load_wrapped(root)
    u_npy = root / "phase_unwrapped_npy"
    u_png = root / "phase_unwrapped_png"
    print(f"[INFO] Found {len(names)} wrapped frames ({mode}). Unwrapping...")

    for name in names:
        phi_w = read_phase((w_npy / name).with_suffix(".npy"),
                           (w_png / name).with_suffix(".png"),
                           is_phase=True)
        phi_u = unwrap_phase(phi_w)
        write_phase(phi_u, (u_npy / name).with_suffix(".npy"),
                          (u_png / name).with_suffix(".png"))
    print("[DONE] Unwrap complete.")

if __name__ == "__main__":
    main()
