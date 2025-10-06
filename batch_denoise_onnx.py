#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch denoise PNGs with ONNX Runtime (prefer DirectML on AMD APUs/GPUs).
Usage:
pip install onnxruntime-directml  (Windows/AMD)
# or: pip install onnxruntime (CPU fallback)

python batch_denoise_onnx.py ^
  --onnx "C:\...\dncnn_lite_eca.onnx" ^
  --input "C:\...\W01_ESPI_90db-Noisy" ^
  --output "C:\...\W01_ESPI_90db-Denoised" ^
  --tile 256 --overlap 32
"""
import argparse, numpy as np, onnxruntime as ort
from pathlib import Path
from PIL import Image

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def providers():
    prov = []
    # If DirectML package is installed, ORT will allow this provider name; otherwise CPU fallback
    prov.append("DmlExecutionProvider")
    prov.append("CPUExecutionProvider")
    return prov

def imread_uint01(path: Path) -> np.ndarray:
    im = Image.open(path)
    if im.mode in ("I;16","I;16B","I"):
        arr = np.array(im, dtype=np.uint16).astype(np.float32) / 65535.0
    else:
        arr = np.array(im.convert("L"), dtype=np.uint8).astype(np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)

def imsave_uint01(arr: np.ndarray, path: Path):
    arr = np.clip(arr, 0.0, 1.0)
    ensure_dir(path.parent)
    Image.fromarray((arr*255.0+0.5).astype(np.uint8), mode="L").save(path)

def hann2d(h: int, w: int, eps: float = 1e-6) -> np.ndarray:
    wx = np.hanning(w).reshape(1,w)
    wy = np.hanning(h).reshape(h,1)
    return (wy @ wx) + eps

def _ensure_nchw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        x = x[np.newaxis, np.newaxis, ...]  # [1,1,H,W]
    return x

def denoise_tiled_sess(sess: ort.InferenceSession, image: np.ndarray, tile: int = 256, overlap: int = 32) -> np.ndarray:
    x = _ensure_nchw(image.astype(np.float32))
    _, _, H, W = x.shape
    out = np.zeros_like(x, dtype=np.float32)
    weight = np.zeros_like(x, dtype=np.float32)
    step = tile - overlap
    win = hann2d(tile, tile).astype(np.float32).reshape(1,1,tile,tile)
    feed_name = sess.get_inputs()[0].name
    fetch_name = sess.get_outputs()[0].name
    for top in range(0, H, step):
        for left in range(0, W, step):
            bottom = min(top + tile, H); right = min(left + tile, W)
            pad_h = tile - (bottom - top); pad_w = tile - (right - left)
            patch = np.zeros((1,1,tile,tile), dtype=np.float32)
            patch[:,:,:tile-pad_h,:tile-pad_w] = x[:,:,top:bottom,left:right]
            pred = sess.run([fetch_name], {feed_name: patch})[0] * win
            out[:,:,top:bottom,left:right] += pred[:,:,:tile-pad_h,:tile-pad_w]
            weight[:,:,top:bottom,left:right] += win[:,:,:tile-pad_h,:tile-pad_w]
    return out / (weight + 1e-6)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    args = ap.parse_args()

    sess = ort.InferenceSession(args.onnx, providers=providers())
    in_dir = Path(args.input); out_dir = Path(args.output)
    files = sorted(in_dir.rglob("*.png"))
    if not files:
        print("No PNGs under", in_dir); return
    for f in files:
        arr = imread_uint01(f)
        den = denoise_tiled_sess(sess, arr, tile=args.tile, overlap=args.overlap)
        rel = f.relative_to(in_dir)
        imsave_uint01(den.squeeze(0).squeeze(0), out_dir / rel)
        print("saved", out_dir / rel)

if __name__ == "__main__":
    main()
