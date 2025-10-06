# -*- coding: utf-8 -*-
import argparse, sys
from pathlib import Path
import numpy as np
from PIL import Image


def load_image_u16_aware(path: Path) -> np.ndarray:
    im = Image.open(path)
    arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    # Work in native integer range to preserve scale
    if arr.dtype == np.uint16:
        return arr.astype(np.float32)
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) * (65535.0 / 255.0))
    return arr.astype(np.float32)


def save_u16(path: Path, arr: np.ndarray) -> None:
    arr16 = np.clip(np.rint(arr), 0, 65535).astype(np.uint16)
    Image.fromarray(arr16, mode="I;16").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--glob", required=True, help="Pattern like '0125Hz_90.0db_*.png'")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_path = Path(args.output)
    files = sorted(in_dir.glob(args.glob))
    if not files:
        print(f"[ERR] No files matched: {in_dir} / {args.glob}")
        sys.exit(1)

    acc = None
    for p in files:
        x = load_image_u16_aware(p)
        if acc is None:
            acc = np.zeros_like(x, dtype=np.float32)
        acc += x

    avg = acc / float(len(files))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_u16(out_path, avg)
    print(f"[OK] Averaged {len(files)} frames -> {out_path}")


if __name__ == "__main__":
    main()




