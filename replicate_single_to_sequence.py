# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from shutil import copy2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source single PNG, e.g., Averaged/0125Hz_90.0db.png")
    ap.add_argument("--dst-dir", required=True, help="Destination directory for replicated sequence")
    ap.add_argument("--count", type=int, default=21, help="Number of frames to create")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    stem = src.stem  # e.g., 0125Hz_90.0db
    for i in range(args.count):
        out = dst / f"{stem}_{i:02d}.png"
        copy2(src, out)
    print(f"[OK] Replicated {src} -> {args.count} frames in {dst}")


if __name__ == "__main__":
    main()




