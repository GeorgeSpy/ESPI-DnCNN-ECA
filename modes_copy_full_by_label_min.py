#!/usr/bin/env python3
# modes_copy_full_by_label_min.py (ASCII-only, Windows-safe)
# Copy FULL PNGs (not thumbnails) into per-label folders using predicted_modes.csv.
#
# Usage:
#   python modes_copy_full_by_label_min.py ^
#     --phase-root "C:\...\W01_PhaseOut_STRICT\band18_ff100_cs16" ^
#     --pred-csv   "C:\...\W01_modes\predicted_modes.csv" ^
#     --out-root   "C:\...\W01_modes\by_label_fullpng" ^
#     --kind unwrapped_png --max-per-label 0
#
# Requires: pandas

import argparse, re, shutil
from pathlib import Path
import pandas as pd

def slugify(s: str) -> str:
    s = re.sub(r'[^\w\-]+', '_', s.strip())
    s = re.sub(r'_+', '_', s)
    return s.strip('_') or "unknown"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-root", required=True, help="Folder with phase_*_png subfolders")
    ap.add_argument("--pred-csv",   required=True, help="predicted_modes.csv (id,pred_label)")
    ap.add_argument("--out-root",   required=True, help="Output root where label folders will be created")
    ap.add_argument("--kind", default="unwrapped_png", choices=["unwrapped_png","wrapped_png","quality_png"])
    ap.add_argument("--max-per-label", type=int, default=0, help="0=no limit; otherwise cap per label")
    ap.add_argument("--missing-log", default="", help="Optional path to write list of missing PNGs")
    args = ap.parse_args()

    phase_root = Path(args.phase_root)
    src_dir = phase_root / f"phase_{args.kind}"
    if not src_dir.exists():
        raise SystemExit(f"[ERR] not found: {src_dir}")

    out_root = Path(args.out_root); ensure_dir(out_root)

    df = pd.read_csv(args.pred_csv)
    if "id" not in df.columns or "pred_label" not in df.columns:
        raise SystemExit("[ERR] predicted_modes.csv must have columns: id, pred_label")

    missing = []
    total_copied = 0
    label_counts = []

    for label, d in df.groupby("pred_label"):
        lbl = str(label); lbl_slug = slugify(lbl)
        out_dir = out_root / lbl_slug
        ensure_dir(out_dir)

        taken = 0
        cap = int(args.max_per_label)
        for _, row in d.iterrows():
            stem = str(row["id"])
            src = src_dir / f"{stem}.png"
            if not src.exists():
                src2 = src_dir / f"{stem}.PNG"
                if src2.exists():
                    src = src2
                else:
                    missing.append(str(src))
                    continue
            dst = out_dir / f"{stem}.png"
            try:
                shutil.copy2(src, dst)
                taken += 1; total_copied += 1
            except Exception as e:
                print(f"[WARN] failed to copy {src} -> {dst}: {e}")
            if cap > 0 and taken >= cap:
                break

        label_counts.append((lbl, taken))
        print(f"[LABEL] {lbl}: copied {taken} file(s)")

    # write counts
    with open(out_root / "copy_counts.txt", "w", encoding="utf-8") as f:
        for lbl, cnt in sorted(label_counts, key=lambda x: x[0]):
            f.write(f"{lbl}\t{cnt}\n")
        f.write(f"\nTOTAL\t{total_copied}\n")

    if args.missing_log:
        Path(args.missing_log).parent.mkdir(parents=True, exist_ok=True)
        with open(args.missing_log, "w", encoding="utf-8") as f:
            for m in missing:
                f.write(m + "\n")
        print(f"[MISS] wrote list to: {args.missing_log}")
    else:
        if missing:
            print(f"[MISS] {len(missing)} file(s) missing (pass --missing-log path to save list).")

    print("[DONE] Copied:", total_copied, "file(s)")

if __name__ == "__main__":
    main()
