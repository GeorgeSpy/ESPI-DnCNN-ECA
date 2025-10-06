#!/usr/bin/env python3
# modes_gallery_export_min.py (ASCII-only, Windows-safe)
# Group frames by predicted label and create HTML galleries with thumbnails.
# - Inputs: predicted_modes.csv (id,pred_label), phase-root (folder with phase_*_png)
# - kind: which image family to use: unwrapped_png (default) | wrapped_png | quality_png
# - Creates out-root/<label>/*.png (thumbnails) and out-root/index.html
#
# Usage example:
#   python modes_gallery_export_min.py ^
#     --phase-root "C:\...\W01_PhaseOut_STRICT\band18_ff100_cs16" ^
#     --pred-csv   "C:\...\W01_modes\predicted_modes.csv" ^
#     --out-root   "C:\...\W01_modes\gallery" ^
#     --kind unwrapped_png --thumb 384 --max-per-label 80
#
# Requires: pandas, pillow

import argparse, re
from pathlib import Path
import pandas as pd
from PIL import Image

def slugify(s: str) -> str:
    s = re.sub(r'[^\w\-]+', '_', s.strip())
    s = re.sub(r'_+', '_', s)
    return s.strip('_') or "unknown"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def thumb_copy(src: Path, dst: Path, max_side: int):
    try:
        im = Image.open(src)
        im = im.convert("L") if im.mode != "L" else im
        w, h = im.size
        if max(w, h) > max_side:
            if w >= h:
                new_w = max_side
                new_h = int(round(h * (max_side / float(w))))
            else:
                new_h = max_side
                new_w = int(round(w * (max_side / float(h))))
            im = im.resize((new_w, new_h), Image.BILINEAR)
        ensure_dir(dst.parent)
        im.save(dst)
        return True
    except Exception as e:
        print(f"[WARN] failed to create thumb {dst.name}: {e}")
        return False

def build_label_page(label_dir: Path, label: str, rel_prefix: str = "."):
    imgs = sorted([p for p in label_dir.glob("*.png")])
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append(f"<title>Mode {label}</title>")
    html.append("<style>body{font-family:Arial, sans-serif} .grid{display:flex;flex-wrap:wrap;gap:8px} .grid img{border:1px solid #ccc;padding:2px}</style>")
    html.append("</head><body>")
    html.append(f"<h2>Mode: {label} ({len(imgs)} frame(s))</h2>")
    html.append("<div class='grid'>")
    for p in imgs:
        html.append(f"<a href='{p.name}' target='_blank'><img src='{p.name}' alt='{p.name}'></a>")
    html.append("</div>")
    html.append(f"<p><a href='{rel_prefix}/index.html'>&larr; Back to index</a></p>")
    html.append("</body></html>")
    (label_dir / "index.html").write_text("\n".join(html), encoding="utf-8")

def build_index(out_root: Path, label_counts):
    html = []
    html.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    html.append("<title>ESPI Modes Gallery</title>")
    html.append("<style>body{font-family:Arial, sans-serif} ul{line-height:1.8}</style>")
    html.append("</head><body>")
    html.append("<h1>ESPI Modes Gallery</h1>")
    html.append("<ul>")
    for label, count in label_counts:
        lbl = slugify(label)
        html.append(f"<li><a href='./{lbl}/index.html'>{label}</a> &mdash; {count} frame(s)</li>")
    html.append("</ul>")
    html.append("</body></html>")
    (out_root / "index.html").write_text("\n".join(html), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-root", required=True, help="Folder with phase_*_png subfolders")
    ap.add_argument("--pred-csv",   required=True, help="predicted_modes.csv (id,pred_label)")
    ap.add_argument("--out-root",   required=True, help="Output gallery root")
    ap.add_argument("--kind", default="unwrapped_png", choices=["unwrapped_png","wrapped_png","quality_png"])
    ap.add_argument("--thumb", type=int, default=384, help="max side for thumbnails")
    ap.add_argument("--max-per-label", type=int, default=80, help="limit per label (0 = no limit)")
    args = ap.parse_args()

    phase_root = Path(args.phase_root)
    src_dir = phase_root / f"phase_{args.kind}"
    if not src_dir.exists():
        raise SystemExit(f"[ERR] not found: {src_dir}")

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    df = pd.read_csv(args.pred_csv)
    if "id" not in df.columns or "pred_label" not in df.columns:
        raise SystemExit("[ERR] predicted_modes.csv must have columns: id, pred_label")

    label_counts = []
    for label, d in df.groupby("pred_label"):
        lbl = str(label)
        lbl_slug = slugify(lbl)
        out_dir = out_root / lbl_slug
        ensure_dir(out_dir)

        taken = 0
        max_take = int(args.max_per_label)
        for _, row in d.iterrows():
            stem = str(row["id"])
            src = src_dir / f"{stem}.png"
            if not src.exists():
                src2 = src_dir / f"{stem}.PNG"
                if src2.exists():
                    src = src2
                else:
                    print(f"[MISS] {src}")
                    continue
            dst = out_dir / f"{stem}.png"
            if thumb_copy(src, dst, max_side=args.thumb):
                taken += 1
            if max_take > 0 and taken >= max_take:
                break

        label_counts.append((lbl, taken))
        build_label_page(out_dir, lbl, rel_prefix="..")

    label_counts_sorted = sorted(label_counts, key=lambda x: x[0])
    build_index(out_root, label_counts_sorted)

    # counts CSV
    import csv
    with open(out_root / "label_counts.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["label","count"])
        for lbl, cnt in label_counts_sorted:
            w.writerow([lbl, cnt])

    print("[DONE] Gallery at:", out_root)

if __name__ == "__main__":
    main()
