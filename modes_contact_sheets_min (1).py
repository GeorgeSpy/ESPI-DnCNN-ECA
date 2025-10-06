#!/usr/bin/env python3
# modes_contact_sheets_min.py (ASCII-only, Windows-safe)
# Build per-label contact sheets (big grids of thumbnails) from phase_*_png and predicted_modes.csv
#
# Usage:
#   python modes_contact_sheets_min.py ^
#     --phase-root "C:\...\W01_PhaseOut_STRICT\band18_ff100_cs16" ^
#     --pred-csv   "C:\...\W01_modes\predicted_modes.csv" ^
#     --out-root   "C:\...\W01_modes\contact_sheets" ^
#     --kind unwrapped_png --thumb 256 --cols 12 --max-per-label 240
#
# Requires: pandas, pillow

import argparse, re
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def slugify(s: str) -> str:
    s = re.sub(r'[^\w\-]+', '_', s.strip())
    s = re.sub(r'_+', '_', s)
    return s.strip('_') or "unknown"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_thumb(src: Path, max_side: int) -> Image.Image:
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
    return im

def draw_header(canvas: Image.Image, text: str, pad: int = 8):
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.rectangle([0,0,canvas.width, 28], fill=240)
    draw.text((pad, 6), text, fill=0, font=font)

def paste_grid(images, cols: int, cell_w: int, cell_h: int, margin: int, head_h: int, title: str):
    rows = (len(images) + cols - 1) // cols
    W = margin + cols*(cell_w+margin)
    H = head_h + margin + rows*(cell_h+margin)
    canvas = Image.new("L", (W, H), color=255)
    draw_header(canvas, title, pad=8)
    y = head_h + margin
    x0 = margin
    i = 0
    for r in range(rows):
        x = x0
        for c in range(cols):
            if i >= len(images): break
            im = images[i]
            # center in cell
            ox = x + (cell_w - im.width)//2
            oy = y + (cell_h - im.height)//2
            canvas.paste(im, (ox, oy))
            x += cell_w + margin
            i += 1
        y += cell_h + margin
    return canvas

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-root", required=True, help="Folder with phase_*_png subfolders")
    ap.add_argument("--pred-csv",   required=True, help="predicted_modes.csv (id,pred_label)")
    ap.add_argument("--out-root",   required=True, help="Output sheets root")
    ap.add_argument("--kind", default="unwrapped_png", choices=["unwrapped_png","wrapped_png","quality_png"])
    ap.add_argument("--thumb", type=int, default=256, help="max side for thumbnails")
    ap.add_argument("--cols", type=int, default=12, help="columns per sheet")
    ap.add_argument("--max-per-label", type=int, default=240, help="limit per label (0 = no limit)")
    ap.add_argument("--margin", type=int, default=8, help="outer/inner margin pixels")
    args = ap.parse_args()

    src_dir = Path(args.phase_root) / f"phase_{args.kind}"
    if not src_dir.exists():
        raise SystemExit(f"[ERR] not found: {src_dir}")
    out_root = Path(args.out_root); ensure_dir(out_root)

    df = pd.read_csv(args.pred_csv)
    if "id" not in df.columns or "pred_label" not in df.columns:
        raise SystemExit("[ERR] predicted_modes.csv must have columns: id, pred_label")

    head_h = 36  # header band height
    cell_w = args.thumb
    cell_h = args.thumb

    for label, d in df.groupby("pred_label"):
        lbl = str(label); lbl_slug = slugify(lbl)
        ids = [str(x) for x in d["id"].tolist()]
        max_take = int(args.max_per_label)
        if max_take > 0:
            ids = ids[:max_take]

        thumbs = []
        for stem in ids:
            src = src_dir / f"{stem}.png"
            if not src.exists():
                src2 = src_dir / f"{stem}.PNG"
                if src2.exists():
                    src = src2
                else:
                    print(f"[MISS] {src}")
                    continue
            try:
                thumbs.append(load_thumb(src, max_side=args.thumb))
            except Exception as e:
                print(f"[WARN] {stem}: {e}")

        if not thumbs:
            continue

        # Compose into one or more pages if too many images
        cols = max(1, int(args.cols))
        per_page = cols * max(1, (4000 - head_h) // (cell_h + args.margin))  # rough cap for tall pages
        if per_page <= 0:
            per_page = cols * 10

        pages = [thumbs[i:i+per_page] for i in range(0, len(thumbs), per_page)]
        for pi, page_imgs in enumerate(pages):
            rows = (len(page_imgs) + cols - 1) // cols
            canvas = paste_grid(page_imgs, cols=cols, cell_w=cell_w, cell_h=cell_h,
                                margin=args.margin, head_h=head_h,
                                title=f"Mode {lbl}  (page {pi+1}/{len(pages)})")
            out_path = out_root / f"{lbl_slug}_page{pi+1}.png"
            canvas.save(out_path)
            print("[WRITE]", out_path)

    print("[DONE] Contact sheets at:", out_root)

if __name__ == "__main__":
    main()
