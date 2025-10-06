#!/usr/bin/env python3
# modes_contact_sheets_caption.py (ASCII-only, Windows-safe)
# Build per-label contact sheets (big grids of thumbnails) with ID captions using
# phase_*_png and predicted_modes.csv.
#
# Usage:
#   python modes_contact_sheets_caption.py ^
#     --phase-root "C:\...\W01_PhaseOut_STRICT\band18_ff100_cs16" ^
#     --pred-csv   "C:\...\W01_modes\predicted_modes.csv" ^
#     --out-root   "C:\...\W01_modes\contact_sheets_captioned" ^
#     --kind unwrapped_png --thumb 256 --cols 12 --max-per-label 240 ^
#     --caption --font-size 14 --caption-h 22
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

def compose_cell(thumb: Image.Image, cell_w: int, cell_h: int, caption: str, cap_h: int, font):
    # Create cell with thumbnail centered and caption below
    bg = Image.new("L", (cell_w, cell_h), color=255)
    # place image
    top_pad = max(2, (cell_h - cap_h - thumb.height) // 2)
    ox = (cell_w - thumb.width)//2
    bg.paste(thumb, (ox, top_pad))
    # caption band
    draw = ImageDraw.Draw(bg)
    y_text = cell_h - cap_h + (cap_h - font.size)//2 - 1 if hasattr(font, "size") else cell_h - cap_h + 4
    # truncate caption if too long
    cap = caption
    # ensure it fits roughly: compute text width with font.getlength if available
    try:
        wtxt = font.getlength(cap)
    except Exception:
        wtxt = len(cap) * (font.size if hasattr(font,"size") else 8)
    if wtxt > (cell_w - 8):
        # attempt to shorten middle
        keep = max(6, int((cell_w - 16) / max(6, (font.size if hasattr(font,"size") else 8))))
        if keep < len(cap):
            cap = cap[:keep//2] + "..." + cap[-keep//2:]
    # draw text centered
    try:
        wtxt = font.getlength(cap)
    except Exception:
        wtxt = len(cap) * (font.size if hasattr(font,"size") else 8)
    x_text = (cell_w - int(wtxt))//2
    draw.rectangle([0, cell_h - cap_h, cell_w, cell_h], fill=245)
    draw.text((max(2,x_text), y_text), cap, fill=0, font=font)
    return bg

def paste_grid(cells, cols: int, cell_w: int, cell_h: int, margin: int, head_h: int, title: str):
    rows = (len(cells) + cols - 1) // cols
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
            if i >= len(cells): break
            cell = cells[i]
            canvas.paste(cell, (x, y))
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
    ap.add_argument("--caption", action="store_true", help="draw ID caption under each thumbnail")
    ap.add_argument("--caption-h", type=int, default=22, help="caption band height in pixels")
    ap.add_argument("--font", default="", help="optional TTF font path")
    ap.add_argument("--font-size", type=int, default=14, help="caption font size")
    args = ap.parse_args()

    src_dir = Path(args.phase_root) / f"phase_{args.kind}"
    if not src_dir.exists():
        raise SystemExit(f"[ERR] not found: {src_dir}")
    out_root = Path(args.out_root); ensure_dir(out_root)

    df = pd.read_csv(args.pred_csv)
    if "id" not in df.columns or "pred_label" not in df.columns:
        raise SystemExit("[ERR] predicted_modes.csv must have columns: id, pred_label")

    # font
    if args.font:
        try:
            font = ImageFont.truetype(args.font, args.font_size)
        except Exception:
            print("[WARN] failed to load font; falling back to default.")
            font = ImageFont.load_default()
    else:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    head_h = 36  # header band height
    img_h = args.thumb
    cap_h = int(args.caption_h) if args.caption else 0
    cell_w = args.thumb
    cell_h = img_h + cap_h

    for label, d in df.groupby("pred_label"):
        lbl = str(label); lbl_slug = slugify(lbl)
        ids = [str(x) for x in d["id"].tolist()]
        max_take = int(args.max_per_label)
        if max_take > 0:
            ids = ids[:max_take]

        cells = []
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
                th = load_thumb(src, max_side=args.thumb)
                if args.caption:
                    cap = stem
                    cell = compose_cell(th, cell_w=cell_w, cell_h=cell_h, caption=cap, cap_h=cap_h, font=font)
                else:
                    # no caption: center in cell_h anyway
                    bg = Image.new("L", (cell_w, cell_h), color=255)
                    ox = (cell_w - th.width)//2
                    oy = (cell_h - th.height)//2
                    bg.paste(th, (ox, oy))
                    cell = bg
                cells.append(cell)
            except Exception as e:
                print(f"[WARN] {stem}: {e}")

        if not cells:
            continue

        cols = max(1, int(args.cols))
        # rough cap for tall pages; each cell ~ cell_h high + margin
        per_page = max(cols * 6, cols * max(1, (4200 - head_h) // (cell_h + args.margin)))
        pages = [cells[i:i+per_page] for i in range(0, len(cells), per_page)]
        for pi, page_cells in enumerate(pages):
            canvas = paste_grid(page_cells, cols=cols, cell_w=cell_w, cell_h=cell_h,
                                margin=args.margin, head_h=head_h,
                                title=f"Mode {lbl}  (page {pi+1}/{len(pages)})")
            out_path = out_root / f"{lbl_slug}_page{pi+1}.png"
            canvas.save(out_path)
            print("[WRITE]", out_path)

    print("[DONE] Contact sheets at:", out_root)

if __name__ == "__main__":
    main()
