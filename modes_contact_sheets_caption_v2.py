#!/usr/bin/env python3
# modes_contact_sheets_caption_v2.py (ASCII-only, Windows-safe)
# Build per-label contact sheets (big grids of thumbnails) with ID captions,
# customizable header text, optional diagonal watermark, and custom DPI.
#
# Usage (example):
#   python modes_contact_sheets_caption_v2.py ^
#     --phase-root "C:\...\W01_PhaseOut_STRICT\band18_ff100_cs16" ^
#     --pred-csv   "C:\...\W01_modes\predicted_modes.csv" ^
#     --out-root   "C:\...\W01_modes\contact_sheets_captioned" ^
#     --kind unwrapped_png --thumb 256 --cols 12 --max-per-label 240 ^
#     --caption --font-size 14 --caption-h 22 ^
#     --header "ESPI Modes | {label} | W01 | {page}/{pages}" --header-font-size 16 ^
#     --watermark "Bouzouki Top" --wm-size 64 --wm-alpha 64 --wm-diagonal ^
#     --dpi 300
#
# Requires: pandas, pillow

import argparse, re, math
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

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

def load_font(font_path: str, size: int):
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def draw_header(canvas: Image.Image, text: str, font, pad: int = 10, band: int = 32):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0,0,canvas.width, band], fill=240)
    # Basic centering
    x = pad
    y = max(4, (band - (getattr(font, "size", 14))) // 2)
    draw.text((x, y), text, fill=0, font=font)

def compose_cell(thumb: Image.Image, cell_w: int, cell_h: int, caption: str, cap_h: int, font):
    bg = Image.new("L", (cell_w, cell_h), color=255)
    top_pad = max(2, (cell_h - cap_h - thumb.height) // 2)
    ox = (cell_w - thumb.width)//2
    bg.paste(thumb, (ox, top_pad))
    if cap_h > 0:
        draw = ImageDraw.Draw(bg)
        # background band
        draw.rectangle([0, cell_h - cap_h, cell_w, cell_h], fill=245)
        # truncate caption if too long
        cap = caption
        avg_w = getattr(font, "size", 8)
        approx = len(cap) * avg_w
        if approx > (cell_w - 12):
            keep = max(6, int((cell_w - 16) / max(6, avg_w)))
            if keep < len(cap):
                cap = cap[:keep//2] + "..." + cap[-keep//2:]
        # center text
        try:
            tw = font.getlength(cap)
        except Exception:
            tw = len(cap) * avg_w
        tx = max(2, int((cell_w - tw)//2))
        ty = cell_h - cap_h + max(2, (cap_h - avg_w)//2)
        draw.text((tx, ty), cap, fill=0, font=font)
    return bg

def paste_grid(cells, cols: int, cell_w: int, cell_h: int, margin: int, head_h: int, header_text: str, header_font):
    rows = (len(cells) + cols - 1) // cols
    W = margin + cols*(cell_w+margin)
    H = head_h + margin + rows*(cell_h+margin)
    canvas = Image.new("L", (W, H), color=255)
    draw_header(canvas, header_text, header_font, pad=10, band=head_h)
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

def add_watermark(canvas: Image.Image, text: str, size: int, alpha: int, diagonal: bool):
    if not text:
        return canvas
    # convert to RGBA for alpha text
    rgba = canvas.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(None, size)
    except Exception:
        font = ImageFont.load_default()
    # draw repeated watermark or single diagonal
    if diagonal:
        # create a single large text across diagonal
        # make a big text image then rotate and center
        txt = Image.new("RGBA", rgba.size, (0,0,0,0))
        d2 = ImageDraw.Draw(txt)
        # estimate repeated text to span width
        rep = max(1, math.ceil((rgba.width * 1.5) / max(10, size*6)))
        msg = (" " * 4 + text) * rep
        d2.text((10, rgba.height//2), msg, fill=(0,0,0,alpha), font=font, anchor=None)
        txt = txt.rotate( -30, resample=Image.BICUBIC, expand=1 )
        # center onto overlay
        ox = (overlay.width - txt.width)//2
        oy = (overlay.height - txt.height)//2
        overlay.alpha_composite(txt, (ox, oy))
    else:
        # tiled watermark every 300 px
        step = max(200, size*6)
        for y in range(0, rgba.height, step):
            for x in range(0, rgba.width, step):
                draw.text((x, y), text, fill=(0,0,0,alpha), font=font)
    out = Image.alpha_composite(rgba, overlay).convert("L")
    return out

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
    # captions
    ap.add_argument("--caption", action="store_true", help="draw ID caption under each thumbnail")
    ap.add_argument("--caption-h", type=int, default=22, help="caption band height in pixels")
    ap.add_argument("--font", default="", help="optional TTF font path for captions")
    ap.add_argument("--font-size", type=int, default=14, help="caption font size")
    # header
    ap.add_argument("--header", default="Mode {label} | Page {page}/{pages}", help="header template")
    ap.add_argument("--header-font", default="", help="optional TTF font path for header")
    ap.add_argument("--header-font-size", type=int, default=16, help="header font size")
    ap.add_argument("--header-h", type=int, default=32, help="header band height")
    # watermark
    ap.add_argument("--watermark", default="", help="optional watermark text")
    ap.add_argument("--wm-size", type=int, default=64, help="watermark font size")
    ap.add_argument("--wm-alpha", type=int, default=64, help="watermark alpha 0..255")
    ap.add_argument("--wm-diagonal", action="store_true", help="diagonal watermark across the page")
    # DPI
    ap.add_argument("--dpi", type=int, default=0, help="PNG save DPI (0 = default)")

    args = ap.parse_args()

    src_dir = Path(args.phase_root) / f"phase_{args.kind}"
    if not src_dir.exists():
        raise SystemExit(f"[ERR] not found: {src_dir}")
    out_root = Path(args.out_root); ensure_dir(out_root)

    df = pd.read_csv(args.pred_csv)
    if "id" not in df.columns or "pred_label" not in df.columns:
        raise SystemExit("[ERR] predicted_modes.csv must have columns: id, pred_label")

    # fonts
    cap_font = load_font(args.font, args.font_size)
    head_font = load_font(args.header_font, args.header_font_size)

    head_h = int(args.header_h)
    img_h = int(args.thumb)
    cap_h = int(args.caption_h) if args.caption else 0
    cell_w = int(args.thumb)
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
                th = load_thumb(src, max_side=img_h)
                cell = compose_cell(th, cell_w=cell_w, cell_h=cell_h, caption=stem, cap_h=cap_h, font=cap_font)
                cells.append(cell)
            except Exception as e:
                print(f"[WARN] {stem}: {e}")

        if not cells:
            continue

        cols = max(1, int(args.cols))
        per_page = max(cols * 6, cols * max(1, (4200 - head_h) // (cell_h + args.margin)))
        pages = [cells[i:i+per_page] for i in range(0, len(cells), per_page)]
        for pi, page_cells in enumerate(pages, start=1):
            header_text = args.header.format(label=lbl, page=pi, pages=len(pages))
            canvas = paste_grid(page_cells, cols=cols, cell_w=cell_w, cell_h=cell_h,
                                margin=args.margin, head_h=head_h,
                                header_text=header_text, header_font=head_font)
            # watermark
            if args.watermark:
                canvas = add_watermark(canvas, args.watermark, size=int(args.wm_size),
                                       alpha=max(0, min(255, int(args.wm_alpha))),
                                       diagonal=bool(args.wm_diagonal))

            out_path = out_root / f"{lbl_slug}_page{pi}.png"
            if args.dpi and int(args.dpi) > 0:
                canvas.save(out_path, dpi=(int(args.dpi), int(args.dpi)))
            else:
                canvas.save(out_path)
            print("[WRITE]", out_path)

    print("[DONE] Contact sheets at:", out_root)

if __name__ == "__main__":
    main()
