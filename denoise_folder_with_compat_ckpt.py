import argparse, os
import numpy as np
from PIL import Image

def load_u16(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.dtype != np.uint16:
        # αν είναι 8-bit, ανέβασέ το σε 16-bit για ομοιομορφία
        arr = (arr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    return arr

def save_u16(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr.astype(np.uint16)).save(path, compress_level=0)

def apply_mask_and_norm(img_u16, mask_u8, pmin=1.0, pmax=99.0):
    mask = (mask_u8 > 0).astype(np.float32)
    img = img_u16.astype(np.float32)
    img *= mask  # εφαρμόζουμε ROI

    # Υπολογισμός percentiles μόνο στα μη-μηδενικά της μάσκας
    valid = img[mask > 0]
    if valid.size < 10:
        # fallback: επιστρέφουμε απλά masked
        return img_u16 * (mask > 0)

    vmin = np.percentile(valid, pmin)
    vmax = np.percentile(valid, pmax)
    if vmax <= vmin:
        vmax = vmin + 1.0

    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0.0, 1.0)
    img *= 65535.0
    return img.astype(np.uint16)

def process_file(in_path, out_path, mask_path, pmin, pmax):
    img = load_u16(in_path)
    mask = np.array(Image.open(mask_path).convert("L"))
    out = apply_mask_and_norm(img, mask, pmin, pmax)
    save_u16(out_path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, help="single image path")
    ap.add_argument("--out", type=str, help="single output path")
    ap.add_argument("--in-dir", type=str, help="input directory")
    ap.add_argument("--out-dir", type=str, help="output directory")
    ap.add_argument("--mask", type=str, required=True, help="roi mask png path")
    ap.add_argument("--pmin", type=float, default=1.0)
    ap.add_argument("--pmax", type=float, default=99.0)
    args = ap.parse_args()

    if args.img and args.out:
        process_file(args.img, args.out, args.mask, args.pmin, args.pmax)
    elif args.in_dir and args.out_dir:
        for name in os.listdir(args.in_dir):
            if name.lower().endswith(".png"):
                in_path  = os.path.join(args.in_dir, name)
                out_path = os.path.join(args.out_dir, name)
                process_file(in_path, out_path, args.mask, args.pmin, args.pmax)
    else:
        raise SystemExit("Specify either (--img & --out) or (--in-dir & --out-dir).")

if __name__ == "__main__":
    main()
