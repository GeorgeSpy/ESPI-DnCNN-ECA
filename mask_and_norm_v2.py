import argparse, os
import numpy as np
from PIL import Image, ImageFilter

def load_u16(path):
    im = Image.open(path)
    arr = np.array(im)
    if arr.dtype == np.uint8:
        arr = (arr.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.float32)
        mx = max(float(arr.max()), 1.0)
        arr = np.clip(arr, 0, mx) / mx
        arr = (arr * 65535.0).astype(np.uint16)
    return arr

def save_u16(path, arr_u16):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(arr_u16.astype(np.uint16)).save(path, compress_level=0)

def make_soft_mask(mask_gray_u8, blur_radius=8):
    # 0/255 -> 0..1, Gaussian blur για feather στα όρια
    m = Image.fromarray(mask_gray_u8).filter(ImageFilter.GaussianBlur(radius=blur_radius))
    m = np.array(m).astype(np.float32) / 255.0
    # κανονικοποίηση σε [0,1]
    m = (m - m.min()) / max(1e-6, (m.max() - m.min()))
    # clamp για να αποφύγουμε «τρύπες»
    return np.clip(m, 0.0, 1.0)

def process_single(img_path, out_path, mask_path):
    img_u16 = load_u16(img_path).astype(np.float32)
    mask_u8 = np.array(Image.open(mask_path).convert("L"))
    soft = make_soft_mask(mask_u8, blur_radius=8)   # feather ~8 px

    # εφαρμογή μάσκας
    x = img_u16 * soft

    # αφαίρεση DC ΜΟΝΟ μέσα στο ROI (weighted mean)
    denom = soft.sum()
    if denom < 10:
        # ασφάλεια: σώσε απλά masked
        y = (x / 65535.0)
    else:
        mean_roi = (x * soft).sum() / denom
        x = (x - mean_roi) * soft  # ξανά εφαρμογή soft για να σβήσουν τα άκρα

        # δυναμικό εύρος γύρω από 0 → χαρτογράφηση σε [0,1]
        # robust scale με p99(|x|)
        valid = np.abs(x[soft > 1e-3])
        scale = np.percentile(valid, 99.0) if valid.size > 10 else 1.0
        if scale < 1e-6: scale = 1.0
        y = 0.5 + 0.5 * np.clip(x / scale, -1.0, 1.0)  # 0..1

    out = (np.clip(y, 0.0, 1.0) * 65535.0).astype(np.uint16)
    save_u16(out_path, out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--in-dir", dest="in_dir", type=str)
    ap.add_argument("--out-dir", dest="out_dir", type=str)
    ap.add_argument("--mask", type=str, required=True)
    args = ap.parse_args()

    if args.img and args.out:
        process_single(args.img, args.out, args.mask)
    elif args.in_dir and args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for name in os.listdir(args.in_dir):
            if name.lower().endswith(".png"):
                process_single(os.path.join(args.in_dir, name),
                               os.path.join(args.out_dir, name), args.mask)
    else:
        raise SystemExit("Specify either (--img & --out) or (--in-dir & --out-dir).")

if __name__ == "__main__":
    import os
    main()
