# -*- coding: utf-8 -*-
import numpy as np, argparse, os
from PIL import Image

def probe(img_path, center_suppress=16, annulus=(8,300)):
    # 16-bit aware loader (no convert("L"))
    im = Image.open(img_path)
    arr = np.asarray(im)
    if arr.ndim == 3:
        # average channels to grayscale while preserving bit depth
        arr = arr.mean(axis=2)
    if arr.dtype == np.uint16:
        x = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    else:
        x = arr.astype(np.float32)
    x = x - x.mean()

    F = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(F)

    H,W = mag.shape
    cy,cx = H//2, W//2

    # center suppress
    yy,xx = np.ogrid[:H,:W]
    r = np.sqrt((yy-cy)**2 + (xx-cx)**2)
    mag = mag.copy()
    mag[r < float(center_suppress)] = 0.0

    # annulus mask
    rmin, rmax = annulus
    mask = (r >= rmin) & (r <= rmax)
    mag_masked = np.where(mask, mag, 0.0)

    # pick peak
    iy,ix = np.unravel_index(np.argmax(mag_masked), mag_masked.shape)
    rr = float(np.sqrt((iy-cy)**2 + (ix-cx)**2))
    theta = float(np.degrees(np.arctan2(iy-cy, ix-cx)))  # deg
    return (iy,ix,rr,theta),(H,W,cy,cx)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--cs", type=float, default=16)
    ap.add_argument("--rmin", type=float, default=8)
    ap.add_argument("--rmax", type=float, default=300)
    args = ap.parse_args()
    (iy,ix,rr,theta),(H,W,cy,cx) = probe(args.img, args.cs, (args.rmin,args.rmax))
    print(f"[{os.path.basename(args.img)}] peak @ (x={ix}, y={iy})  r={rr:.2f}  θ={theta:.1f}deg   (center=({cx},{cy}))")

if __name__ == "__main__":
    main()

