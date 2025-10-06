#!/usr/bin/env python3
# patch_reflect_pad_batch_denoise_SIMPLE.py
# Replaces the function `reflect_pad_patch` in batch_denoise_pytorch_v3.py
# with a safer version: uses 'reflect' when valid, otherwise falls back to 'replicate'.
#
# Usage (CMD):
#   python patch_reflect_pad_batch_denoise_SIMPLE.py --src "C:\ESPI_DnCNN\batch_denoise_pytorch_v3.py" --dst "C:\ESPI_DnCNN\batch_denoise_pytorch_v3_REFLECTFIX.py"

import argparse
from pathlib import Path

NEW_FUNC = (
    "def reflect_pad_patch(x, top, left, tile):\n"
    "    patch = x[..., top:top+tile, left:left+tile]\n"
    "    ph = max(0, tile - patch.shape[-2])  # pad bottom\n"
    "    pw = max(0, tile - patch.shape[-1])  # pad right\n"
    "    if ph > 0 or pw > 0:\n"
    "        mode = 'reflect'\n"
    "        # reflect requires pad < dimension; if not possible, fallback to replicate\n"
    "        if (patch.shape[-2] <= 1 or patch.shape[-1] <= 1 or\n"
    "            ph >= patch.shape[-2] or pw >= patch.shape[-1]):\n"
    "            mode = 'replicate'\n"
    "        patch = F.pad(patch, (0, pw, 0, ph), mode=mode)\n"
    "    return patch\n"
)

def replace_function(src_text):
    lines = src_text.splitlines(keepends=True)
    # find start of function
    start_idx = -1
    for i, line in enumerate(lines):
        if line.lstrip().startswith("def reflect_pad_patch(") and line.startswith("def"):
            start_idx = i
            break
        # also allow leading spaces before 'def'
        if line.lstrip().startswith("def reflect_pad_patch(") and start_idx == -1:
            # capture even if indented (but expected to be top-level)
            start_idx = i
            break
    if start_idx == -1:
        return None  # not found

    # find end of function: next top-level 'def ' after start
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        # a new function usually starts with 'def ' at column 0
        if lines[j].startswith("def ") or lines[j].startswith("class "):
            end_idx = j
            break

    # Build new text
    new_block = NEW_FUNC.replace("\n", "\n")  # keep LF
    new_text = "".join(lines[:start_idx]) + new_block + "".join(lines[end_idx:])
    return new_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to source batch_denoise_pytorch_v3.py")
    ap.add_argument("--dst", required=True, help="Path to write patched copy")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f"[ERR] Source not found: {src}")

    txt = src.read_text(encoding="utf-8")
    patched = replace_function(txt)
    if patched is None:
        raise SystemExit("[ERR] Could not find function 'reflect_pad_patch' in source.")

    dst.write_text(patched, encoding="utf-8")
    print(f"[OK] Patched file written to: {dst}")

if __name__ == "__main__":
    main()
