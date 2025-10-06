#!/usr/bin/env python3
# ckpt_inspect_and_convert.py
# Inspect a PyTorch .pth and convert it to {'model': <state_dict>} format.
# Optional: strip a prefix from keys (e.g., 'module.').
# Usage:
#   python ckpt_inspect_and_convert.py --in "path\to\best_finetune.pth" --out "path\to\best_finetune_MODEL.pth"
#   python ckpt_inspect_and_convert.py --in in.pth --out out.pth --strip-prefix module.

import argparse
from collections import OrderedDict
from pathlib import Path
import sys

try:
    import torch
except Exception as e:
    print("[ERR] PyTorch not available:", e)
    sys.exit(1)

COMMON_KEYS = ("model", "state_dict", "model_state_dict", "net", "network")

def is_tensor_like(v):
    try:
        return torch.is_tensor(v) or hasattr(v, "shape")
    except Exception:
        return False

def looks_like_state_dict(obj):
    if not isinstance(obj, (dict, OrderedDict)):
        return False
    # if any value looks tensor-like, we assume it's a state_dict
    for k, v in obj.items():
        if isinstance(k, str) and is_tensor_like(v):
            return True
    return False

def strip_prefix_from_state_dict(sd, prefix):
    if not prefix:
        return sd
    new_sd = OrderedDict()
    plen = len(prefix)
    for k, v in sd.items():
        if k.startswith(prefix):
            new_sd[k[plen:]] = v
        else:
            new_sd[k] = v
    return new_sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input .pth path")
    ap.add_argument("--out", dest="out", required=True, help="Output .pth path")
    ap.add_argument("--strip-prefix", default="", help="Optional key prefix to strip, e.g. 'module.'")
    args = ap.parse_args()

    inp = Path(args.inp); out = Path(args.out)
    if not inp.exists():
        print("[ERR] Input not found:", inp)
        sys.exit(2)

    print("[INFO] Loading:", inp)
    obj = torch.load(str(inp), map_location="cpu")
    print("[INFO] Loaded type:", type(obj))

    model_sd = None

    if isinstance(obj, (dict, OrderedDict)):
        top_keys = list(obj.keys())
        print("[INFO] Top-level keys:", top_keys[:10], "... total", len(top_keys))
        # Try common container keys
        for key in COMMON_KEYS:
            if key in obj and looks_like_state_dict(obj[key]):
                model_sd = obj[key]
                print("[OK] Found state_dict under key:", key)
                break
        # Try bare state_dict at top level
        if model_sd is None and looks_like_state_dict(obj):
            model_sd = obj
            print("[OK] Top-level object looks like a state_dict.")
    else:
        # Sometimes checkpoints save a bare OrderedDict directly
        if looks_like_state_dict(obj):
            model_sd = obj
            print("[OK] Non-dict object treated as state_dict.")

    if model_sd is None:
        print("[ERR] Could not locate a model state_dict in this file.")
        sys.exit(3)

    # Optionally strip prefix (e.g. 'module.')
    if args.strip_prefix:
        before = next(iter(model_sd.keys()))
        model_sd = strip_prefix_from_state_dict(model_sd, args.strip_prefix)
        after = next(iter(model_sd.keys()))
        print("[INFO] Stripped prefix:", args.strip_prefix)
        print("[INFO] Example key before:", before)
        print("[INFO] Example key after :", after)

    # Save in the desired format
    torch.save({"model": model_sd}, str(out))
    print("[DONE] Wrote converted checkpoint:", out)
    print("[INFO] Total params:", sum(p.numel() for p in model_sd.values() if is_tensor_like(p)))

if __name__ == "__main__":
    main()
