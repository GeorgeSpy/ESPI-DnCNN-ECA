#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phase_extract_fft_preset.py
---------------------------
Thin wrapper that calls your existing phase_extract_fft.py with tuned defaults
(center-suppress, flatfield, etc.) so you don't have to remember flags.

Usage:
python phase_extract_fft_preset.py ^
  --input-dir "C:\...\W01_ESPI_90db-Averaged_masked" ^
  --output-dir "C:\...\W01_PhaseOut_masked" ^
  --roi-mask  "C:\...\roi_mask.png"  (optional)

You can override defaults via flags.
"""

import argparse, subprocess, sys
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase-script", default="phase_extract_fft.py", help="path to the extractor script")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--roi-mask", default="")
    # tuned defaults
    ap.add_argument("--flatfield", type=int, default=60)
    ap.add_argument("--band", type=int, default=26)
    ap.add_argument("--center-suppress", type=int, default=14)
    ap.add_argument("--unwrap", default="auto")
    args = ap.parse_args()

    cmd = [sys.executable, args.phase_script,
           "--input-dir", args.input_dir,
           "--output-dir", args.output_dir,
           "--flatfield", str(args.flatfield),
           "--band", str(args.band),
           "--center-suppress", str(args.center_suppress),
           "--unwrap", args.unwrap]
    if args.roi_mask:
        cmd += ["--roi-mask", args.roi_mask]

    print("[RUN]", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("Extractor returned", rc)
        sys.exit(rc)
    print("[DONE]")

if __name__ == "__main__":
    main()
