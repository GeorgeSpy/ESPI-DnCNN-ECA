#!/usr/bin/env python3
# paired_sanity_check.py — checks that every clean has a matching noisy (same filename) and reports counts.
# Example:
#   python paired_sanity_check.py --clean "C:\...\W01_ESPI_90db-Averaged" --noisy "C:\...\W01_ESPI_90db-PseudoNoisy"

import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noisy", required=True)
    ap.add_argument("--ext", default=".png")
    args = ap.parse_args()

    croot = Path(args.clean); nroot = Path(args.noisy)
    cset = {p.name for p in croot.glob(f"*{args.ext}")}
    nset = {p.name for p in nroot.glob(f"*{args.ext}")}
    inter = cset & nset
    print(f"Clean: {len(cset)} | Noisy: {len(nset)} | Paired: {len(inter)} | MissingNoisy: {len(cset-nset)} | OrphanNoisy: {len(nset-cset)}")
    if cset-nset:
        print("Missing noisy for:", sorted(list(cset-nset))[:10], "...")
    if nset-cset:
        print("Orphan noisy:", sorted(list(nset-cset))[:10], "...")

if __name__ == "__main__":
    main()
