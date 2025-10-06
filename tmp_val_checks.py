#!/usr/bin/env python3
import os
import json
import math
import pandas as pd

CSV = r"C:\ESPI_TEMP\pairs\pairs_mix_80_20_FULL.csv"


def check_usable_rows():
    df = pd.read_csv(CSV)
    df["noisy_ok"] = df["noisy"].map(os.path.exists)
    df["clean_ok"] = df["clean"].map(os.path.exists)
    usable = df[df["noisy_ok"] & df["clean_ok"]]
    out = {
        "csv_rows": int(len(df)),
        "usable_rows": int(len(usable)),
        "missing_noisy": int((~df["noisy_ok"]).sum()),
        "missing_clean": int((~df["clean_ok"]).sum()),
    }
    print("\n[CHECK1] USABLE ROWS")
    print(json.dumps(out, indent=2))


def check_expected_steps():
    N = len(pd.read_csv(CSV))
    train = int(N * 0.85)
    val = N - train
    B = 12
    out = {
        "N": int(N),
        "train": int(train),
        "val": int(val),
        "train_steps_drop_last": int(train // B),
        "val_steps_drop_last": int(val // B),
        "train_exact_steps": int((train + B - 1) // B),
        "val_exact_steps": int((val + B - 1) // B),
        "batch": B,
    }
    print("\n[CHECK2] EXPECTED STEPS")
    print(json.dumps(out, indent=2))


def check_val_dynamic_range():
    from PIL import Image
    import numpy as np

    df = pd.read_csv(CSV)
    N = len(df)
    tr = int(N * 0.85)
    val = df.iloc[tr:][["noisy", "clean"]].head(64)

    def dynrng(p: str) -> int:
        try:
            with Image.open(p) as im:
                a = np.array(im)
            return int(a.max()) - int(a.min())
        except Exception:
            return -1

    noisy_dr = [dynrng(p) for p in val["noisy"].tolist()]
    clean_dr = [dynrng(p) for p in val["clean"].tolist()]
    suspects = sum((nd <= 1) or (cd <= 1) for nd, cd in zip(noisy_dr, clean_dr))

    out = {
        "val_rows_checked": int(len(val)),
        "noisy_dr_min": int(min(noisy_dr) if noisy_dr else -1),
        "noisy_dr_median": int(np.median(noisy_dr) if noisy_dr else -1),
        "clean_dr_min": int(min(clean_dr) if clean_dr else -1),
        "clean_dr_median": int(np.median(clean_dr) if clean_dr else -1),
        "flat_suspects": int(suspects),
    }
    print("\n[CHECK3] VAL DYNAMIC RANGE (first 64)")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    check_usable_rows()
    check_expected_steps()
    check_val_dynamic_range()


