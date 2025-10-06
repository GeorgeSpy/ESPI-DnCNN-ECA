import argparse, json, math
import numpy as np
import pandas as pd

LABEL_MAP = {
    "mode_(1,1)H": 0,
    "mode_(1,1)T": 1,
    "mode_(1,2)": 2,
    "mode_(2,1)": 3,
    "mode_higher": 4,
    "other_unknown": 5,
}

# Fixed bins for TETRACHORD, as defined:
BINS_WOOD = [
    ("mode_(1,1)H", (155.0, 175.0)),
    ("mode_(1,1)T", (320.0, 345.0)),
    ("mode_(1,2)" , (500.0, 525.0)),
    ("mode_(2,1)" , (540.0, 570.0)),
    ("mode_higher", (680.0, 1500.0)),
]
BINS_CARBON = [
    ("mode_(1,1)H", (170.0, 190.0)),
    ("mode_(1,1)T", (340.0, 365.0)),
    ("mode_(1,2)" , (520.0, 550.0)),
    ("mode_(2,1)" , (570.0, 600.0)),
    ("mode_higher", (710.0, 1500.0)),
]

def choose_bins(material: str):
    m = (material or "").strip().lower()
    if m in ("carbon","cf","carbonfiber","carbon_fiber"): 
        return BINS_CARBON
    return BINS_WOOD

def assign_class(freq_hz: float, bins, halo_hz: float = 0.0):
    # allow "cushion" ±halo_hz if you want more lenient boundaries
    for name, (fmin, fmax) in bins:
        if (freq_hz >= fmin - halo_hz) and (freq_hz <= fmax + halo_hz):
            return name
    return "other_unknown"

def passes_qc(row, rmse_max=4.0, pct_pi2_max=50.0, min_valid_ratio=0.20, max_zero_frac=0.80):
    # Uses available columns, otherwise skips corresponding checks
    ok = True

    # IRLS / QC metrics
    for col, thr, mode in [
        ("rmse", rmse_max, "max"),
        ("rmse_unwrapped", rmse_max, "max"),
        ("pct_pi2", pct_pi2_max, "max"),
    ]:
        if col in row and pd.notna(row[col]):
            v = float(row[col])
            if mode == "max" and v > thr:
                ok = False

    # valid ratio (valid_px / roi_area) if available
    if "valid_px" in row and "roi_area" in row and pd.notna(row["valid_px"]) and pd.notna(row["roi_area"]):
        area = float(row["roi_area"])
        vr = float(row["valid_px"]) / area if area > 0 else 0.0
        if vr < min_valid_ratio:
            ok = False

    # zero_frac (smaller is better)
    if "zero_frac" in row and pd.notna(row["zero_frac"]):
        zf = float(row["zero_frac"])
        if zf > max_zero_frac:
            ok = False

    return ok

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)  # all_features_QCpass.csv
    ap.add_argument("--out", required=True)       # labels_fixed_bins.csv
    ap.add_argument("--labelmap", required=False, default="") # optional write-out JSON
    ap.add_argument("--material-col", default="material")     # if exists in CSV
    ap.add_argument("--instrument-col", default="instrument") # or "set" / "dataset" 
    ap.add_argument("--inst2material", default="")            # e.g. "W01:wood,W02:wood,W03:carbon"
    ap.add_argument("--halo-hz", type=float, default=0.0)     # if you want ±5 Hz margin
    ap.add_argument("--use-qc", action="store_true")          # enable QC gate
    # thresholds
    ap.add_argument("--rmse-max", type=float, default=4.0)
    ap.add_argument("--pct-pi2-max", type=float, default=50.0)
    ap.add_argument("--min-valid-ratio", type=float, default=0.20)
    ap.add_argument("--max-zero-frac", type=float, default=0.80)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    if "freq_hz" not in df.columns:
        raise SystemExit("Missing column freq_hz in features CSV")

    # mapping instrument -> material, if no material column exists
    inst_map = {}
    if args.inst2material.strip():
        for token in args.inst2material.split(","):
            if ":" in token:
                k,v = token.split(":",1)
                inst_map[k.strip()] = v.strip()

    out_rows = []
    for _, row in df.iterrows():
        freq = float(row["freq_hz"])
        # from material:
        material = None
        if args.material_col in df.columns and pd.notna(row.get(args.material_col, np.nan)):
            material = str(row[args.material_col]).strip()
        elif args.instrument_col in df.columns and pd.notna(row.get(args.instrument_col, np.nan)):
            material = inst_map.get(str(row[args.instrument_col]).strip(), "wood")
        else:
            material = "wood"

        bins = choose_bins(material)
        cname = assign_class(freq, bins, halo_hz=args.halo_hz)

        # optional QC gating
        if args.use_qc and cname != "other_unknown":
            if not passes_qc(row, args.rmse_max, args.pct_pi2_max, args.min_valid_ratio, args.max_zero_frac):
                cname = "other_unknown"

        out_rows.append({
            "class_id": LABEL_MAP[cname],
            "class_name": cname,
            "material": material,
            "freq_hz": freq,
        })

    lab = pd.DataFrame(out_rows)
    merged = pd.concat([df.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
    merged.to_csv(args.out, index=False)
    if args.labelmap:
        with open(args.labelmap, "w", encoding="ascii") as f:
            json.dump(LABEL_MAP, f, indent=2)
    print("[OK] wrote", args.out)
    if args.labelmap:
        print("[OK] wrote", args.labelmap)

if __name__ == "__main__":
    main()
