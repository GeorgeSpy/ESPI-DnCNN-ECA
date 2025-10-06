import argparse, pandas as pd, numpy as np
from pathlib import Path

def read_auto(p):
    return pd.read_csv(p, sep=None, engine="python")

def ensure_columns(df, all_cols):
    for c in all_cols:
        if c not in df.columns:
            df[c] = ""
    return df[all_cols]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real",   required=True)
    ap.add_argument("--pseudo", required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--real-frac", type=float, default=0.8,
                    help="στόχος κλάσμα real στο τελικό set (default 0.8)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--oversample-pseudo", dest="oversample_pseudo", action="store_true",
                    help="αν δεν υπάρχουν αρκετά pseudo, κάνε oversample με επανάληψη για να πετύχεις target ratio")
    args = ap.parse_args()

    R = read_auto(args.real)
    P = read_auto(args.pseudo)

    must = ["noisy","clean"]
    for m in must:
        if m not in R.columns: raise ValueError(f"Real CSV: λείπει στήλη '{m}'")
        if m not in P.columns: raise ValueError(f"Pseudo CSV: λείπει στήλη '{m}'")

    preferred_order = ["noisy","clean","material","instrument","hz","db","frame_id","real_root","source"]
    all_cols = list(dict.fromkeys(preferred_order + list(R.columns) + list(P.columns)))

    R = ensure_columns(R.copy(), all_cols)
    P = ensure_columns(P.copy(), all_cols)
    if "source" not in R.columns: R["source"] = "real"
    else: R.loc[R["source"].eq(""), "source"] = "real"
    if "source" not in P.columns: P["source"] = "pseudo"
    else: P.loc[P["source"].eq(""), "source"] = "pseudo"

    R = R.drop_duplicates(subset=["noisy","clean"]).reset_index(drop=True)
    P = P.drop_duplicates(subset=["noisy","clean"]).reset_index(drop=True)

    nR, nP = len(R), len(P)
    if nR == 0: raise ValueError("Real CSV άδειο μετά το καθάρισμα.")
    if nP == 0: raise ValueError("Pseudo CSV άδειο μετά το καθάρισμα.")

    # θέλουμε: nR / (nR + nPsel) = real_frac  => nPsel = nR*(1-real_frac)/real_frac
    nP_needed = int(round(nR * (1.0 - args.real_frac) / args.real_frac))

    if args.oversample_pseudo and nP_needed > nP:
        # oversample με replacement για να φτάσουμε nP_needed
        reps = nP_needed // nP
        rem  = nP_needed %  nP
        P_aug = pd.concat([P] * reps + [P.sample(rem, random_state=args.seed, replace=True)], axis=0)
        nPsel = len(P_aug)
        Psel  = P_aug
    else:
        nPsel = min(nP, nP_needed)
        Psel  = P.sample(n=nPsel, random_state=args.seed) if nPsel < nP else P

    M = pd.concat([R, Psel], axis=0).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    M.to_csv(out, index=False)

    real_final = (M["source"]=="real").sum()
    pseudo_final = (M["source"]=="pseudo").sum()
    frac_real = real_final / len(M)
    print(f"[OK] mix written -> {out}  (rows={len(M)})")
    print(f"[INFO] used: real={nR}, pseudo_final={pseudo_final} (available pseudo={nP}, need={nP_needed}, oversample={'YES' if (args.oversample_pseudo and nP_needed>nP) else 'NO'})")
    print(f"[INFO] final composition: real={real_final}, pseudo={pseudo_final}, real_frac={frac_real:.3f}")
    if "instrument" in M.columns:
        br = M.groupby(["source","instrument"]).size().reset_index(name="count")
        print("[INFO] by source/instrument:\n", br.to_string(index=False))

if __name__ == "__main__":
    main()
