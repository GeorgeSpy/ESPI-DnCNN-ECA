import argparse, os, re, csv
from pathlib import Path

def key_from_name(n):
    # πχ "0125Hz_90.0db.png" -> ("0125","90.0")
    m = re.match(r"^(\d{4})Hz_(\d+(?:\.\d+)?)db", n, re.I)
    return m.group(1), m.group(2) if m else (None,None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noisy-root", required=True)
    ap.add_argument("--avg-root", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    avg_map = {}
    for p in Path(args.avg_root).rglob("*.png"):
        k = key_from_name(p.stem)
        if k[0] is None: continue
        avg_map[k] = str(p)

    rows = []
    for p in Path(args.noisy_root).rglob("*.png"):
        k = key_from_name(p.stem)
        if k[0] is None: continue
        if k in avg_map:
            rows.append((str(p), avg_map[k]))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f); wr.writerow(["noisy","clean"])
        wr.writerows(rows)
    print(f"[OK] wrote {len(rows)} pairs to {args.out_csv}")

if __name__ == "__main__":
    main()
