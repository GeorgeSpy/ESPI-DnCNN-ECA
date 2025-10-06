import argparse, re, csv
from pathlib import Path

def parse_name(name:str):
    # Handle both _full.png and .png suffixes
    m = re.match(r'(?P<hz>\d{4}Hz)_(?P<db>\d+\.0db)(?:_full)?\.png$', name)
    if not m: return None, None
    return m.group("hz"), m.group("db")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pseudo-root", action="append", required=True,
                    help="φάκελος με pseudo-noisy PNGs (δώσε πολλαπλές φορές για περισσότερους φακέλους)")
    ap.add_argument("--avg-root", required=True,
                    help="αντίστοιχος averaged φάκελος (π.χ. C:\\ESPI\\data\\wood_Averaged\\W01_ESPI_90db-Averaged)")
    ap.add_argument("--instrument", default="W01")
    ap.add_argument("--material",   default="wood")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    avg_root = Path(args.avg_root)
    if not avg_root.exists():
        raise SystemExit(f"[ERR] avg-root not found: {avg_root}")

    rows = []; total = miss = 0
    for pr in args.pseudo_root:
        pr = Path(pr)
        if not pr.exists():
            print(f"[WARN] pseudo-root not found: {pr} (skip)")
            continue
        for f in pr.glob("*.png"):
            total += 1
            hz, db = parse_name(f.name)
            if not hz:
                miss += 1; continue
            clean = avg_root / f"{hz}_{db}.png"
            if not clean.exists():
                # fallback: πρώτο που ταιριάζει με Hz_
                cands = list(avg_root.glob(f"{hz}_*.png"))
                if not cands:
                    miss += 1; continue
                clean = cands[0]
            rows.append({
                "noisy": str(f),
                "clean": str(clean),
                "material": args.material,
                "instrument": args.instrument,
                "hz": hz, "db": db,
                "frame_id": "",
                "real_root": "",
                "source": "pseudo"
            })

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as g:
        wr = csv.DictWriter(g, fieldnames=list(rows[0].keys()) if rows else
                             ["noisy","clean","material","instrument","hz","db","frame_id","real_root","source"])
        wr.writeheader()
        for r in rows: wr.writerow(r)

    print(f"[OK] pseudo pairs: {len(rows)} / seen {total} -> {out}")
    print(f"[INFO] misses (no matching averaged): {miss}")

if __name__ == "__main__":
    main()
