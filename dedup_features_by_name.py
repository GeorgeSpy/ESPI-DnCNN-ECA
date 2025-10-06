import csv, argparse
import numpy as np
from collections import Counter
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()

    sums = {}
    cnts = Counter()
    with open(args.inp, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = next(rd)
        for row in rd:
            name = row[0]
            vec = np.array([float(x) for x in row[1:]], np.float64)
            sums[name] = sums.get(name, 0.0) + vec
            cnts[name] += 1

    outp = Path(args.outp)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        for name in sorted(sums.keys()):
            mean_vec = (sums[name] / cnts[name]).astype(float)
            wr.writerow([name] + [f"{x:.6g}" for x in mean_vec])

    print(f"[OK] Dedup wrote {len(sums)} unique rows to {outp} (from {sum(cnts.values())} rows)")

if __name__ == "__main__":
    main()
