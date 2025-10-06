import csv, argparse
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument("--pred", required=True)   # predictions.csv
ap.add_argument("--assign-base", required=True)  # π.χ. cluster_assignments_merged.csv
ap.add_argument("--out-assign", required=True)   # merged_out.csv
ap.add_argument("--thr", type=float, default=0.90)
args=ap.parse_args()

def maxprob(row):
    probs=[float(x) for x in row[2:]] if len(row)>2 else []
    return max(probs) if probs else 1.0

# διάβασε υπάρχοντες
base={}
with open(args.assign_base, newline="", encoding="utf-8") as f:
    rd=csv.reader(f); header=next(rd)
    for name,c in rd: base[name]=int(c)

added=0
with open(args.pred, newline="", encoding="utf-8") as f:
    rd=csv.reader(f); header=next(rd)
    for row in rd:
        name=row[0]; c=int(row[1])
        if name in base: continue
        if maxprob(row) >= args.thr:
            base[name]=c; added+=1

outp=Path(args.out_assign); outp.parent.mkdir(parents=True, exist_ok=True)
with open(outp, "w", newline="", encoding="utf-8") as g:
    wr=csv.writer(g); wr.writerow(["name","cluster"])
    for k in sorted(base.keys()):
        wr.writerow([k, base[k]])

print(f"[OK] wrote {outp}  (added={added} new confident labels, thr={args.thr})")
