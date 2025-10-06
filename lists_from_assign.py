import csv, os, collections

assign = r"C:\ESPI_TEMP\features\cluster_assignments_merged.csv"  # ή το αρχικό αν δεν κάνεις merge
outdir = r"C:\ESPI_TEMP\features\per_cluster_lists"
os.makedirs(outdir, exist_ok=True)

groups = collections.defaultdict(list)
with open(assign, newline="", encoding="utf-8") as f:
    rd = csv.reader(f); next(rd)
    for name, c in rd:
        groups[int(c)].append(name)

for c, lst in groups.items():
    with open(os.path.join(outdir, f"cluster_{c}.txt"), "w", encoding="utf-8") as g:
        g.write("\n".join(sorted(lst)))

print("[OK] wrote lists to", outdir)
