import csv

inp = r"C:\ESPI_TEMP\features\cluster_assignments.csv"
out = r"C:\ESPI_TEMP\features\cluster_assignments_merged.csv"

# αντιστοίχιση που θες: 2 -> 3 (άλλαξε εδώ αν προτιμάς άλλο merge)
merge_map = {2: 3}

with open(inp, newline="", encoding="utf-8") as f, open(out, "w", newline="", encoding="utf-8") as g:
    rd = csv.reader(f); wr = csv.writer(g)
    header = next(rd); wr.writerow(header)
    for name, c in rd:
        c = int(c)
        c = merge_map.get(c, c)
        wr.writerow([name, c])

print("[OK] wrote merged labels to", out)
