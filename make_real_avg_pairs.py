import re, csv, sys
from pathlib import Path

def norm_freq_dir(name:str):
    """
    Παίρνει π.χ. '0280Hz_90.0db', '0280Hz_90.0db1' και επιστρέφει ('0280Hz','90.0db').
    Αγνοεί τυχόν ψηφία στο τέλος μετά το 'db'.
    """
    # πιάσε το βασικό μοτίβο
    m = re.match(r"(?P<hz>\d{4}Hz)_(?P<db>\d+\.0db)", name)
    if not m: return None
    return m.group("hz"), m.group("db")

def find_averaged(averaged_root:Path, hz:str, db:str):
    """
    Βρίσκει το averaged png: π.χ. <averaged_root>/<hz>_<db>.png
    Αν δεν υπάρχει, δοκιμάζει με μόνο Hz (fallback) – επιστρέφει None αν δεν βρεθεί.
    """
    base = f"{hz}_{db}.png"
    p = averaged_root / base
    if p.exists(): return p
    # fallback: πρώτο που ξεκινά με Hz_
    cand = list(averaged_root.glob(f"{hz}_*.png"))
    return cand[0] if cand else None

def material_of(rootname:str):
    return "wood" if rootname.lower().startswith("w") else "carbon"

def main():
    if len(sys.argv) < 3:
        print("Usage: python make_real_avg_pairs.py <DATA_ROOT> <OUT_CSV>")
        sys.exit(1)
    data_root = Path(sys.argv[1]); out_csv = Path(sys.argv[2])
    if not data_root.exists():
        print("Data root not found:", data_root); sys.exit(2)

    # φάκελοι averaged
    avg_wood = data_root / "wood_Averaged"
    avg_carbon = data_root / "carbon_Averaged"
    if not avg_wood.exists() or not avg_carbon.exists():
        print("Expected 'wood_Averaged' and 'carbon_Averaged' under", data_root); sys.exit(3)

    # index averaged ανά (instrument, hz, db)
    avg_index = {}  # (mat, inst, hz, db) -> path
    for mat, avg_root in [("wood", avg_wood), ("carbon", avg_carbon)]:
        for inst_dir in sorted(avg_root.glob("*_ESPI_90db-Averaged")):
            inst = inst_dir.name.split("_")[0]  # W01 ή C01...
            for f in inst_dir.glob("*.png"):
                m = re.match(r"(?P<hz>\d{4}Hz)_(?P<db>\d+\.0db)\.png$", f.name)
                if not m: continue
                key = (mat, inst, m.group("hz"), m.group("db"))
                avg_index[key] = f

    # σάρωση real
    real_roots = [
        data_root / "wood_real_A",
        data_root / "wood_real_B",
        data_root / "wood_real_C",
        data_root / "carbon_real_A",
        data_root / "carbon_real_B",
        data_root / "carbon_real_C",
    ]
    rows = []
    miss_exact = miss_all = 0

    for rr in real_roots:
        if not rr.exists(): continue
        for inst_dir in sorted(rr.glob("*_ESPI_90db")):
            inst = inst_dir.name.split("_")[0]  # W01/W02/W03 ή C01/C02/C03
            mat  = material_of(inst)
            # averaged αντίστοιχο instrument
            if mat == "wood":
                averaged_root = data_root / "wood_Averaged" / f"{inst}_ESPI_90db-Averaged"
            else:
                averaged_root = data_root / "carbon_Averaged" / f"{inst}_ESPI_90db-Averaged"

            for freq_dir in sorted(inst_dir.glob("*Hz_*db*")):
                nf = norm_freq_dir(freq_dir.name)
                if not nf: continue
                hz, db = nf
                # exact match από index (γρήγορο)
                clean = avg_index.get((mat, inst, hz, db))
                if clean is None:
                    # fallback: ψάξε μέσα στο averaged_root
                    clean = find_averaged(averaged_root, hz, db)
                    if clean is None:
                        miss_all += 1
                        continue
                    else:
                        miss_exact += 1

                # όλα τα frames μέσα στο frequency folder
                for noisy_png in sorted(freq_dir.glob("*.png")):
                    # frame id (π.χ. ..._00.png → 00)
                    m = re.search(r"_(\d+)\.png$", noisy_png.name)
                    frame_id = m.group(1) if m else ""
                    rows.append({
                        "noisy":   str(noisy_png),
                        "clean":   str(clean),
                        "material": mat,
                        "instrument": inst,   # W01/C02...
                        "hz": hz, "db": db,
                        "frame_id": frame_id,
                        "real_root": str(rr)
                    })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
            ["noisy","clean","material","instrument","hz","db","frame_id","real_root"])
        wr.writeheader()
        for r in rows: wr.writerow(r)

    print(f"[OK] pairs written: {len(rows)} → {out_csv}")
    print(f"[INFO] avg exact-miss: {miss_exact} (found via fallback by Hz)")
    print(f"[INFO] avg total-miss: {miss_all} (no averaged found)")

if __name__ == "__main__":
    main()
