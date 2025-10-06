import argparse, csv, re
from pathlib import Path

PAT_FREQ = re.compile(r"^(\d{4})Hz_(\d+(?:\.\d+)?)db$", re.I)
PAT_INSTR = re.compile(r"([WC]\d{2})_ESPI_90db", re.I)

def find_instr(parts):
    # Επιστρέφει π.χ. ("W01","wood") ή ("C03","carbon"), αλλιώς (None,None)
    joined = "/".join(parts)
    m = PAT_INSTR.search(joined)
    if not m: return None, None
    instr = m.group(1).upper()
    material = "wood" if instr.startswith("W") else "carbon"
    return instr, material

def find_session(parts):
    # ψάχνει wood_real_A/B/C ή carbon_real_A/B/C
    for p in parts:
        if p.lower().endswith("_real_a"): return "A"
        if p.lower().endswith("_real_b"): return "B"
        if p.lower().endswith("_real_c"): return "C"
    return "UNK"

def find_avg_root(data_root: Path, material: str, instr: str) -> Path:
    # π.χ. wood_Averaged\W01_ESPI_90db-Averaged
    top = data_root / (f"{material}_Averaged")
    cand = top / f"{instr}_ESPI_90db-Averaged"
    return cand if cand.exists() else None

def pick_avg_file(avg_root: Path, freq_dir_name: str) -> Path|None:
    # Προτίμηση exact match, αλλιώς οποιοδήποτε που ξεκινά με freq_dir_name
    exact = avg_root / f"{freq_dir_name}.png"
    if exact.exists(): return exact
    for p in (avg_root).glob(f"{freq_dir_name}*.png"):
        return p
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="π.χ. C:\\ESPI\\data")
    ap.add_argument("--out-csv", required=True, help="που θα γραφτεί το pairs CSV")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    # Βρες όλους τους φακέλους συχνότητας κάτω από *_real_A/B/C
    real_roots = []
    for m in ("wood", "carbon"):
        for s in ("A","B","C"):
            real_roots += list((data_root / f"{m}_real_{s}").glob("*_ESPI_90db"))

    rows = []
    misses = []
    for rr in real_roots:
        # rr = ...\W01_ESPI_90db
        instr, material = find_instr(rr.parts)
        session = find_session(rr.parts)
        if not instr:
            continue
        for freq_dir in rr.iterdir():
            if not freq_dir.is_dir(): continue
            m = PAT_FREQ.match(freq_dir.name)
            if not m: 
                continue
            hz = int(m.group(1))
