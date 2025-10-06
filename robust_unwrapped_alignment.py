import numpy as np, json, os
from pathlib import Path
from PIL import Image

TEST_ROOT = Path(r"C:\ESPI_TEMP\SMOKE\W01_PhaseOut_DENFT_b18_cs16_ff100_0050_FULL")
REF_ROOT  = Path(r"C:\ESPI_TEMP\SMOKE\W01_PhaseRef0050_b18_cs16_ff100")
ROI_PATH  = Path(r"C:\ESPI_TEMP\roi_mask.png")
BASENAME  = r"0050Hz_90.0db"
OUT_DIR   = Path(r"C:\ESPI_TEMP\SMOKE\W01_PhaseOut_DENFT_b18_cs16_ff100_0050_FULL\qc_align"); OUT_DIR.mkdir(parents=True, exist_ok=True)

pi = np.pi

def load_npy(root, sub, name):
    p = root / sub / f"{name}.npy"
    a = np.load(p).astype(np.float32)
    return a

def load_roi(png_path):
    m = np.array(Image.open(png_path))
    if m.ndim == 3:  # RGB → grayscale
        m = m[...,0]
    m = m > 0
    return m

def rmse(x, roi):
    d = x[roi]
    return float(np.sqrt(np.mean(d*d)))

def pct_gt(x, thr, roi):
    d = x[roi]
    return float(100.0*np.mean(np.abs(d) > thr))

def fit_plane(res, roi):
    H,W = res.shape
    yy,xx = np.mgrid[0:H,0:W]
    X = np.stack([xx[roi], yy[roi], np.ones(np.count_nonzero(roi), dtype=np.float32)], axis=1)
    y = res[roi][:,None]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)  # a*x + b*y + c
    a,b,c = coef.ravel().astype(np.float32)
    plane = (a*xx + b*yy + c).astype(np.float32)
    return plane

def unwrap2d(a):
    # 2x 1D unwrap (cols then rows): γρήγορο & αξιόπιστο στις ESPI φάσεις
    a1 = np.unwrap(a, axis=1)
    a2 = np.unwrap(a1, axis=0)
    return a2.astype(np.float32)

roi = load_roi(ROI_PATH)

# --- Load wrapped/unwrapped ---
w_ref  = load_npy(REF_ROOT,  "phase_wrapped_npy",   BASENAME)
w_test = load_npy(TEST_ROOT, "phase_wrapped_npy",   BASENAME)
u_ref  = load_npy(REF_ROOT,  "phase_unwrapped_npy", BASENAME)
u_test = load_npy(TEST_ROOT, "phase_unwrapped_npy", BASENAME)

results = {}

# === A) Align σε ήδη-unwrapped ===
bestA = None
for sgn in (+1.0, -1.0):
    diff = (sgn*u_test - u_ref)
    # integer 2π offset (robust μέσω median)
    k = np.round(np.median(diff[roi])/(2*pi))
    diff -= (2*pi*k)
    # προαιρετικό plane (piston/tip/tilt)
    plane = fit_plane(diff, roi)
    diff2 = diff - plane
    metrics = dict(
        rmse = rmse(diff2, roi),
        pct_gt_pi2 = pct_gt(diff2, pi/2, roi),
        pct_gt_pi4 = pct_gt(diff2, pi/4, roi),
        sgn = int(sgn), k = int(k),
    )
    if (bestA is None) or (metrics["rmse"] < bestA["metrics"]["rmse"]):
        bestA = dict(metrics=metrics, diff=diff2, plane=plane)

results["A_align_unwrapped"] = bestA["metrics"]

np.save(OUT_DIR/"unwrapped_align_diff_A.npy", bestA["diff"])
np.save(OUT_DIR/"unwrapped_align_plane_A.npy", bestA["plane"])

# === B) Unwrap της wrapped διαφοράς ===
# δ = angle(exp(j*(w_test - w_ref))) ∈ (-π,π], μετά 2D unwrap
d_wrapped = np.angle(np.exp(1j*(w_test - w_ref))).astype(np.float32)
d_unw = unwrap2d(d_wrapped)
# επιτρέπουμε plane removal
planeB = fit_plane(d_unw, roi)
d_unw2 = d_unw - planeB
metricsB = dict(
    rmse = rmse(d_unw2, roi),
    pct_gt_pi2 = pct_gt(d_unw2, pi/2, roi),
    pct_gt_pi4 = pct_gt(d_unw2, pi/4, roi),
)
results["B_unwrap_wrapped_diff"] = metricsB
np.save(OUT_DIR/"unwrap_wrapped_diff_B.npy", d_unw2)
np.save(OUT_DIR/"unwrap_wrapped_plane_B.npy", planeB)

# --- pick best ---
pick = "A_align_unwrapped" if results["A_align_unwrapped"]["rmse"] <= results["B_unwrap_wrapped_diff"]["rmse"] else "B_unwrap_wrapped_diff"
results["_best"] = pick

with open(OUT_DIR/"qc_unwrapped_metrics.json","w",encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("[OK] wrote metrics ->", OUT_DIR/"qc_unwrapped_metrics.json", "| best:", pick)
