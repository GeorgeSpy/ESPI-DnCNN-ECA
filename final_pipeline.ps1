# Final Pipeline - Execute after phase/QC/IRLS/Features complete
# Run this when harvesting is done for all datasets

Write-Host "=== FINAL PIPELINE FOR THESIS RESULTS ===" -ForegroundColor Cyan

$VENV = "C:\ESPI_VENV2\Scripts\python.exe"
$ROOTS = @(
    "C:\ESPI_TEMP\GPU_FULL2\W01_PhaseOut_b18_cs16_ff100",
    "C:\ESPI_TEMP\GPU_FULL2\W02_PhaseOut_b18_cs16_ff100",
    "C:\ESPI_TEMP\GPU_FULL2\W03_PhaseOut_b18_cs16_ff100"
)
$FEAT_DIR = "C:\ESPI_TEMP\features"

# Create features dir
if (!(Test-Path $FEAT_DIR)) { New-Item -ItemType Directory -Path $FEAT_DIR | Out-Null }

Write-Host "`n[1/4] Merging all features..." -ForegroundColor Yellow
& $VENV C:\ESPI_DnCNN\merge_all_features.py --roots $ROOTS --out "$FEAT_DIR\all_features_QCpass.csv"

Write-Host "`n[2/4] Creating labels..." -ForegroundColor Yellow
& $VENV C:\ESPI_DnCNN\create_labels_corrected.py --in "$FEAT_DIR\all_features_QCpass.csv" --out "$FEAT_DIR\all_features_labeled.csv"

Write-Host "`n[3/4] Deduplicating features..." -ForegroundColor Yellow
& $VENV C:\ESPI_DnCNN\simple_dedup.py --in "$FEAT_DIR\all_features_labeled.csv" --out "$FEAT_DIR\all_features_labeled_dedup.csv"

Write-Host "`n[4/4] Training final RF model..." -ForegroundColor Yellow
& $VENV C:\ESPI_DnCNN\hierarchical_rf_classifier.py --in "$FEAT_DIR\all_features_labeled_dedup.csv" --outdir "$FEAT_DIR\rf_final"

Write-Host "`n=== PIPELINE COMPLETE ===" -ForegroundColor Green
Write-Host "Results saved to: $FEAT_DIR\rf_final" -ForegroundColor Green

