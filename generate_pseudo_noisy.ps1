# Generate pseudo-noisy (profiles × seeds × instruments)
# Uses patched make_pseudo_noisy_plus.py with --profile and --bitdepth 16

# === Paths ===
$AvgW01 = "C:\ESPI\data\wood_Averaged\W01_ESPI_90db-Averaged"
$AvgW02 = "C:\ESPI\data\wood_Averaged\W02_ESPI_90db-Averaged"
$AvgW03 = "C:\ESPI\data\wood_Averaged\W03_ESPI_90db-Averaged"

$PseudoRoot   = "C:\ESPI_TEMP\pseudo"
$MetricsRoot  = "C:\ESPI_TEMP\pseudo\metrics"
New-Item -ItemType Directory -Force -Path $PseudoRoot,$MetricsRoot | Out-Null

# === Profiles & Seeds ===
$profiles = @("lite","mid","heavy")
$seeds    = 1..3
$jobs = @(
  @{inst="W01"; avg=$AvgW01},
  @{inst="W02"; avg=$AvgW02},
  @{inst="W03"; avg=$AvgW03}
)

# === Generate ===
foreach ($job in $jobs) {
  foreach ($p in $profiles) {
    foreach ($s in $seeds) {
      $out = Join-Path $PseudoRoot ("{0}\Pseudo_{1}_s{2}" -f $job.inst,$p,$s)
      New-Item -ItemType Directory -Force -Path $out | Out-Null
      $mcsv = Join-Path $MetricsRoot ("{0}_{1}_s{2}.csv" -f $job.inst,$p,$s)

      Write-Host "Generating: $($job.inst) - $p - seed $s"
      C:\ESPI_VENV2\Scripts\python.exe C:\ESPI_DnCNN\make_pseudo_noisy_plus.py `
        --input    $job.avg `
        --output   $out `
        --profile  $p `
        --seed     $s `
        --bitdepth 16 `
        --export-metrics $mcsv
    }
  }
}

Write-Host "Pseudo-noisy generation completed!"
Write-Host "Output directory: $PseudoRoot"
Write-Host "Metrics directory: $MetricsRoot"
