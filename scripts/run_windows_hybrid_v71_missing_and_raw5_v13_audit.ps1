param(
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/adpep-hybrid-missing-benchmark-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

Write-Host ""
Write-Host "======================================================================"
Write-Host " HYBRID V7.1 EXACT RECONSTRUCTION + RAW5/V13 VALID COMPLEMENTARITY"
Write-Host " RAW5 FROZEN PREDICTIONS ONLY / NO TRAINING / NO NEW TEST FORWARD"
Write-Host "======================================================================"
Write-Host ""

$rawRoot = ".\result\missing_baseline\cfcompat_prediction_ensemble_v1\mosi"
$missing = @()
foreach ($seed in @(1111,1112,1113,1114,1115)) {
    foreach ($split in @("valid","test")) {
        $path = Join-Path $rawRoot ("online_seed{0}_{1}_predictions.csv" -f $seed,$split)
        if (-not (Test-Path $path)) {
            $missing += $path
        }
    }
}
if ($missing.Count -gt 0) {
    Write-Host "The previous five-checkpoint recovery did not leave all raw prediction CSVs."
    Write-Host "Missing:"
    $missing | ForEach-Object { Write-Host "  $_" }
    throw "Raw5 frozen prediction set is incomplete. Do not run Hybrid reconstruction yet."
}

$v13Raw = ".\result\missing_baseline\cfcompat_adam_step_safety_v13\mosi\valid_screen\seed1113_dev\adam_step_safety_v13_candidate_raw_valid_events.csv"
if (-not (Test-Path $v13Raw)) {
    Write-Warning "v13 raw Valid events are absent. Hybrid reconstruction will still run; Raw5/v13 blend audit will be skipped."
}

python -m py_compile .\reconstruct_hybrid_v71_missing_and_raw5_v13_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "py_compile failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\reconstruct_hybrid_v71_missing_and_raw5_v13_audit.py",
    "--dataset", "mosi"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Hybrid reconstruction / Valid complementarity audit failed with exit code $LASTEXITCODE"
}

$root = ".\result\missing_baseline\hybrid_v71_missing_reconstruction_v1\mosi"
$summary = Join-Path $root "hybrid_v71_missing_reconstruction_summary.json"
$metrics = Join-Path $root "hybrid_raw5_v13_valid_metrics.csv"
$blend = Join-Path $root "raw5_v13_valid_blend_grid.csv"
$corr = Join-Path $root "raw5_v13_valid_error_correlation.csv"

if (-not (Test-Path $summary)) {
    throw "Summary missing: $summary"
}

Write-Host ""
Write-Host "================ Hybrid / Raw5 / v13 metrics ========================="
Import-Csv $metrics |
    Where-Object { $_.Mode -in @("LAV","MissingMacro") } |
    Select-Object Method, Split, Mode, J, MAE, Corr, acc_2, F1_score |
    Format-Table -AutoSize

if (Test-Path $blend) {
    Write-Host ""
    Write-Host "================ Raw5 + v13 Valid-only blend grid ===================="
    Import-Csv $blend |
        Select-Object Raw5Weight, V13Weight, ValidJ, LAV_MAE, MissingMacroMAE, LA_MAE, LV_MAE, L_MAE |
        Format-Table -AutoSize

    Write-Host ""
    Write-Host "================ Raw5 / v13 error complementarity ==================="
    Import-Csv $corr | Format-Table -AutoSize
}

$data = Get-Content $summary -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "================ Frozen historical replay ============================"
Write-Host ("Base teacher seed:                {0}" -f $data.historical_reconstruction.base_teacher_seed)
Write-Host ("Committee selected:               {0}" -f $data.historical_reconstruction.committee_selected)
Write-Host ("Hybrid beta:                      {0}" -f $data.historical_reconstruction.hybrid_valid_selected.beta)
Write-Host ("Hybrid valid MAE replay:          {0}" -f $data.historical_reconstruction.hybrid_valid_selected.mae)
Write-Host ("Historical Test LAV replay pass:  {0}" -f $data.historical_reconstruction.test_lav_replay_passed)
Write-Host ("Hybrid missing Test J:            {0}" -f $data.hybrid_missing_metrics.test.J)
Write-Host ("Hybrid Test MissingMacro MAE:     {0}" -f $data.hybrid_missing_metrics.test.MissingMacro.MAE)
Write-Host ("Raw5 PE5 Valid J:                 {0}" -f $data.raw5_valid.J)
if ($null -ne $data.blend_valid) {
    Write-Host ("50/50 Raw5-v13 Valid J:           {0}" -f $data.blend_valid.fixed_50_50_row.ValidJ)
    Write-Host ("Best grid Raw5 weight:            {0}" -f $data.blend_valid.best_grid_row.Raw5Weight)
    Write-Host ("Best grid Valid J:                {0}" -f $data.blend_valid.best_grid_row.ValidJ)
    Write-Host ("Blend improves best single:       {0}" -f $data.blend_valid.improves_best_single)
}

Write-Host ""
Write-Host "NO TRAINING: True"
Write-Host "NEW TEST MODEL FORWARD: False"
Write-Host "BLEND TEST EVALUATED: False"
Write-Host "SAMPLE-LEVEL TEST OUTPUT WRITTEN: False"
Write-Host "Result root: $root"
