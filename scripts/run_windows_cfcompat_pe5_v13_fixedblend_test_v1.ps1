param(
    [switch]$AcknowledgeExploratoryContaminatedTest,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/cfcompat-pe5-v13-fixedblend-test-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

if (-not $AcknowledgeExploratoryContaminatedTest) {
    throw "This Test is exploratory/contaminated and the 0.5/0.5 weight is frozen. Re-run with -AcknowledgeExploratoryContaminatedTest."
}

Write-Host ""
Write-Host "======================================================================"
Write-Host " CFCompat Raw5-PE5 + v13 FIXED 0.5/0.5 TEST TRANSFER CHECK"
Write-Host " EXPLORATORY / TEST ALREADY HISTORICALLY ACCESSED"
Write-Host " NO TEST WEIGHT SEARCH / AGGREGATE OUTPUT ONLY"
Write-Host "======================================================================"
Write-Host ""

$rawRoot = ".\result\missing_baseline\cfcompat_prediction_ensemble_v1\mosi"
$missing = @()
foreach ($seed in @(1111,1112,1113,1114,1115)) {
    $path = Join-Path $rawRoot ("online_seed{0}_test_predictions.csv" -f $seed)
    if (-not (Test-Path $path)) {
        $missing += $path
    }
}
if ($missing.Count -gt 0) {
    Write-Host "Missing recovered Raw5 Test members:"
    $missing | ForEach-Object { Write-Host "  $_" }
    throw "Raw5 frozen Test prediction set is incomplete."
}

$historicalV13 = ".\result\missing_baseline\cfcompat_exploratory_test_viability_v13\mosi\exploratory_test\seed1113\exploratory_test_viability_v13_comparison.csv"
if (-not (Test-Path $historicalV13)) {
    throw "Historical v13 aggregate replay source is missing: $historicalV13"
}

python -m py_compile .\evaluate_cfcompat_pe5_v13_fixedblend_test_v1.py
if ($LASTEXITCODE -ne 0) {
    throw "Fixed-blend evaluator py_compile failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\evaluate_cfcompat_pe5_v13_fixedblend_test_v1.py",
    "--dataset", "mosi",
    "--num-workers", "1",
    "--acknowledge-exploratory-contaminated-test"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Fixed 0.5/0.5 Test transfer check failed with exit code $LASTEXITCODE"
}

$root = ".\result\missing_baseline\cfcompat_pe5_v13_fixedblend_test_v1\mosi\exploratory_test_fixed_blend"
$comparison = Join-Path $root "fixed_blend_test_comparison.csv"
$summary = Join-Path $root "fixed_blend_test_summary.json"

if (-not (Test-Path $comparison)) {
    throw "Comparison output missing: $comparison"
}
if (-not (Test-Path $summary)) {
    throw "Summary output missing: $summary"
}

Write-Host ""
Write-Host "================ Fixed-blend aggregate Test comparison ==============="
Import-Csv $comparison |
    Select-Object Method, TestJ, LAV_MAE, MissingMacroMAE, LA_MAE, LV_MAE, L_MAE, DeltaJVsRaw5, DeltaLAVMAEVsRaw5, DeltaMissingMacroMAEVsRaw5 |
    Format-Table -AutoSize

$data = Get-Content $summary -Raw | ConvertFrom-Json
Write-Host ""
Write-Host "================ Frozen protocol ====================================="
Write-Host ("Valid Raw5 J:              {0}" -f $data.frozen_valid_evidence.raw5_J)
Write-Host ("Valid v13 J:               {0}" -f $data.frozen_valid_evidence.v13_J)
Write-Host ("Valid fixed 50/50 J:       {0}" -f $data.frozen_valid_evidence.fixed_50_50_J)
Write-Host ("Test Raw5 J:               {0}" -f $data.test_metrics.raw5_pe5.TestJ)
Write-Host ("Test v13 J:                {0}" -f $data.test_metrics.adam_step_safety_v13.TestJ)
Write-Host ("Test fixed 50/50 J:        {0}" -f $data.test_metrics.fixed_blend.TestJ)
Write-Host ("Delta J blend vs Raw5:     {0}" -f $data.test_deltas_fixed_blend_vs_raw5.delta_J)
Write-Host ("Delta LAV MAE vs Raw5:     {0}" -f $data.test_deltas_fixed_blend_vs_raw5.delta_LAV_MAE)
Write-Host ("Delta MissingMacro vs Raw5:{0}" -f $data.test_deltas_fixed_blend_vs_raw5.delta_MissingMacro_MAE)
Write-Host ("Verdict:                   {0}" -f $data.verdict)
Write-Host ""
Write-Host "WEIGHT SEARCH ON TEST: False"
Write-Host "ONLY TESTED WEIGHT: Raw5=0.5, v13=0.5"
Write-Host "NEW V13 TEST MODEL FORWARD COUNT: 1"
Write-Host "SAMPLE-LEVEL TEST OUTPUT WRITTEN: False"
Write-Host "Result root: $root"
