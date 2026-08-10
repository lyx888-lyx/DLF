param(
    [switch]$Overwrite,
    [switch]$AcknowledgeTestInference
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/adpep-hybrid-missing-benchmark-v1"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

if (-not $AcknowledgeTestInference) {
    throw "This recovery performs frozen-checkpoint valid/test inference. Re-run with -AcknowledgeTestInference. It does NOT train or select checkpoints."
}

Write-Host ""
Write-Host "================================================================"
Write-Host " REBUILD HISTORICAL ADPEP ARTIFACTS FROM FROZEN CHECKPOINTS"
Write-Host " INFERENCE ONLY / NO TRAINING / NO CHECKPOINT SELECTION"
Write-Host " OFFICIAL TEST IS ALREADY EXPLORATORY-HISTORICALLY ACCESSED"
Write-Host "================================================================"
Write-Host ""

$checkpoints = @(
    ".\pt\missing_baseline\cf_compat_kd_v1\DLF_mosi_seed1111_best_valid.pth",
    ".\pt\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed1112\DLF_mosi_seed1112_best_valid.pth",
    ".\pt\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed1113\DLF_mosi_seed1113_best_valid.pth",
    ".\pt\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed1114\DLF_mosi_seed1114_best_valid.pth",
    ".\pt\missing_baseline\cf_compat_kd_v1\benchmark_multiseed\seed1115\DLF_mosi_seed1115_best_valid.pth"
)
foreach ($path in $checkpoints) {
    if (-not (Test-Path $path)) {
        throw "Required frozen validation-best checkpoint missing: $path"
    }
    if ($path -match "diagnostic|best_test|smoke") {
        throw "Forbidden checkpoint path: $path"
    }
}

$recentProbe = ".\result\missing_baseline\cfcompat_exploratory_test_viability_v13\mosi\exploratory_test\seed1113\exploratory_test_viability_v13_comparison.csv"
if (-not (Test-Path $recentProbe)) {
    Write-Warning "Recent exploratory probe aggregate is absent. Historical PE5 replay will still be mandatory; seed1113 recent-Original cross-check will be skipped."
}

python -m py_compile `
    .\rebuild_adpep_frozen_artifacts_from_cfcompat_checkpoints.py `
    .\eval_anchor_decision_preserving_ensemble.py `
    .\aggregate_anchor_decision_preserving_ensemble.py
if ($LASTEXITCODE -ne 0) {
    throw "py_compile failed with exit code $LASTEXITCODE"
}

$stage9Root = ".\result\missing_baseline\cfcompat_prediction_ensemble_v1\mosi"
$stage9BRoot = ".\result\missing_baseline\anchor_decision_preserving_ensemble_v1\mosi"

if ((Test-Path $stage9Root) -and (-not $Overwrite)) {
    throw "Recovered Stage9A directory already exists: $stage9Root. Inspect it or rerun with -Overwrite."
}
if ((Test-Path $stage9BRoot) -and (-not $Overwrite)) {
    throw "ADPEP Stage9B directory already exists: $stage9BRoot. Inspect it or rerun with -Overwrite."
}
if ($Overwrite -and (Test-Path $stage9BRoot)) {
    Remove-Item $stage9BRoot -Recurse -Force
}

Write-Host ""
Write-Host "================ Step 1/3: Recover Stage9A predictions ========="
$recoverArgs = @(
    ".\rebuild_adpep_frozen_artifacts_from_cfcompat_checkpoints.py",
    "--dataset", "mosi",
    "--num-workers", "0"
)
if ($Overwrite) {
    $recoverArgs += "--overwrite"
}
python @recoverArgs
if ($LASTEXITCODE -ne 0) {
    throw "Stage9A recovery failed with exit code $LASTEXITCODE. ADPEP projection was NOT run."
}

Write-Host ""
Write-Host "================ Step 2/3: Frozen label-free ADPEP projection =="
python .\eval_anchor_decision_preserving_ensemble.py --dataset mosi
if ($LASTEXITCODE -ne 0) {
    throw "Stage9B ADPEP projection failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "================ Step 3/3: Original Stage9B aggregate audit ===="
python .\aggregate_anchor_decision_preserving_ensemble.py `
    --dataset mosi `
    --verify-decisions `
    --compute-retention
if ($LASTEXITCODE -ne 0) {
    throw "Stage9B aggregate audit failed with exit code $LASTEXITCODE"
}

$requiredOutputs = @(
    (Join-Path $stage9Root "individual_predictions_manifest.json"),
    (Join-Path $stage9Root "ensemble_predictions_test.csv"),
    (Join-Path $stage9Root "individual_model_metrics.csv"),
    (Join-Path $stage9BRoot "output_prediction_manifest.json"),
    (Join-Path $stage9BRoot "adpep_all_predictions_test.csv"),
    (Join-Path $stage9BRoot "adpep_metrics.csv"),
    (Join-Path $stage9BRoot "stage9b_adpep_final_audit.md")
)
foreach ($path in $requiredOutputs) {
    if (-not (Test-Path $path)) {
        throw "Recovery completed but required output is missing: $path"
    }
}

Write-Host ""
Write-Host "================ Recovery complete ============================="
Write-Host "Stage9A root: $stage9Root"
Write-Host "Stage9B root: $stage9BRoot"
Write-Host ""
Write-Host "ADPEP final audit:"
Get-Content (Join-Path $stage9BRoot "stage9b_adpep_final_audit.md")
Write-Host ""
Write-Host "No training performed: True"
Write-Host "No checkpoint selected on Test: True"
Write-Host "Diagnostic/best-Test checkpoints used: False"
Write-Host "Frozen valid-best checkpoints used: 5"
Write-Host ""
Write-Host "Next command:"
Write-Host ".\scripts\run_windows_adpep_hybrid_missing_benchmark_v1.ps1"
