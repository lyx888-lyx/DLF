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
Write-Host "==============================================================="
Write-Host " ADPEP / HYBRID MISSING BENCHMARK V1"
Write-Host " OFFLINE FROZEN TEST PREDICTIONS ONLY"
Write-Host " NO NEW TEST DATALOADER / NO MODEL FORWARD / NO TRAINING"
Write-Host "==============================================================="
Write-Host ""

$required = @(
    ".\result\missing_baseline\cfcompat_exploratory_test_viability_v13\mosi\exploratory_test\seed1113\exploratory_test_viability_v13_summary.json",
    ".\result\missing_baseline\cfcompat_exploratory_test_viability_v13\mosi\exploratory_test\seed1113\exploratory_test_viability_v13_comparison.csv",
    ".\result\missing_baseline\anchor_decision_preserving_ensemble_v1\mosi\output_prediction_manifest.json"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen artifact missing: $path"
    }
}

$stage9Manifest = ".\result\missing_baseline\cfcompat_prediction_ensemble_v1\mosi\individual_predictions_manifest.json"
$stage8Root = ".\result\missing_baseline\cfcompat_stability_v1\mosi"
if (Test-Path $stage9Manifest) {
    Write-Host "Frozen member source preflight: Stage9A manifest found."
} else {
    Write-Warning "Stage9A intermediate manifest is absent. Falling back to frozen Stage8 Online Test predictions."
    foreach ($seed in 1111,1112,1113,1114,1115) {
        $seedDir = Join-Path $stage8Root ("seed{0}" -f $seed)
        foreach ($name in "online_test_predictions.csv","online_replay_audit.json","per_seed_all_methods.csv") {
            $path = Join-Path $seedDir $name
            if (-not (Test-Path $path)) {
                throw "Stage8 fallback artifact missing: $path"
            }
        }
    }
    Write-Host "Frozen member source preflight: Stage8 fallback complete for seeds 1111-1115."
}

$hybridRoot = ".\result\complementarity_v71\mosi\seed_1111"
if (-not (Test-Path (Join-Path $hybridRoot "complementarity_v71_summary.json"))) {
    Write-Warning "Historical Hybrid local artifacts were not found under $hybridRoot. ADPEP-All will still run; Hybrid will be reported unavailable."
}

python -m py_compile `
    .\benchmark_adpep_hybrid_missing_offline_v1.py `
    .\benchmark_adpep_hybrid_missing_offline_v1_stage8fallback.py `
    .\smoke_test_adpep_hybrid_missing_benchmark_v1.py
if ($LASTEXITCODE -ne 0) {
    throw "py_compile failed with exit code $LASTEXITCODE"
}

python .\smoke_test_adpep_hybrid_missing_benchmark_v1.py
if ($LASTEXITCODE -ne 0) {
    throw "synthetic smoke failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\benchmark_adpep_hybrid_missing_offline_v1_stage8fallback.py",
    "--dataset", "mosi",
    "--hybrid-root", $hybridRoot
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "offline benchmark failed with exit code $LASTEXITCODE"
}

$root = ".\result\missing_baseline\adpep_hybrid_missing_benchmark_v1\mosi\offline_frozen_test_predictions"
$summary = Join-Path $root "adpep_hybrid_missing_summary.json"
$comparison = Join-Path $root "adpep_hybrid_missing_comparison.csv"
foreach ($path in @($summary, $comparison)) {
    if (-not (Test-Path $path)) {
        throw "Required benchmark output missing: $path"
    }
}

Write-Host ""
Write-Host "================ Common missing-modality comparison ==========="
Import-Csv $comparison |
    Select-Object Method, TestJ, MissingMacroMAE, LAV_MAE, LA_MAE, LV_MAE, L_MAE, DeltaJVsOriginal, DeltaMissingMacroMAEVsOriginal, RankByTestJ, RankByMissingMacroMAE |
    Format-Table -AutoSize

$data = Get-Content $summary -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "================ Frozen member source =========================="
Write-Host ("source root: {0}" -f $data.stage9a_root)
$data.stage9a_members | ConvertTo-Json -Depth 5

Write-Host ""
Write-Host "================ ADPEP-All paired vs Original ================="
$data.adpep_paired_vs_original | Format-List

Write-Host ""
Write-Host "================ Historical Hybrid reconstruction ============="
$data.hybrid_status | Format-List

if ($data.hybrid_status.available) {
    Write-Host ""
    Write-Host "================ Hybrid-MissingExt paired vs Original =========="
    $data.hybrid_paired_vs_original | Format-List
}

Write-Host ""
Write-Host "OFFLINE ONLY: True"
Write-Host "New official Test DataLoader constructed: False"
Write-Host "New model forward: False"
Write-Host "New sample-level Test output written: False"
Write-Host "Result root: $root"
