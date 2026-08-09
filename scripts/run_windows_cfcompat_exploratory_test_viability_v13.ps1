param(
    [int]$GpuId = 0,
    [switch]$AcknowledgeExploratoryTestAccess,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-exploratory-test-viability-v13"

if (-not $AcknowledgeExploratoryTestAccess) {
    throw @"
This command intentionally accesses official MOSI Test.
It is exploratory only; Test was already historically accessed and must not be
used to tune later methods. Re-run with -AcknowledgeExploratoryTestAccess only
if you consciously accept that protocol status.
"@
}

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

Write-Host ""
Write-Host "==============================================================="
Write-Host " EXPLORATORY_TEST_VIABILITY_ONLY"
Write-Host " TEST_ALREADY_HISTORICALLY_ACCESSED"
Write-Host " NO_TEST_DRIVEN_TUNING_ALLOWED"
Write-Host "==============================================================="
Write-Host ""

$required = @(
    ".\result\missing_baseline\cfcompat_student_safe_abstain_v2\mosi\valid_screen\student_safe_abstain_source_manifest.json",
    ".\result\missing_baseline\cfcompat_regret_preserve_v4\mosi\valid_screen\seed1113_dev\regret_preserve_v4_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_sample_conditioned_residual_v8\mosi\valid_screen\seed1113_dev\sample_residual_v8_candidate_grid.csv",
    ".\result\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev\gradient_surgery_v12_fold_manifest.csv",
    ".\result\missing_baseline\cfcompat_adam_step_safety_v13\mosi\valid_screen\seed1113_dev\adam_step_safety_v13_fold_manifest.csv",
    ".\result\missing_baseline\cfcompat_adam_step_safety_v13\mosi\valid_screen\seed1113_dev\adam_step_safety_v13_valid_screen_summary.json"
)
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen pre-Test artifact missing: $path"
    }
}

# These checks do not load any MOSI split.
python -m py_compile `
    .\trains\singleTask\cfcompat_exploratory_test_utils.py `
    .\evaluate_cfcompat_exploratory_test_viability_v13.py `
    .\smoke_test_cfcompat_exploratory_test_viability.py
if ($LASTEXITCODE -ne 0) {
    throw "Exploratory Test probe py_compile failed with exit code $LASTEXITCODE"
}

python .\smoke_test_cfcompat_exploratory_test_viability.py
if ($LASTEXITCODE -ne 0) {
    throw "Exploratory Test aggregate smoke failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Synthetic aggregate checks passed. Official Test access starts now."
Write-Host "No sample-level Test artifact will be written."
Write-Host ""

$argsList = @(
    ".\evaluate_cfcompat_exploratory_test_viability_v13.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Exploratory Test viability probe failed with exit code $LASTEXITCODE"
}

$root = ".\result\missing_baseline\cfcompat_exploratory_test_viability_v13\mosi\exploratory_test\seed1113"
$summary = Join-Path $root "exploratory_test_viability_v13_summary.json"
$comparison = Join-Path $root "exploratory_test_viability_v13_comparison.csv"
$transfer = Join-Path $root "exploratory_test_viability_v13_transfer.csv"
$manifest = Join-Path $root "exploratory_test_viability_v13_checkpoint_manifest.json"

foreach ($path in @($summary, $comparison, $transfer, $manifest)) {
    if (-not (Test-Path $path)) {
        throw "Probe completed without required aggregate artifact: $path"
    }
}

Write-Host ""
Write-Host "================ Test comparison =============================="
Import-Csv $comparison |
    Select-Object Method, TestJ, MissingMacroMAE, LAV_MAE, LA_MAE, LV_MAE, L_MAE, OverallMeanGain, OverallNTR, OverallSevereNTR, BeneficialNTR, BeneficialSevereNTR, NonbeneficialNTR, DeltaJVsOriginal, OverallNTRReductionVsOriginal, BeneficialNTRReductionVsOriginal, SevereNTRReductionVsOriginal |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ Transfer groups ==============================="
Import-Csv $transfer |
    Select-Object Method, Group, N, mean_gain_vs_baseline, positive_transfer_rate, negative_transfer_rate, severe_negative_transfer_rate, not_improved_rate |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ Route decision ================================"
$data = Get-Content $summary -Raw | ConvertFrom-Json
Write-Host ("route verdict:                              {0}" -f $data.route_decision.verdict)
Write-Host ("v13 Test-J minus Original:                 {0}" -f $data.route_decision.delta_J_v13_minus_original)
Write-Host ("v13 beneficial NTR reduction vs Original: {0}" -f $data.route_decision.beneficial_NTR_reduction_v13_vs_original)
Write-Host ("v13 overall NTR reduction vs Original:    {0}" -f $data.route_decision.overall_NTR_reduction_v13_vs_original)
Write-Host ("v13 severe NTR reduction vs Original:     {0}" -f $data.route_decision.severe_NTR_reduction_v13_vs_original)
Write-Host ""
Write-Host "Checks:"
$data.route_decision.checks | Format-List

Write-Host ""
Write-Host "================ v13 vs v12 ==================================="
$data.v13_vs_v12 | Format-List

Write-Host ""
Write-Host "EXPLORATORY_TEST_VIABILITY_ONLY"
Write-Host "TEST_ALREADY_HISTORICALLY_ACCESSED"
Write-Host "NO_TEST_DRIVEN_TUNING_ALLOWED"
Write-Host "Sample-level Test artifacts written: False"
Write-Host "Aggregate result root: $root"
