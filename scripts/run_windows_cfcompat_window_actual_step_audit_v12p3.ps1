param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-window-actual-step-audit-v12p3"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$v12Result = ".\result\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev"
$v12Model = ".\pt\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev\seed1113"
$required = @(
    (Join-Path $v12Result "gradient_surgery_v12_fold_manifest.csv"),
    (Join-Path $v12Result "gradient_surgery_v12_fold_assignment.csv"),
    (Join-Path $v12Result "gradient_surgery_v12_update_windows.csv"),
    ".\result\missing_baseline\cfcompat_trajectory_audit_v12p2\mosi\train_oof\seed1113_dev\trajectory_v12p2_summary.json"
)
for ($fold = 0; $fold -lt 5; $fold++) {
    $required += (Join-Path $v12Model "fold$fold\conservative_train_holdout_bank.pth")
}
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen v12/v12.2 input missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_window_actual_step_audit_utils.py `
    .\audit_cfcompat_window_actual_step_train_oof_v12p3.py `
    .\smoke_test_cfcompat_window_actual_step_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "v12.3 py_compile failed with exit code $LASTEXITCODE"
}

python .\smoke_test_cfcompat_window_actual_step_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "v12.3 smoke failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\audit_cfcompat_window_actual_step_train_oof_v12p3.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "v12.3 formal window audit failed with exit code $LASTEXITCODE; result printing aborted"
}

$root = ".\result\missing_baseline\cfcompat_window_actual_step_audit_v12p3\mosi\train_oof\seed1113_dev"
$summary = Join-Path $root "window_v12p3_summary.json"
$replay = Join-Path $root "window_v12p3_exact_replay_manifest.csv"
$geometry = Join-Path $root "window_v12p3_optimizer_geometry.csv"
$byWindow = Join-Path $root "window_v12p3_mechanism_by_window_group.csv"
$byEpoch = Join-Path $root "window_v12p3_mechanism_epoch_summary.csv"

foreach ($path in @($summary, $replay, $geometry, $byWindow, $byEpoch)) {
    if (-not (Test-Path $path)) {
        throw "v12.3 completed without required artifact: $path"
    }
}

Write-Host ""
Write-Host "================ v12.3 summary ==============================="
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v12.3 exact replay =========================="
Import-Csv $replay |
    Select-Object Fold, ReplaySelectedEpoch, FormalSelectedEpoch, ReplayAbsoluteBestEpoch, FormalAbsoluteBestEpoch, missing_sequence_hash_exact, selected_state_tensor_exact, CapturedWindowN |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.3 Adam transform geometry ==============="
$g = Import-Csv $geometry
Write-Host ("captured windows: {0}" -f $g.Count)
$meanRawAdam = (($g | ForEach-Object { [double]$_.raw_to_adam_effective_cosine }) | Measure-Object -Average).Average
$minRawAdam = (($g | ForEach-Object { [double]$_.raw_to_adam_effective_cosine }) | Measure-Object -Minimum).Minimum
$conflictRate = (($g | Where-Object { $_.formal_conflict -eq "True" }).Count) / [double]$g.Count
Write-Host ("mean cosine raw surgery vs Adam effective direction: {0:N6}" -f $meanRawAdam)
Write-Host ("minimum cosine raw surgery vs Adam effective direction: {0:N6}" -f $minRawAdam)
Write-Host ("formal surgery conflict-window rate in audited interval: {0:P2}" -f $conflictRate)

Write-Host ""
Write-Host "================ v12.3 mechanism counts ======================"
$rows = Import-Csv $byWindow
foreach ($group in @("OOF_TEACHER_BENEFICIAL", "OOF_S0_BENEFICIAL", "OOF_TEACHER_NONBENEFICIAL")) {
    $local = $rows | Where-Object { $_.OOFGroup -eq $group }
    $raw = ($local | Where-Object { $_.mechanism_class -eq "RAW_SURGERY_DIRECTION_HARM" }).Count
    $adam = ($local | Where-Object { $_.mechanism_class -eq "ADAM_TRANSFORM_HARM" }).Count
    $nonlinear = ($local | Where-Object { $_.mechanism_class -eq "NONLINEAR_FINITE_STEP_HARM" }).Count
    $safe = ($local | Where-Object { $_.mechanism_class -eq "SAFE_OR_IMPROVING" }).Count
    Write-Host ("{0}: N={1} raw_harm={2} adam_harm={3} nonlinear_harm={4} safe={5}" -f $group, $local.Count, $raw, $adam, $nonlinear, $safe)
}

Write-Host ""
Write-Host "================ v12.3 epoch mechanism timeline =============="
Import-Csv $byEpoch |
    Where-Object { $_.OOFGroup -in @("OOF_TEACHER_BENEFICIAL", "OOF_S0_BENEFICIAL", "OOF_TEACHER_NONBENEFICIAL") } |
    Sort-Object {[int]$_.Epoch}, OOFGroup |
    Select-Object Epoch, OOFGroup, WindowCount, raw_harm_window_fraction, adam_first_order_harm_window_fraction, finite_harm_window_fraction, mean_raw_surgery_cosine, mean_adam_effective_cosine, mean_raw_to_adam_effective_cosine |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.3 interpretation protocol ==============="
Write-Host "RAW_SURGERY_DIRECTION_HARM: the finite window's post-surgery gradient is already first-order harmful."
Write-Host "ADAM_TRANSFORM_HARM: raw surgery is first-order safe, but Adam's actual parameter displacement is first-order harmful."
Write-Host "NONLINEAR_FINITE_STEP_HARM: both raw and actual-step first-order signs are safe, but realized OOF loss increases."
Write-Host "SAFE_OR_IMPROVING: none of those harm conditions occurs."
Write-Host "All optimizer windows in epochs 4..12 are included; no result-dependent window selection or magnitude threshold is used."
Write-Host "Official Test was never constructed or accessed."
Write-Host "Result root: $root"
