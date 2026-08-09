param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-trajectory-audit-v12p2"

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
    ".\result\missing_baseline\cfcompat_post_surgery_audit_v12p1\mosi\train_oof\seed1113_dev\post_surgery_audit_v12p1_summary.json"
)
for ($fold = 0; $fold -lt 5; $fold++) {
    $required += (Join-Path $v12Model "fold$fold\conservative_train_holdout_bank.pth")
}
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen v12/v12.1 input missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_trajectory_audit_utils.py `
    .\audit_cfcompat_trajectory_train_oof_v12p2.py `
    .\smoke_test_cfcompat_trajectory_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "v12.2 py_compile failed with exit code $LASTEXITCODE"
}

python .\smoke_test_cfcompat_trajectory_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "v12.2 smoke failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\audit_cfcompat_trajectory_train_oof_v12p2.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "v12.2 formal trajectory audit failed with exit code $LASTEXITCODE; result printing aborted"
}

$root = ".\result\missing_baseline\cfcompat_trajectory_audit_v12p2\mosi\train_oof\seed1113_dev"
$summary = Join-Path $root "trajectory_v12p2_summary.json"
$replay = Join-Path $root "trajectory_v12p2_exact_replay_manifest.csv"
$gradient = Join-Path $root "trajectory_v12p2_fixed_milestone_gradient_summary.csv"
$groups = Join-Path $root "trajectory_v12p2_epoch_group_summary.csv"
$onset = Join-Path $root "trajectory_v12p2_failure_onset.csv"
$projection = Join-Path $root "trajectory_v12p2_milestone_projection_geometry.csv"

foreach ($path in @($summary, $replay, $gradient, $groups, $onset, $projection)) {
    if (-not (Test-Path $path)) {
        throw "v12.2 completed without required artifact: $path"
    }
}

Write-Host ""
Write-Host "================ v12.2 summary =============================="
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v12.2 exact replay ========================="
Import-Csv $replay |
    Select-Object Fold, ReplaySelectedEpoch, FormalSelectedEpoch, ReplayAbsoluteBestEpoch, FormalAbsoluteBestEpoch, selected_epoch_exact, absolute_best_epoch_exact, missing_sequence_hash_exact, selected_state_tensor_exact |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.2 surgery-update gradient timeline ====="
Import-Csv $gradient |
    Where-Object {
        $_.Mode -eq "ALL" -and
        $_.TrainComponent -eq "SURGERY_UPDATE" -and
        $_.OOFGroup -in @("OOF_TEACHER_BENEFICIAL", "OOF_S0_BENEFICIAL", "OOF_TEACHER_NONBENEFICIAL")
    } |
    Sort-Object {[int]$_.Epoch}, OOFGroup |
    Select-Object Epoch, OOFGroup, FoldCount, mean_gradient_cosine, harm_fold_count, improve_fold_count |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.2 OOF transfer trajectory =============="
$fixed = @(0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
Import-Csv $groups |
    Where-Object {
        $fixed -contains [int]$_.Epoch -and
        $_.Group -in @("TEACHER_BENEFICIAL", "TEACHER_NONBENEFICIAL", "S0_BENEFICIAL")
    } |
    Sort-Object {[int]$_.Epoch}, Group |
    Select-Object Epoch, Group, FoldCount, N, mean_gain_vs_baseline, negative_transfer_rate, severe_negative_transfer_rate, mean_abs_residual, crossed_label_from_s0_rate |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.2 interpretation protocol =============="
Write-Host "The five-fold replay must be exact before any trajectory conclusion is valid."
Write-Host "If early fixed milestones show SURGERY_UPDATE harming beneficial OOF groups and the selected endpoint later improves them, that supports a trajectory sign-reversal / accumulated-drift mechanism."
Write-Host "If SURGERY_UPDATE is already beneficial-safe at all early milestones while beneficial OOF failures still accumulate, expected full-Train first-order geometry is insufficient; inspect finite Adam windows, momentum/second moments, and microbatch heterogeneity next."
Write-Host "No threshold, model, checkpoint, or Test decision is tuned by v12.2."
Write-Host "Official Test was never constructed or accessed."
Write-Host "Result root: $root"
