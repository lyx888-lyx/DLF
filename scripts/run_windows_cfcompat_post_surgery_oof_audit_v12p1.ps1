param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "agent/cfcompat-post-surgery-oof-audit-v12p1"

$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$v12Root = ".\result\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev"
$v10Root = ".\result\missing_baseline\cfcompat_conservative_crossfit_residual_v10\mosi\valid_screen\seed1113_dev"
$required = @(
    (Join-Path $v12Root "gradient_surgery_v12_fold_manifest.csv"),
    (Join-Path $v12Root "gradient_surgery_v12_valid_screen_summary.json"),
    (Join-Path $v10Root "conservative_crossfit_v10_fold_assignment.csv")
)
for ($fold = 0; $fold -lt 5; $fold++) {
    $required += ".\pt\missing_baseline\cfcompat_gradient_surgery_v12\mosi\valid_screen\seed1113_dev\seed1113\fold$fold\conservative_train_holdout_bank.pth"
}
foreach ($path in $required) {
    if (-not (Test-Path $path)) {
        throw "Required frozen v12.1 audit input missing: $path"
    }
}

python -m py_compile `
    .\trains\singleTask\cfcompat_objective_audit_utils.py `
    .\trains\singleTask\cfcompat_post_surgery_audit_utils.py `
    .\audit_cfcompat_objective_train_oof_v11.py `
    .\audit_cfcompat_post_surgery_train_oof_v12p1.py `
    .\smoke_test_cfcompat_post_surgery_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "v12.1 py_compile failed with exit code $LASTEXITCODE"
}

python .\smoke_test_cfcompat_post_surgery_audit.py
if ($LASTEXITCODE -ne 0) {
    throw "v12.1 smoke failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\audit_cfcompat_post_surgery_train_oof_v12p1.py",
    "--gpu-ids", "$GpuId"
)
if ($Overwrite) {
    $argsList += "--overwrite"
}
python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "v12.1 formal audit failed with exit code $LASTEXITCODE; result printing aborted"
}

$root = ".\result\missing_baseline\cfcompat_post_surgery_audit_v12p1\mosi\train_oof\seed1113_dev"
$summary = Join-Path $root "post_surgery_audit_v12p1_summary.json"
$headline = Join-Path $root "post_surgery_audit_v12p1_headline_gradient_matrix.csv"
$projection = Join-Path $root "post_surgery_audit_v12p1_projection_geometry.csv"
$eventSummary = Join-Path $root "post_surgery_audit_v12p1_event_summary.csv"
$influence = Join-Path $root "post_surgery_audit_v12p1_gradient_influence_summary.csv"

foreach ($path in @($summary, $headline, $projection, $eventSummary, $influence)) {
    if (-not (Test-Path $path)) {
        throw "v12.1 completed without required result artifact: $path"
    }
}

Write-Host ""
Write-Host "================ v12.1 post-surgery audit summary ================"
Get-Content $summary -Raw

Write-Host ""
Write-Host "================ v12.1 headline gradient matrix =================="
Import-Csv $headline |
    Sort-Object OOFGroup, TrainComponent |
    Select-Object OOFGroup, TrainComponent, mean_gradient_dot, mean_gradient_cosine, harm_fold_count, improve_fold_count |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.1 ALL-mode projection geometry ============="
Import-Csv $projection |
    Where-Object { $_.Mode -eq "ALL" } |
    Select-Object Fold, conflict, gradient_cosine_before, post_projection_cosine, supervised_l2_removed_fraction |
    Format-Table -AutoSize

Write-Host ""
Write-Host "================ v12.1 key interpretation rows =================="
$rows = Import-Csv $headline
$keys = @(
    @("OOF_TEACHER_BENEFICIAL", "SUPERVISED_PROJECTED"),
    @("OOF_TEACHER_BENEFICIAL", "SURGERY_UPDATE"),
    @("OOF_S0_BENEFICIAL", "SUPERVISED_PROJECTED"),
    @("OOF_S0_BENEFICIAL", "SURGERY_UPDATE"),
    @("OOF_TEACHER_NONBENEFICIAL", "SURGERY_UPDATE")
)
foreach ($key in $keys) {
    $row = $rows | Where-Object { $_.OOFGroup -eq $key[0] -and $_.TrainComponent -eq $key[1] }
    if ($null -eq $row) {
        throw "Missing headline row: $($key[0]) / $($key[1])"
    }
    Write-Host ("{0} <- {1}: cosine={2} harm={3}/5 improve={4}/5" -f `
        $key[0], $key[1], $row.mean_gradient_cosine, $row.harm_fold_count, $row.improve_fold_count)
}

Write-Host ""
Write-Host "================ v12.1 interpretation protocol =================="
Write-Host "If SUPERVISED_PROJECTED still harms OOF beneficial groups in 5/5 folds, the selective gradient is not a sufficient functional-safety anchor."
Write-Host "If SURGERY_UPDATE improves OOF beneficial groups in 5/5 folds while frozen v12 Valid remains unsafe, local final-checkpoint geometry is insufficient and trajectory/nonlinear drift becomes the stronger hypothesis."
Write-Host "No thresholds are tuned from this audit; it is diagnostic only."
Write-Host "Official Test was never constructed or accessed."
Write-Host "Result root: $root"
