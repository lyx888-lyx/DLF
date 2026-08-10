param(
    [string]$ResultRoot = ".\result",
    [string]$ModelRoot = ".\pt",
    [string]$DatasetPath = ".\dataset\MOSEI\Processed\aligned_50.pkl"
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/fixedblend-dp57-mosei-ready-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

Write-Host "================ MOSEI FixedBlend-DP57 preflight ==================="
Write-Host ("Dataset: {0} -> {1}" -f $DatasetPath, (Test-Path $DatasetPath))

$raw5Root = Join-Path $ResultRoot "missing_baseline\cfcompat_prediction_ensemble_v1\mosei"
$raw5Missing = @()
foreach ($split in @("valid","test")) {
    foreach ($seed in @(1111,1112,1113,1114,1115)) {
        $path = Join-Path $raw5Root ("online_seed{0}_{1}_predictions.csv" -f $seed,$split)
        if (-not (Test-Path $path)) { $raw5Missing += $path }
    }
}
Write-Host ""
Write-Host "Raw5 prediction root: $raw5Root"
if ($raw5Missing.Count -eq 0) {
    Write-Host "Raw5 predictions: COMPLETE (5 seeds x valid/test)"
} else {
    Write-Host ("Raw5 predictions: INCOMPLETE ({0}/10 missing)" -f $raw5Missing.Count)
    $raw5Missing | ForEach-Object { Write-Host "  MISSING $_" }
}

Write-Host ""
Write-Host "Candidate MOSEI CFCompat best-valid checkpoints:"
$checkpointHits = @()
if (Test-Path $ModelRoot) {
    $checkpointHits = Get-ChildItem $ModelRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "mosei" -and
            $_.Name -match "best_valid" -and
            $_.FullName -match "cf_compat|cfcompat"
        }
}
if ($checkpointHits.Count -eq 0) {
    Write-Host "  NONE FOUND"
} else {
    $checkpointHits | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Candidate frozen MOSEI v13 prediction/event files:"
$v13Hits = @()
if (Test-Path $ResultRoot) {
    $v13Hits = Get-ChildItem $ResultRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -match "mosei" -and
            $_.FullName -match "v13|adam_step_safety" -and
            $_.Name -match "prediction|event|raw"
        }
}
if ($v13Hits.Count -eq 0) {
    Write-Host "  NONE FOUND"
} else {
    $v13Hits | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Other MOSEI prediction artifacts (for recovery/discovery):"
$predictionHits = @()
if (Test-Path $ResultRoot) {
    $predictionHits = Get-ChildItem $ResultRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.FullName -match "mosei" -and
            $_.Name -match "prediction"
        } |
        Select-Object -First 80
}
if ($predictionHits.Count -eq 0) {
    Write-Host "  NONE FOUND"
} else {
    $predictionHits | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
}

Write-Host ""
if (-not (Test-Path $DatasetPath)) {
    Write-Host "STATUS: BLOCKED_DATASET_MISSING"
} elseif ($raw5Missing.Count -eq 0 -and $v13Hits.Count -gt 0) {
    Write-Host "STATUS: LIKELY_READY_FOR_MOSEI_DP57_EVALUATION"
    Write-Host "Next: choose the exact frozen v13 Valid/Test files, run Valid first, freeze anchor seed, then Test."
} elseif ($checkpointHits.Count -ge 5) {
    Write-Host "STATUS: CHECKPOINTS_PRESENT_BUT_PREDICTION_PIPELINE_INCOMPLETE"
    Write-Host "Next: regenerate Raw5 predictions and/or port the frozen v13 inference/training chain without Test selection."
} else {
    Write-Host "STATUS: UPSTREAM_MOSEI_ASSETS_INCOMPLETE"
    Write-Host "Next: build the MOSEI CFCompat five-seed + v13 upstream chain before DP57 evaluation."
}
