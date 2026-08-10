param(
    [string]$ResultRoot = ".\result",
    [string]$ModelRoot = ".\pt"
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/fixedblend-dp57-mosei-ready-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$seeds = @(1111,1112,1113,1114,1115)

Write-Host "================ MOSEI Stage-0 teacher / ModDrop preflight ================"

Write-Host ""
Write-Host "Clean DLF teacher checkpoints expected by clean_checkpoint_path():"
$cleanMissing = @()
foreach ($seed in $seeds) {
    $path = Join-Path $ModelRoot ("DLF_mosei_seed{0}_best.pth" -f $seed)
    if (Test-Path $path) {
        $item = Get-Item $path
        Write-Host ("  PASS seed{0}: {1} ({2} bytes)" -f $seed,$item.FullName,$item.Length)
    } else {
        Write-Host ("  MISSING seed{0}: {1}" -f $seed,$path)
        $cleanMissing += $path
    }
}

Write-Host ""
Write-Host "Stage1 ModDrop checkpoints (single-run canonical path):"
$moddropCanonicalMissing = @()
foreach ($seed in $seeds) {
    $path = Join-Path $ModelRoot ("missing_baseline\moddrop\DLF_mosei_seed{0}_best.pth" -f $seed)
    if (Test-Path $path) {
        $item = Get-Item $path
        Write-Host ("  PASS seed{0}: {1} ({2} bytes)" -f $seed,$item.FullName,$item.Length)
    } else {
        Write-Host ("  MISSING seed{0}: {1}" -f $seed,$path)
        $moddropCanonicalMissing += $path
    }
}

Write-Host ""
Write-Host "Stage1 ModDrop result CSV candidates:"
$canonicalCsv = Join-Path $ResultRoot "missing_baseline\moddrop\train\mosei_per_seed.csv"
if (Test-Path $canonicalCsv) {
    Write-Host "  PASS canonical: $canonicalCsv"
    try {
        Import-Csv $canonicalCsv |
            Select-Object Seed,BestEpoch,Checkpoint |
            Format-Table -AutoSize
    } catch {
        Write-Host "  WARN: canonical CSV exists but summary display failed: $($_.Exception.Message)"
    }
} else {
    Write-Host "  MISSING canonical: $canonicalCsv"
}

Write-Host ""
Write-Host "Stage1 multiseed ModDrop assets (if historical replication layout was used):"
$multiHits = @()
$multiRoot = Join-Path $ResultRoot "missing_baseline\moddrop_benchmark_multiseed_v1"
if (Test-Path $multiRoot) {
    $multiHits = Get-ChildItem $multiRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "mosei|seed111[1-5]" }
}
if ($multiHits.Count -eq 0) {
    Write-Host "  NONE FOUND"
} else {
    $multiHits | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
}

Write-Host ""
Write-Host "Any other clean/ModDrop MOSEI checkpoints under pt (discovery only):"
$discovery = @()
if (Test-Path $ModelRoot) {
    $discovery = Get-ChildItem $ModelRoot -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Name -match "mosei" -and
            $_.Extension -eq ".pth" -and
            ($_.Name -match "DLF" -or $_.FullName -match "moddrop")
        } |
        Select-Object -First 80
}
if ($discovery.Count -eq 0) {
    Write-Host "  NONE FOUND"
} else {
    $discovery | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
}

Write-Host ""
if ($cleanMissing.Count -eq 0 -and (Test-Path $canonicalCsv)) {
    Write-Host "STATUS: READY_TO_PORT_AND_RUN_MOSEI_CFCOMPAT"
    Write-Host "All five clean DLF teachers and canonical Stage1 evaluator manifest are present."
} elseif ($cleanMissing.Count -eq 0 -and $multiHits.Count -gt 0) {
    Write-Host "STATUS: CLEAN_TEACHERS_PRESENT_MODDROP_LAYOUT_NEEDS_BINDING"
    Write-Host "Clean DLF teachers exist; inspect the printed multiseed Stage1 artifacts and bind the validation-best evaluators."
} elseif ($cleanMissing.Count -eq 0) {
    Write-Host "STATUS: CLEAN_TEACHERS_PRESENT_MODDROP_MISSING"
    Write-Host "Next: train the five MOSEI Stage1 ModDrop validation-best evaluators, then CFCompat."
} else {
    Write-Host "STATUS: CLEAN_DLF_TEACHERS_MISSING"
    Write-Host "Next: train the five clean MOSEI DLF validation-best teachers first, then Stage1 ModDrop, then CFCompat."
}
