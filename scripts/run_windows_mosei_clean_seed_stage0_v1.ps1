param(
    [ValidateSet(1111,1112,1113,1114,1115)]
    [int]$Seed = 1111,
    [int]$BatchSize = 16,
    [int]$UpdateEpochs = 10,
    [int]$NumWorkers = 0,
    [ValidateSet("highest","high","medium")]
    [string]$MatmulPrecision = "high",
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/fixedblend-dp57-mosei-ready-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

python -m py_compile .\train_mosei_clean_seed_stage0_v1.py
if ($LASTEXITCODE -ne 0) {
    throw "MOSEI clean Stage-0 runner py_compile failed"
}

$argsList = @(
    ".\train_mosei_clean_seed_stage0_v1.py",
    "--seed", "$Seed",
    "--gpu-id", "0",
    "--batch-size", "$BatchSize",
    "--update-epochs", "$UpdateEpochs",
    "--num-workers", "$NumWorkers",
    "--matmul-precision", $MatmulPrecision
)
if ($Overwrite) { $argsList += "--overwrite" }

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MOSEI clean Stage-0 seed$Seed failed with exit code $LASTEXITCODE"
}

$checkpoint = ".\pt\DLF_mosei_seed${Seed}_best.pth"
$manifest = ".\result\missing_baseline\mosei_clean_stage0_v1\seed${Seed}\run_manifest.json"
foreach ($path in @($checkpoint,$manifest)) {
    if (-not (Test-Path $path)) { throw "Required output missing: $path" }
}

Write-Host ""
Write-Host "================ Stage-0 output ====================================="
Get-Item $checkpoint | Select-Object FullName,Length,LastWriteTime | Format-Table -AutoSize
Get-Content $manifest -Raw
