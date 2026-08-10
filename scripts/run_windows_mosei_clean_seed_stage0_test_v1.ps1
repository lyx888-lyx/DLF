param(
    [int]$Seed = 1111,
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ExpectedBranch = "analysis/fixedblend-dp57-mosei-ready-v1"
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne $ExpectedBranch) {
    throw "Wrong branch: $branch. Expected $ExpectedBranch"
}

$checkpoint = ".\pt\DLF_mosei_seed${Seed}_best.pth"
$manifest = ".\result\missing_baseline\mosei_clean_stage0_v1\seed${Seed}\run_manifest.json"
foreach ($path in @($checkpoint, $manifest)) {
    if (-not (Test-Path $path)) { throw "Missing required frozen Stage-0 asset: $path" }
}

python -m py_compile .\evaluate_mosei_clean_seed_stage0_test_v1.py
if ($LASTEXITCODE -ne 0) {
    throw "MOSEI clean Test evaluator py_compile failed with exit code $LASTEXITCODE"
}

$argsList = @(
    ".\evaluate_mosei_clean_seed_stage0_test_v1.py",
    "--seed", "$Seed",
    "--gpu-id", "$GpuId",
    "--num-workers", "0",
    "--matmul-precision", "high"
)
if ($Overwrite) { $argsList += "--overwrite" }

python @argsList
if ($LASTEXITCODE -ne 0) {
    throw "MOSEI clean DLF Test reproduction failed with exit code $LASTEXITCODE"
}
