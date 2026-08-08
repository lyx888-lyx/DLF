param(
    [int]$GpuId = 0,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$hotfix = Join-Path $PSScriptRoot "run_windows_cfcompat_gradient_surgery_v12_hotfix.ps1"
if (-not (Test-Path $hotfix)) {
    throw "Missing v12 numerical-hotfix runner: $hotfix"
}

Write-Host "Using v12 numerical hotfix for scale-stable post-projection validation."
if ($Overwrite) {
    & $hotfix -GpuId $GpuId -Overwrite
} else {
    & $hotfix -GpuId $GpuId
}
