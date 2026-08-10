$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    Write-Host "================ MOSEI ModDrop Stage-1 preflight ================="

    $dataset = ".\dataset\MOSEI\Processed\aligned_50.pkl"
    Write-Host ("MOSEI dataset: {0}  exists={1}" -f $dataset, (Test-Path $dataset))
    if (-not (Test-Path $dataset)) {
        throw "MOSEI aligned dataset is missing: $dataset"
    }

    $allClean = $true
    foreach ($seed in 1111,1112,1113,1114,1115) {
        $ckpt = ".\pt\DLF_mosei_seed${seed}_best.pth"
        $manifest = ".\result\missing_baseline\mosei_clean_stage0_v1\seed${seed}\run_manifest.json"
        $ckptOk = Test-Path $ckpt
        $manifestOk = Test-Path $manifest
        $shaOk = $false
        $sha = "MISSING"
        $recorded = "MISSING"

        if ($ckptOk) {
            $sha = (Get-FileHash $ckpt -Algorithm SHA256).Hash.ToLower()
        }
        if ($manifestOk) {
            $m = Get-Content $manifest -Raw | ConvertFrom-Json
            $recorded = [string]$m.checkpoint_sha256
            if ($ckptOk) {
                $shaOk = ($sha -eq $recorded.ToLower())
            }
        }

        [PSCustomObject]@{
            Seed = $seed
            CleanCheckpoint = $ckptOk
            Stage0Manifest = $manifestOk
            SHA_match = $shaOk
            BatchSize = if ($manifestOk) { $m.batch_size } else { $null }
            UpdateEpochs = if ($manifestOk) { $m.update_epochs } else { $null }
            TestConstructed = if ($manifestOk) { $m.test_constructed } else { $null }
        } | Format-List

        if (-not ($ckptOk -and $manifestOk -and $shaOk)) {
            $allClean = $false
        }
    }

    if (-not $allClean) {
        throw "At least one clean Stage-0 checkpoint/manifest binding failed."
    }

    Write-Host "---------------- Existing formal Stage-1 outputs -----------------"
    $canonical = ".\result\missing_baseline\moddrop\train\mosei_per_seed.csv"
    if (Test-Path $canonical) {
        Write-Host "Canonical CSV exists: $canonical"
        Import-Csv $canonical | Select-Object Seed,BestEpoch,J_val,Checkpoint | Format-Table -AutoSize
    }
    else {
        Write-Host "Canonical CSV does not exist yet (expected before first run)."
    }

    foreach ($seed in 1111,1112,1113,1114,1115) {
        $stage1 = ".\pt\missing_baseline\moddrop\DLF_mosei_seed${seed}_best.pth"
        Write-Host ("seed {0}: Stage1 checkpoint exists={1}  {2}" -f $seed, (Test-Path $stage1), $stage1)
    }

    Write-Host "=================================================================="
    Write-Host "STATUS: CLEAN_DLF_X5_READY_FOR_MOSEI_MODDROP_STAGE1"
    Write-Host "TEST WILL NOT BE CONSTRUCTED BY STAGE-1 TRAINING"
}
finally {
    Pop-Location
}
