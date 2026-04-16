param(
    [string]$ModalExe = "",
    [string]$LocalCaseDirs = "",
    [string]$RemoteCaseRoot = "training_cases",
    [string]$OutputSubdir = "",
    [string]$DownloadLocalDir = "",
    [string]$ConfigName = "4k4d_prior_case",
    [string]$ExpName = "4k4d_prior_case",
    [string]$PretrainedRepo = "facebook/VGGT-1B",
    [string]$PretrainedFilename = "model.pt",
    [int]$MaxEpochs = 5,
    [int]$LimitTrainBatches = 100,
    [int]$LimitValBatches = 10,
    [int]$ValEpochFreq = 1,
    [double]$LearningRate = 1e-5,
    [int]$MaxImgPerGpu = 13,
    [int]$FixImgNum = -1,
    [int]$ImgNumsMin = 7,
    [int]$ImgNumsMax = 13,
    [int]$LenTrain = 200,
    [int]$LenTest = 20,
    [string]$ModalGpu = "A100-40GB",
    [double]$ModalCpu = 8.0,
    [int]$ModalMemoryMb = 98304,
    [int]$ModalTimeoutSec = 43200,
    [string]$DataVolume = "",
    [string]$OutputVolume = "",
    [double]$PreflightMinFreeMemoryGb = 4.0,
    [switch]$SkipPreflight,
    [switch]$StopExistingApps,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

function Resolve-ModalExe([string]$Preferred) {
    if (-not [string]::IsNullOrWhiteSpace($Preferred) -and (Test-Path $Preferred)) {
        return (Resolve-Path $Preferred).Path
    }

    $candidates = @(
        ".venv5080\\Scripts\\modal.exe",
        ".venv\\Scripts\\modal.exe",
        "venv\\Scripts\\modal.exe",
        "D:\\anaconda\\envs\\phreg\\Scripts\\modal.exe",
        "D:\\anaconda\\Scripts\\modal.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "modal"
}

function Get-ActiveModalApps([string]$ModalCmd, [string]$DescriptionFilter) {
    $raw = & $ModalCmd app list --json
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to query Modal app list."
    }

    $items = @()
    if (-not [string]::IsNullOrWhiteSpace($raw)) {
        $parsed = $raw | ConvertFrom-Json
        if ($parsed -is [System.Array]) {
            $items = $parsed
        } elseif ($null -ne $parsed) {
            $items = @($parsed)
        }
    }

    return @(
        $items | Where-Object {
            $description = "$($_.'Description')"
            $state = "$($_.'State')".ToLowerInvariant()
            $isActive = $state -notin @("stopped", "stopping", "completed", "failed")
            $matchesDescription = [string]::IsNullOrWhiteSpace($DescriptionFilter) -or $description -eq $DescriptionFilter
            $isActive -and $matchesDescription
        }
    )
}

function Wait-ModalAppsToStop([string]$ModalCmd, [string]$DescriptionFilter, [int]$TimeoutSec = 120) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    do {
        $active = Get-ActiveModalApps -ModalCmd $ModalCmd -DescriptionFilter $DescriptionFilter
        if ($active.Count -eq 0) {
            return @()
        }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)

    return (Get-ActiveModalApps -ModalCmd $ModalCmd -DescriptionFilter $DescriptionFilter)
}

if ([string]::IsNullOrWhiteSpace($LocalCaseDirs)) {
    throw "LocalCaseDirs is required."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$entryScript = Join-Path $repoRoot "modal_4k4d_vggt_train.py"
$preflightScript = Join-Path $repoRoot "scripts\\invoke_modal_4k4d_preflight.ps1"
$modalAppDescription = "vggt-4k4d-train"

if (-not $SkipPreflight) {
    $preflightArgs = @(
        "-ExecutionPolicy", "Bypass", "-File", $preflightScript,
        "-ModalExe", $modal,
        "-MinFreeMemoryGb", $PreflightMinFreeMemoryGb,
        "-StopRepoProcesses"
    )
    if ($StopExistingApps) {
        $preflightArgs += "-StopExistingApps"
    }
    & powershell @preflightArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Modal preflight failed with exit code $LASTEXITCODE."
    }
}

if (-not [string]::IsNullOrWhiteSpace($ModalGpu)) {
    $env:VGGT_MODAL_GPU = $ModalGpu
}
if ($ModalCpu -gt 0) {
    $env:VGGT_MODAL_CPU = [string]$ModalCpu
}
if ($ModalMemoryMb -gt 0) {
    $env:VGGT_MODAL_MEMORY_MB = [string]$ModalMemoryMb
}
if ($ModalTimeoutSec -gt 0) {
    $env:VGGT_MODAL_TIMEOUT_SEC = [string]$ModalTimeoutSec
}
if (-not [string]::IsNullOrWhiteSpace($DataVolume)) {
    $env:VGGT_MODAL_DATA_VOLUME = $DataVolume
}
if (-not [string]::IsNullOrWhiteSpace($OutputVolume)) {
    $env:VGGT_MODAL_OUTPUT_VOLUME = $OutputVolume
}

if ([string]::IsNullOrWhiteSpace($DownloadLocalDir)) {
    $DownloadLocalDir = Join-Path $repoRoot "output\\modal_training_results"
}

$argList = @(
    "run",
    "$entryScript::run_cases_from_local",
    "--local-case-dirs", $LocalCaseDirs,
    "--remote-case-root", $RemoteCaseRoot,
    "--download-local-dir", $DownloadLocalDir,
    "--config-name", $ConfigName,
    "--exp-name", $ExpName,
    "--pretrained-repo", $PretrainedRepo,
    "--pretrained-filename", $PretrainedFilename,
    "--max-epochs", $MaxEpochs,
    "--limit-train-batches", $LimitTrainBatches,
    "--limit-val-batches", $LimitValBatches,
    "--val-epoch-freq", $ValEpochFreq,
    "--learning-rate", $LearningRate,
    "--max-img-per-gpu", $MaxImgPerGpu,
    "--fix-img-num", $FixImgNum,
    "--img-nums-min", $ImgNumsMin,
    "--img-nums-max", $ImgNumsMax,
    "--len-train", $LenTrain,
    "--len-test", $LenTest
)

if (-not [string]::IsNullOrWhiteSpace($OutputSubdir)) {
    $argList += @("--output-subdir", $OutputSubdir)
}

Write-Host "[modal-4k4d-train] repo_root=$repoRoot"
Write-Host "[modal-4k4d-train] modal=$modal"
Write-Host "[modal-4k4d-train] entry=$entryScript"
Write-Host "[modal-4k4d-train] local_case_dirs=$LocalCaseDirs"
Write-Host "[modal-4k4d-train] download_local_dir=$DownloadLocalDir"
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_MODAL_GPU)) {
    Write-Host "[modal-4k4d-train] modal_gpu=$env:VGGT_MODAL_GPU"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_MODAL_CPU)) {
    Write-Host "[modal-4k4d-train] modal_cpu=$env:VGGT_MODAL_CPU"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_MODAL_MEMORY_MB)) {
    Write-Host "[modal-4k4d-train] modal_memory_mb=$env:VGGT_MODAL_MEMORY_MB"
}
if (-not [string]::IsNullOrWhiteSpace($env:VGGT_MODAL_TIMEOUT_SEC)) {
    Write-Host "[modal-4k4d-train] modal_timeout_sec=$env:VGGT_MODAL_TIMEOUT_SEC"
}

if ($DryRun) {
    Write-Host "[modal-4k4d-train] dry run command:"
    Write-Host "$modal $($argList -join ' ')"
    return
}

Push-Location $repoRoot
try {
    & $modal @argList
} finally {
    Pop-Location
}

$activeAppsAfter = Wait-ModalAppsToStop -ModalCmd $modal -DescriptionFilter $modalAppDescription -TimeoutSec 120
if ($activeAppsAfter.Count -gt 0) {
    $summary = ($activeAppsAfter | ForEach-Object { "$($_.'App ID'):$($_.'State')" }) -join ", "
    throw "Run finished but active Modal apps still remain: $summary"
}

Write-Host "[modal-4k4d-train] active_modal_app_count_after_finish=0"
