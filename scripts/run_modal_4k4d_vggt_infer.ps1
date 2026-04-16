param(
    [string]$ModalExe = "",
    [string]$LocalSceneDir = "",
    [string]$RemoteSceneSubdir = "",
    [string]$OutputSubdir = "",
    [string]$ImageMode = "pad",
    [string]$HfRepo = "facebook/VGGT-1B",
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
        "D:\\anaconda\\Scripts\\modal.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    return "modal"
}

if ([string]::IsNullOrWhiteSpace($LocalSceneDir)) {
    throw "LocalSceneDir is required."
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$entryScript = Join-Path $repoRoot "modal_4k4d_vggt_infer.py"

$argList = @(
    "run",
    "$entryScript::run_scene_from_local",
    "--local-scene-dir", $LocalSceneDir,
    "--image-mode", $ImageMode,
    "--hf-repo", $HfRepo
)

if (-not [string]::IsNullOrWhiteSpace($RemoteSceneSubdir)) {
    $argList += @("--remote-scene-subdir", $RemoteSceneSubdir)
}
if (-not [string]::IsNullOrWhiteSpace($OutputSubdir)) {
    $argList += @("--output-subdir", $OutputSubdir)
}

Write-Host "[modal-4k4d] repo_root=$repoRoot"
Write-Host "[modal-4k4d] modal=$modal"
Write-Host "[modal-4k4d] entry=$entryScript"

if ($DryRun) {
    Write-Host "[modal-4k4d] dry run command:"
    Write-Host "$modal $($argList -join ' ')"
    return
}

Push-Location $repoRoot
try {
    & $modal @argList
} finally {
    Pop-Location
}
