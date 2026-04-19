param(
    [string]$DatasetRoot = "",
    [string]$Seq = "0012_11",
    [int]$Frame = 0,
    [string]$TargetCamera = "00",
    [int]$AutoSources = 6,
    [string]$OutputBase = "D:\vggt\vggt-main\output",
    [int]$TargetSize = 518,
    [string]$SmplxModelDir = "G:\数据集\datasets\smplx",
    [string]$CheckpointRelpath = "",
    [string]$ModalExe = "",
    [switch]$OverwriteScene,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $detected = Get-ChildItem 'G:\' -Directory -Recurse -Depth 3 -Filter 'data_used_in_4K4D' -ErrorAction SilentlyContinue |
        Select-Object -First 1 -ExpandProperty FullName
    if ([string]::IsNullOrWhiteSpace($detected)) {
        throw "DatasetRoot was not provided and no data_used_in_4K4D directory could be auto-detected under G:\\."
    }
    $DatasetRoot = $detected
}

$sceneName = "{0}_frame{1:D4}_{2}views" -f $Seq, $Frame, ($AutoSources + 1)
$sceneDir = Join-Path $OutputBase ("4k4d_scenes\" + $sceneName)
$modalOutputSubdir = "vggt_4k4d_infer/$sceneName"

$exportArgs = @(
    "tools/export_4k4d_scene.py",
    "--dataset-root", $DatasetRoot,
    "--seq", $Seq,
    "--frame", $Frame,
    "--target-camera", $TargetCamera,
    "--auto-sources", $AutoSources,
    "--target-size", $TargetSize,
    "--smplx-model-dir", $SmplxModelDir,
    "--output-dir", $sceneDir
)
if ($OverwriteScene) {
    $exportArgs += "--overwrite"
}

$modalScript = Join-Path $repoRoot "scripts\run_modal_4k4d_vggt_infer.ps1"
$modalArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", $modalScript,
    "-LocalSceneDir", $sceneDir,
    "-OutputSubdir", $modalOutputSubdir
)
if (-not [string]::IsNullOrWhiteSpace($ModalExe)) {
    $modalArgs += @("-ModalExe", $ModalExe)
}
if (-not [string]::IsNullOrWhiteSpace($CheckpointRelpath)) {
    $modalArgs += @("-CheckpointRelpath", $CheckpointRelpath)
}
if ($DryRun) {
    $modalArgs += "-DryRun"
}

Write-Host "[4k4d-vggt] repo_root=$repoRoot"
Write-Host "[4k4d-vggt] dataset_root=$DatasetRoot"
Write-Host "[4k4d-vggt] scene_name=$sceneName"
Write-Host "[4k4d-vggt] scene_dir=$sceneDir"
Write-Host "[4k4d-vggt] modal_output_subdir=$modalOutputSubdir"

Push-Location $repoRoot
try {
    if ($DryRun) {
        Write-Host "[4k4d-vggt] dry run export command:"
        Write-Host ("python " + ($exportArgs -join " "))
        & powershell @modalArgs
    } else {
        & python @exportArgs
        & powershell @modalArgs
    }
} finally {
    Pop-Location
}
