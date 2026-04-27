param(
    [string]$PredictionsNpz = "",
    [string]$SceneDir = "",
    [string]$OutputDir = "",
    [ValidateSet("world_points", "depth_unprojection")]
    [string]$PointSource = "world_points",
    [int]$MaxPoints = 400000,
    [double]$ConfPercentile = 40.0,
    [switch]$HumanOnly,
    [ValidateSet("full", "head", "face")]
    [string]$Roi = "full",
    [int]$Width = 1600,
    [int]$Height = 1200,
    [double]$PointSize = 3.0,
    [double]$VoxelSize = 0.0,
    [int]$StatisticalOutlierNeighbors = 0,
    [double]$StatisticalOutlierStdRatio = 1.5,
    [int]$RadiusOutlierNbPoints = 0,
    [double]$RadiusOutlierRadius = 0.01,
    [string]$CameraViewIndices = "",
    [switch]$Interactive,
    [switch]$ProjectionOnly,
    [string]$PythonExe = "D:\anaconda\envs\g3splat\python.exe"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

if ([string]::IsNullOrWhiteSpace($PredictionsNpz)) {
    throw "PredictionsNpz is required."
}
if ([string]::IsNullOrWhiteSpace($SceneDir)) {
    throw "SceneDir is required."
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    throw "OutputDir is required."
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RenderScript = Join-Path $RepoRoot "tools\render_open3d_pointcloud.py"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $RenderScript)) {
    throw "Render script not found: $RenderScript"
}

$argsList = @(
    $RenderScript,
    "--predictions-npz", $PredictionsNpz,
    "--scene-dir", $SceneDir,
    "--output-dir", $OutputDir,
    "--point-source", $PointSource,
    "--max-points", "$MaxPoints",
    "--conf-percentile", "$ConfPercentile",
    "--roi", $Roi,
    "--width", "$Width",
    "--height", "$Height",
    "--point-size", "$PointSize",
    "--voxel-size", "$VoxelSize",
    "--statistical-outlier-neighbors", "$StatisticalOutlierNeighbors",
    "--statistical-outlier-std-ratio", "$StatisticalOutlierStdRatio",
    "--radius-outlier-nb-points", "$RadiusOutlierNbPoints",
    "--radius-outlier-radius", "$RadiusOutlierRadius"
)

if (-not [string]::IsNullOrWhiteSpace($CameraViewIndices)) {
    $argsList += @("--camera-view-indices", $CameraViewIndices)
}

if ($HumanOnly) {
    $argsList += "--human-only"
}
if ($Interactive) {
    $argsList += "--interactive"
}
if ($ProjectionOnly) {
    $argsList += "--projection-only"
}

Write-Host "[render-open3d] python=$PythonExe"
Write-Host "[render-open3d] predictions_npz=$PredictionsNpz"
Write-Host "[render-open3d] scene_dir=$SceneDir"
Write-Host "[render-open3d] output_dir=$OutputDir"
Write-Host "[render-open3d] roi=$Roi"

& $PythonExe @argsList
