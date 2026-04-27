param(
    [string]$InputPath = "D:\vggt\vggt-main\output\modal_results\0012_11_frame0000_60views_smplxsurfacepose_a10080_e2_r2\pointcloud_dense_p40_hires"
)

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = "D:\anaconda\envs\g3splat\python.exe"
$ViewerScript = Join-Path $RepoRoot "tools\open3d_view_pointcloud.py"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

if (-not (Test-Path $ViewerScript)) {
    throw "Viewer script not found: $ViewerScript"
}

& $PythonExe $ViewerScript $InputPath
