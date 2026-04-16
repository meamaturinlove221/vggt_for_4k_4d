param(
    [string]$ModalExe = "",
    [string]$AppDescription = "vggt-4k4d-train",
    [double]$MinFreeMemoryGb = 4.0,
    [int]$MinStaleDetachMinutes = 5,
    [switch]$Detach,
    [switch]$StopExistingApps,
    [switch]$StopRepoProcesses
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

    $command = Get-Command modal -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Path
    }

    throw "Unable to resolve modal.exe."
}

function Get-AncestorProcessIds([int]$StartPid) {
    $ids = @()
    $currentPid = $StartPid
    while ($true) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $currentPid" -ErrorAction SilentlyContinue
        if (-not $proc) {
            break
        }
        $parentPid = [int]$proc.ParentProcessId
        if ($parentPid -le 0 -or $ids -contains $parentPid) {
            break
        }
        $ids += $parentPid
        $currentPid = $parentPid
    }
    return $ids
}

function Get-RepoProcesses([string]$RepoRoot, [int[]]$IgnoredPids = @()) {
    $escapedRepo = [regex]::Escape($RepoRoot)
    Get-CimInstance Win32_Process | Where-Object {
        $_.ProcessId -ne $PID `
            -and $_.ProcessId -notin $IgnoredPids `
            -and $_.Name -match "powershell|python|modal" `
            -and $_.CommandLine `
            -and $_.CommandLine -match $escapedRepo `
            -and $_.CommandLine -notmatch "invoke_modal_4k4d_preflight\.ps1"
    } | Select-Object Name, ProcessId, ParentProcessId, CreationDate, CommandLine
}

function Get-ProcessTreeIds([int]$RootPid) {
    $children = @(Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $RootPid })
    $ids = @()
    foreach ($child in $children) {
        $ids += Get-ProcessTreeIds -RootPid $child.ProcessId
        $ids += $child.ProcessId
    }
    return $ids
}

function Stop-ProcessTree([int]$RootPid) {
    $allIds = @((Get-ProcessTreeIds -RootPid $RootPid) + $RootPid) |
        Sort-Object -Descending -Unique
    foreach ($procId in $allIds) {
        try {
            Stop-Process -Id $procId -Force -ErrorAction Stop
            Write-Host "[modal-4k4d-preflight] stopped local process pid=$procId"
        } catch {
        }
    }
}

function Get-ProcessAgeMinutes($Proc) {
    try {
        if (-not $Proc -or [string]::IsNullOrWhiteSpace([string]$Proc.CreationDate)) {
            return $null
        }
        $createdAt = [Management.ManagementDateTimeConverter]::ToDateTime([string]$Proc.CreationDate)
        return ((Get-Date) - $createdAt).TotalMinutes
    } catch {
        return $null
    }
}

function Test-StaleDetachedLauncher($Proc, [string]$RepoRoot, [int]$MinAgeMinutes) {
    if (-not $Proc -or [string]::IsNullOrWhiteSpace($Proc.CommandLine)) {
        return $false
    }

    $cmd = [string]$Proc.CommandLine
    $isRepoLauncher = $cmd -match [regex]::Escape($RepoRoot) `
        -and $cmd -match "modal_4k4d_vggt_train\.py|run_modal_4k4d_vggt_train\.ps1"
    $usesDetach = $cmd -match "(^|[\s`"'])--detach($|[\s`"'])" -or $cmd -match "(^|[\s`"'])-Detach($|[\s`"'])"
    if (-not ($isRepoLauncher -and $usesDetach)) {
        return $false
    }

    $ageMinutes = Get-ProcessAgeMinutes $Proc
    return $ageMinutes -ne $null -and $ageMinutes -ge $MinAgeMinutes
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$modal = Resolve-ModalExe $ModalExe
$os = Get-CimInstance Win32_OperatingSystem
$freeMemoryGb = [math]::Round($os.FreePhysicalMemory / 1MB, 2)
$usedMemoryGb = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 2)

Write-Host "[modal-4k4d-preflight] repo_root=$repoRoot"
Write-Host "[modal-4k4d-preflight] modal=$modal"
Write-Host "[modal-4k4d-preflight] free_memory_gb=$freeMemoryGb used_memory_gb=$usedMemoryGb"

if ($freeMemoryGb -lt $MinFreeMemoryGb) {
    throw "Free system memory is only ${freeMemoryGb}GB. Abort before launching cloud work."
}

$ignoredPids = @(Get-AncestorProcessIds -StartPid $PID)
$repoProcesses = @(Get-RepoProcesses -RepoRoot $repoRoot -IgnoredPids $ignoredPids)
if ($repoProcesses.Count -gt 0) {
    $staleDetachLaunchers = @(
        $repoProcesses | Where-Object {
            Test-StaleDetachedLauncher -Proc $_ -RepoRoot $repoRoot -MinAgeMinutes $MinStaleDetachMinutes
        }
    )

    if ($StopRepoProcesses -and $staleDetachLaunchers.Count -gt 0) {
        Write-Host "[modal-4k4d-preflight] stopping stale detached repo launchers:"
        $staleDetachLaunchers | Format-Table -AutoSize | Out-String | Write-Host
        foreach ($proc in ($staleDetachLaunchers | Sort-Object ProcessId -Unique)) {
            Stop-ProcessTree -RootPid $proc.ProcessId
        }
        Start-Sleep -Seconds 2
        $repoProcesses = @(Get-RepoProcesses -RepoRoot $repoRoot -IgnoredPids $ignoredPids)
    }
}

if ($repoProcesses.Count -gt 0) {
    $repoProcesses | Format-Table -AutoSize | Out-String | Write-Host
    throw "Detected repo-scoped local powershell/python/modal processes. Stop them before launching cloud work."
}

$profileName = & $modal profile current
Write-Host "[modal-4k4d-preflight] modal_profile=$profileName"

$appJson = & $modal app list --json
$apps = @()
if (-not [string]::IsNullOrWhiteSpace($appJson)) {
    $apps = @($appJson | ConvertFrom-Json)
}

$activeApps = @(
    $apps | Where-Object {
        $description = "$($_.'Description')"
        $state = "$($_.'State')".ToLowerInvariant()
        $isActive = $state -notin @("stopped", "stopping", "completed", "failed")
        $description -eq $AppDescription -and $isActive
    }
)

if ($activeApps.Count -gt 0) {
    Write-Host "[modal-4k4d-preflight] active matching apps detected:"
    $activeApps | Format-Table -AutoSize | Out-String | Write-Host
    if ($StopExistingApps) {
        foreach ($app in $activeApps) {
            Write-Host "[modal-4k4d-preflight] stopping app $($app.'App ID')"
            & $modal app stop $app.'App ID'
        }
    } else {
        throw "Found active Modal apps for $AppDescription. Re-run with -StopExistingApps or stop them manually."
    }
}

if ($Detach) {
    Write-Host "[modal-4k4d-preflight] detach enabled"
} else {
    Write-Host "[modal-4k4d-preflight] detach disabled"
}

Write-Host "[modal-4k4d-preflight] checks passed"
