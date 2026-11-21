[CmdletBinding()]
param([string]$Python = 'python')

$ErrorActionPreference = 'Stop'
$launcher = Join-Path $PSScriptRoot 'confine.py'
$root = Join-Path $env:LOCALAPPDATA 'desman-research'
$runtime = Join-Path $root 'python'
$workspace = Join-Path $root 'ws'
$profilePath = $env:USERPROFILE
$userTemp = $env:TEMP
$publicPath = $env:PUBLIC

$baseJson = & $Python -c 'import json, sys; assert sys.version_info >= (3, 11); print(json.dumps(sys.base_prefix))'
if ($LASTEXITCODE -ne 0 -or -not $baseJson) { throw 'Python 3.11+ is required.' }
$base = ConvertFrom-Json -InputObject $baseJson
if (Test-Path -LiteralPath $runtime) { throw "Move the previous private runtime aside: $runtime" }
if (Test-Path -LiteralPath $workspace) { throw "Move the previous workspace aside: $workspace" }
if (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'acl-journal.json')) {
    throw 'Run confine.py --cleanup to finish the previous run first.'
}
$fullBase = [IO.Path]::GetFullPath($base).TrimEnd('\')
$fullRoot = [IO.Path]::GetFullPath($root).TrimEnd('\')
if ($fullRoot.Equals($fullBase, [StringComparison]::OrdinalIgnoreCase) -or
    $fullRoot.StartsWith("$fullBase\", [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The private copy cannot be placed inside the source Python installation.'
}
New-Item -ItemType Directory -Path $workspace -Force | Out-Null
Copy-Item -LiteralPath $base -Destination $runtime -Recurse
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'probe.py') -Destination $workspace

$listener = $null
$hostPipe = $null
$needsCleanup = $false
$failed = $false
try {
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
    $listener.Start()
    $port = $listener.LocalEndpoint.Port
    $pipeName = 'desman-research-' + [Guid]::NewGuid().ToString('N')
    $hostPipe = [IO.Pipes.NamedPipeServerStream]::new($pipeName, [IO.Pipes.PipeDirection]::Out)

    foreach ($mode in @('AppContainer', 'LPAC')) {
        $options = @('--workspace', $workspace, '--runtime', $runtime, '--timeout', '60')
        if ($mode -eq 'LPAC') { $options += '--lpac' }
        $needsCleanup = $true
        Write-Host "=== $mode ==="
        $output = & $Python $launcher @options -- python -B -u -X utf8 probe.py `
            --user-profile $profilePath --user-temp $userTemp --public $publicPath `
            --loopback-port $port --pipe "\\.\pipe\$pipeName"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "$mode launcher failed with exit code $LASTEXITCODE."
            $failed = $true
            continue
        }
        $output | Write-Output
        try {
            $result = $output | ConvertFrom-Json
            $rows = @($result.stdout -split '\r?\n' | Where-Object { $_ } |
                ForEach-Object { ConvertFrom-Json -InputObject $_ })
            $rows | Format-Table -AutoSize | Out-Host
            $write = $rows | Where-Object { $_.check -eq 'workspace_write' }
            if ($result.exit_code -ne 0 -or $result.timed_out -or
                $rows.Count -ne 17 -or -not $write.allowed) {
                Write-Warning "$mode did not complete all probes with workspace write access."
                $failed = $true
            }
        } catch {
            Write-Warning "$mode produced unreadable results: $_"
            $failed = $true
        }
    }
} finally {
    try {
        if ($needsCleanup) {
            & $Python $launcher --cleanup
            if ($LASTEXITCODE -ne 0) { throw "ACL/profile cleanup failed with exit code $LASTEXITCODE." }
        }
    } finally {
        if ($null -ne $hostPipe) { $hostPipe.Dispose() }
        if ($null -ne $listener) { $listener.Stop() }
    }
}
if ($failed) { exit 1 }
