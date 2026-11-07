# Windows PowerShell 5.1; per-user install, no administrator shell required.
[CmdletBinding()]
param(
    [string]$Ref = $env:DESMAN_REF,
    [switch]$Yes, [switch]$NoStart, [switch]$Uninstall, [switch]$Purge,
    [switch]$DryRun
)
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false)
$ErrorActionPreference = 'Stop'
$Repository = 'https://github.com/Temporary-Bot-5/DESMAN-P.git'
$App = Join-Path $env:LOCALAPPDATA 'DESMAN\app'
if ($env:DESMAN_HOME) { $App = $env:DESMAN_HOME }
$MinGit = Join-Path $env:LOCALAPPDATA 'DESMAN\mingit'
$HomeConfig = Join-Path $env:USERPROFILE '.config\desman'
$Config = $HomeConfig
if ($env:DESMAN_CONFIG_DIR) { $Config = $env:DESMAN_CONFIG_DIR }
$Data = Join-Path $env:USERPROFILE '.local\share\desman'
if ($env:DESMAN_DATA_DIR) { $Data = $env:DESMAN_DATA_DIR }
function Confirm-Action([string]$Message) {
    Write-Host $Message
    if ($Yes) { return }
    if ([Console]::IsInputRedirected) { throw 'Non-interactive input: use -Yes.' }
    if ((Read-Host 'Continue? (y/N)') -notmatch '^[yY]$') { throw 'Cancelled.' }
}
function Run([string]$Command, [string[]]$Arguments) {
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Command failed ($LASTEXITCODE)." }
}
function Safe-Directory([string]$Path) {
    $Full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    if ($Full -eq [IO.Path]::GetPathRoot($Full).TrimEnd('\', '/') -or
        $Full -eq $env:USERPROFILE.TrimEnd('\', '/')) { throw "Unsafe directory: $Path" }
    if ((Test-Path -LiteralPath $Full) -and
        ((Get-Item -LiteralPath $Full).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Refusing linked directory: $Full"
    }
}
# DryRun never downloads, prompts, writes files or invokes an installed command.
if ($DryRun) {
    if ($Uninstall) {
        Write-Output 'Verify installer-owned app directory (.desman-install, .git, src/desman/cli.py); if absent remove nothing, including purge paths.'
        Write-Output 'Check each .lnk TargetPath is under App before removing it; preserve foreign shortcuts.'
        Write-Output "Confirm removal: $App; Desktop/DESMAN.lnk; Start menu/DESMAN.lnk. Preserve config and data."
        if ($Purge) { Write-Output "Separate purge confirmation: $Data; $Config; $App\config.json" }
    } else {
        Write-Output "Install to $App; refuse non-DESMAN directories; confirm uv from https://astral.sh/uv/install.ps1"
        Write-Output "Confirm git: winget install --id Git.Git -e --source winget; otherwise official git-for-windows MinGit x64 zip to $MinGit"
        Write-Output "git ls-remote --tags (newest v*, fallback master); override -Ref / DESMAN_REF: $Ref"
        Write-Output 'git clone; git fetch origin --tags; git checkout -B desman-installed REF'
        Write-Output 'uv venv -p 3.12; uv pip install --python .venv\Scripts\python.exe -e .; desman gui --shortcut'
        if (!$NoStart) { Write-Output 'desman gui' }
        Write-Output 'Updates: desman global update --show; desman global update --apply NUMBER'
    }
    return
}
try {
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Safe-Directory $App
    $App = [IO.Path]::GetFullPath($App).TrimEnd('\', '/')
    foreach ($Path in @($Data, $Config, $HomeConfig)) {
        $Full = [IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
        if ($Full -eq $App -or $Full.StartsWith($App + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Data/config must be outside the install directory.'
        }
    }
    if (Test-Path -LiteralPath $App) {
        foreach ($Required in @('.desman-install', '.git', 'src\desman\cli.py')) {
            if (!(Test-Path -LiteralPath (Join-Path $App $Required))) { throw "Refusing non-DESMAN directory: $App" }
        }
    }
    if ($Uninstall) {
        if (!(Test-Path -LiteralPath $App)) {
            Write-Host 'No verified installer-owned app directory; nothing removed.'
            return
        }
        Confirm-Action "Remove $App and DESMAN shortcuts; keep config and data."
        if ($Purge) {
            Safe-Directory $Data
            Safe-Directory $Config
            Confirm-Action "PURGE permanently removes $Data ; $Config ; $App\config.json (keys, history, workspaces, node key)."
        } elseif (Test-Path -LiteralPath "$App\config.json") {
            if (Test-Path -LiteralPath "$HomeConfig\config.json") { throw 'Config backup exists; preserve app/config.json manually first.' }
            New-Item -ItemType Directory -Force -Path $HomeConfig | Out-Null
            Move-Item -LiteralPath "$App\config.json" -Destination "$HomeConfig\config.json"
        }
        $Shell = New-Object -ComObject WScript.Shell
        foreach ($Folder in @([Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs'))) {
            $Link = Join-Path $Folder 'DESMAN.lnk'
            if (Test-Path -LiteralPath $Link) {
                $Target = $Shell.CreateShortcut($Link).TargetPath
                if ($Target -and [IO.Path]::GetFullPath($Target).StartsWith($App + '\', [StringComparison]::OrdinalIgnoreCase)) {
                    Remove-Item -LiteralPath $Link -Force
                }
            }
        }
        if (Test-Path -LiteralPath $App) { Remove-Item -LiteralPath $App -Recurse -Force }
        if ($Purge) {
            foreach ($Path in @($Data, $Config)) {
                if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Recurse -Force }
            }
        }
        Write-Host "DESMAN removed. uv, git and $MinGit were kept."
        return
    }
    if ($Purge) { throw '-Purge requires -Uninstall.' }
    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        Confirm-Action 'Install uv (and Python 3.12 as needed) from https://astral.sh/uv/install.ps1.'
        # Only the official uv installer is evaluated as PowerShell source.
        irm https://astral.sh/uv/install.ps1 | iex
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
        if (!(Get-Command uv -ErrorAction SilentlyContinue)) { throw 'uv not found after installation.' }
    }
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        if (Test-Path -LiteralPath "$MinGit\cmd\git.exe") {
            $env:PATH = "$MinGit\cmd;$env:PATH"
        } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
            Confirm-Action 'Install Git with winget install --id Git.Git -e --source winget.'
            Run 'winget' @('install', '--id', 'Git.Git', '-e', '--source', 'winget')
            $env:PATH = $env:PATH + ';' + [Environment]::GetEnvironmentVariable('PATH', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('PATH', 'User')
        } else {
            Confirm-Action "Download official portable MinGit x64 to $MinGit (no winget found)."
            if (Test-Path -LiteralPath $MinGit) { throw "Refusing existing incomplete MinGit directory: $MinGit" }
            $Release = Invoke-RestMethod -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest'
            $Asset = @($Release.assets | Where-Object { $_.name -match '^MinGit-[0-9.]+-64-bit\.zip$' })
            if ($Asset.Count -ne 1) { throw 'Expected one official MinGit x64 zip.' }
            $Url = [string]$Asset[0].browser_download_url
            if ($Url -notmatch '^https://github\.com/git-for-windows/git/releases/download/[^/]+/MinGit-[0-9.]+-64-bit\.zip$') { throw 'Unexpected MinGit URL.' }
            $Zip = Join-Path ([IO.Path]::GetTempPath()) ([IO.Path]::GetRandomFileName() + '.zip')
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $Zip
                Expand-Archive -LiteralPath $Zip -DestinationPath $MinGit
            } finally { if (Test-Path -LiteralPath $Zip) { Remove-Item -LiteralPath $Zip -Force } }
            $env:PATH = "$MinGit\cmd;$env:PATH"
        }
        if (!(Get-Command git -ErrorAction SilentlyContinue)) { throw 'git not found; reopen PowerShell and retry.' }
    }
    # Portable git must remain discoverable by later signed-update commands.
    if (Test-Path -LiteralPath "$MinGit\cmd\git.exe") {
        $UserPath = [Environment]::GetEnvironmentVariable('PATH', 'User')
        if (@($UserPath -split ';') -notcontains "$MinGit\cmd") {
            [Environment]::SetEnvironmentVariable('PATH', "$MinGit\cmd;$UserPath", 'User')
            Write-Host 'Portable git added to user PATH. Sign out and back in before using shortcuts for signed updates.'
        }
    }
    if (Test-Path -LiteralPath "$App\.git") {
        $Origin = & git -C $App remote get-url origin
        if ($LASTEXITCODE -ne 0 -or $Origin -ne $Repository) { throw 'Unexpected origin.' }
        $Dirty = & git -C $App status --porcelain --untracked-files=no
        if ($LASTEXITCODE -ne 0 -or $Dirty) { throw 'Tracked changes exist; preserve them before reinstalling.' }
    }
    if (!$Ref) {
        $Tags = & git ls-remote --tags --refs --sort=-version:refname $Repository 'v*'
        if ($LASTEXITCODE -ne 0) { throw 'Could not list tags.' }
        if ($Tags) { $Ref = (($Tags | Select-Object -First 1) -split 'refs/tags/')[1] }
        if (!$Ref) { $Ref = 'master'; Write-Host 'No v* tag found; using master until the first release tag exists.' }
    }
    Run 'git' @('check-ref-format', "refs/heads/$Ref")
    if (!(Test-Path -LiteralPath "$App\.git")) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $App) | Out-Null
        Run 'git' @('clone', $Repository, $App)
        New-Item -ItemType File -Path "$App\.desman-install" | Out-Null
    }
    Push-Location -LiteralPath $App
    try {
        Run 'git' @('fetch', 'origin', '--tags')
        & git show-ref --verify --quiet "refs/tags/$Ref"
        $Target = "refs/remotes/origin/$Ref"
        if ($LASTEXITCODE -eq 0) { $Target = "refs/tags/$Ref" }
        Run 'git' @('checkout', '-B', 'desman-installed', $Target)
        if (!(Test-Path -LiteralPath '.venv')) { Run 'uv' @('venv', '-p', '3.12') }
        Run 'uv' @('pip', 'install', '--python', '.venv\Scripts\python.exe', '-e', '.')
        Run '.\.venv\Scripts\desman.exe' @('gui', '--shortcut')
        Write-Host 'Installed. Updates: desman global update --show, then --apply NUMBER.'
        if (!$NoStart) { Run '.\.venv\Scripts\desman.exe' @('gui') }
    } finally { Pop-Location }
} catch {
    Write-Error $_
    exit 1
}
