$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Get-ChildItem -LiteralPath $root -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'main.py') } | Select-Object -First 1 -ExpandProperty FullName
if (-not $source) { throw 'Could not find the source directory containing main.py.' }
$python = Join-Path $root '.venv\Scripts\python.exe'
$dist = Join-Path $source 'dist'
$release = Join-Path $root 'release'
$version = '1.3.1'

if (-not (Test-Path $python)) { throw 'Create the project .venv and install requirements.txt first.' }
Push-Location $source
try {
    & $python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name AstraCurrentMonitor main.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
} finally { Pop-Location }

New-Item -ItemType Directory -Path $release -Force | Out-Null
$bundle = Join-Path $release "AstraCurrentMonitor-v$version-offline"
if (Test-Path $bundle) { Remove-Item -LiteralPath $bundle -Recurse -Force }
Copy-Item -LiteralPath (Join-Path $dist 'AstraCurrentMonitor') -Destination $bundle -Recurse
Copy-Item -LiteralPath (Join-Path $root 'USER_GUIDE.txt') -Destination $bundle
Copy-Item -LiteralPath (Join-Path $root 'VALIDATION.txt') -Destination $bundle
Copy-Item -LiteralPath (Join-Path $root 'THIRD_PARTY_NOTICES.txt') -Destination $bundle
Copy-Item -LiteralPath (Join-Path $root 'LICENSE') -Destination $bundle
Copy-Item -LiteralPath (Join-Path $root 'THIRD_PARTY_LICENSES') -Destination $bundle -Recurse
Compress-Archive -LiteralPath $bundle -DestinationPath (Join-Path $release "AstraCurrentMonitor-v$version-windows-offline.zip") -Force

$online = Join-Path $release "AstraCurrentMonitor-v$version-online"
if (Test-Path $online) { Remove-Item -LiteralPath $online -Recurse -Force }
New-Item -ItemType Directory -Path $online -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $root 'Install-Online.ps1') -Destination $online
Copy-Item -LiteralPath (Join-Path $root 'ONLINE_README.txt') -Destination $online
Compress-Archive -LiteralPath $online -DestinationPath (Join-Path $release "AstraCurrentMonitor-v$version-windows-online.zip") -Force
Write-Host (Join-Path $release "AstraCurrentMonitor-v$version-windows-offline.zip")
Write-Host (Join-Path $release "AstraCurrentMonitor-v$version-windows-online.zip")
