$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$source = Get-ChildItem -LiteralPath $root -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'main.py') } | Select-Object -First 1 -ExpandProperty FullName
if (-not $source) { throw 'Could not find the source directory containing main.py.' }
$python = Join-Path $root '.venv\Scripts\python.exe'
$dist = Join-Path $source 'dist'
$release = Join-Path $root 'release'

if (-not (Test-Path $python)) { throw 'Create the project .venv and install requirements.txt first.' }
Push-Location $source
try {
    & $python -m PyInstaller --noconfirm --clean --onedir --windowed `
        --name AstraCurrentMonitor main.py
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
} finally { Pop-Location }

if (Test-Path $release) { Remove-Item -LiteralPath $release -Recurse -Force }
New-Item -ItemType Directory -Path $release | Out-Null
$bundle = Join-Path $release 'AstraCurrentMonitor-v1.3.0-windows'
Copy-Item -LiteralPath (Join-Path $dist 'AstraCurrentMonitor') -Destination $bundle -Recurse
$docs = @(Get-ChildItem -LiteralPath $root -File -Filter '*.txt' | Where-Object { $_.Length -gt 2000 } | Sort-Object Length -Descending)
Copy-Item -LiteralPath $docs[0].FullName -Destination (Join-Path $bundle 'USER_GUIDE.txt')
Copy-Item -LiteralPath $docs[1].FullName -Destination (Join-Path $bundle 'VALIDATION.txt')
Copy-Item -LiteralPath (Join-Path $root 'THIRD_PARTY_NOTICES.txt') -Destination $bundle
Copy-Item -LiteralPath (Join-Path $root 'LICENSE') -Destination $bundle
Copy-Item -LiteralPath (Join-Path $root 'THIRD_PARTY_LICENSES') -Destination $bundle -Recurse
Compress-Archive -LiteralPath $bundle -DestinationPath (Join-Path $release 'AstraCurrentMonitor-v1.3.0-windows.zip') -Force
Write-Host (Join-Path $release 'AstraCurrentMonitor-v1.3.0-windows.zip')
