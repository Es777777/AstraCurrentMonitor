$ErrorActionPreference = 'Stop'
$version = 'v1.3.1'
$repository = 'Es777777/AstraCurrentMonitor'
$assetName = 'AstraCurrentMonitor-v1.3.1-windows-offline.zip'

if ($env:OS -ne 'Windows_NT') { throw 'This installer supports Windows only.' }
$architecture = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
if ($architecture -notin @('AMD64', 'ARM64')) { throw "Unsupported Windows architecture: $architecture" }

$api = "https://api.github.com/repos/$repository/releases/tags/$version"
Write-Host "Checking Windows $architecture, network access, and release $version..."
$release = Invoke-RestMethod -Uri $api -Headers @{ 'User-Agent' = 'AstraCurrentMonitor-Online-Installer' }
$asset = $release.assets | Where-Object { $_.name -eq $assetName } | Select-Object -First 1
if (-not $asset) { throw "Release asset '$assetName' was not found." }

$installRoot = Join-Path $env:LOCALAPPDATA "Programs\AstraCurrentMonitor\$version"
$drive = [System.IO.Path]::GetPathRoot($installRoot)
$driveName = $drive.Substring(0,1)
$freeBytes = (Get-PSDrive -Name $driveName -ErrorAction Stop).Free
if ($freeBytes -lt 400MB) { throw 'At least 400 MB of free space is required for download and installation.' }

$work = Join-Path $env:TEMP ("AstraCurrentMonitor-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $work | Out-Null
try {
    $archive = Join-Path $work $assetName
    Write-Host "Downloading the complete offline runtime ($([math]::Round($asset.size / 1MB)) MB)..."
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $archive
    if (-not $asset.digest -or -not $asset.digest.StartsWith('sha256:')) { throw 'The release has no SHA-256 digest. Refusing to install an unverifiable package.' }
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    $expected = $asset.digest.Substring(7).ToLowerInvariant()
    if ($actual -ne $expected) { throw 'Downloaded package SHA-256 does not match the published release digest.' }

    $expanded = Join-Path $work 'expanded'
    Expand-Archive -LiteralPath $archive -DestinationPath $expanded
    $exe = Get-ChildItem -LiteralPath $expanded -Filter 'AstraCurrentMonitor.exe' -Recurse | Select-Object -First 1
    if (-not $exe) { throw 'The downloaded package does not contain the application executable.' }
    New-Item -ItemType Directory -Path $installRoot -Force | Out-Null
    Get-ChildItem -LiteralPath $exe.Directory.FullName | Copy-Item -Destination $installRoot -Recurse -Force
    Write-Host "Installed to $installRoot"
    Start-Process -FilePath (Join-Path $installRoot 'AstraCurrentMonitor.exe')
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
