# Astra Current Monitor

Windows desktop software for reading a two-row seven-segment display from a camera. The upper blue row is voltage (V); the lower red row is current (A). It records current peaks, voltage at the peak, and maximum voltage, and exports CSV.

## Recognition

The recognizer uses color separation, seven-segment geometry, slant correction, threshold ensemble checks, and rejection of ambiguous frames. It does not use CNN inference or a network service. This suits fixed displays and a small calibration set: recognition stays inspectable and offline, and uncertain values are rejected instead of guessed.

## Run from source

Requires Windows and Python 3.12.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
$source = Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'main.py') } | Select-Object -First 1 -ExpandProperty FullName
pip install -r (Join-Path $source 'requirements.txt')
python (Join-Path $source 'main.py') --demo
```

Run camera mode without `--demo`. To run tests, use `Push-Location $source`, then `python -m unittest discover -p "test_*.py" -v`, then `Pop-Location`. To include the migrated photo set, set `ASTRA_TEST_IMAGE_DIR` to the absolute path of `test_images` before running tests.

## Windows release

Choose one of the two release packages:

- Offline: `AstraCurrentMonitor-v1.3.1-windows-offline.zip` includes the executable, Python runtime, Qt, OpenCV, and dependencies. Extract the whole folder and run `AstraCurrentMonitor.exe`; no network or Python installation is needed.
- Online: `AstraCurrentMonitor-v1.3.1-windows-online.zip` contains a small PowerShell installer. It checks Windows architecture, network access, and free space, downloads the complete offline package, verifies the GitHub SHA-256 digest, installs under `%LOCALAPPDATA%\Programs`, and launches the app.

Both versions support selectable DirectShow/MSMF cameras. Refresh the camera list after connecting or disconnecting a device. The application stores records and settings in a writable `data` directory beside the executable.

## Validation

Automated tests cover measurement timing and storage, seven-segment readings, edge clipping, synthetic dual-color readings, and the migrated ten-photo regression set in `test_images`. Actual camera accuracy still depends on focus, exposure, display type, and framing; verify readings against the physical meter before measuring.

## License

GPL-3.0-only. Third-party license texts are in `THIRD_PARTY_LICENSES/`; notices are in `THIRD_PARTY_NOTICES.txt`.
