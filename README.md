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

Download `AstraCurrentMonitor-v1.3.0-windows.zip` from GitHub Releases, extract the complete folder, and run `AstraCurrentMonitor.exe`. The application stores records and settings in a `data` directory next to the executable. Keep the extracted folder writable.

## Validation

Automated tests cover measurement timing and storage, seven-segment readings, edge clipping, synthetic dual-color readings, and the migrated ten-photo regression set in `test_images`. Actual camera accuracy still depends on focus, exposure, display type, and framing; verify readings against the physical meter before measuring.

## License

GPL-3.0-only. Third-party license texts are in `THIRD_PARTY_LICENSES/`; notices are in `THIRD_PARTY_NOTICES.txt`.
