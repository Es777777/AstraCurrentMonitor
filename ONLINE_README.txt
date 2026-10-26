AstraCurrentMonitor v1.3.2 - Online installer

Requirements: Windows 10/11 x64 or ARM64 and an internet connection.
No Python installation is required. The installer downloads the complete
application and bundled runtime from the official GitHub release, verifies the
published SHA-256 digest, installs under LocalAppData\Programs, and starts it.

Extract this ZIP, then run Install-Online.ps1 from PowerShell:

  powershell -NoProfile -ExecutionPolicy Bypass -File .\Install-Online.ps1

For a computer without internet access, download the separate offline ZIP on
another computer and copy it over. Extract the entire folder before starting
AstraCurrentMonitor.exe; keep the _internal folder beside the executable.
