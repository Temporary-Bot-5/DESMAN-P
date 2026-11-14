@echo off
setlocal
set "DESMAN_INSTALLER_REF=master"
set "DESMAN_INSTALLER=%~dp0install.ps1"
if exist "%DESMAN_INSTALLER%" goto run
set "DESMAN_INSTALLER=%TEMP%\desman-install-%RANDOM%-%RANDOM%.ps1"
echo Downloading install.ps1 from GitHub over HTTPS (ref %DESMAN_INSTALLER_REF%) to "%DESMAN_INSTALLER%".
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/Temporary-Bot-5/DESMAN-P/%DESMAN_INSTALLER_REF%/install/install.ps1' -OutFile $env:DESMAN_INSTALLER"
if errorlevel 1 goto failed
:run
echo Running "%DESMAN_INSTALLER%".
rem One line, read whole before it runs: -Uninstall deletes this very file, and cmd would read
rem the next line from it ("The system cannot find the path specified"). (goto) ends the batch
rem without reading on; exit /b would close a terminal the person typed the command in.
powershell -NoProfile -ExecutionPolicy Bypass -File "%DESMAN_INSTALLER%" %* && (if not exist "%~f0" ((goto) 2>nul & ver >nul) else exit /b 0) || (pause & if not exist "%~f0" ((goto) 2>nul) else exit /b 1)
:failed
pause
exit /b 1
