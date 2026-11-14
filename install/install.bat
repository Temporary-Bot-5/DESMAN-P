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
rem One line, read whole before it runs: -Uninstall deletes this very file, and cmd would look
rem for it again after PowerShell returns, even for exit /b ("The system cannot find the path
rem specified"). When it is gone, (goto) ends the batch without reading it; a plain exit would
rem close the terminal the person typed the command in. The exit code is set explicitly.
powershell -NoProfile -ExecutionPolicy Bypass -File "%DESMAN_INSTALLER%" %* && (if not exist "%~f0" ((goto) 2>nul & ver >nul) else exit /b 0) || (pause & if not exist "%~f0" ((goto) 2>nul & cmd /c exit 1) else exit /b 1)
:failed
pause
exit /b 1
