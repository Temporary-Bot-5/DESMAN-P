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
powershell -NoProfile -ExecutionPolicy Bypass -File "%DESMAN_INSTALLER%" %*
if errorlevel 1 goto failed
exit /b 0
:failed
pause
exit /b 1
