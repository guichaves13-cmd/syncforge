@echo off
setlocal
echo === SyncForge installer ===

set INSTALL_DIR=%LOCALAPPDATA%\SyncForge
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo Copying files to %INSTALL_DIR%...
xcopy /E /I /Y "%~dp0syncforge" "%INSTALL_DIR%\backend" >nul
xcopy /E /I /Y "%~dp0frontend" "%INSTALL_DIR%\frontend" >nul
copy /Y "%~dp0LICENSE" "%INSTALL_DIR%\LICENSE" >nul
copy /Y "%~dp0README.md" "%INSTALL_DIR%\README.md" >nul

echo Creating Desktop shortcut...
set SHORTCUT=%USERPROFILE%\Desktop\SyncForge.lnk
powershell -NoProfile -Command ^
  "$s = (New-Object -ComObject WScript.Shell).CreateShortcut('%SHORTCUT%');" ^
  "$s.TargetPath = '%INSTALL_DIR%\backend\syncforge.exe';" ^
  "$s.WorkingDirectory = '%INSTALL_DIR%';" ^
  "$s.IconLocation = '%INSTALL_DIR%\backend\syncforge.exe,0';" ^
  "$s.Save()"

echo.
echo Installed to: %INSTALL_DIR%
echo Launch from the Desktop shortcut.
endlocal
pause
