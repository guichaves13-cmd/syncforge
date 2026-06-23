@echo off
setlocal
cd /d "%~dp0.."
echo === SyncForge build ===

REM Backend → single-folder bundle
if not exist backend\venv (
    echo Creating venv...
    python -m venv backend\venv
)
call backend\venv\Scripts\activate
pip install --quiet --upgrade pip
pip install --quiet -r backend\requirements.txt
pip install --quiet pyinstaller

pyinstaller --noconfirm --clean --workpath dist\.build --distpath dist\out dist\syncforge.spec

REM Frontend → static export
pushd frontend
if not exist node_modules (
    call npm install
)
call npm run build
popd

REM Bundle everything into dist\out\SyncForge\
mkdir dist\out\SyncForge\frontend 2>nul
xcopy /E /I /Y frontend\.next dist\out\SyncForge\frontend\.next >nul
xcopy /E /I /Y frontend\public dist\out\SyncForge\frontend\public >nul
copy /Y LICENSE dist\out\SyncForge\LICENSE >nul
copy /Y README.md dist\out\SyncForge\README.md >nul
copy /Y dist\install.bat dist\out\SyncForge\install.bat >nul

echo.
echo === Build complete ===
echo Output:  dist\out\SyncForge\
endlocal
