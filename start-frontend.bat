@echo off
cd /d "%~dp0frontend"
if not exist node_modules (
  echo Installing deps...
  call npm install
)
call npm run dev
