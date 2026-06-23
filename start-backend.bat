@echo off
cd /d "%~dp0backend"
if not exist venv (
  echo Creating venv...
  python -m venv venv
  call venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call venv\Scripts\activate
)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
