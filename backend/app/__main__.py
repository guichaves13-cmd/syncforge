"""Entry point for PyInstaller: `python -m app` (or compiled .exe).

Launches uvicorn against the FastAPI app.
"""
from __future__ import annotations
import os
import sys

import uvicorn


def main() -> None:
    host = os.getenv("SYNCFORGE_HOST", "127.0.0.1")
    port = int(os.getenv("SYNCFORGE_PORT", "8000"))
    reload = "--reload" in sys.argv
    uvicorn.run("app.main:app", host=host, port=port, reload=reload,
                 log_level=os.getenv("SYNCFORGE_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
