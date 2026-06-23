# PyInstaller spec — bundles the FastAPI backend as a single .exe.
# Frontend ships separately as a static Next.js export.
# Build:  pyinstaller dist/syncforge.spec
from __future__ import annotations
import os
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 — SPECPATH injected by PyInstaller
BACKEND = ROOT / "backend"

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("fastapi")
    + collect_submodules("starlette")
    + collect_submodules("edge_tts")
    + ["app.main", "app.services.runner",
       "app.auth.users", "app.billing.stripe_handler",
       "app.core.license", "app.core.quota", "app.core.audit",
       "app.updater.check"]
)

datas = []
datas += collect_data_files("certifi")        # TLS roots for HTTPS APIs
datas += collect_data_files("edge_tts")       # voices list, certificates

a = Analysis(
    [str(BACKEND / "app" / "__main__.py")],
    pathex=[str(BACKEND)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "PIL.ImageTk", "pytest"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="syncforge",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,                 # keep console: backend logs are useful
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "dist" / "icon.ico") if (ROOT / "dist" / "icon.ico").exists() else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="syncforge",
)
