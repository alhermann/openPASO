"""
Pytest configuration: ensure all solver backends are discoverable.

Environment variables (set these for your machine, or they'll be auto-detected):
  FOURC_ROOT   — path to 4C source tree
  FOURC_BINARY — path to 4C binary
  LD_LIBRARY_PATH — include 4C dependency libs if needed
"""
import os
import shutil

# 4C Multiphysics — auto-detect if not set
if not os.environ.get("FOURC_ROOT"):
    # Try common locations
    for candidate in [os.path.expanduser("~/4C"), "/opt/4C", "/usr/local/4C"]:
        if os.path.isdir(candidate):
            os.environ["FOURC_ROOT"] = candidate
            break

if not os.environ.get("FOURC_BINARY"):
    # Try finding in FOURC_ROOT/build or on PATH
    root = os.environ.get("FOURC_ROOT", "")
    for d in ["build", "build/release", "build/debug"]:
        candidate = os.path.join(root, d, "4C")
        if os.path.isfile(candidate):
            os.environ["FOURC_BINARY"] = candidate
            break
    else:
        p = shutil.which("4C")
        if p:
            os.environ["FOURC_BINARY"] = p

# 4C runtime dependencies (if present)
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
for dep_lib in ["/opt/4C-dependencies/lib", os.path.join(os.environ.get("FOURC_ROOT", ""), "lib")]:
    if os.path.isdir(dep_lib) and dep_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{dep_lib}:{ld_path}" if ld_path else dep_lib
        break

# PyVista off-screen rendering
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

# The solver probe lives in tests/backend_probe.py. conftest is imported before
# pytest puts the tests directory on sys.path, so it has to add it itself.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from backend_probe import backend_state, need_backend  # noqa: E402,F401
