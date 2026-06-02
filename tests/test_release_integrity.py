from __future__ import annotations

import subprocess
import sys


def test_release_integrity_contract() -> None:
    subprocess.run([sys.executable, "scripts/verify_release.py"], check=True)
