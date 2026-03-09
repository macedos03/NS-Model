from __future__ import annotations

import sys
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent
YC_DIR = SRC_DIR / "yc"

if str(YC_DIR) not in sys.path:
	sys.path.insert(0, str(YC_DIR))
