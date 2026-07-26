#!/usr/bin/env python3
"""Compatibility entry point for the released TPTAF validation protocol."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tptaf.validation import (  # noqa: E402
    DetectionRunner,
    FusionRunner,
    LossRunner,
    ValidationOptions,
    ValidationResult,
    validate,
)

run = validate

__all__ = [
    "DetectionRunner",
    "FusionRunner",
    "LossRunner",
    "ValidationOptions",
    "ValidationResult",
    "run",
    "validate",
]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "tools" / "val.py"), run_name="__main__")
