#!/usr/bin/env python3
"""Launch an unmodified legacy OSTrack test entrypoint on modern PyTorch.

The upstream test-data import references the removed torch._six.string_classes
symbol. This process-local shim restores that symbol only; it does not change
the tracker, its checkpoint, dataset, result writer, or evaluator.
"""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: run_legacy_ostrack.py PATH_TO_TRACKING_TEST.py [test arguments]")
    test_script = Path(sys.argv[1]).resolve()
    legacy = types.ModuleType("torch._six")
    legacy.string_classes = (str, bytes)
    sys.modules["torch._six"] = legacy
    setattr(torch, "_six", legacy)
    sys.argv = [str(test_script), *sys.argv[2:]]
    runpy.run_path(str(test_script), run_name="__main__")


if __name__ == "__main__":
    main()
