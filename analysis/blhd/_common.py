"""Shared helpers for the BL-HD analysis scripts: repo-root / results-dir resolution."""

from pathlib import Path

__all__ = ["REPO_ROOT", "RESULTS_DIR"]

# analysis/blhd/_common.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "blhd"
