"""Canonical project paths. Everything is relative to the repository root."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
RESULTS = REPORTS / "results"
SQL = ROOT / "sql"


def ensure_dirs() -> None:
    for p in (RAW, PROCESSED, FIGURES, RESULTS):
        p.mkdir(parents=True, exist_ok=True)
