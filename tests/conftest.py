"""Shared pytest fixtures for the CEE+ test suite.

This file:
  - Puts the project root on sys.path so tests can `import main`.
  - Loads `main.py` once per session (the import has side effects: builds a
    Dash app; reading it once amortises that cost).
  - Exposes the raw source text of main.py for prompt-string introspection.
  - Lists the 70 push_test ground-truth files for parametrized C-series tests.

Tests are READ-ONLY against main.py and against GT files. See TESTS.md and
tests/README.md for the contract.
"""
from __future__ import annotations

import os as _os
# Unit tests must not load the optional sentence-transformers model (a 5s display-only
# diagnostic); the live app still computes it. Cleared by unsetting CEE_DISABLE_SEMANTIC.
_os.environ.setdefault("CEE_DISABLE_SEMANTIC", "1")

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import pytest


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = PROJECT_ROOT / "main.py"
GT_PUSH_TEST = PROJECT_ROOT / "exports" / "ground_truth" / "candidates" / "push_test"
GT_VERIFIED = PROJECT_ROOT / "exports" / "ground_truth" / "verified"
MEMORY_DIR = Path.home() / ".claude" / "projects" / "-Users-sunny-Documents-CEE-" / "memory"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def main_module():
    """Import main.py once per session. Raises if import fails — that
    failure surfaces as G2 (all-required-imports-resolve) in test_g_codebase."""
    import importlib
    return importlib.import_module("main")


@pytest.fixture(scope="session")
def main_source() -> str:
    """Raw text of main.py for prompt-string and regex introspection."""
    return MAIN_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="session")
def main_prompt(main_source: str) -> str:
    """The MAIN Qwen prompt block (state vocab + effect vocab + rules).
    Bounded by 'State vocabulary' header (~line 31) and the next major
    section that follows the rules block. We grab a generous superset
    of the prompt so substring assertions work without false negatives.
    """
    start = main_source.find("## State vocabulary")
    assert start != -1, "Could not locate '## State vocabulary' in main.py — main prompt missing?"
    # The main prompt ends where the GRAPH_B_PROMPT begins (line ~302 area).
    end = main_source.find("GRAPH_B_PROMPT")
    assert end != -1, "Could not locate GRAPH_B_PROMPT in main.py"
    return main_source[start:end]


@pytest.fixture(scope="session")
def graph_b_prompt(main_module) -> str:
    """The GRAPH_B_PROMPT string itself, as defined in main.py."""
    return getattr(main_module, "GRAPH_B_PROMPT")


# ---------------------------------------------------------------------------
# Ground-truth file fixtures
# ---------------------------------------------------------------------------
def _list_gt_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(folder.glob("*.gt.json"))


@pytest.fixture(scope="session")
def push_test_gt_paths() -> list[Path]:
    """The 70 push_test GT files used for C-series parametrization."""
    paths = _list_gt_files(GT_PUSH_TEST)
    return paths


def _load_gt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# Parametrization helpers used by C-series tests.
def all_push_test_gts() -> Iterable[tuple[str, Path]]:
    """Yield (id, path) pairs for pytest.mark.parametrize. Lazy: only walks
    the directory at collection time, so test discovery is cheap when the
    folder is absent."""
    paths = _list_gt_files(GT_PUSH_TEST)
    for p in paths:
        yield pytest.param(p, id=p.name)


@pytest.fixture
def gt_loader():
    """Function fixture for tests that want to load a GT lazily."""
    def _load(path: Path) -> dict[str, Any]:
        return _load_gt(path)
    return _load


# ---------------------------------------------------------------------------
# Sample GTs (hand-built) for E-series synthetic tests
# ---------------------------------------------------------------------------
SAMPLE_GTS_DIR = Path(__file__).parent / "fixtures" / "sample_gts"


@pytest.fixture
def sample_gt_minimal_burning_house() -> dict[str, Any]:
    """Two-node hand-built fixture: burning house + uninjured person, one
    may_harm edge. Used for E-series synonym / soft-tier tests."""
    return {
        "image_filename": "synthetic_burning_house.jpg",
        "nodes": [
            {"id": "house_1", "label": "house", "state": "burning", "hazardous": True, "inferred": False},
            {"id": "person_1", "label": "person", "state": "stationary", "hazardous": False, "inferred": False},
        ],
        "edges": [
            {"source": "house_1", "target": "person_1", "effect": "may_harm", "via_state": "burning"},
        ],
    }


@pytest.fixture
def sample_gt_mutual_worsens() -> dict[str, Any]:
    """Two burning houses with mutual worsens edges (E7 fixture)."""
    return {
        "image_filename": "synthetic_mutual.jpg",
        "nodes": [
            {"id": "house_1", "label": "house", "state": "burning", "hazardous": True, "inferred": False},
            {"id": "house_2", "label": "house", "state": "burning", "hazardous": True, "inferred": False},
        ],
        "edges": [
            {"source": "house_1", "target": "house_2", "effect": "worsens", "via_state": "burning"},
            {"source": "house_2", "target": "house_1", "effect": "worsens", "via_state": "burning"},
        ],
    }
