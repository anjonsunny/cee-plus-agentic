"""Hermetic tests for the scene runner's caption resolution.

Run:  pytest agentic/test_run_scenes.py -q
"""
from __future__ import annotations

from pathlib import Path

from agentic.run_scenes import SCENE_NAMES, resolve_caption


def test_scene_roster_is_the_agreed_six():
    assert SCENE_NAMES == [
        "A_fire", "B_pool", "C_tanker_fire",
        "D_aerial_spill", "E_collapse", "F_park_control",
    ]


def test_sidecar_caption_read_and_stripped(tmp_path):
    img = tmp_path / "A_fire.png"
    img.write_bytes(b"fake")
    (tmp_path / "A_fire.txt").write_text("  A house on fire at night.  \n")
    assert resolve_caption(img) == "A house on fire at night."


def test_missing_sidecar_gives_empty_caption(tmp_path):
    img = tmp_path / "B_pool.jpg"
    img.write_bytes(b"fake")
    assert resolve_caption(img) == ""


def test_repo_sidecars_exist_for_all_six_scenes():
    """The authored captions are versioned next to the images. This test
    runs against the real repo layout; skip when scenes are absent (CI
    environments without the images)."""
    scenes_dir = Path(__file__).resolve().parent.parent / "experiments" / "agentic_scenes"
    import pytest

    if not scenes_dir.exists():
        pytest.skip("scene directory not present")
    missing = [n for n in SCENE_NAMES if not (scenes_dir / f"{n}.txt").exists()]
    assert not missing, f"missing sidecar captions: {missing}"
