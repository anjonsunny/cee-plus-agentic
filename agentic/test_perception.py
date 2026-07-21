"""Hermetic Stage 1 tests: vocabulary, contract, id assignment, overlay.

No VLM, no detector weights, no network: the grounding and mask steps are
monkeypatched. What IS tested is everything deterministic that Stage 1
promises: canonicalization behavior, the escape hatch, state kinds, id
form, the Pydantic contract, and overlay rendering.

Run:  pytest agentic/test_perception.py -q
"""
from __future__ import annotations

import json

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from agentic import perception  # noqa: E402
from agentic.vocabulary import canonicalize_label, family_of, is_life  # noqa: E402


def test_canonicalize_verbatim():
    assert canonicalize_label("house") == ("house", "", True)


def test_canonicalize_synonym_logged():
    label, note, ok = canonicalize_label("sedan")
    assert label == "car" and ok and note == "synonym:sedan->car"


def test_canonicalize_case_and_spaces():
    assert canonicalize_label(" Tanker Truck ")[0] == "tanker_truck"


def test_canonicalize_trivial_plural():
    assert canonicalize_label("cars")[0] == "car"


def test_escape_hatch():
    label, note, ok = canonicalize_label("zamboni")
    assert label == "other" and not ok and note == "extension:zamboni"


def test_families_and_life():
    assert family_of("child") == "person"
    assert family_of("smoke") == "hazard_media"
    assert is_life("dog") and is_life("hazmat_worker") and not is_life("fence")


def test_state_kinds():
    assert perception.state_kind("burning") == "hazard_bearing"
    assert perception.state_kind("drowning") == "at_risk"
    assert perception.state_kind("intact") == "normal"
    assert perception.state_kind("submerged") == "hazard_bearing"  # synonym->flooded
    assert perception.state_kind("floating") == "unknown"


def test_perception_prompt_contains_vocab_and_no_bbox_rule():
    prompt = perception.build_perception_prompt()
    assert "tanker_truck" in prompt and "lifeguard_chair" in prompt
    assert "Do NOT output bounding boxes" in prompt


@pytest.fixture()
def fake_models(monkeypatch):
    """Stub the detector and SAM so the pipeline runs offline."""

    def fake_ground(image, entities):
        for i, e in enumerate(entities):
            e.pop("_phrase", None)
            e["bbox"] = [10 + i * 20, 10, 100 + i * 20, 120]
            e["box_confidence"] = 0.9 - i * 0.1
            e["box_source"] = "grounding_dino"
        # Last entity deliberately unlocalized when there are 3+.
        if len(entities) >= 3:
            entities[-1].update(bbox=None, box_confidence=0.0, box_source="none")
        return entities

    def fake_mask(image, bbox):
        return Image.new("L", image.size, 255)

    monkeypatch.setattr(perception, "ground_entities", fake_ground)
    monkeypatch.setattr(perception, "mask_for_box", fake_mask)


def test_full_standin_run(tmp_path, fake_models):
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (400, 300), "gray").save(img_path)
    entities = [
        {"label": "house", "state": "burning", "description": "left"},
        {"label": "sedan", "state": "intact", "description": "foreground"},
        {"label": "zamboni", "state": "stationary", "description": "odd machine"},
    ]
    result = perception.run_perception(
        img_path, entities=entities, out_dir=tmp_path / "out"
    )
    ids = [o.object_id for o in result.detected_objects]
    assert ids == ["house_1", "car_1", "other_1"]           # canonicalized + ids
    assert result.detected_objects[0].state_kind == "hazard_bearing"
    assert result.detected_objects[1].label_note == "synonym:sedan->car"
    assert result.detected_objects[2].vocab_extension        # escape hatch flagged
    assert result.unlocalized == ["other_1"]                 # honest about no box
    assert result.detected_objects[0].mask_path              # mask captured
    assert (tmp_path / "out" / "scene__overlay.png").exists()
    saved = json.loads((tmp_path / "out" / "scene__perception.json").read_text())
    assert saved["entity_source"] == "standin"
    assert len(saved["detected_objects"]) == 3


def test_duplicate_labels_get_distinct_ids(tmp_path, fake_models):
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (300, 200), "gray").save(img_path)
    entities = [
        {"label": "house", "state": "burning"},
        {"label": "house", "state": "intact"},
    ]
    result = perception.run_perception(
        img_path, entities=entities, with_masks=False, out_dir=tmp_path / "out"
    )
    assert [o.object_id for o in result.detected_objects] == ["house_1", "house_2"]
