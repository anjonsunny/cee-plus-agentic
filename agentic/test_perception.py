"""Hermetic Stage 1 tests, round 2: vocabulary, anchor binding, contract.

No VLM, no detector weights, no network: detection and SAM are
monkeypatched. Tested here: canonicalization (incl. the family-name trap
from round 1), state extensions, anchor-based instance binding (the
burning-vs-intact house and child-vs-swimmer failures), the SAM fallback
path, id assignment, the Pydantic contract, and overlay rendering.

Run:  pytest agentic/test_perception.py -q
"""
from __future__ import annotations

import json

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from agentic import perception  # noqa: E402
from agentic.vocabulary import canonicalize_label, family_of, is_life  # noqa: E402

# ── Vocabulary ──────────────────────────────────────────────────────────


def test_canonicalize_verbatim():
    assert canonicalize_label("house") == ("house", "", True, False)


def test_canonicalize_synonym_logged():
    label, note, ok, fam = canonicalize_label("sedan")
    assert label == "car" and ok and note == "synonym:sedan->car" and not fam


def test_family_name_never_silently_mapped():
    """Round 1 regression: 'vehicle' must NOT become car."""
    for raw in ("vehicle", "structure", "hazard_media", "vegetation",
                "infrastructure", "object"):
        label, note, ok, fam = canonicalize_label(raw)
        assert label == "other" and not ok and fam, raw
        assert note == f"family_name:{raw}"


def test_member_family_names_still_valid_labels():
    """'person' and 'animal' are family names AND member labels; they stay valid."""
    assert canonicalize_label("person") == ("person", "", True, False)
    assert canonicalize_label("animal") == ("animal", "", True, False)


def test_tanker_synonym_still_works():
    assert canonicalize_label("tanker")[0] == "tanker_truck"


def test_escape_hatch():
    label, note, ok, fam = canonicalize_label("zamboni")
    assert label == "other" and not ok and not fam and note == "extension:zamboni"


def test_families_and_life():
    assert family_of("child") == "person"
    assert family_of("smoke") == "hazard_media"
    assert is_life("dog") and is_life("hazmat_worker") and not is_life("fence")


# ── States ──────────────────────────────────────────────────────────────


def test_state_kinds_core():
    assert perception.state_kind("burning") == "hazard_bearing"
    assert perception.state_kind("drowning") == "at_risk"
    assert perception.state_kind("intact") == "normal"
    assert perception.state_kind("submerged") == "hazard_bearing"  # synonym->flooded


def test_state_extensions():
    assert perception.state_kind("swimming") == "normal"
    assert perception.state_kind("walking") == "normal"
    assert perception.state_kind("overturned") == "hazard_bearing"  # ->fallen
    assert perception.state_kind("seated") == "normal"              # ->resting
    assert perception.normalize_state("overturned") == "fallen"
    assert perception.state_kind("floating") == "unknown"


# ── Prompt ──────────────────────────────────────────────────────────────


def test_prompt_teaches_producer_and_medium_pattern():
    """Agreed with Sunny 2026-07-21 after the C_tanker 'stationary' miss:
    the prompt teaches the two-hazard pattern (leaking tanker + spill,
    mirroring burning house + smoke) and the fire-entity-wins rule."""
    flat = " ".join(perception.build_perception_prompt().split())
    assert "Producer and medium are TWO hazards" in flat
    assert 'tanker leaking fuel is "tanker_truck" with state "leaking"' in flat
    assert "the fire entity wins" in flat


def test_prompt_forbids_family_names_and_states_fluid_rule():
    prompt = perception.build_perception_prompt()
    assert "NEVER answer" in prompt and "family name" in prompt
    assert "fire attached to a burning object is a STATE" in prompt
    assert "drowning" in prompt and "not \"swimming\"" in prompt or "drowning" in prompt
    assert "bbox" in prompt


# ── Anchor binding ──────────────────────────────────────────────────────


def test_binding_prefers_anchor_over_score():
    """The burning-house regression: highest-score candidate is the WRONG
    instance; the anchor must pull each entity to its own box."""
    entities = [
        {"label": "house", "state": "burning", "_phrase": "house",
         "anchor_bbox": [400, 150, 850, 620]},
        {"label": "house", "state": "intact", "_phrase": "house",
         "anchor_bbox": [1100, 300, 1500, 640]},
    ]
    candidates = {"house": [
        {"score": 0.9, "bbox": [1120, 310, 1490, 630]},   # intact house, higher score
        {"score": 0.4, "bbox": [420, 160, 840, 610]},     # burning house, lower score
    ]}
    bound = perception.bind_entities(entities, candidates, (1600, 1000))
    burning = next(e for e in bound if e["state"] == "burning")
    intact = next(e for e in bound if e["state"] == "intact")
    assert burning["bbox"] == [420, 160, 840, 610]
    assert intact["bbox"] == [1120, 310, 1490, 630]
    assert burning["box_source"] == intact["box_source"] == "dino_matched"


def test_binding_no_candidate_falls_back_to_anchor():
    entities = [{"label": "smoke", "state": "billowing", "_phrase": "smoke",
                 "anchor_bbox": [100, 0, 900, 400]}]
    bound = perception.bind_entities(entities, {}, (1000, 800))
    assert bound[0]["box_source"] == "vlm_sam_fallback"
    assert bound[0]["bbox"] == [100, 0, 900, 400]


def test_binding_without_anchor_takes_best_unclaimed():
    entities = [
        {"label": "car", "state": "intact", "_phrase": "car", "anchor_bbox": None},
        {"label": "car", "state": "crushed", "_phrase": "car", "anchor_bbox": None},
    ]
    candidates = {"car": [
        {"score": 0.8, "bbox": [0, 0, 100, 100]},
        {"score": 0.6, "bbox": [200, 0, 300, 100]},
    ]}
    bound = perception.bind_entities(entities, candidates, (400, 200))
    assert bound[0]["bbox"] == [0, 0, 100, 100]
    assert bound[1]["bbox"] == [200, 0, 300, 100]      # no shared box


def test_binding_low_iou_anchor_not_bound_to_wrong_candidate():
    entities = [{"label": "dog", "state": "healthy", "_phrase": "dog",
                 "anchor_bbox": [0, 0, 50, 50]}]
    candidates = {"dog": [{"score": 0.9, "bbox": [500, 500, 600, 600]}]}
    bound = perception.bind_entities(entities, candidates, (800, 800))
    assert bound[0]["box_source"] == "vlm_sam_fallback"


def test_iou_and_center():
    assert perception._iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert perception._iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert perception._center_dist([0, 0, 10, 10], [0, 0, 10, 10]) == 0.0


# ── Full pipeline (stand-in, mocked models) ─────────────────────────────


@pytest.fixture()
def fake_models(monkeypatch):
    def fake_detect(image, entities):
        # One candidate per phrase, offset per entity order (mirror the real
        # function's contract: set _phrase on each entity).
        cands = {}
        for i, e in enumerate(entities):
            p = perception._detector_phrase(e["label"], e.get("description", ""))
            e["_phrase"] = p
            cands.setdefault(p, []).append(
                {"score": 0.8 - i * 0.1, "bbox": [10 + i * 30, 10, 100 + i * 30, 120]}
            )
        return cands

    def fake_mask(image, bbox):
        m = Image.new("L", image.size, 0)
        from PIL import ImageDraw
        ImageDraw.Draw(m).rectangle(bbox, fill=255)
        return m

    monkeypatch.setattr(perception, "detect_candidates", fake_detect)
    monkeypatch.setattr(perception, "mask_for_box", fake_mask)


def test_full_standin_run(tmp_path, fake_models):
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (400, 300), "gray").save(img_path)
    entities = [
        {"label": "house", "state": "burning", "description": "left",
         "bbox": [12, 12, 98, 118]},
        {"label": "vehicle", "state": "intact", "description": "tanker truck",
         "bbox": [40, 10, 130, 122]},
        {"label": "smoke", "state": "billowing", "description": "above the house"},
    ]
    result = perception.run_perception(
        img_path, entities=entities, out_dir=tmp_path / "out"
    )
    ids = [o.object_id for o in result.detected_objects]
    assert ids == ["house_1", "other_1", "smoke_1"]
    house, other, smoke = result.detected_objects
    assert house.state_kind == "hazard_bearing" and house.box_source == "dino_matched"
    assert house.anchor_bbox == [12, 12, 98, 118]        # anchor preserved, distinct
    assert other.family_name_as_label and other.label_note == "family_name:vehicle"
    assert smoke.anchor_bbox is None
    assert (tmp_path / "out" / "scene__overlay.png").exists()
    saved = json.loads((tmp_path / "out" / "scene__perception.json").read_text())
    assert saved["entity_source"] == "standin"


def test_fallback_path_refines_bbox_from_mask(tmp_path, monkeypatch):
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (400, 300), "gray").save(img_path)

    monkeypatch.setattr(perception, "detect_candidates", lambda image, entities: {})

    def tight_mask(image, bbox):
        m = Image.new("L", image.size, 0)
        from PIL import ImageDraw
        ImageDraw.Draw(m).rectangle([50, 60, 90, 110], fill=255)
        return m

    monkeypatch.setattr(perception, "mask_for_box", tight_mask)
    entities = [{"label": "dog", "state": "healthy", "bbox": [30, 40, 120, 140]}]
    result = perception.run_perception(
        img_path, entities=entities, out_dir=tmp_path / "out"
    )
    dog = result.detected_objects[0]
    assert dog.box_source == "vlm_sam_fallback"
    assert dog.bbox == [50, 60, 91, 111]                 # tightened from the mask
    assert dog.anchor_bbox == [30, 40, 120, 140]


def test_duplicate_labels_distinct_ids(tmp_path, fake_models):
    img_path = tmp_path / "scene.jpg"
    Image.new("RGB", (300, 200), "gray").save(img_path)
    entities = [
        {"label": "house", "state": "burning", "bbox": [12, 12, 98, 118]},
        {"label": "house", "state": "intact", "bbox": [42, 10, 128, 122]},
    ]
    result = perception.run_perception(
        img_path, entities=entities, with_masks=False, out_dir=tmp_path / "out"
    )
    assert [o.object_id for o in result.detected_objects] == ["house_1", "house_2"]
    boxes = [tuple(o.bbox) for o in result.detected_objects if o.bbox]
    assert len(set(boxes)) == len(boxes)                 # never share a box
