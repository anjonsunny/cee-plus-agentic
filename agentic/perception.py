"""Stage 1 — Perception (AGENTIC_PLAN.md Stage 1), round 2.

Division of labor (revised after the round 1 review with Sunny):
  - The VLM (Qwen via Ollama) owns `label` and `state`, and now also emits a
    ROUGH bbox per entity. That box is an instance ANCHOR only: it binds the
    entity to the right detector candidate (which house is the burning one),
    and it never becomes the output geometry.
  - Grounding DINO proposes candidate boxes per label; each entity binds to
    the candidate that best agrees with its anchor (IoU, then center).
  - SAM refines the bound box into a mask. When no candidate matches, SAM is
    prompted directly with the VLM's rough box (recorded as a fallback).

Fluid convention (matches the rulebook in main.py): fire attached to a
burning object is a STATE on that object, never a separate entity. Only
diffuse media (smoke, water, gas, dust, spills, free-burning fire) are
entities of their own.

Not agentic: no planning, no loops, no retrieval. Deterministic tools only.

CLI (stand-in mode, no VLM):
    python -m agentic.perception --image scene.jpg --entities entities.json
Live mode (Ollama running):
    python -m agentic.perception --image scene.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentic.vocabulary import (  # noqa: E402
    OTHER_LABEL,
    canonicalize_label,
    family_of,
    vocabulary_prompt_block,
)

# State vocabulary stays owned by main.py.
from main import (  # noqa: E402
    AT_RISK_STATES,
    HAZARD_BEARING_STATES,
    NORMAL_STATES,
    canonicalize_state,
    clamp_bbox,
)

# Perception-side state extensions, pending Sunny's motion-vocab pass in
# main.py (PROJECT_STATE 8.3). Kept here so main.py stays untouched; migrate
# these into NORMAL_STATES / STATE_SYNONYMS there when that pass happens.
EXTRA_NORMAL_STATES = {"swimming", "walking", "running", "parked", "driving"}
EXTRA_STATE_SYNONYMS = {
    "overturned": "fallen",
    "seated": "resting",
    "sitting": "resting",
    "spilled": "seeping",
    "spilling": "leaking",
}

# ── Contract ─────────────────────────────────────────────────────────────


class DetectedObject(BaseModel):
    """The Stage 1 output contract. Violations fail at this boundary."""

    object_id: str                      # "<label>_<N>", N 1-indexed per label
    label: str                          # canonical vocabulary noun
    family: str                         # vocabulary family
    state: str                          # from main.py state vocabulary (+ extensions)
    state_kind: str                     # hazard_bearing | at_risk | normal | unknown
    description: str = ""               # VLM prose; REQUIRED for label 'other'
    bbox: Optional[list[int]] = None    # [x1,y1,x2,y2] final geometry; None = not localized
    box_source: str = "none"            # dino_matched | vlm_sam_fallback | none
    box_confidence: float = 0.0         # DINO score, or 0.0 on the fallback path
    anchor_bbox: Optional[list[int]] = None  # the VLM's rough box (never the output)
    mask_path: Optional[str] = None     # saved mask PNG, for future inpainting
    label_note: str = ""                # '' | synonym:x->y | extension:<raw> | family_name:<raw>
    vocab_extension: bool = False
    family_name_as_label: bool = False  # VLM answered with a family name


class PerceptionResult(BaseModel):
    image_path: str
    image_size: list[int]               # [width, height]
    caption: str = ""
    entity_source: str                  # "vlm" | "standin"
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    unlocalized: list[str] = Field(default_factory=list)   # ids with no box
    notes: list[str] = Field(default_factory=list)
    # Loop 1 history (None = loop not run, e.g. stand-in mode without a
    # repair function). See agentic/repair_loop.py for the full story.
    repair_trace: Optional[dict[str, Any]] = None


def state_kind(state: str) -> str:
    raw = str(state or "").strip().lower()
    raw = EXTRA_STATE_SYNONYMS.get(raw, raw)
    s = canonicalize_state(raw)
    if s in HAZARD_BEARING_STATES:
        return "hazard_bearing"
    if s in AT_RISK_STATES:
        return "at_risk"
    if s in NORMAL_STATES or s in EXTRA_NORMAL_STATES:
        return "normal"
    return "unknown"


def normalize_state(state: str) -> str:
    raw = str(state or "").strip().lower()
    return EXTRA_STATE_SYNONYMS.get(raw, raw)


# ── VLM entity naming (labels + states + rough anchor boxes) ────────────

PERCEPTION_PROMPT_TEMPLATE = """Analyze the image (and caption, if given).
List every distinct entity relevant to safety assessment: people, animals,
vehicles, structures, hazard media, vegetation, infrastructure, and
significant objects.

{vocab_block}

CRITICAL LABEL RULES:
- The label must be ONE SPECIFIC NOUN from the lists above. NEVER answer
  with a family name (the word left of the colon). "vehicle", "structure",
  "hazard_media", "vegetation", "infrastructure" are NOT valid labels: write
  "tanker_truck", "house", "smoke", "brush", "powerline" instead.
- Fluid convention: fire attached to a burning object is a STATE, not an
  entity. A house on fire is one entity: label "house", state "burning". Do
  NOT add a separate fire entity for it. Emit a hazard-media entity only for
  diffuse media: smoke ("smoke", state "billowing"), floodwater or pool
  water containing a victim ("water", state "rising"/"engulfing"), leaked
  liquid ("spill", state "leaking"/"seeping"), dust ("dust", state
  "billowing"), gas, and FREE-BURNING fire not attached to one object (a
  grass or brush fire: label "fire", state "spreading").

For each entity give:
- label: one specific noun from the allowed labels
- state: a single lowercase word for its current condition (examples:
  burning, burnt, collapsed, collapsing, fallen, crushed, flooded, leaking,
  spreading, billowing, rising, seeping, engulfing / injured, bleeding,
  fleeing, trapped, cowering, drowning, suffocating, unconscious / intact,
  standing, upright, dry, sealed, stationary, resting, healthy, stable,
  swimming, walking, running)
- description: one short phrase locating it ("silver sedan in the
  foreground", "child mid-splash near the pool center")
- bbox: rough pixel bounding box [x_min, y_min, x_max, y_max]. Approximate
  is fine; it anchors WHICH instance you mean, not exact geometry.

READ DISTRESS CAREFULLY. A person in water with arms up, splashing, head
low is "drowning", not "swimming". A person pinned, wedged, or unable to
move is "trapped". Distress states matter more than any other field.

Model each instance separately when instances differ causally (a burning
house and an intact house are two entries). Count people individually. Do
NOT invent entities you cannot see; unseen occupants are handled elsewhere.

Return valid JSON only:
{{"entities": [{{"label": "...", "state": "...", "description": "...",
"bbox": [x1, y1, x2, y2]}}]}}
"""


def build_perception_prompt() -> str:
    return PERCEPTION_PROMPT_TEMPLATE.format(vocab_block=vocabulary_prompt_block())


def _query_vlm_raw(text: str, image_data_url: str) -> list[dict[str, Any]]:
    """One VLM call (text + image) returning the parsed entity list. Shared
    by the first perception ask and by Loop 1's repair rounds, so both speak
    to the model identically."""
    import requests

    from main import extract_json_block  # lazy: keeps offline import light

    api_url = os.getenv("QWEN_API_URL", "http://localhost:11434/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("QWEN_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": os.getenv("QWEN_MODEL_NAME", "qwen2.5vl:7b"),
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    r = requests.post(api_url, headers=headers, json=payload,
                      timeout=int(os.getenv("QWEN_TIMEOUT", "600")))
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    raw = extract_json_block(content)
    return list(raw.get("entities") or [])


def query_vlm_entities(image_data_url: str, caption: str = "") -> list[dict[str, Any]]:
    """Ask the VLM for labels + states + rough anchor boxes."""
    text = build_perception_prompt()
    if caption:
        text += f"\n\nCaption:\n{caption}"
    return _query_vlm_raw(text, image_data_url)


# ── Grounding DINO candidate detection ──────────────────────────────────

_DETECTOR = {"model": None, "processor": None}
DETECTOR_MODEL_ID = os.getenv("GDINO_MODEL", "IDEA-Research/grounding-dino-base")
BOX_THRESHOLD = float(os.getenv("GDINO_BOX_THRESHOLD", "0.25"))
TEXT_THRESHOLD = float(os.getenv("GDINO_TEXT_THRESHOLD", "0.20"))


def _load_detector():
    if _DETECTOR["model"] is None:
        import torch  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        _DETECTOR["processor"] = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
        _DETECTOR["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTOR_MODEL_ID
        )
    return _DETECTOR["model"], _DETECTOR["processor"]


def _detector_phrase(label: str, description: str) -> str:
    """Plain label phrase. Instance choice comes from anchor binding, not
    from state-qualified phrasing (which mis-bound in round 1)."""
    if label == OTHER_LABEL:
        return (description or "object").lower()
    return label.replace("_", " ")


def detect_candidates(image, entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """One DINO pass over all phrases. Returns phrase -> candidate list,
    each candidate {'score': float, 'bbox': [x1,y1,x2,y2]}."""
    import torch

    model, processor = _load_detector()
    phrases: list[str] = []
    for e in entities:
        p = _detector_phrase(e["label"], e.get("description", ""))
        e["_phrase"] = p
        if p not in phrases:
            phrases.append(p)

    text = ". ".join(phrases) + "."
    inputs = processor(images=image, text=text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]],
    )[0]

    candidates: dict[str, list[dict[str, Any]]] = {}
    for score, box, lbl in zip(
        results["scores"].tolist(),
        results["boxes"].tolist(),
        results.get("text_labels", results.get("labels", [])),
    ):
        key = str(lbl).strip().lower()
        candidates.setdefault(key, []).append(
            {"score": float(score), "bbox": [int(round(v)) for v in box]}
        )
    for v in candidates.values():
        v.sort(key=lambda c: -c["score"])
    return candidates


# ── Anchor binding ──────────────────────────────────────────────────────


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _center_dist(a: list[int], b: list[int]) -> float:
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


MIN_BIND_IOU = 0.10


def bind_entities(
    entities: list[dict[str, Any]],
    candidates: dict[str, list[dict[str, Any]]],
    image_size: tuple[int, int],
) -> list[dict[str, Any]]:
    """Bind each entity to its best detector candidate via the anchor box.

    With an anchor: candidates for the entity's phrase are ranked by IoU
    against the anchor (center distance as tie-break); a bound candidate is
    removed from the pool so two same-label entities cannot share a box.
    Binding requires IoU >= MIN_BIND_IOU; otherwise the entity keeps its
    anchor for the SAM fallback. Without an anchor: highest unclaimed score.
    """
    diag = (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5

    def pool_for(phrase: str) -> list[dict[str, Any]]:
        key = phrase.strip().lower()
        if key in candidates:
            return candidates[key]
        head = key.split()[-1] if key.split() else key
        for bkey, dets in candidates.items():
            if head and head in bkey:
                return dets
        return []

    # Entities with anchors bind first (they carry the strongest evidence).
    ordered = sorted(entities, key=lambda e: e.get("anchor_bbox") is None)
    for e in ordered:
        if "_phrase" not in e:
            e["_phrase"] = _detector_phrase(e["label"], e.get("description", ""))
        pool = pool_for(e["_phrase"])
        anchor = e.get("anchor_bbox")
        chosen = None
        if anchor and pool:
            scored = [
                (_iou(anchor, c["bbox"]), -_center_dist(anchor, c["bbox"]) / diag, c)
                for c in pool
            ]
            scored.sort(key=lambda t: (-t[0], -t[1]))
            if scored and scored[0][0] >= MIN_BIND_IOU:
                chosen = scored[0][2]
        elif pool:
            chosen = pool[0]
        if chosen is not None:
            pool.remove(chosen)
            e["bbox"] = chosen["bbox"]
            e["box_confidence"] = chosen["score"]
            e["box_source"] = "dino_matched"
        elif anchor:
            e["bbox"] = anchor                  # provisional; SAM refines below
            e["box_confidence"] = 0.0
            e["box_source"] = "vlm_sam_fallback"
        else:
            e["bbox"] = None
            e["box_confidence"] = 0.0
            e["box_source"] = "none"
        e.pop("_phrase", None)
    return entities


# ── SAM masks ───────────────────────────────────────────────────────────

_SAM = {"model": None, "processor": None}
SAM_MODEL_ID = os.getenv("SAM_MODEL", "facebook/sam-vit-base")


def _load_sam():
    if _SAM["model"] is None:
        from transformers import SamModel, SamProcessor

        _SAM["processor"] = SamProcessor.from_pretrained(SAM_MODEL_ID)
        _SAM["model"] = SamModel.from_pretrained(SAM_MODEL_ID)
    return _SAM["model"], _SAM["processor"]


def mask_for_box(image, bbox: list[int]):
    """Best SAM mask for a box prompt; returns a PIL 'L' image or None."""
    import numpy as np
    import torch
    from PIL import Image

    model, processor = _load_sam()
    inputs = processor(image, input_boxes=[[bbox]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]
    scores = outputs.iou_scores.cpu().reshape(-1)
    best = masks[int(scores.argmax())].numpy().astype(np.uint8) * 255
    return Image.fromarray(best, mode="L")


def mask_bbox(mask) -> list[int] | None:
    """Tight bbox of a mask's nonzero region (refines the fallback path)."""
    import numpy as np

    arr = np.array(mask)
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


# ── Assembly ────────────────────────────────────────────────────────────


def assign_ids(entities: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for e in entities:
        counts[e["label"]] = counts.get(e["label"], 0) + 1
        e["object_id"] = f"{e['label']}_{counts[e['label']]}"


def run_perception(
    image_path: str | Path,
    caption: str = "",
    entities: list[dict[str, Any]] | None = None,
    with_masks: bool = True,
    out_dir: str | Path | None = None,
    repair: bool = True,
    repair_query_fn: Any = None,
) -> PerceptionResult:
    """Full Stage 1 on one image. `entities` supplied = stand-in mode.

    Loop 1 (agentic/repair_loop.py) runs after the first VLM answer when
    `repair` is True: violations of the interface contract (family names,
    out-of-vocab words, missing anchors) are cited back to the model for a
    bounded number of rounds. `repair_query_fn` overrides the live model
    call for tests. The full loop history lands in result.repair_trace.
    """
    import base64
    import mimetypes

    from PIL import Image

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    out_dir = Path(out_dir) if out_dir else image_path.parent / "perception"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    data_url: str | None = None
    if entities is None:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        entities = query_vlm_entities(data_url, caption)
        source = "vlm"
    else:
        entities = [dict(e) for e in entities]
        source = "standin"

    # ── Loop 1: evidence-triggered repair of the model's answer ─────────
    repair_trace = None
    if repair and (repair_query_fn is not None or source == "vlm"):
        from agentic.repair_loop import repair_entities

        if repair_query_fn is None:
            # Live path: each repair round is one targeted VLM call that
            # includes the image, so the model can re-look while fixing.
            def repair_query_fn(prompt_text: str) -> list[dict[str, Any]]:  # noqa: F811
                return _query_vlm_raw(prompt_text, data_url)

        entities, trace = repair_entities(entities, repair_query_fn)
        repair_trace = trace.model_dump()

    # Canonicalize labels; log drift; clamp anchor boxes.
    for e in entities:
        canon, note, in_vocab, is_family = canonicalize_label(e.get("label", ""))
        if note:
            notes.append(f"label {e.get('label')!r}: {note}")
        e["label"] = canon
        e["label_note"] = note
        e["vocab_extension"] = not in_vocab
        e["family_name_as_label"] = is_family
        e.setdefault("description", "")
        e["state"] = normalize_state(e.get("state", "unknown"))
        e["anchor_bbox"] = clamp_bbox(e.get("bbox"), image.width, image.height)
        e.pop("bbox", None)

    assign_ids(entities)
    candidates = detect_candidates(image, entities)
    entities = bind_entities(entities, candidates, (image.width, image.height))

    objects: list[DetectedObject] = []
    unlocalized: list[str] = []
    for e in entities:
        mask_path = None
        if with_masks and e.get("bbox"):
            mask = mask_for_box(image, e["bbox"])
            if e["box_source"] == "vlm_sam_fallback":
                refined = mask_bbox(mask)
                if refined:
                    e["bbox"] = refined
            mask_path = str(out_dir / f"{image_path.stem}__{e['object_id']}_mask.png")
            mask.save(mask_path)
        if not e.get("bbox"):
            unlocalized.append(e["object_id"])
        objects.append(
            DetectedObject(
                object_id=e["object_id"],
                label=e["label"],
                family=family_of(e["label"]),
                state=str(e.get("state", "unknown")),
                state_kind=state_kind(e.get("state", "")),
                description=str(e.get("description", "")),
                bbox=e.get("bbox"),
                box_source=e.get("box_source", "none"),
                box_confidence=float(e.get("box_confidence", 0.0)),
                anchor_bbox=e.get("anchor_bbox"),
                mask_path=mask_path,
                label_note=e.get("label_note", ""),
                vocab_extension=bool(e.get("vocab_extension", False)),
                family_name_as_label=bool(e.get("family_name_as_label", False)),
            )
        )

    result = PerceptionResult(
        image_path=str(image_path),
        image_size=[image.width, image.height],
        caption=caption,
        entity_source=source,
        detected_objects=objects,
        unlocalized=unlocalized,
        notes=notes,
        repair_trace=repair_trace,
    )
    (out_dir / f"{image_path.stem}__perception.json").write_text(
        result.model_dump_json(indent=2)
    )
    render_overlay(image, result, out_dir / f"{image_path.stem}__overlay.png")
    return result


# ── Overlay rendering (first artifact of the scene-is-the-interface UI) ─

STATE_KIND_COLORS = {
    "hazard_bearing": (239, 68, 68),    # red
    "at_risk": (249, 115, 22),          # orange
    "normal": (59, 130, 246),           # blue
    "unknown": (148, 163, 184),         # slate
}


def render_overlay(image, result: PerceptionResult, out_path: str | Path) -> None:
    from PIL import ImageDraw, ImageFont

    img = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            max(14, img.width // 90),
        )
    except OSError:
        font = ImageFont.load_default()

    for obj in result.detected_objects:
        if not obj.bbox:
            continue
        color = STATE_KIND_COLORS.get(obj.state_kind, STATE_KIND_COLORS["unknown"])
        x1, y1, x2, y2 = obj.bbox
        width = max(3, img.width // 400)
        if obj.box_source == "vlm_sam_fallback":
            _dashed_rectangle(draw, x1, y1, x2, y2, color, width)
        else:
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        tag = f"{obj.object_id} · {obj.state}"
        tb = draw.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = y1 - th - 8 if y1 - th - 8 > 0 else y1 + 4
        draw.rectangle([x1, ty - 2, x1 + tw + 10, ty + th + 6], fill=color)
        draw.text((x1 + 5, ty), tag, fill="white", font=font)

    missing = [o.object_id for o in result.detected_objects if not o.bbox]
    if missing:
        note = "not localized: " + ", ".join(missing)
        draw.text((10, img.height - 28), note, fill=(239, 68, 68), font=font)
    img.save(out_path)


def _dashed_rectangle(draw, x1, y1, x2, y2, color, width, dash=14):
    """Dashed box marks the vlm_sam_fallback path visually."""
    def _dash_line(a, b):
        (ax, ay), (bx, by) = a, b
        length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        if length == 0:
            return
        n = max(1, int(length // dash))
        for i in range(0, n, 2):
            t0, t1 = i / n, min((i + 1) / n, 1.0)
            draw.line(
                [ax + (bx - ax) * t0, ay + (by - ay) * t0,
                 ax + (bx - ax) * t1, ay + (by - ay) * t1],
                fill=color, width=width,
            )
    _dash_line((x1, y1), (x2, y1))
    _dash_line((x2, y1), (x2, y2))
    _dash_line((x2, y2), (x1, y2))
    _dash_line((x1, y2), (x1, y1))


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CEE+ agentic Stage 1: perception.")
    p.add_argument("--image", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--entities", help="JSON file of [{label, state, description, bbox}] "
                                      "(stand-in mode, skips the VLM).")
    p.add_argument("--no-masks", action="store_true")
    p.add_argument("--out-dir", default=None)
    args = p.parse_args(argv)

    entities = None
    if args.entities:
        entities = json.loads(Path(args.entities).read_text())

    result = run_perception(
        args.image,
        caption=args.caption,
        entities=entities,
        with_masks=not args.no_masks,
        out_dir=args.out_dir,
    )
    print(f"{len(result.detected_objects)} entities "
          f"({len(result.unlocalized)} unlocalized) [{result.entity_source}]")
    for o in result.detected_objects:
        box = f"bbox={o.bbox}" if o.bbox else "NO BOX"
        flags = []
        if o.vocab_extension:
            flags.append("vocab_extension")
        if o.family_name_as_label:
            flags.append("family_name")
        flag_txt = f" [{','.join(flags)}]" if flags else ""
        print(f"  {o.object_id:<22} {o.state:<12} {o.box_source:<17} "
              f"conf={o.box_confidence:.2f} {box}{flag_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
