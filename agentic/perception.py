"""Stage 1 — Perception (AGENTIC_PLAN.md Stage 1).

One source per field:
  - The VLM (Qwen via Ollama) owns `label` and `state`. Its boxes are never
    used; it is not asked for any.
  - Grounding DINO owns `bbox`, queried with the VLM's own state-qualified
    label phrase ("burning house").
  - SAM owns the mask, prompted with the detector's box. Masks are captured
    now because visual suppression (inpainting) will consume them later.

Not agentic: no planning, no loops, no retrieval. Deterministic tools only.

CLI (development / stand-in mode, no VLM needed):
    python -m agentic.perception --image scene.jpg --entities entities.json
where entities.json is [{"label": "house", "state": "burning"}, ...].
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
)

# ── Contract ─────────────────────────────────────────────────────────────


class DetectedObject(BaseModel):
    """The Stage 1 output contract. Violations fail at this boundary."""

    object_id: str                      # "<label>_<N>", N 1-indexed per label
    label: str                          # canonical vocabulary noun
    family: str                         # vocabulary family
    state: str                          # from main.py state vocabulary
    state_kind: str                     # hazard_bearing | at_risk | normal | unknown
    description: str = ""               # VLM prose; REQUIRED for label 'other'
    bbox: Optional[list[int]] = None    # [x1,y1,x2,y2]; None = not localized
    box_source: str = "none"            # grounding_dino | none
    box_confidence: float = 0.0
    mask_path: Optional[str] = None     # saved mask PNG, for future inpainting
    label_note: str = ""                # '' | synonym:x->y | extension:<raw>
    vocab_extension: bool = False


class PerceptionResult(BaseModel):
    image_path: str
    image_size: list[int]               # [width, height]
    caption: str = ""
    entity_source: str                  # "vlm" | "standin"
    detected_objects: list[DetectedObject] = Field(default_factory=list)
    unlocalized: list[str] = Field(default_factory=list)   # ids with no box
    notes: list[str] = Field(default_factory=list)


def state_kind(state: str) -> str:
    s = canonicalize_state(str(state or "").strip())
    if s in HAZARD_BEARING_STATES:
        return "hazard_bearing"
    if s in AT_RISK_STATES:
        return "at_risk"
    if s in NORMAL_STATES:
        return "normal"
    return "unknown"


# ── VLM entity naming (labels + states only, no boxes) ──────────────────

PERCEPTION_PROMPT_TEMPLATE = """Analyze the image (and caption, if given).
List every distinct entity relevant to safety assessment: people, animals,
vehicles, structures, hazard media (fire, smoke, water, dust, gas, spills,
debris), vegetation, infrastructure, and significant objects.

{vocab_block}

For each entity give:
- label: one noun from the allowed labels above
- state: a single lowercase word for its current condition (examples:
  burning, burnt, collapsed, collapsing, fallen, crushed, flooded, leaking,
  spreading, billowing, rising, seeping, engulfing / injured, bleeding,
  fleeing, trapped, cowering, drowning, suffocating, unconscious / intact,
  standing, upright, dry, sealed, stationary, resting, healthy, stable)
- description: one short phrase locating it in the scene (e.g. "silver sedan
  in the foreground", "child mid-splash near the pool center")

Model each instance separately when instances differ causally (a burning
house and an intact house are two entries). Do NOT output bounding boxes.
Do NOT invent entities you cannot see; unseen occupants are handled
elsewhere. Count people individually.

Return valid JSON only:
{{"entities": [{{"label": "...", "state": "...", "description": "..."}}]}}
"""


def build_perception_prompt() -> str:
    return PERCEPTION_PROMPT_TEMPLATE.format(vocab_block=vocabulary_prompt_block())


def query_vlm_entities(image_data_url: str, caption: str = "") -> list[dict[str, Any]]:
    """Ask the VLM for labels + states only. Requires the Ollama endpoint."""
    import requests

    from main import extract_json_block  # lazy: keeps offline import light

    api_url = os.getenv("QWEN_API_URL", "http://localhost:11434/v1/chat/completions")
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("QWEN_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    text = build_perception_prompt()
    if caption:
        text += f"\n\nCaption:\n{caption}"
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


# ── Grounding DINO boxes ────────────────────────────────────────────────

_DETECTOR = {"model": None, "processor": None}
DETECTOR_MODEL_ID = os.getenv("GDINO_MODEL", "IDEA-Research/grounding-dino-base")
BOX_THRESHOLD = float(os.getenv("GDINO_BOX_THRESHOLD", "0.30"))
TEXT_THRESHOLD = float(os.getenv("GDINO_TEXT_THRESHOLD", "0.25"))


def _load_detector():
    if _DETECTOR["model"] is None:
        import torch  # noqa: F401
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        _DETECTOR["processor"] = AutoProcessor.from_pretrained(DETECTOR_MODEL_ID)
        _DETECTOR["model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
            DETECTOR_MODEL_ID
        )
    return _DETECTOR["model"], _DETECTOR["processor"]


def _detector_phrase(label: str, state: str, description: str) -> str:
    """State-qualified phrase for grounding. 'burning house' localizes better
    than 'house' when two houses differ exactly by state."""
    label_text = label.replace("_", " ")
    state_text = str(state or "").strip().replace("_", " ")
    if label == OTHER_LABEL:
        return (description or "object").lower()
    qualifiable = {"burning", "burnt", "collapsed", "collapsing", "fallen",
                   "crushed", "flooded", "leaking", "overturned", "intact"}
    if state_text in qualifiable:
        return f"{state_text} {label_text}"
    return label_text


def ground_entities(
    image, entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run Grounding DINO once with all phrases; greedily assign boxes.

    Returns entities enriched with bbox / box_confidence / box_source.
    Instances sharing a phrase are assigned boxes in descending score order.
    """
    import torch

    model, processor = _load_detector()
    phrases = []
    for e in entities:
        p = _detector_phrase(e["label"], e.get("state", ""), e.get("description", ""))
        e["_phrase"] = p
        if p not in phrases:
            phrases.append(p)

    # transformers grounding-dino expects "a. b. c." style text.
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

    # Bucket detections by matched phrase text.
    buckets: dict[str, list[tuple[float, list[int]]]] = {}
    for score, box, lbl in zip(
        results["scores"].tolist(),
        results["boxes"].tolist(),
        results.get("text_labels", results.get("labels", [])),
    ):
        key = str(lbl).strip().lower()
        buckets.setdefault(key, []).append(
            (float(score), [int(round(v)) for v in box])
        )
    for v in buckets.values():
        v.sort(key=lambda t: -t[0])

    def _take(phrase: str) -> tuple[float, list[int]] | None:
        key = phrase.strip().lower()
        # exact bucket, else any bucket containing the phrase's head noun
        if buckets.get(key):
            return buckets[key].pop(0)
        head = key.split()[-1]
        for bkey, dets in buckets.items():
            if head in bkey and dets:
                return dets.pop(0)
        return None

    for e in entities:
        det = _take(e.pop("_phrase"))
        if det is not None:
            e["box_confidence"], e["bbox"] = det
            e["box_source"] = "grounding_dino"
        else:
            e["bbox"] = None
            e["box_confidence"] = 0.0
            e["box_source"] = "none"
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
    )[0][0]                              # (num_masks, H, W) bool
    scores = outputs.iou_scores.cpu().reshape(-1)
    best = masks[int(scores.argmax())].numpy().astype(np.uint8) * 255
    return Image.fromarray(best, mode="L")


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
) -> PerceptionResult:
    """Full Stage 1 on one image. `entities` supplied = stand-in mode
    (development without the VLM); omitted = live VLM naming."""
    import base64
    import mimetypes

    from PIL import Image

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    out_dir = Path(out_dir) if out_dir else image_path.parent / "perception"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if entities is None:
        mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        entities = query_vlm_entities(f"data:{mime};base64,{b64}", caption)
        source = "vlm"
    else:
        entities = [dict(e) for e in entities]
        source = "standin"

    # Canonicalize labels (VLM owns them; we normalize and log drift).
    for e in entities:
        canon, note, in_vocab = canonicalize_label(e.get("label", ""))
        if note:
            notes.append(f"label {e.get('label')!r}: {note}")
        e["label"] = canon
        e["label_note"] = note
        e["vocab_extension"] = not in_vocab
        e.setdefault("description", "")
        e.setdefault("state", "unknown")

    assign_ids(entities)
    entities = ground_entities(image, entities)

    objects: list[DetectedObject] = []
    unlocalized: list[str] = []
    for e in entities:
        mask_path = None
        if with_masks and e.get("bbox"):
            mask = mask_for_box(image, e["bbox"])
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
                mask_path=mask_path,
                label_note=e.get("label_note", ""),
                vocab_extension=bool(e.get("vocab_extension", False)),
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
    )
    (out_dir / f"{image_path.stem}__perception.json").write_text(
        result.model_dump_json(indent=2)
    )
    render_overlay(image, result, out_dir / f"{image_path.stem}__overlay.png")
    return result


# ── Overlay rendering (the first artifact of the scene-is-the-interface UI) ─

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
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(3, img.width // 400))
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


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CEE+ agentic Stage 1: perception.")
    p.add_argument("--image", required=True)
    p.add_argument("--caption", default="")
    p.add_argument("--entities", help="JSON file of [{label, state, description}] "
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
        ext = " [vocab_extension]" if o.vocab_extension else ""
        print(f"  {o.object_id:<22} {o.state:<12} conf={o.box_confidence:.2f} {box}{ext}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
