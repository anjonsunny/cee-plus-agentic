"""Frozen Stage 1 label vocabulary (agreed with Sunny, 2026-07-20).

Closed by default, with a logged escape hatch: a label outside the vocabulary
and its synonym map becomes `other` and is flagged vocab_extension, boxed
anyway, and counted. A rising extension rate is a finding about the
vocabulary, never silent noise. (Same philosophy as the protocol's
`uncodable` effect label.)

States are NOT defined here; the state vocabulary stays owned by main.py.
"""
from __future__ import annotations

# ── The closed vocabulary: 9 families, 55 nouns ─────────────────────────

LABEL_FAMILIES: dict[str, list[str]] = {
    "person": ["person", "man", "woman", "child", "boy", "girl"],
    "responder": [
        "responder", "police_officer", "firefighter", "paramedic", "hazmat_worker",
    ],
    "animal": ["dog", "cat", "animal"],
    "vehicle": [
        "car", "truck", "tanker_truck", "pickup_truck", "van", "bus",
        "bicycle", "motorcycle", "fire_truck", "ambulance", "police_car",
    ],
    "structure": ["house", "building", "shed", "garage"],
    "vegetation": ["tree", "bush", "grass", "brush"],
    "hazard_media": [
        "fire", "smoke", "water", "dust", "gas", "spill", "debris", "rubble",
    ],
    "infrastructure": [
        "road", "bridge", "sidewalk", "fence", "pole", "powerline",
        "traffic_light", "street_lamp", "sign", "hydrant", "ladder",
        "handrail", "traffic_cone", "caution_tape", "pool",
    ],
    "object": [
        "bench", "chair", "lifeguard_chair", "canister", "tank",
        "umbrella", "cart", "swing",
    ],
}

ALL_LABELS: set[str] = {lbl for fam in LABEL_FAMILIES.values() for lbl in fam}

FAMILY_OF: dict[str, str] = {
    lbl: fam for fam, labels in LABEL_FAMILIES.items() for lbl in labels
}

# Families whose members count as lives for consequence weighting downstream.
LIFE_FAMILIES = {"person", "responder", "animal"}

# ── Synonym map: drift → canonical. Conservative, only clear synonyms. ──

LABEL_SYNONYMS: dict[str, str] = {
    # person
    "people": "person", "human": "person", "pedestrian": "person",
    "bystander": "person", "adult": "person", "kid": "child",
    "toddler": "child", "infant": "child", "teenager": "child",
    "guy": "man", "male": "man", "female": "woman", "lady": "woman",
    "swimmer": "person", "driver": "person", "occupant": "person",
    "resident": "person", "victim": "person", "worker": "person",
    # responder
    "police": "police_officer", "policeman": "police_officer",
    "cop": "police_officer", "officer": "police_officer",
    "fireman": "firefighter", "emt": "paramedic",
    "rescuer": "responder", "first_responder": "responder",
    "hazmat": "hazmat_worker",
    # animal
    "puppy": "dog", "kitten": "cat", "pet": "animal",
    # vehicle  (NOTE: "vehicle" itself is a family name, NOT a synonym; round 1
    # mapped it to car and silently turned a tanker truck into car_1)
    "sedan": "car", "suv": "car", "automobile": "car",
    "hatchback": "car", "jeep": "car", "lorry": "truck",
    "tanker": "tanker_truck", "tank_truck": "tanker_truck",
    "fuel_truck": "tanker_truck", "semi": "truck", "pickup": "pickup_truck",
    "bike": "bicycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "cruiser": "police_car", "firetruck": "fire_truck",
    # structure  ("structure" itself is a family name, not a synonym)
    "home": "house", "residence": "house", "dwelling": "house",
    "apartment": "building", "apartment_building": "building",
    "barn": "shed", "warehouse": "building",
    # vegetation
    "trees": "tree", "branch": "tree", "shrub": "bush", "plant": "bush",
    "foliage": "bush", "lawn": "grass", "shrubbery": "brush",
    "brushfire_fuel": "brush",
    # hazard media
    "flame": "fire", "flames": "fire", "blaze": "fire", "wildfire": "fire",
    "smog": "smoke", "fumes": "smoke", "fume": "smoke", "haze": "smoke",
    "plume": "smoke",
    "flood": "water", "floodwater": "water", "flood_water": "water",
    "river": "water", "surge": "water", "current": "water", "puddle": "spill",
    "leak": "spill", "leakage": "spill", "liquid": "spill", "oil": "spill",
    "fuel": "spill", "chemical": "spill", "wreckage": "debris",
    # Compound spill names (D_aerial ui_c7b362ef: the model answered
    # "chemical_spill" — more specific than our noun, and the map
    # couldn't hear it; the entity fell to 'other', DINO had nothing to
    # ground, the caption ticket stood, and a whole petition ran on a
    # missing synonym).
    "chemical_spill": "spill", "oil_spill": "spill", "fuel_spill": "spill",
    "spillage": "spill", "hazmat_spill": "spill",
    "rubble_pile": "rubble", "dust_cloud": "dust",
    # "brush fire" / "grass fire" name ONE free-burning fire, not a
    # vegetation entity plus a fire entity (C_tanker rule-5 artifact).
    "brush_fire": "fire", "grass_fire": "fire", "brushfire": "fire",
    # infrastructure
    "street": "road", "highway": "road", "lane": "road", "pavement": "sidewalk",
    "wall": "fence", "railing": "handrail", "rail": "handrail",
    "power_line": "powerline", "powerlines": "powerline", "wire": "powerline",
    "cable": "powerline", "utility_pole": "pole", "lamppost": "street_lamp",
    "streetlight": "street_lamp", "lamp": "street_lamp",
    "stoplight": "traffic_light", "signal": "traffic_light",
    "cone": "traffic_cone", "tape": "caution_tape",
    "police_tape": "caution_tape", "barrier_tape": "caution_tape",
    "swimming_pool": "pool", "stop_sign": "sign", "signboard": "sign",
    # object
    "seat": "chair", "park_bench": "bench", "gas_canister": "canister",
    "cylinder": "canister", "propane_tank": "tank", "gas_tank": "tank",
    "swing_set": "swing", "swingset": "swing", "stroller": "cart",
    "vendor_cart": "cart", "parasol": "umbrella",
}

OTHER_LABEL = "other"

# Family names that are NOT also member labels ("person", "responder", "dog"
# are members; "vehicle", "structure", "hazard_media", "vegetation",
# "infrastructure", "object", "animal" minus members are pure family names).
PURE_FAMILY_NAMES: set[str] = set(LABEL_FAMILIES) - ALL_LABELS


def canonicalize_label(raw: str) -> tuple[str, str, bool, bool]:
    """Return (canonical_label, mapping_note, in_vocab, family_name_as_label).

    mapping_note records what happened ('' = verbatim; 'synonym:x->y';
    'extension:<raw>'; 'family_name:<raw>'). A family name used as a label is
    NEVER silently mapped to a member (round 1 turned a tanker truck into
    car_1 that way); it falls to OTHER_LABEL with the family_name flag set so
    the drift is visible and countable. Out-of-vocab labels become
    OTHER_LABEL with in_vocab=False; the raw text is preserved in the note.
    """
    label = str(raw or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not label:
        return OTHER_LABEL, "extension:<empty>", False, False
    if label in PURE_FAMILY_NAMES:
        return OTHER_LABEL, f"family_name:{label}", False, True
    if label in ALL_LABELS:
        return label, "", True, False
    if label in LABEL_SYNONYMS:
        canon = LABEL_SYNONYMS[label]
        return canon, f"synonym:{label}->{canon}", True, False
    # Singularize the trivial plural, resolving through BOTH the vocabulary
    # and the synonym map ("chemicals" -> "chemical" -> spill; the D_aerial
    # caption miss).
    if label.endswith("s"):
        singular = label[:-1]
        if singular in ALL_LABELS:
            return singular, f"synonym:{label}->{singular}", True, False
        if singular in LABEL_SYNONYMS:
            canon = LABEL_SYNONYMS[singular]
            return canon, f"synonym:{label}->{canon}", True, False
    return OTHER_LABEL, f"extension:{label}", False, False


def family_of(label: str) -> str:
    return FAMILY_OF.get(label, "unknown")


def is_life(label: str) -> bool:
    return family_of(label) in LIFE_FAMILIES


def vocabulary_prompt_block() -> str:
    """The vocabulary as a prompt fragment for the perception call."""
    lines = ["Allowed labels (choose the most specific; grouped by family):"]
    for fam, labels in LABEL_FAMILIES.items():
        lines.append(f"- {fam}: {', '.join(labels)}")
    lines.append(
        "If an entity genuinely fits none of these labels, use label "
        f"'{OTHER_LABEL}' and describe it in the entity's description field. "
        "Do not force a wrong label."
    )
    return "\n".join(lines)
