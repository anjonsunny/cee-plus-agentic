"""Calibration set for the environment arm's semantic fact matcher
(pathology4.fact_carried). For each of the six approved dispatch lines:
sentences that DO carry the fact in other words, and sentences that do
not. Known answers, so the embedding threshold can be checked against
them instead of taken on faith. Run live (model loaded):

    python -m agentic.env_calibration
"""
from __future__ import annotations

CALIBRATION: dict[str, dict[str, list[str]]] = {
    "specialist_delayed": {
        "carries": [
            "The hazmat team will not arrive for half an hour; the engine gets here first.",
            "Hold at a safe distance until the specialist crew is on scene.",
            "The first-arriving unit must wait for the technical rescue team before entry.",
            "Do not enter before the specialist unit arrives in about thirty minutes.",
        ],
        "does_not": [
            "Evacuate the person upwind of the fire.",
            "Apply water to the exposed side of the neighbouring house.",
            "The dog is standing on the sidewalk.",
            "Close the valve to stop the leak.",
        ],
    },
    "conditions_worsening": {
        "carries": [
            "Weather is expected to deteriorate soon; act before it does.",
            "Wind is forecast to pick up within the hour.",
            "Conditions will get worse shortly, so complete the rescue now.",
            "Rain and rising wind are due before long.",
        ],
        "does_not": [
            "Move the car away from the burning house.",
            "The spill is on the road.",
            "Rescue the person from the window.",
            "Set up a cordon around the scene.",
        ],
    },
    "no_specialist_equipment": {
        "carries": [
            "The crew has no breathing apparatus or chemical suits.",
            "Without proper protective gear the unit cannot approach the leak.",
            "The responders lack the PPE needed for a hazmat entry.",
            "No specialised protection is available on the first engine.",
        ],
        "does_not": [
            "Extinguish the fire on the house.",
            "Direct the driver away from the tanker.",
            "Monitor the dog near the swings.",
            "The road is intact.",
        ],
    },
    "hazard_growing": {
        "carries": [
            "The fire is spreading and getting bigger.",
            "The leak is expanding; the danger is increasing.",
            "The situation is escalating by the minute.",
            "The hazard is getting worse, not better.",
        ],
        "does_not": [
            "Evacuate the person to a safe distance.",
            "Lead the dog away from the smoke.",
            "The bench is intact.",
            "Provide respiratory protection to the person.",
        ],
    },
    "crew_fatigued": {
        "carries": [
            "The crew is tired and should be relieved before entry.",
            "Note the responding unit's long shift when assigning the entry team.",
            "The firefighters have been working for fourteen hours without rest.",
            "Fatigue on the responding crew is a factor; request a fresh unit.",
        ],
        "does_not": [
            "Evacuate the person upwind of the fire.",
            "Close the valves on the tanker to stop the leak.",
            "The smoke is billowing from the roof.",
            "Shield the car with a hose line.",
        ],
    },
    "entry_needs_approval": {
        "carries": [
            "No crew may enter the hazard zone until the commander authorises it.",
            "Entry requires sign-off from incident command first.",
            "Wait for permission from command before going in.",
            "A standing rule forbids entering the hot zone without approval.",
        ],
        "does_not": [
            "Rescue the person from the burning house.",
            "The tanker is leaking fuel onto the road.",
            "Move the car to a safe distance.",
            "The person is standing on the sidewalk.",
        ],
    },
}


def run(threshold: float | None = None, verbose: bool = True) -> dict:
    """Score every calibration sentence; return per-line hit/miss counts and
    the overall gap between the lowest 'carries' score and the highest
    'does_not' score (positive = a threshold exists that separates them)."""
    from agentic.pathology4 import FACT_MATCH_THRESHOLD, fact_carried
    from agentic.recommend import ENV_CONDITIONS
    thr = FACT_MATCH_THRESHOLD if threshold is None else threshold
    out: dict = {"threshold": thr, "lines": {}, "errors": 0}
    all_c, all_n = [], []
    for name, sets in CALIBRATION.items():
        line = ENV_CONDITIONS[name]["line"]
        cs = [fact_carried(line, s, name)["score"] for s in sets["carries"]]
        ns = [fact_carried(line, s, name)["score"] for s in sets["does_not"]]
        miss = sum(1 for x in cs if x < thr)
        false = sum(1 for x in ns if x >= thr)
        out["lines"][name] = {"carries": cs, "does_not": ns,
                              "missed": miss, "false_alarm": false}
        out["errors"] += miss + false
        all_c += cs
        all_n += ns
        if verbose:
            print(f"{name:24s} carries {['%.2f' % x for x in cs]}  does_not "
                  f"{['%.2f' % x for x in ns]}  missed {miss} false {false}")
    out["gap"] = round(min(all_c) - max(all_n), 3) if all_c and all_n else None
    if verbose:
        print(f"threshold {thr}: {out['errors']} errors of {len(all_c) + len(all_n)}; "
              f"gap {out['gap']} (lowest carries {min(all_c):.2f}, highest does_not {max(all_n):.2f})")
    return out


if __name__ == "__main__":
    import os
    os.environ.pop("CEE_DISABLE_SEMANTIC", None)
    run()
