# CEE+ Agentic — briefing for Claude Code

Read this first. Then read `agentic/FINDINGS.md` (the findings ledger —
the fix taxonomy table at the top is the fastest way to understand what
has happened so far).

## What this project is

CEE+ (Causal Explanation Engine) measures whether a vision-language
model's emergency-response recommendations are **causally grounded** —
via a hazard-suppression intervention (remove the hazard from the scene,
check the recommendation changes accordingly).

There are two arms:

- **Arm A** = `main.py` at the repo root. The legacy monolithic
  pipeline with FROZEN scoring. **NEVER edit main.py. Import only.**
  It owns the state vocabulary and the causal-quad ontology.
- **Arm B** = everything in `agentic/`. The agentic conversion:
  staged, ledgered, revision-capable. All new work happens here,
  on git branch `agentic`. Sunny pushes git himself — never push.

Arm B is scored by IMPORTING Arm A's frozen metrics, so the two arms
stay comparable.

## Where things are

```
main.py                          Arm A. Frozen. Import only.
agentic/                         Arm B — all the new code
  perception.py                  Stage 1: VLM naming -> repair loop ->
                                 DINO grounding -> SAM masks -> assemble
                                 (+ medium-bound hazard derivation,
                                  duplicate-human merge)
  repair_loop.py                 Loop 1: P1-P6 checks on raw VLM output
  vocabulary.py                  closed label vocabulary + synonyms
  assessment.py                  Stage 2: merged scene assessment
                                 (verdict + threats + at-risk), checks
                                 S1-S8, G2/G4; text-only by design
  uncertainty.py                 measured uncertainty: 5 probes at
                                 temp 0.7, granular per-field/per-entity
  reflection.py                  the reflection loop (capped at 2,
                                 evidence-quoted, anti-rumination)
  petition.py                    contextual re-perception + routing:
                                 stage-1 (re-look at image) vs stage-2
                                 (re-ask the question once, fresh)
  rulebook.py / rulebook_rag.py  one law, two engines: code detects,
                                 rulebook text teaches (quoted into
                                 reflection prompts)
  evals.py                       GT eval, quadrant, citation counts,
                                 pairwise/runoff judges, rubric
  geometry.py                    deterministic bbox math (spatial hints)
  ui.py                          the Dash UI (the main way runs happen)
  dialogue.py, agent_tools.py    "ask the analyst" chat over run records
  test_*.py                      297 hermetic tests, no models needed
  FINDINGS.md                    THE LEDGER. Findings F1-F14 + fix
                                 taxonomy. Read it.
experiments/agentic_scenes/      the six frozen calibration scenes:
  A_fire, B_pool, C_tanker_fire, D_aerial_spill, E_collapse,
  F_park_control (.jpg + .txt caption each)
  gt_stage2.json                 verified ground truth (dev-only!)
  perception/  assessment/       frozen stage outputs
exports/agentic_runs/ui_*/       EVERY UI RUN LANDS HERE:
  events.jsonl                   the flight recorder — every event of
                                 the run; the UI is a pure function of
                                 this stream (best debugging source)
  *__perception.json             the final perception record
  assessment.json                the final assessment
  *_mask.png, *__overlay.png     masks and overlay images
```

## How to run things

```
pytest agentic/ -q               all tests, hermetic, ~10s, no models
python agentic/ui.py             the Dash UI (hot reload on;
                                 AGENTIC_UI_DEBUG=0 to disable)
python -m agentic.assessment     assess frozen perception records
python -m agentic.evals          eval battery vs GT (--judge for LLM judges)
```

Models are LOCAL via Ollama (Sunny runs them; they may not be running):
subject VLM `qwen2.5vl:7b`, dialogue/explainer `qwen2.5:7b`,
judge `llama3.1:8b` (deliberately a different family).

## The architecture in one picture

```
STAGE 1  PERCEPTION (image -> shared record)
  VLM names entities -> Loop 1 repairs format (P1-P6, capped 2,
  model may STAND its ground) -> DINO grounds boxes -> SAM masks ->
  assemble: medium-bound hazard derivation (drowning -> the pool is
  the hazard, derived IN CODE), duplicate-human merge (IoU >= 0.8)

STAGE 2  ASSESSMENT (record -> verdict; TEXT-ONLY, never sees image)
  one merged judgment: disaster Yes/No, type, level 0-10 -> bucket,
  threats[], at_risk[{kind: distress|proximity}]
  -> code checks S1-S8 + geometry hints
  -> 5 probes measure uncertainty (granular, per-entity votes)
  -> REFLECTION loop (violations + evidence quoted back, capped 2)
  -> judges (runoff, pairwise, rubric) — ADVISE ONLY, never overwrite
  -> PETITION if pressure survives:
       entity list suspect -> stage 1 re-look (two-witness merge,
                              NO-ERASURE: petitions add, never delete)
       facts fine, sorting wrong -> re-ask the question once, fresh

STAGE 4  (NEXT, not built) recommendations / causal quads — where
  Arm A's frozen scoring meets Arm B. Ratified design: states stay on
  nodes (the computational currency); mechanism-on-edge is ADDITIVE,
  only where natural language and the ontology diverge.
```

## Iron rules (violating these has burned us before)

1. **Never edit `main.py`.** Import only.
2. **Every code change ships with tests** — including malformed-model-
   output cases at every boundary that consumes raw VLM answers.
3. **Discuss before building.** Propose, get Sunny's explicit yes
   ("do it", "build it"), then build.
4. **Easy language with Sunny.** Short sentences, examples, small
   diagrams. No jargon walls. He has rejected dense replies angrily.
5. **Prompt neutrality:** prompts never reference specific scenes or
   contain id-shaped tokens; corrections quote only the model's OWN
   prior words as evidence, never say what the right answer is.
6. **Judges advise, never overwrite.** Only reflection carries the
   message, only the model revises.
7. **GT and evals NEVER feed the pipeline.** GT is dev-only
   calibration; production has none.
8. **No-erasure:** petitions add, never delete; omissions are recorded
   as disputes. Deterministic code may derive/merge, but always with a
   note and an event — nothing silent.
9. **Keep the flowchart in sync.** Sunny maintains a hand-drawn system
   flowchart ("two local loops + a two-route petition"). Whenever a
   change touches pipeline structure — a loop, a check family, a route,
   a stage — say whether the chart needs updating and, if so, describe
   the exact box + text edit in plain words so Sunny can append it.
   Box-internal refinements (one new synonym) usually don't need an
   edit; new checks/loops/routes/stages do.

## Current state (2026-07-22)

Stage 2 is nearly closed. Five of six scenes ran live and produced
findings F7-F14 (each fixed + regression-tested). Remaining:

- Run `F_park_control` in the UI — the silence test: the whole
  apparatus must produce nothing on a safe scene.
- Re-run the other scenes to confirm the latest fixes.
- Then: judge-bias decision (F4), false-certainty mitigations (F5),
  six-scene synthesis from the taxonomy, close Stage 2, start Stage 4.

Key open observations: "reflection jitter" (measured uncertainty rises
when reflection installs a claim the model's own polls don't
reproduce — 5 sightings); the runoff judge convenes on pre-reflection
uncertainty only.

The central research finding so far: **most defects were ours, not the
model's** — "fixing the interview, not the witness." The model's true
deficits: it capitulates under authoritative pressure, its second looks
are unstable, and its self-reported confidence is flat 0.95 (useless).
