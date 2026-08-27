# CEE+ — Causal Explanation Engine for VLMs

CEE+ tests whether a Vision Language Model's recommendations are mechanistically anchored to scene evidence or only declaratively coherent with it. It is a working evaluation framework for safety critical decision support, focused on fire disaster scenarios as a controlled proxy domain.

The core claim: a baseline VLM can produce coherent threat identification, recommendations, and structured reasoning, but that reasoning often remains *declarative* rather than *mechanistically verified*. CEE+ exposes the causal structure explicitly via intervention based transparency, and detects a recurring set of model honesty pathologies.

---

## Two arms

CEE+ runs as two arms that share one scoring vocabulary, so results stay directly comparable:

- **Legacy pipeline (`main.py`).** The original monolithic pipeline with frozen scoring and the intervention tab. It owns the state vocabulary and the causal-quad ontology.
- **Agentic pipeline (`agentic/`).** A staged, self-correcting conversion of the same evaluation, built as perception, assessment, and recommendation stages, each with its own checks and revision loops. It is scored by *importing* the legacy pipeline's frozen metrics, so the two arms measure the same thing.

## What it does

For every scene, CEE+ builds three views of the model's reasoning and scores agreement across them.

| View | Source |
|---|---|
| 1. Recommendations | What the model would act on |
| 2. Stated beliefs | A separate causal graph prompt asking what the model believes |
| 3. Reference truth | A human validated ground truth scene |

Then it computes:

| Metric | Question it answers |
|---|---|
| **A fidelity** | Do recommendations match the model's stated beliefs? |
| **B coverage** | Do the model's beliefs surface in what it recommends? |
| **Internal alignment** | Is the brief self consistent on its own terms? |
| **Measured uncertainty** | Re-sampling each step, do the same claims reproduce, or are they unstable? |
| **Trust Score** | Consequence-weighted operational roll-up with Low / Moderate / High bands |

## Agentic pipeline

The agentic arm processes each scene in stages and self-corrects rather than answering in one shot:

- **Stage 1 — Perception.** The VLM names entities; a repair loop fixes malformed output (the model may stand its ground); boxes and masks are attached; hazards are derived as states and duplicate detections merged. Nothing is deleted silently, every derivation is recorded with a note and an event.
- **Stage 2 — Assessment.** One merged judgment (disaster yes/no, type, level, threats, at-risk entities), checked against the ontology and geometry, then re-sampled to measure uncertainty and refined through a reflection loop (capped, evidence-quoted). Judges advise; only the model revises.
- **Stage 4 — Recommendations (in progress).** The model recommends actions; measured uncertainty re-samples each step to flag recommendations that don't reproduce; a causal graph is built two ways, from code and from the model's own structured declaration, and intervention candidates are selected from both plus direct model asks. A consequence-weighted Trust Score rolls the signals up with Low / Moderate / High bands, minus deductions from a priced library of singular errors (an invented emergency, a victim left behind, an unaddressed hazard, a non-action).
- **Judges (advisory only).** Every verdict is advice; none moves a score. Built so far: a card judge (does each explanation actually explain its action?), a runoff judge over disagreeing probe answers (two applications: recommendations and Graph B), and a combined A-vs-B judge (which causal account better describes the scene, and whose endangered set faces the graver harm?). Judges run as twins — the same model answering text-only (official) and image-aware (witness) — and only their agreement or disagreement is displayed. All judge reasoning is captured verbatim, alongside preference pairs and repair pairs, as training data for a future fine-tuned overseer.

Two self-correction loops and a two-route petition (re-look at the image, or re-ask the question once, fresh) drive revision. The whole agentic pipeline runs as byte-identical LangGraph and Python controls and is covered by a hermetic test suite (`pytest agentic/ -q`, no models needed).

## Intervention based grounding

Causal grounding is verified by intervention. Suppress a hazard in the scene (visual inpainting, caption redaction, or both) and re-run the pipeline. Six shift signals measure what changed:

1. Hazard shift
2. Causal graph shift
3. Recommendation shift
4. Structural alignment shift
5. Semantic alignment shift
6. Cross modal consistency shift

If a recommendation does not change after the hazard it cited is removed, that recommendation was never anchored in the hazard. It was anchored in priors or in surface phrasing.

This intervention gate is operational in the legacy arm. In the agentic arm it is the next stage to port; today the agentic pipeline runs through recommendation and intervention-candidate selection.

## Pathology framework

CEE+ detects five named model honesty pathologies. Each has a cross metric signature and an inferred ML mechanism cause.

| Pathology | What it does |
|---|---|
| Sycophancy | Gives the asker the answer they seem to want; does not push back on the framing. |
| Rationalized Minimization | Stacks defensible qualifiers until a real threat reads as ambiguous. |
| Truth Suppression for Peace | Softens findings that would create social or diplomatic friction. |
| Tribal Mirroring | Shades the same facts toward each audience's preferred framing. |
| Safety Theater | Refusal training as surface filter; reframed requests bypass it. |

The framework operates as a behavior level lie detector that requires no access to model internals.

## Schema innovations

- **Hazard as state grounding.** Hazards are encoded as states on entities (`fire_on_house_1`, not `burning_house`) so a specific mechanism can be suppressed.
- **Four part causal links.** Every causal claim is written as `(source, state, effect, target)` so the exact mechanism is suppressible.
- **Ground truth corpus.** 89 human validated reference scenes anchor the evaluation.

## Headline findings

69 scene Qwen2.5-VL batch (May 2026):

- A fidelity median: **0.33**
- B coverage median: **0.11**
- Internal alignment median: **0.87**
- 48 percent of scenes route to human review under the Trust Score

Briefs read coherent on top of broken reasoning. The model is producing unjustified confidence.

A pre-build audit of the imported scoring metrics (finding F15) surfaced four defects that made the trust read misleadingly low: saturation, a wrong denominator, double-counting of coupled violations, and severity-blindness. The agentic Trust layer corrects all four, so its score is non-saturating, uses a principled denominator, de-duplicates coupled violations, and weights by severity.

---

## Setup

```bash
brew install ollama
ollama serve
ollama pull qwen2.5vl:7b   # subject
ollama pull gemma4:26b     # judges (text + image twins)

conda activate clip_dash
pip install -r requirements.txt

export QWEN_API_URL="http://localhost:11434/v1/chat/completions"
export QWEN_MODEL_NAME="qwen2.5vl:7b"

# legacy pipeline (Dash) — frozen scoring + intervention tab
python main.py

# agentic pipeline (Dash UI)
python agentic/ui.py

# tests (hermetic, no models needed)
pytest agentic/ -q
```

## Project layout

- `main.py` — legacy pipeline (Dash) calling Qwen2.5-VL via Ollama; frozen scoring + intervention tab
- `agentic/` — the agentic pipeline
  - `perception.py`, `repair_loop.py`, `vocabulary.py` — Stage 1 (naming, repair, grounding, masks, hazard derivation)
  - `assessment.py`, `uncertainty.py`, `reflection.py`, `petition.py` — Stage 2 (merged judgment, measured uncertainty, reflection, two-route petition)
  - `recommend.py`, `evals4.py`, `graph_s4.py` — Stage 4 (recommendations, consequence-weighted trust, LangGraph twin)
  - `judge_card.py`, `judge_runoff.py`, `judge_graph.py` — the Stage 4 judges (card, runoff twins, combined A-vs-B)
  - `errors4.py`, `models.py` — singular-error library priced by consequence; model seats (any subject, any judge)
  - `JUDGES.md`, `PANELS.md` — the judge spec and the finding-to-objective map
  - `rulebook.py` / `rulebook_rag.py` — one law, two engines: code detects, rulebook text teaches
  - `evals.py`, `geometry.py`, `ui.py`, `dialogue.py` — GT eval, bbox geometry, Dash UI, chat over run records
  - `FINDINGS.md` — the findings ledger (F1–F54 across six fix categories, A–F)
  - `test_*.py` — hermetic tests, no models needed
- `experiments/agentic_scenes/` — frozen calibration scenes + verified ground truth
- `GROUND_TRUTH_PROTOCOL.md` — schema rules and validation conventions
- `CEE_plus_discussion_notes.md` — failure taxonomy and worked examples

## Status

- **Legacy arm (`main.py`)** — operational, including the intervention gate with six shift signals.
- **Agentic Stage 1 (Perception)** — operational.
- **Agentic Stage 2 (Assessment)** — operational; closing formally (the silence test on the safe scene plus scene re-runs).
- **Agentic Stage 4 (Recommendations)** — Phase 1a/1b built plus the judges' bench: recommend, measure uncertainty, build Graph A (from code) and Graph B (from the model), pick intervention targets, roll up a consequence-weighted Trust Score with the singular-error library, and run the advisory judges (card, runoff twins, combined A-vs-B) with all reasoning captured as training data. Python and LangGraph controls are byte-identical; 714 hermetic tests pass. Calibration on the six scenes is in progress (A fire, C tanker, D aerial done on the current prompts).

Roadmap, in order:

1. **Calibrate Stage 4 on the six scenes** (live, via Ollama). Tune the trust and consequence weights and calibrate the judges scene by scene. This same live pass closes Stage 2. In progress.
2. **Reflection loop for Stage 4** (gated by measured uncertainty). The remaining judges land here (pathology detector and judge, then the overseer); judges advise, reflection carries the message, and only the model revises; petitions are the second self-correction route. A light re-calibration follows, since reflection changes the outputs.
3. **Pathology (S5).** Map the calibrated, reflection-stable signals onto the five named pathologies.
4. **Intervention gate (S6).** Port the counterfactual gate (declared-vs-operative groundedness, faithfulness) from the legacy arm into the agentic arm, then extend it to multi-step interventions and to video and audio.

The three-arm comparison harness is research infrastructure for the write-up and slots in around S5 and S6; it does not block the sequence.

## Publication

Assessing the Causal Reliability of AI-Generated Emergency Explanations: An Intervention-Based Evaluation Framework. *HCI International* 2026 (forthcoming; Springer LNAI vol. 16744).

---

## Acknowledgment

Research was sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-25-2-0116. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
