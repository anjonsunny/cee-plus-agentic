# Agentic CEE+

An agentic rebuild of **CEE+**, a framework that measures whether a Vision-Language
Model's safety recommendations are *causally grounded* or only *fluently coherent*.

CEE+ suppresses a hazard from a disaster scene and checks whether the model's advice
actually moves in response. Recommendations that stay put when their justification is
removed are coherent rung-1 association wearing the costume of rung-3 counterfactual
reasoning. This repository rebuilds that measurement as an **agentic system**: planning,
loops, reflection, tool use, memory, multi-agent verification, RAG, and a conversational
agent, orchestrated with LangGraph.

> This is the **agentic** repo. The original single-shot CEE+ pipeline lives at
> [`anjonsunny/CEE`](https://github.com/anjonsunny/CEE). The scoring functions in
> `main.py` / `intervention.py` are reused unchanged; the agentic layer never
> re-implements a number.

---

## Two documents define the project

| Document | What it is |
|---|---|
| [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) | The **controlled experiment**. Two immutable arms, controls, calibration anchors, fair-test validation, go/no-go gates. This carries the scientific claim. |
| [`AGENTIC_PLAN.md`](AGENTIC_PLAN.md) | The **engineering and capability plan**. All 25 stages, the loops, the evaluation strategy, and the build order. Subordinate to the protocol: nothing here may change a number in it. |

The governing rule throughout: **agents decide, tools compute.** Determinism lives in the
scoring functions, so a stage can be agentic without losing reproducibility.

---

## What is built so far

The system is being built stage by stage against a frozen six-scene worked example.
Current state, honestly:

| Component | File | Status |
|---|---|---|
| **Stage 1 — Perception** | `agentic/perception.py` | Built. VLM emits label + state; a detector owns the box; closed vocabulary with a logged escape hatch. |
| **Loop 1 — Local repair** | `agentic/repair_loop.py` | Built. A rule-conformance critic drives repair, the external feedback that reflection needs. |
| **Perception rulebook (RAG seam)** | `agentic/rulebook.py` | v0. Rules as retrievable chunks; exact lookup now, embeddings later. |
| **Dialogue agent** | `agentic/dialogue.py` | v0 on LangGraph. Read-only and terminal; queries run records, never re-enters the pipeline. |
| **Dialogue tools** | `agentic/agent_tools.py` | Built. Structured lookup over run records (Stage 17). |
| **State vocabulary** | `agentic/vocabulary.py` | Frozen Stage-1 label set. |
| **Live UI** | `agentic/ui.py` | v0. Watch the perception pipeline and dialogue agent work. |
| **Scene runner** | `agentic/run_scenes.py` | Runs Stage 1 over the six worked-example scenes. |
| Stages 2–25 | — | Designed in `AGENTIC_PLAN.md`, not yet built. |

Everything downstream of perception (scene assessment, threats, quads, Graph B, the
counterfactual core, synthesis, progressive suppression, fine-tuning) is planned, not
implemented. This README will not claim otherwise.

---

## Stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph (state machine, loops, checkpointing) |
| Typed state | Pydantic |
| Memory | LangGraph checkpointer + store |
| RAG | rulebook chunks; exact lookup now, embeddings planned |
| Subject VLM | Qwen2.5-VL via Ollama |
| Scoring | reused from `main.py` / `intervention.py` |

---

## Layout

```
agentic/              the agentic package (Stage 1 + dialogue agent, above)
main.py               baseline VLM pipeline + scoring (reused, not re-run as the subject)
intervention.py       suppression enumeration, shift signals, groundedness adjudication
tests/                baseline test suite
AGENTIC_PLAN.md       25-stage engineering plan
RESEARCH_PROTOCOL.md  the controlled experiment
PROJECT_STATE.md      full state of the baseline system
```

## Running

```bash
conda activate clip_dash
pip install -r requirements.txt
pip install -r agentic/requirements.txt      # langgraph, pydantic, etc.

export QWEN_API_URL="http://localhost:11434/v1/chat/completions"
export QWEN_MODEL_NAME="qwen2.5vl:7b"

python -m agentic.run_scenes      # Stage 1 perception over the worked example
python -m agentic.ui              # live UI
pytest agentic/                   # agentic tests
```

---

## Status

Early. Stage 1 works end to end on the worked example, and the design for the rest is
complete and reviewed. Progress is deliberately gated: each stage keeps a deterministic
baseline, so every agentic addition is a *measured* improvement rather than an assumed one.
