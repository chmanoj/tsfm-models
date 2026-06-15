# TETRIS — Working Agreement (read before contributing / resuming)

This captures *how* we build TETRIS so the process is reproducible across machines
and sessions. It is process, not architecture — the architecture lives in the three
source-of-truth docs below.

## Source of truth (read in this order)

1. `docs/tetris/tetris_decision_log.html` — the architecture & decisions **D1–D15**
   (the "why"). Do not redesign anything in it.
2. `docs/tetris/implementation_plan.md` — the build plan: module layout, tensor
   shapes, build order **S0–S13**, the four pre-training gates, and the pinned
   **reconciliation blocks** recording every decision already taken.
3. `docs/tetris/tetris_pipeline_walkthrough.html` — the pinned end-to-end
   type/shape reference (Stages 1–9, exact signatures/shapes).

**Tiebreaker: where the plan and the walkthrough conflict, the walkthrough wins.**

## Rules

- **Build in order S0 → S13, one stage at a time.** Get the stage's smoke/unit test
  green before starting the next. The four mandatory pre-training gates are
  **S1, S2, S5, S9** (plus the required `test_aux_boundary`); all are green.
- **Stop and ask on ANY cross-doc ambiguity or conflict.** Do not silently pick an
  architectural direction — even when "walkthrough wins" would resolve it. State the
  ambiguity, give a recommendation with pros/cons, and let the maintainer decide
  (use a focused multiple-choice question).
- **Don't over-build / don't jump ahead.** Implement the stage in front of you; no
  speculative multi-stage builds. Check in at natural milestones (e.g. for a big
  stage, after the first sub-component and after the first full forward).
- **Commit per stage, only when asked.** The rhythm is: stage test green → report →
  maintainer says "commit Sx + docs, then proceed". Branch off `main` for work;
  don't push unless asked. End commit messages with the `Co-Authored-By` trailer.
- **Record every decision in the plan.** After a decision, add/extend a pinned
  `**<Topic> reconciliation (post-Sx — pinned, do not re-flag):**` block in
  `implementation_plan.md`, and fix any now-stale in-body references so the doc
  can't mislead later.

## Conventions

- **Match the surrounding code style:** dataclasses + type hints, concise docstrings
  citing the relevant D-decision/Stage. **numpy at CPU pack-time, torch on the model
  side.**
- **Backends (D14):** CUDA → FlexAttention + `torch.compile`; Mac/CPU → SDPA +
  materialized bool mask + eager. The two paths must be numerically equal (tested).
  Dev machines here are Mac/no-CUDA → the SDPA/eager path runs locally.
- **Static shapes (D14):** everything compiles at static `L = L_pack`; per-sample
  variability lives in tensor *contents*, never shapes. No per-batch dynamic dims.
- **D8 hard rule:** no buffer-index positional encoding anywhere; all token geometry
  comes from side tensors (`t_center`, `tier_id`, `variate_uid`, `role`, …).
  `test_pack_invariance` (S9) is the keystone guard.

## Tooling

- `uv` for env/deps (`uv sync`); `pytest` for the gates (`uv run pytest`).
- Run one test: `uv run pytest tests/test_<name>.py -v`.
- Config is dataclasses ↔ YAML via OmegaConf (`configs/{shakedown,base}.yaml`).
- **Python pinned to 3.13** (`.python-version`). Do **not** use Python 3.14 — it
  breaks `torch.compile` (its functorch path imports `networkx`, which fails to
  import on CPython 3.14.1). Revisit when the 3.14 toolchain catches up.
- No-recompile (D14) is tested with `torch.compile(model, backend=CompileCounter(),
  dynamic=True)` + `mark_dynamic` on the per-step-varying dims (`R`, `n_var`),
  asserting a flat frame count after warmup — eager backend, so no CPU inductor.
