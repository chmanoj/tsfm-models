# TETRIS — Implementation Plan (v1)

**Status:** planning artifact. No model/training code yet. Source of truth: `docs/tetris/tetris_decision_log.html` (rev 17, D1–D15 + v2 backlog). **The pinned end-to-end type/shape reference is `docs/tetris/tetris_pipeline_walkthrough.html`** (Stages 1–9 with exact signatures and tensor shapes); where this plan and the walkthrough differ, the walkthrough wins. This plan does **not** redesign any decided item; it sequences them into buildable modules with smoke/unit tests.

**Walkthrough reconciliation (post-S4 — pinned, do not re-flag):**
- **`SegmentSpec` carries a per-channel role list** `channel_roles: list[C] ∈ {FEATURE, TARGET, KFF}` (plus scalar `C, n_features, n_targets, origin, p, Q, K, Q_total, n, counts[6]`). It is CPU pack-time data and never reaches the GPU, so a variable-length list there does **not** affect the static compiled shapes (the *Batch* tensors stay `[B,L]`). No positional KFF assumption.
- **Normalization stats are per-(sample,channel) constants**, not per-step: `Stats = {a: float, sigma_delta: float[6]}`. `a:[n_samples,C]`, `σΔ:[n_samples,C,6]`. `sigma_delta[0]` is the base scale `1.4826·median|Δx|` used for the input `norm_values` and Stage-9 inversion; `[1..5]` are per-tier scales for the locally-reanchored aux targets (D10) and the D4 variate scale-receipt. `norm_values[R]` is the only per-timestep quantity.
- **Three token enums retained (decision-log faithful):** `role ∈ {CTX, QRY}` (D6/D9, drives the mask), `content_state ∈ {OBSERVED, MASK, NA}` (D7, selects the content slot), `role_ft ∈ {FEATURE, TARGET}` (D11, the added role embedding). **KFF is not a token role** — it is an observed CTX *feature* token at `t_center > 0` (D11: distinguished by time, not a role), encoded by the P_out tier encoder. The embedding is a 5-part sum (content, time, span, variate, **role_ft**); `content_state` *selects* the content slot (encoder output | `[MASK]` | `[NA]`) and is not itself added; `role`/CTX-QRY is structural (mask + content + masks at loss). The target→feature *demotion* half of D11 augmentation is deferred; KFF-reveal among native features is kept.
- **Everything compiles at static shape `L`.** No per-batch `Qmax`. `horizon_target`/`target_valid` are dense `[B,L,P_out]`; encoders run at static `ENCODER_CAP` (config `model.encoder_cap`, default `L`), `[CAP,P_k,2]→[CAP,D]`, sentinel-padded; heads run dense over all `L`, gather/mask at loss time. The encoder input "2" = (normalized value, observed-indicator) per timestep (D7). Encoder-routed tokens are exactly `content_state==OBSERVED`.
- **Attention mask (S5)** is the decision-log D9 truth table (KFF behaves as CTX: queries read it, it never attends queries). Earlier `tier_caps`→`tier_alloc_per_channel` (ratio prior) and `n_ctx_cap`→`encoder_cap` (= `ENCODER_CAP`).

**Collator reconciliation (post-S6 — pinned, do not re-flag):** the plan §2.1 once
wrote `pack(items, specs)`; the **walkthrough Stage 4** (`collate` over *already
assembled* per-segment inputs) wins. The frozen signature is:

```
pack(buffers: list[list[AssembledSegment]], *, l_pack, p_out, num_buffers=None) -> Batch
```

- **Input = assembled segments**, not raw `(item, spec)`. `assemble()` (S4) runs
  first; `pack` does **no** tokenization.
- **Caller owns buffer grouping.** `pack` is pure tensor materialization of a
  *pre-grouped* `list[list[AssembledSegment]]` (one inner list per buffer); it does
  **no** best-fit packing. Best-fit-decreasing lives solely in the reservoir (S11);
  the trivial path (S6/S10) and the reservoir path both call this same frozen `pack`.
- **`Batch` is the 16-field record** (walkthrough Stage 4 = §1 table; the stale
  "13 fields" was a miscount). Implemented as a dataclass of CPU `torch` tensors
  (matches `AssembledSegment`/`SegmentSpec` style; `.to(device)` in the loop).
- **Per-segment → buffer-global rebasing in `pack`:** `raw_start` offset by the
  segment's base in the buffer's concatenated `norm_values` store; `variate_uid`
  offset so every `(sample, channel)` is **buffer-unique** (D4); `stats_a`/
  `stats_sigma` broadcast per token from `(a, σΔ[0])`. `channel_idx` stays
  **sample-local** (unlike `variate_uid`).
- **`R` = max per-buffer store length, ragged→padded** (the only non-`L`-bounded
  dim, walkthrough Stage 3). Pad tokens get `sample_id=-1`, `content_state=NA`
  (so they never route to an encoder), `raw_start=-1`, `variate_uid=-1`.
- **No `aux_target` Batch field.** Aux targets are gathered from `norm_values` at
  loss time (S8) — the collator carries only `valid_aux`.

**Model reconciliation (post-S7 — pinned, do not re-flag):**
- **Variate IDs (D4) are eager-QR, hoisted out of the compiled forward.**
  `sample_orthonormal_basis(B, n_var, d)` builds a **per-buffer** orthonormal pool
  (`[B, n_var, d]`, QR; L2-normalized-Gaussian fallback when `n_var > d`) — keeps
  QR out of the graph (D14 no-recompile). Resampled per step in training, fixed
  seed at inference. `Tetris.forward(batch, variate_basis=None, *, generator)`
  samples internally when not supplied (standalone smoke), but a train step should
  pass it in. `n_var = max(variate_uid)+1`; basis is per-buffer because
  `variate_uid` is buffer-local-unique (collator), so the same id in two buffers is
  two different variates.
- **Per-tier encoders (D3) are uniform Fourier(value)+observed → 2-layer MLP.**
  Each step: `[sin(vω),cos(vω)]` (log-spaced freqs) ⊕ observed bit; flatten the
  `P_k` window; `Linear→GELU→Linear → d`. No appended summary stats in v1 (D3
  "may append" — deferred). Same recipe all six tiers (only `P_k` differs).
- **Dispatch (Stage 7) is fully static** (D14): per tier, a cumsum-rank +
  `scatter_` into a `[CAP+1]` trash-binned buffer builds the `[CAP]` slot list;
  windows gathered from the flat `norm_values` via `raw_start`; `index_add_`
  scatters encoder outputs back (sentinel rows add 0). Routed = exactly
  `content_state==OBSERVED`; `MASK`/`NA`/pad get learned `[MASK]`/`[NA]` vectors.
- **Time embedding = Fourier(`t_center`) → linear** (own low-freq range,
  wavelengths ~6..6000, since `t_center` spans ±thousands). span/role_ft are small
  `nn.Embedding` tables. **Position is only `t_center`** (D8).
- **Heads run dense over all L** (option-β): `horizon: d→P_out`, six `aux: d→P_k`;
  selection/masking deferred to the S8 loss reduction. `tetris.forward` returns raw
  head outputs (`ModelOutput{horizon, aux}`); loss + inversion are S8.

**Loss reconciliation (post-S8 — pinned, do not re-flag):**
- **Aux targets use the base scale `σΔ[0]` (walkthrough-faithful), NOT D10's
  per-tier locally-reanchored `σΔ[1..5]`.** The decision log D6/D10 (+ `normalize.
  aux_target`) call for per-tier step-vol re-anchoring, but the **frozen S6 Batch**
  carries only base-scale `norm_values` + `valid_aux`; honoring D10 would reopen the
  collator. Decided: keep base scale — aux is auxiliary and damped by `aux_weights`,
  and `test_aux_boundary` (validity masking) is independent of the choice. Therefore
  **`config.norm.loss_target` is not yet wired** (locally_reanchored vs
  global_norm_space is a later iteration); `loss_space` honors `arcsinh` (default),
  `vol_units` raises `NotImplementedError` (deferred).
- **Aux target gather:** for a tier-`k` token, the next `P_k` raw steps
  `[raw_start+P_k, raw_start+2P_k)` from the buffer's `norm_values`; masked by
  `tier_id==k & valid_aux` **and** the target steps' `observed` bit + in-range
  guard. `valid_aux` (collator) is the precise raw-time check (origin-crossing →
  horizon's job; history-edge → unavailable). `total = horizon_MAE + Σ_k
  aux_weights[k]·aux_MAE_k` (the `aux_weights[6]` vector replaces a single λ).
- **Metrics (D13):** v1 records **test loss only** (horizon MAE on a shard);
  **MASE deferred (O4)** — `seasonal_naive_denom`/`mase` are stubs.

**S9 gate (passed) — pinned test convention:** `test_pack_invariance` proves the D8
no-leakage invariant (identical samples solo vs packed-together / permuted /
regrouped → identical per-sample outputs **and** total loss). Because D4 variate IDs
are *relational* (a single forward is not invariant to which orthonormal vectors are
drawn), the **test pins a fixed basis** mapping each sample's channels to the same ID
vectors across layouts (training resamples freely). All four pre-training gates
(S1, S2, S5, S9) + the required `test_aux_boundary` are green.

**Train reconciliation (post-S10 — pinned, do not re-flag):**
- **No-recompile gate is exercised locally via `torch.compile` + `CompileCounter`**
  (decided: do it for real, not CUDA-only). `test_shakedown` compiles the model with
  the eager `CompileCounter` backend + `dynamic=True`, runs varying-data steps, and
  asserts the frame count is **flat after warmup** (D14 one-graph). Eager backend
  avoids slow/flaky CPU inductor while still catching shape-guard recompiles.
- **`step.py` hoists the variate basis out of the compiled forward** (eager QR) and
  marks the only per-step-varying dims dynamic — `R` (`norm_values`/`observed`) and
  `n_var` (basis) — so one graph serves all data. `train_step(forward, batch, basis,
  opt, ...)`; `mark_dynamic_batch`; `make_basis`.
- **Shakedown trivial path = one segment per buffer** (`pack([[s] for s in segs])`),
  streaming `build_loader → sample_window → assemble → pack`. The reservoir path
  (S11) reuses the same `pack`/`train_step`; only the grouping changes.
- **ENV: project pinned to Python 3.13.10** (`.python-version`). Python **3.14**
  breaks `torch.compile` — its functorch path imports `networkx`, which fails to
  import on CPython 3.14.1 (a dataclass-`slots`/`__annotate__` change; `networkx`
  3.6.1 is already newest). 3.13 is the stable ML target; revisit when the
  3.14 toolchain catches up. `Batch.to(device)` added (additive) for device moves.

**Streaming packer reconciliation (post-S11 — pinned, do not re-flag):**
- **Reservoir yields assembled pack-groups; packing lives in an adapter, not the
  loop.** `StreamingReservoir.__next__` yields one step as
  `list[list[AssembledSegment]]` (B buffers). The frozen `pack` is called by the
  thin `packed_batches(groups, *, l_pack, p_out, num_buffers)` generator **between**
  the reservoir and `train/loop.py` — so the loop only iterates `Batch`es and the
  reservoir stays collator-free / unit-testable. Decided over "reservoir returns a
  `Batch`" to keep `pack` the single explicit seam. The collator/`Batch` are
  untouched (S6 frozen).
- **BFD reuses `pack`'s overflow contract as an invariant.** `_form_one_buffer`
  greedily places the largest spec that fits the residual until residual <
  `tail_tolerance·L` (or nothing fits); every spec satisfies `S ≤ L` by the
  window-sampler budget, so giants take a buffer solo. `ΣS ≤ L` always holds ⇒
  `pack`'s `ValueError` is a pure guard, never tripped on this path.
- **Cost-bucketing is from specs, single-rank, before assembly.** `scheduler.py`:
  `buffer_cost = Σ S_i²/2` (D9.4), `cost_bucketed_steps` sorts a window of
  `scheduler_window` (W ∈ [64,256]) buffers and chunks into similar-cost B-buffer
  steps (contiguous in cost order; short final step pad-filled by
  `pack(num_buffers=B)`). Cost needs only `spec.S`, so scheduling runs **before**
  `assemble`; only emitted buffers are assembled. **Single-rank/in-process now**;
  the cross-rank cost all-gather (§9) stays S13 — the `cost_of` callable + the
  deterministic sort are the seam a global schedule reuses.
- **Resume is minimal-but-real (D13).** `state_dict`/`load_state_dict` capture the
  RNG, the loader cursor (`items_pulled`), the pending reservoir specs, and any
  formed-but-unyielded steps → exact resume at the **same** world size (the loader
  is re-derived and the cursor skipped on load). Full re-shard at a *different*
  world size is S13. Per-rank, shuffle seed rank-offset (`default_rng((seed,
  rank))`); base loader already sharded disjointly by `build_loader` (O6).
- **New `packing` config knobs:** `reservoir_k` (K≈1000), `scheduler_window`
  (W∈[64,256]), `tail_tolerance` (residual floor, default 0.05). `cfg.packing.
  reservoir` flips the trivial S10 path (`shakedown.py`) to the reservoir path
  (`loop.run_training`); both call the same `pack`/`train_step`.

**Eval reconciliation (post-S12 — pinned, do not re-flag):**
- **Real GIFT-Eval download is a lazy, dependency-free seam.** `gifteval_download.
  py` fetches `Salesforce/GiftEval` (HF dataset, gluonts arrow; repo id verified,
  O1) via `huggingface_hub.snapshot_download` and splits test windows with the
  official `gift_eval` package — **both lazy-imported, NOT in `pyproject` deps**.
  The network path is never exercised in CI; `test_eval_loss` runs entirely on an
  offline synthetic shard (`make_synthetic_eval_shard`). The 97-config table comes
  from the downloaded tree (`_list_configs`).
- **Held-out scoring reuses the frozen tokenizer at a fixed origin (no S4/S6
  edits).** `eval_loader.eval_batch` rebuilds a training-style segment from
  `cat(context, y_true)` with **origin pinned at the context boundary** and `p =
  len(y_true)` (target rows get `y_true`; feature rows get NaN futures — they carry
  no horizon tokens), then runs the unchanged `assemble`→`pack`→horizon head.
  Query slots are `content_state == MASK` and the causal mask blocks context→
  horizon, so `y_true` lands **only** in `horizon_target`, never the observed store
  — same no-leakage guarantee as training. `eval_spec` mirrors `window_sampler`'s
  budget math deterministically (no random origin/p, no KFF). Decided over a
  duplicate eval-only `assemble`.
- **`EvalItem` never enters the training path.** A separate `build_eval_loader(cfg)`
  factory (`gifteval_test` = lazy download · `synthetic_eval` = offline shard)
  returns the map-style `GiftEvalEvalLoader`; the training `build_loader` stays
  Item-only. `to_train_item` strips the held-out fields before packing.
- **Test loss is record-only (D13), MASE deferred (O4).** `evaluate_test_loss`
  averages horizon MAE (Stage-9 normalized space, `metrics.horizon_test_loss`,
  `@torch.no_grad`) over the first `eval.shard_windows` windows, one segment per
  buffer, with the **inference variate basis fixed** (`basis_seed`, D4). It updates
  no parameters. `loop.run_training` gained an additive, optional eval hook
  (`eval_loader`/`eval_every`/`eval_log`) that fires the *same* collator on
  context-only items — training-loader contract unbroken. `EvalItem.naive_denom`
  stays `None`.

**Distributed reconciliation (post-S13 — pinned, do not re-flag):**
- **`test_distributed` spawns real gloo process groups** (`torch.multiprocessing.
  spawn`, CPU) — decided over an in-process simulation. It asserts the O6
  guarantees directly: disjoint+covering shards per rank, DDP-vs-single-process
  per-sample loss identity, and checkpoint re-shard 2→3.
- **Cross-rank cost scheduling needs no collective.** Each rank's
  `StreamingReservoir` already cost-sorts its scheduler window (S11), so global
  step `t` draws similar-cost steps on every rank by construction — the
  "deterministic global cost-sorted schedule seeded identically" option, chosen
  over an all-gather of per-buffer costs (which would couple the reservoir to the
  process group). No new scheduler code in S13.
- **DDP is wired+tested for real; FSDP is switch-only.** `wrap_model` selects on
  `cfg.distributed.parallel`: DDP (`find_unused_parameters=True` so a buffer
  lacking some tier's tokens doesn't trip the reducer; `device_ids` only on CUDA;
  `torch.compile` applied by the caller *after* the wrap). FSDP is a recognized
  branch wrapped only when selected **and** available — **not exercised on CPU CI**
  (FSDP-on-CPU is fragile), same lazy posture as the GIFT-Eval download.
- **Per-sample numerical identity is a forward/loss property, not a grad claim.**
  No module/collator reads cross-rank or global state, so a fixed sample's
  forward+loss is identical solo vs under a 2-rank DDP job (`test_distributed`
  pins `|ddp − single| < 1e-5`); DDP only averages gradients. This extends the S9
  pack-invariance keystone across ranks.
- **Checkpoints (D13) persist per-rank reservoir state and re-shard.**
  `save_checkpoint` writes one file per rank (model/optimizer identical across DDP
  ranks; reservoir state rank-local). `load_checkpoint` restores model/optimizer
  from the canonical rank-0 file always; if the saved `world_size` matches it
  restores the rank-local reservoir **exactly** (S11 `state_dict`), otherwise it
  **re-shards** — a fresh reservoir for the new `(rank, world_size)` refills from
  its new disjoint shard. `rank_shard` is the deterministic round-robin partition
  `StandInPretrainLoader` already uses. **S13 closes the build order (S0–S13).**

**Sanity stage reconciliation (post-S13, real-data bring-up — pinned, do not re-flag):**
- **Deliberate departure from zero-shot, staged.** The decision log targets
  zero-shot pretraining on `GiftEvalPretrain` → zero-shot GIFT-Eval. Before any of
  that, the maintainer chose a **simple-synthetic in-distribution `train→test`
  bring-up** to prove the architecture has capacity to learn at all: (1) sine
  univariate first, then (2) one case at a time, then (3) all cases together; then
  GIFT-Eval `test→test` overfit; then synthetic + Pretrain + GIFT-Eval `train` →
  GIFT-Eval `test`. This does **not** rewrite the docs' zero-shot framing — it is a
  bring-up ladder in front of it.
- **MASE (O4) is un-deferred — but period detection is NOT in the model.** Scoring
  is GIFT-Eval/gluonts MASE = `MAE_horizon / in-sample seasonal-naive denom`, in
  **raw value space** (the horizon head is inverted out of anchored-arcsinh via
  `sinh(·)·σ+a`, the D10 Stage-9 receipts on the Batch). The season length `m` is
  **dataset metadata** carried on `EvalItem.season_length` (synthetic series are
  generated with known calendar-style periods; GIFT-Eval provides its own), exactly
  as the leaderboard provides seasonality. `metrics.{seasonal_naive_forecast,
  seasonal_naive_denom,mase}` are now real (raw-space, 1-D per channel);
  `evaluate_mase` reports model vs seasonal-naive gmean MASE + the skill ratio.
- **New seams, all additive / frozen seams untouched.** `data/sanity.py`
  (`SanitySpec` = single source of truth for matched train+eval series) +
  `data/sanity_loader.py` (`SanityTrainLoader`, cyclic overfit pool, O6 rank-shard;
  `make_sanity_eval_shard`). Factory keys `sanity` (train) / `sanity_eval` (eval).
  `EvalItem` gained `season_length: Optional[int] = None` (defaulted → GIFT-Eval/
  synthetic-eval constructors unchanged). `DataCfg` gained `case`,
  `season_lengths`, `horizon`, `series_len`, `n_channels`, `local_dir` (additive
  structured-config merge). `run_training` gained an optional `eval_fn` (defaults to
  `evaluate_test_loss`) so the loop can record MASE mid-train. `pack`/`assemble`/
  `Batch`/`window_sampler` are **untouched** — the sanity loader yields the frozen
  `Item`, eval reuses the frozen no-leakage `eval_batch`.
- **Entrypoint + artifacts.** `train/sanity_run.py` ties loader→eval→MASE and writes
  a git-ignored `outputs/<run>_<ts>/` per run: `command.txt`, resolved `config.yaml`,
  `train_log.txt` (stdlib logging, `time | level | module | msg`; logs param counts,
  device, loader sizes, per-eval MASE, total training time), and `samples.png`
  (5 random eval samples: context + actual vs model vs seasonal-naive, all cases).
  **First result (sine univariate, d=64/3L, CPU):** model MASE 6.8 (random) → 0.81,
  beating seasonal naive (0.98) — the architecture learns. `test_sanity_mase`
  (offline, CI-safe) guards the MASE math + the learns-a-sine smoke.

**GIFT-Eval G1 reconciliation (observability — pinned, do not re-flag):**
- **Mid-train eval MASE now logs under `--compile` (GPU-note #1 fixed).** The eager
  B=1 eval used to recompile per item, so `sanity_run` skipped the mid-train hook
  under compile. Fix mirrors the train step inside `evaluate_mase`/`evaluate_test_loss`:
  `mark_dynamic_batch(batch, basis)` marks the only per-item-varying dims (`R`/`n_var`)
  dynamic, and the FlexAttention `BlockMask` is built **eagerly** via
  `make_block_mask(batch)` and passed into the forward (it can't be lowered inside
  `torch.compile` — same hoist as `train_step`). The mid-train hook in `sanity_run`
  is re-enabled unconditionally (`loop_eval = eval_loader`, `eval_every` honored under
  compile). **Verified on the WSL RTX 3070** (sine, 60 steps): periodic eval MASE logs
  at every `eval_every`; **exactly 1** recompile event total (the train→`no_grad`
  `grad_mode` flip, not per-item); compiled and eager-on-CUDA numbers are **identical**
  (init 6.7274; step40 4.6076; step60/FINAL 3.1098). No frozen seam touched.
- **Experiment-tracker seam (new `train/tracking.py`), optional + offline-friendly.**
  A backend-agnostic `Tracker` protocol (`log_config` / `log_scalars(*, step)` /
  `finish`) with `NullTracker` (no-op) and `WandbTracker`. `make_tracker(cfg, …)`
  **never raises** — a tracker problem degrades to `NullTracker` and logs why, so CI +
  Mac dev never depend on it. **Default backend = wandb** (maintainer's choice) with a
  graceful chain: `auto` mode resolves to **online** iff a wandb API key is discoverable
  (`WANDB_API_KEY`/`~/.netrc`) **and** `api.wandb.ai:443` is reachable (3 s TCP probe),
  else **offline** (local `wandb/` dir, no account, sync later); if `wandb` isn't
  importable at all → **disabled** no-op with a logged warning. `WANDB_MODE` env
  overrides. `cfg.tracking.backend == "none"` forces off. `TrackingCfg{backend, project,
  mode}` added to `Config` (additive; validated in `__post_init__`).
- **Wiring.** `run_training` gained an optional `tracker` param (defaults to
  `NullTracker`); it logs `{train_loss, lr, steps_per_sec}` each `log_every` and the
  flattened eval result (`eval/…`, NaNs dropped) each `eval_every`. `sanity_run` builds
  the tracker, logs the resolved config + base/final eval, and `finish()`es. wandb is
  **session-installed, not in `pyproject`** (lazy-dep convention, like the GIFT-Eval
  deps). `test_tracking` (13 tests, no wandb needed — fake-module + `sys.modules` tricks)
  guards the seam; `_tiny_cfg` sets `tracking.backend=none` so existing tests never touch
  wandb.
- **Credentials via `.env` (zero-dep loader).** `tracking._load_dotenv()` parses a
  repo-root `.env` (`KEY=VALUE`, quotes stripped) into the environment **without
  overriding** already-set vars, so `WANDB_API_KEY` is discoverable for `auto`→online +
  `wandb.init`. `.env` is **gitignored** (never committed). **To track GPU runs online
  on WSL you must put `.env` on the box too** — the rsync ship command does **not**
  exclude `.env`, so a normal `rsync` includes it (or `scp .env wsl-gpu:~/tsfm-models/`).
- **All three tracker paths live-verified.** **offline** (local run dir) ✓; **disabled**
  (wandb absent on WSL → logged no-op) ✓; **online** ✓ — a real run synced to project
  **`chmanoj/tetris`** (auto-created on first `init`) and confirmed via the wandb **API**
  (`api.runs("chmanoj/tetris")` → run `finished`, all 8 scalars in the summary, config
  logged).
- **Still open (deferred past G1):** throughput counts compile warmup in the cumulative
  `steps_per_sec`; tracker wired into the sanity entrypoint only, not
  `distributed.run_training`.

**GIFT-Eval G2 reconciliation (real data + leaderboard metric — pinned, do not re-flag):**
- **The S12 `gift_eval` API guess was wrong; re-verified against the installed package.**
  `gift_eval.data.Dataset(name, term, to_univariate, storage_env_var="GIFT_EVAL")` reads
  the storage root from the **`GIFT_EVAL` env var** — there is **no `storage_path` kwarg**
  (the S12 call `Dataset(..., storage_path=local_dir)` would have `TypeError`d). Fixed:
  `gifteval_download._make_dataset` sets `os.environ["GIFT_EVAL"]=local_dir` then constructs
  `Dataset(name, term)`. `term` is the `short|medium|long` enum (horizon ×1/10/15). Verified
  members: `.test_data.{input,label}` (gluonts `TestData`; per-window context/label entries
  with channel-major `["target"]`), `.training_dataset` (gluonts series before the test/val
  windows), `.prediction_length`, `.windows`, `.freq`. Installed for verification with
  `uv pip install --no-deps git+…/gift-eval` + `gluonts` (strict pins won't build on 3.13;
  `--no-deps` is fine since we only read the API; full deps live on WSL).
- **Seasonality has a single canonical, non-silent source.** `gift_eval` does **not** carry
  per-config seasonality; GIFT-Eval's own scoring derives it from the frequency via gluonts'
  `get_seasonality(freq)` (H→24, M→12, D→1, W→1, T→1440, Q→4, B→5; default-1 for unknown,
  gluonts' own warning surfaced verbatim). `gifteval_download._season_length` wraps exactly
  that and wires it to `EvalItem.season_length`. **No silent fallback of our own** (maintainer
  mandate) — multivariate configs share one freq so all channels get that season.
- **Train-split reader added (no factory key yet).** `iter_train_items(local_dir, configs,
  term, max_series_per_config, to_univariate, rank, world_size)` yields the frozen training
  `Item` 3-tuple from `Dataset.training_dataset` series (features-first `nf=0`, NaNs kept),
  round-robin sharded over the flattened `(config, series)` stream by `(rank, world_size)`
  (§9). **Deliberately not wired into `build_loader`** — the test-as-training loader is G3,
  train-split *training* is G5 (mutual exclusivity).
- **Leaderboard MASE = geometric mean across configs (+ per-config breakdown).**
  `eval_loader.evaluate_leaderboard` groups items by `EvalItem.config_id`, geo-means the
  per-channel-item MASEs **within** each config to one number, then geo-means those per-config
  MASEs (every config weighs equally, not every window — the GIFT-Eval convention; confirmed
  with the maintainer; CRPS out of scope, point forecast only). It reuses the exact per-item
  machinery of `evaluate_mase` via the new shared `_score_item` helper (`horizon_forecast_raw`
  → raw space → `metrics.mase` with the dataset season; compile-safe dynamic-mark + hoisted
  block mask). Returns `{leaderboard_mase, snaive_mase, skill, n_configs, n_configs_in_gmean,
  n_configs_finite, skipped, per_config}`. `evaluate_mase` is unchanged in behaviour (still the
  flat single-shard scorer for the synthetic/sanity mid-train eval; both now share `_score_item`).
- **NaN posture (maintainer mandate): model NaN poisons, data NaN is masked like gluonts.**
  The two NaN sources are handled **differently on purpose**. (1) **Data** NaNs — missing
  observations in the `*_with_missing` configs — are masked exactly as gluonts' `Evaluator` does
  (`np.ma.masked_invalid` on the **label** and **past_data**): `_score_item` masks the horizon on
  `isfinite(y_true)` only, and `metrics.seasonal_naive_denom` averages over the **finite** seasonal
  diffs (floored, never NaN). (2) **Model** output is **never** masked — the forecast flows into
  `mase` as-is, so a model NaN/inf propagates → MASE non-finite → poisons the per-config and
  leaderboard geo-mean. `_gmean` deliberately does **not** drop non-finite (only empty→NaN), so
  model breakage is **visible**, not hidden (our guard; the model should never emit NaN). A config
  leaves the cross-config geo-mean **only** when it has zero scorable channel-items (a pure data
  reason), never because the model poisoned it. **Observed at random init on the WSL subset:**
  `leaderboard_mase=inf` (the untrained head overflows `sinh(·)`), all 6 configs scored, snaive
  baseline finite (geo-mean ≈1.9) — i.e. the guard fires as intended; G3's training must bring the
  model output finite for a meaningful number.
- **Eval cap is per-config and configurable (default 10, -1=all).** New `EvalCfg.
  items_per_config` (validated `-1` or `≥1`) — the maintainer chose **10** for fast dev eval
  (deviates from the prompt's default 100) with **`-1` → all items**. It caps the real test
  windows per config in `iter_eval_items`, the train series per config in `iter_train_items`,
  and the per-config items scored in `evaluate_leaderboard`. The legacy `EvalCfg.shard_windows`
  (default 100) is a **distinct** knob: the *global* cap still used by `evaluate_test_loss`/
  `evaluate_mase` over the synthetic/sanity shards (config-ids are unique there, so per-config
  grouping is meaningless).
- **Storage roots come from env vars (maintainer's layout).** `TEST_ENV_VAR="GIFT_EVAL"`
  (the package's own var — one export wires both us and `gift_eval`) → `~/Projects/gifteval/
  test`; `PRETRAIN_ENV_VAR="GIFT_EVAL_PRETRAIN"` → `~/Projects/gifteval/pretrain` (reserved
  for G5). Resolution order for the eval loader: explicit arg → `cfg.data.local_dir` →
  `$GIFT_EVAL` (auto-loaded from the repo-root **`.env`** if not already exported — the same
  `python-dotenv` mechanism `gift_eval` itself uses, so a single `GIFT_EVAL=…` line in `.env`
  wires both us and the package); a missing root raises a clear `ValueError` (dep-free, before any
  lazy import). `_list_configs` walks the tree for `dataset_info.json` leaves (robust to variable
  nesting), returning names relative to the root (e.g. `electricity/15T`); `config_id =
  "{name}/{term}"`.
- **Tests stay CI-safe (same posture as `test_eval_loss`).** `tests/test_leaderboard.py`:
  the geo-mean aggregator + per-config cap (incl. `-1`) + season-skip are exercised **offline**
  on a hand-built multi-config `EvalItem` shard with a tiny real model; the storage-root guard
  is dep-free; the real download/iterators + the get_seasonality values **skip when the deps/
  data are absent**, so CI never touches the network. `uv run pytest` → **115 passed, 1
  skipped** (was 105; +10 `test_leaderboard`, the real-tree smoke skipped) — incl. explicit
  model-NaN-poisons-vs-data-NaN-masked tests.
- **Verified end-to-end on the WSL box (deliverable met).** Installed the extras there
  (`huggingface_hub datasets gluonts python-dotenv` + `gift_eval --no-deps`), set
  `GIFT_EVAL=~/Projects/gifteval/test` in `.env`, and ran our own `download_gifteval` for a small
  subset (`covid_deaths, hospital, m4_yearly, us_births/{D,M,W}` → 6 configs, 4.7 MB). Confirmed:
  `iter_eval_items` yields correct `EvalItem`s (context `[C,t]`, `y_true [p,nt]`, `season_length`
  from `get_seasonality`), `iter_train_items` yields valid frozen `Item`s, and
  `evaluate_leaderboard` runs (geo-mean + per-config; random-init guard = `inf` as above).
- **Not done in G2 (by scope):** no *training* run (G3); covariates (`past_feat_dynamic_real`)
  are ignored (`nf=0`, target-only) — KFF over real covariates is future work; only a 6-config
  dev subset is downloaded (full 97 is a larger fetch, done when needed).

**Decided session conventions** (from clarifying Q&A):

- **Compute / backends:** a backend switch. FlexAttention + `torch.compile` is the CUDA path (the real D14 target); SDPA + materialized bool mask + eager is the Mac (MPS/CPU) path for local unit tests and a reduced shakedown. The two paths must produce numerically equal masking/attention on the same inputs (tested).
- **Data layering (three layers):** `base loader (item stream)` → **streaming `IterableDataset`** doing window-sampling + reservoir + best-fit-decreasing + cost-bucketing → **stateless `pack()` collator**. The collator's relationship to the raw loader item never changes when the real loaders are swapped in.
- **Mask depth:** v1 ships **one depth-invariant** dense-causal `BlockMask`, built once per batch, reused across all layers. The per-layer channel-open schedule in D5 is an **A3-ladder contingency** (deferred), *not* a contradiction with D9/D14. Attention modules take `block_mask` as a **parameter** so a future per-layer schedule is an interface addition, not a refactor.
- **Tooling:** `uv` (env/deps), `pytest` (the mandated unit gates), plain dataclasses + YAML (`OmegaConf`) for the heavy ablation-toggle surface (D10 norm axes, D13 mixture weights, D15).

**Resolved stale references** (recorded so they aren't re-flagged):

- D3 mentions "5 encoders"; **D12 is final → six** per-tier encoders for vocabulary `{4, 8, 16, 64, 256, 512}`, six per-tier σ_Δ statistics.
- D2's `×1` raw tier is **rejected by D12** → finest patch is **4**; there are no per-step tokens.
- D5 "layer mask schedule" vs D9/D14 "depth-invariant in v1": reconciled above (default depth-invariant; schedule is an A3 remedy).

---

**GIFT-Eval G3 reconciliation (test-as-training overfit — PARTIAL, blocked on eval query overflow; pinned, do not re-flag):**
G3 is the in-distribution overfit capacity probe on real GIFT-Eval `test` data. The framing is
**honest and deliberate**: we train on the test split's *context windows* and score the held-out
horizons of those *same* series — this is **NOT zero-shot** (the decision log's target), it is a
"can the architecture learn real data at all" probe. The held-out horizon is never shown to the
model (Item-only into the reservoir; same no-leakage posture as `SanityTrainLoader`).

- **Built and verified (Mac offline + WSL online):**
  - **`data.terms: List[str]`** (config.py; default `[short, medium, long]`, validated subset of
    short|medium|long) — replaces the single-scalar term. The leaderboard scores each config across
    **all applicable terms** (`config_id = "{name}/{term}"`); **every configured term is attempted
    on every config and a (config, term) that yields zero windows / a gluonts split error is
    skipped with a warning, never fabricated** (maintainer's choice: no hardcoded med/long
    applicability list to drift from the benchmark). `iter_eval_items` now takes `terms=(...)`,
    loops `name × term` via `_iter_one_config_term` (materializes the split inside a guard so a
    non-applicable term is skipped, not fatal); `eval_loader.from_download` threads `cfg.data.terms`.
  - **`GiftEvalTestOverfitLoader`** (`data/gifteval_overfit_loader.py`) + factory key
    **`gifteval_test_overfit`** in `build_loader` — materializes the test-window contexts once via
    `iter_eval_items` → `to_train_item` (Item-only), cycles (overfit), rank-shards round-robin (O6),
    raises if a rank's shard is empty. Per-config window cap = `cfg.eval.items_per_config` (the
    universal G2 cap), so training overfits on exactly the windows the leaderboard scores.
  - **`configs/gifteval_test_overfit.yaml`** — d=224/6L/4h, out_patch=16, L_pack=512 → **10,555,980
    params** (confirmed on the RTX 3070; scaled from the d=64/3L=2.06M sanity model). CUDA Flex +
    `torch.compile`; eval/plot eager.
  - **`train/overfit_run.py`** entrypoint — mirrors `sanity_run` but eval = **`evaluate_leaderboard`**
    (geo-mean MASE across configs, per-config breakdown logged), saves a final `model.pt`
    (entrypoint-level; D13 per-rank reservoir checkpointing via `distributed.save_checkpoint` is
    G5 — single-rank here), writes the sample plot, and **uploads the plot to wandb**.
  - **`tracking.log_image`** seam (Tracker protocol + NullTracker no-op + WandbTracker
    `wandb.Image`) — so scalars (leaderboard_mase / snaive_mase / skill / n_configs…) **and** the
    sample-forecast plot land in wandb. `eval_scalars` already flattens the leaderboard dict.
  - **Tests:** `tests/test_overfit_loader.py` (6, all offline: Item-only contract, cycle, no-cycle
    drain, rank-shard disjoint+complete, empty-shard raises, factory-key dispatch via monkeypatched
    `iter_eval_items`). Also fixed a **pre-existing fragility** in
    `test_leaderboard.py::test_real_iterators_need_storage_root`: it `chdir`'d to defeat `.env`, but
    python-dotenv discovers `.env` from the **source-file location, not cwd**, so on a host whose
    `.env` defines `GIFT_EVAL` (the WSL box) the guard didn't fire — now we neutralize
    `dotenv.load_dotenv` in the test. **Mac suite: 121 passed, 1 skipped.**
  - **WSL:** full `test` set downloaded (`download_gifteval()` no-args → 175 files, **1.5 GB**, 56
    base configs → 168 (config,term) pairs); **`wandb` installed** (`uv pip install wandb`; it is a
    **lazy dep, NOT in pyproject** — the box's venv was missing it, which silently degraded tracking
    to disabled — install it on any fresh WSL venv); **online tracking verified** (project
    `chmanoj/tetris`, run `qropuf1u`), model confirmed at 10.56M on the 8 GiB RTX 3070.

- **BLOCKER discovered (the reason G3 isn't closed) — eval query-token overflow:** the first real
  run crashed at **baseline eval** with `ValueError: buffer 0 overflow: placing segment of size 651
  at offset 0 exceeds l_pack=512` (`eval_loader.eval_batch` → `pack`). Root cause: `eval_spec` sizes
  the query budget as `Q_total = n_targets·ceil(p/p_out) (+ KFF)`; for **medium/long terms** the
  horizon `p` is ×10/×15, so `Q_total` alone can exceed `L_pack` and the per-channel context budget
  `n` clamps to 1, giving segment size `C + Q_total > L_pack`. It is **not** a channel-count problem
  (max C in the data is 21). Measured at L_pack=512: only **7/168** (config,term) pairs need > 512,
  **0** need > 1024; worst is `jena_weather/long` (966; C=21, p=720, q_tok=45). `eval_spec`'s
  docstring already flagged the latent assumption ("S ≤ L … guaranteed while Q_total < L (modest
  benchmark p)").
- **Maintainer's decision (rejecting "bump L_pack" / "skip configs"):** add a **max-query-token cap
  alongside L_pack** — for any item/horizon, predict only as many horizon patches as fit in the cap
  and **ignore the rest** (training should never one-shot a 720-step horizon; real inference does an
  iterative rollout for very large horizons). **Interim:** `configs/gifteval_test_overfit.yaml` is
  pinned to **`terms: [short]`** (short horizons don't overflow) so nothing regresses; the proper
  cap fix + the full all-terms run are deferred to **session G3.1** (`prompts/gifteval_G3.1.md` —
  full design, the Mac repro recipe, and the validation path are there).
- **Note (benign):** the eager B=1 eval logs a FlexAttention "called without torch.compile()"
  warning on CUDA — unfused but **correct**; only the training forward is compiled/fused (D14: eval
  stays eager). Not a bug.
- **Not done in G3 (carried to G3.1, now RESOLVED — see the G3.1 block below):** the max-query-token
  cap, the overfit *run*, and the final reconciliation + handoff.

---

**GIFT-Eval G3.1 reconciliation (query-token cap + iterative rollout + NaN-robustness — pinned, do not re-flag):**
G3.1 closed the G3 blocker and ran the real overfit. Three threads: (A) the max-query-token cap +
iterative rollout, (B) overfit-loader correctness, (C) eval/metric NaN-robustness. Verified locally
on Mac (downloaded the real 1.5 GB `test` set + 5k-step CPU overfit). The WSL 20k GPU run is the only
remaining G3.1 deliverable.

- **(A) `packing.max_query_tokens` — a first-class budget (no off-switch).** The maintainer's call:
  it is "as essential as L_pack", set explicitly in **every** config (base=512, sanity/shakedown=128,
  gifteval=128), validated `>= 1`. `SamplerParams.max_query_tokens` carries it (default `1<<30` =
  "bounded only by L_pack" for direct-construction unit tests). The per-pass query budget is
  `q_budget = min(max_query_tokens, L_pack − C)` (so ≥1 context token always fits — guarantees
  `S ≤ L_pack`, killing the overflow). `eval_loader._capped_p` → `q_pred = q_budget // n_horizon`,
  `p_pred = min(p, q_pred·p_out)`; `eval_batch` builds the segment for `p_pred` only; `eval_spec`
  trusts the pre-capped `p`. **Training** (`window_sampler.sample_window`) bounds the sampled horizon
  by the same cap (`max_q_by_cap`). The synthetic/sanity suite is unchanged (cap never binds the small
  sanity horizons; rollout collapses to one pass).
- **Iterative rollout (the maintainer chose to BUILD it now, overriding the prompt's deferral).**
  `eval_loader.rollout_forecast` covers the **full** benchmark horizon in `ceil(p/p_pred)` capped,
  autoregressive passes — each pass predicts `p_pred` steps, feeds its own raw forecast back as the
  next pass's context (KFF feature rows revealed if known, else NaN), and `y_true` never enters the
  context (held-out scoring preserved). `_score_item` (→ `evaluate_mase`/`evaluate_leaderboard`) and
  the sample plot use rollout; `evaluate_test_loss` stays single-pass (a record-only diagnostic). With
  `p_pred ≥ p` (small horizons) rollout is exactly one pass = the old single-shot eval (sanity numbers
  unchanged). **So there is no partial-horizon caveat** — the leaderboard scores the whole horizon.
- **(B) overfit loader trains on the merged ctx+horizon series.** The maintainer caught that the G3
  loader fed the sampler **context-only** Items, so the model never trained on the `y_true` it is
  scored on — you can't overfit data you never see. `gifteval_overfit_loader.merge_context_horizon`
  now builds **one continuous series** `cat(context, y_true)` `[C, t_ctx+p]` (target rows = context ++
  `y_true`; feature rows = context ++ `feature_future` else NaN) and the training sampler crops its
  own ctx/horizon windows from it — exactly like `SanityTrainLoader`. This **deliberately reverses the
  old "Item-only / no-leakage" framing**: G3.1's overfit is an honest *memorization-capacity* probe
  (still NOT zero-shot), and the eval loader is untouched (keeps the held-out split). `min_context`
  dropped from `out_patch+1` → **2** (the sampler's only hard floor): short series (`t_raw < out_patch`)
  are kept and trained with an **incomplete output patch** (e.g. 4 steps → 2 ctx + 2 pred, unused tail
  of the 16-wide patch masked), so the model learns short/1-step horizons. On the real data 0/657
  series are dropped (merging makes even the shortest viable; min merged length 33).
- **(C) the model must never emit non-finite output — and now doesn't.** Diagnosed at random init:
  the model output (arcsinh space) and denorm stats (`a`, `σ`) are **always finite**; the non-finite
  came entirely from the **unbounded inverse transform** `raw = sinh(pred)·σ + a` (`sinh` overflows
  float32 once `|pred| ≳ 89`; the untrained head emitted arcsinh-space values up to ~750 on high-σ
  series). Fix: clamp the prediction to **`±normalize.ARCSINH_INV_CLAMP = 10`** before `sinh` (applied
  only at the forecast/metric boundary in `horizon_forecast_raw`; the math primitives stay exact).
  `CAP=10` is the cross-precision choice: `sinh(10) ≈ 11013 < fp16 max (65504) < bf16/fp32` (15 would
  overflow fp16), and arcsinh-space ±10 ≈ ±11000·σ so it never clips a legitimate prediction. Rollout
  also `nan_to_num`s the fed-back chunk (defence in depth). **Separate, pre-existing baseline bug**
  also fixed: `metrics.seasonal_naive_forecast` didn't impute missing values, so `*_with_missing`/
  `bitbrains` series (NaN at the seasonal lag, up to 100% missing) NaN'd the baseline and poisoned
  `snaive`/`skill`. Now it replicates gluonts' `SeasonalNaivePredictor`: last-value imputation
  (forward-fill; leading NaN → first finite; all-NaN → 0) then seasonal repeat, `nanmean` fallback for
  sub-season contexts (`metrics._impute_last_value`; no gluonts dep in the core module). The geomean
  non-finite guard is **kept** (catches future regressions). Result at random init: all 153/153 configs
  finite, `snaive`/`skill` finite.
- **Optional deps now declared (`pyproject.toml`).** Real GIFT-Eval access + tracking are
  `[project.optional-dependencies]`: **`gifteval`** (`salesforce_gift_eval` @ git, `huggingface_hub`,
  `datasets`, `gluonts`, `python-dotenv`) and **`tracking`** (`wandb`) — `uv sync --extra gifteval
  --extra tracking`. `[tool.uv] override-dependencies` loosens `salesforce_gift_eval`'s over-tight
  `gluonts<0.16`/`matplotlib<3.10` pins (it runs fine on the project's versions); `allow-direct-references`
  enables the git URL; `uv.lock` updated. **Test-count note:** with the `gifteval` extra installed,
  `test_leaderboard.py::test_download_requires_hf_when_absent` (the *ImportError-path* test) correctly
  **skips** because `huggingface_hub` is now present → suite reads **130 passed, 2 skipped** on a
  gifteval-equipped box (vs 131 passed, 1 skipped without the extra). Not a regression.
- **Local Mac validation (CPU, 10.56M, items_per_config=5, all terms).** Downloaded the real `test`
  set (175 files, 1.5 GB). Baseline eval no longer crashes (290+ items, C≤21, p≤900, skipped=0). 5k
  overfit: leaderboard MASE **226 → 9.15** (≈25× in 5k steps), skill **118 → 4.78**, all 153 configs
  finite throughout, **37/153 already beat seasonal-naive** (e.g. `bizitobs_l2c/5T/short` 0.48). Does
  not beat naive in aggregate yet (low-freq weekly/monthly configs still poor) — expected for a 5k CPU
  probe; the full run is 20k / items_per_config=10 on the WSL GPU.
- **Tests:** `tests/test_eval_query_cap.py` (cap math, no-overflow repro red→green, rollout full-horizon
  coverage + single-pass collapse, inversion clamp, rollout sanitize, training cap); `test_overfit_loader`
  updated for the merge + the drop floor; `test_sanity_mase` gains the gluonts-style snaive imputation
  case. **Mac suite (with extras): 130 passed, 2 skipped.**
- **Still open (G3.1 remainder):** the **WSL 20k GPU run** (compiled Flex path, all terms, the experiments-doc
  section + skill log) — bring the box up, `uv sync --extra gifteval --extra tracking`, rsync (exclude `.env`),
  flip `configs/gifteval_test_overfit.yaml` `eval.items_per_config` back to 10, run 20k.

---

## 1. Notation & global shape constants

All per-batch shapes are static (D14: one compile graph; per-sample variability lives in tensor *contents*, never shapes).

| Symbol | Meaning | Shakedown | Target |
|---|---|---|---|
| `B` | buffers (segmented packs) per step | 2 | data/throughput-tuned |
| `L` | `L_pack` tokens per buffer | 256 | 1024 (→2048) |
| `d` | `d_model` | 32 | 256 (base) |
| `H` | attention heads | 2 | tuned |
| `n_layers` | backbone depth | 2 | 8 (base) |
| `V` | telescope tiers / patch vocab size | 6 | 6 |
| `PATCH` | patch vocabulary | `{4,8,16,64,256,512}` | same |
| `P_out` | horizon output patch | 8 | 16 |
| `ENCODER_CAP` | static per-tier encoder dispatch capacity (`model.encoder_cap`, 0→L) | = L | = L |
| `R` | per-buffer raw-store length (`norm_values[B,R]`); raw steps, padded | data | data |

**Variate id (`variate_uid`)** = a buffer-local id unique per `(sample_id, channel)`. It is the binding handle for D4: the same random orthonormal ID and the same normalization stats attach to every token (tier, query, KFF) of one variate via `variate_uid`.

**Side tensors** — the *only* source of token geometry (D8 hard rule: **no buffer-index positional encoding anywhere**). Full `Batch` field list (walkthrough Stage 4); per token `[B, L]` unless noted:

| Name | dtype / shape | Meaning |
|---|---|---|
| `sample_id` | int32 `[B,L]` | packing segment id within the buffer; `-1` = pad |
| `channel_idx` | int32 `[B,L]` | channel within its sample (features first, then targets) |
| `t_center` | float32 `[B,L]` | continuous time vs forecast origin (negative = past, ≥0 = horizon) |
| `tier_id` | int8 `[B,L]` | 0–5; selects per-tier encoder & span embedding |
| `role` | int8 `[B,L]` | `{CTX, QRY}` (D6/D9, drives the mask) |
| `content_state` | int8 `[B,L]` | `{OBSERVED, MASK, NA}` (D7, selects content slot) |
| `role_ft` | int8 `[B,L]` | `{FEATURE, TARGET}` (D11, added role embedding) |
| `raw_start` | int32 `[B,L]` | offset into `norm_values[B,R]` for this token's window (`-1` unless OBSERVED) |
| `variate_uid` | int32 `[B,L]` | per-(sample,channel) id → random orthonormal ID (D4) |
| `valid_aux` | bool `[B,L]` | is the next-patch aux target well-defined here? (boundary/origin/NaN) |
| `norm_values` | float32 `[B,R]` | per-buffer normalized raw store (base scale σΔ[0]); indexed by `raw_start` |
| `observed` | bool `[B,R]` | observed-indicator for `norm_values` (D7) |
| `stats_a`, `stats_sigma` | float32 `[B,L]` | per-token anchor & base scale (broadcast from `Stats`) for Stage-9 inversion |
| `horizon_target` | float32 `[B,L,P_out]` | GT horizon patch per slot; real only at QUERY slots, rest masked (dense ⇒ static) |
| `target_valid` | bool `[B,L,P_out]` | true only at QUERY slots with non-NaN GT |

---

## 2. Repo / module layout

```
tsfm-models/
  pyproject.toml                 # uv; deps: torch>=2.5, numpy, omegaconf, pytest
  uv.lock
  configs/
    shakedown.yaml               # §7 tiny end-to-end config
    base.yaml                    # full-size defaults (D12 design point)
    norm_ablations.yaml          # D10 toggle matrix
  src/tetris/
    __init__.py
    config.py                    # dataclasses for every toggle; YAML<->dataclass
    constants.py                 # PATCH vocab, enums Role/ContentState/RoleFT + ChannelRole, dtypes
    backend.py                   # backend switch: flex+compile (CUDA) | sdpa+eager (Mac)

    normalize.py                 # D10: anchored-arcsinh, fallback chain, per-tier σ_Δ,
                                 #      exact inversion, local-reanchor loss-target builders
    telescope.py                 # D2/D12: allocation, coverage<->tokens, per-tier gather/scatter index build

    tokenize/
      spec.py                    # SegmentSpec (size computable from spec alone, no tokenization)
      window_sampler.py          # D9.2: origin/p sampling, per-tier counts, content-state assignment
      assemble.py                # raw->normalized patch windows + side tensors (pure, per spec)

    packing/
      collator.py                # STATELESS pack(buffers: list[list[AssembledSegment]]) -> Batch
      reservoir.py               # streaming IterableDataset: reservoir + best-fit-decreasing (D9.3)
      scheduler.py               # D9.4 cost-bucketed step grouping (Σ S_i²/2), cross-rank (multi-node)

    masks.py                     # D9 truth table -> BlockMask (flex) / bool [L,L] (sdpa); equality-tested

    model/
      embeddings.py              # time/span/role/content-state embeddings (+ MASK/NA learned)
      variate_id.py              # D4: random orthonormal IDs per variate_uid + scale-receipt MLP(a,σΔ)
      encoders.py                # D3: six per-tier encoders; D14 grouped gather->encode->scatter
      attention.py               # attn(e, block_mask, score_mod=None); backend-routed
      blocks.py                  # transformer block (attn + FFN); pre-norm
      heads.py                   # horizon point head; six per-tier raw-time aux heads (D6)
      tetris.py                  # full forward: embed -> encode -> backbone -> decode

    losses.py                    # D6 MAE horizon + per-tier-weighted aux; D10 loss-space toggles; valid masking
    metrics.py                   # D13: test loss on deterministic shard (record-only); MASE deferred (O4)

    data/
      contract.py               # Item type/Protocol; EvalItem (superset); factory build_loader(cfg)
      synthetic.py              # generators: trend/seasonal/AR/intermittent + Mystic-B shared-factor + lag-probe
      standin_loader.py         # StandInPretrainLoader (synthetic; GiftEvalPretrain is NOT downloadable)
      gifteval_download.py      # fetch real GIFT-Eval (train+test) from the internet; config-table source
      eval_loader.py            # GiftEvalEvalLoader: real test split; context + held-out horizon (record-only)

    train/
      distributed.py            # rank/world init (DDP/FSDP), (node,rank) data sharding, cross-rank cost scheduling
      step.py                   # one compiled train step (forward+loss+backward+opt)
      loop.py                   # training loop; reservoir + scheduler wiring; checkpoints (D13 discipline)
      shakedown.py              # tiny end-to-end runner (entrypoint)

  tests/
    test_normalize.py           # [UNIT GATE] D10 round-trip / inversion / fallback
    test_telescope.py           # [UNIT GATE] allocation + per-tier gather/scatter dispatch
    test_masks.py               # [UNIT GATE] D9 truth table (flex==sdpa)
    test_pack_invariance.py     # [UNIT GATE] packed==unpacked loss invariance (no buffer-index leakage)
    test_aux_boundary.py        # [REQUIRED] raw-time aux validity: origin-crossing & history-edge masking
    test_synthetic.py           # generators honor contract; cross-channel structure present
    test_collator_shapes.py     # Batch shapes/capacities; specs predict sizes exactly
    test_encoders.py            # grouped per-tier dispatch correctness (part of telescope gate)
    test_model_smoke.py         # forward finite; one static graph (no recompiles)
    test_distributed.py         # single-process == 2-rank per-sample loss; disjoint shards; ckpt restore
    test_eval_loss.py           # held-out horizon path; record-only test loss on toy shard
```

### 2.1 Key tensor shapes flowing between modules

```
base loader item:           (data_tensor [n_i, t_i] float32, num_features int, num_targets int)
                            n_i = num_features + num_targets (features first); raw, may contain NaN

window_sampler(item,rng) -> SegmentSpec:
                            { C, n_features, n_targets, origin, p, channel_roles[C],
                              Q, K, Q_total, n, counts[6] }   # channel_roles ∈ {FEATURE,TARGET,KFF}
                            S = C·Σcounts + Q_total, exact from the spec alone (D9: packable w/o tokenizing)

assemble(item, spec, P_out) -> AssembledSegment (pure; walkthrough Stages 2–3):
                            per-token (len S): tier_id, channel, raw_start, role, content_state,
                              role_ft, variate_uid, t_center, valid_aux
                            flat store: norm_values[R_seg], observed[R_seg]   (base scale σΔ[0])
                            per-channel: stats_a[C], stats_sigma_delta[C,6]
                            dense GT: horizon_target[S,P_out], target_valid[S,P_out]

reservoir IterableDataset -> pack-group (~B buffers' worth): specs chosen + assembled,
                            then grouped into buffers (best-fit-decreasing, S11)
                            -> buffers: list[list[AssembledSegment]]

pack(buffers, *, l_pack, p_out, num_buffers) -> Batch (all static at L; see §1 side-tensor table):
  side tensors            : sample_id, channel_idx, t_center, tier_id, role, content_state,
                            role_ft, raw_start, variate_uid, valid_aux       each [B, L]
  raw store               : norm_values [B, R], observed [B, R]
  inversion stats         : stats_a [B, L], stats_sigma [B, L]
  dense horizon GT        : horizon_target [B, L, P_out], target_valid [B, L, P_out]

model.forward(Batch) :
  content (Stage 7)       : route content_state==OBSERVED by tier — encoderₖ [CAP, P_k, 2]→[CAP, d],
                            scatter → [B, L, d]; MASK/NA slots get learned [MASK]/[NA] vectors
  embeddings (Stage 6)    : e = content + time(t_center) + span(tier_id) + variate(variate_uid) + role_ft
  backbone (n_layers ×)   : attn(e, block_mask) + FFN -> e [B, L, d]
  horizon head (Stage 9)  : dense [B, L, d] -> [B, L, P_out]   (gather/mask at loss time)
  aux heads (6, Stage 9)  : dense [B, L, d] -> [B, L, P_k] ×6; each token tier-selects its own

losses.compute           : scalar  = horizon_MAE(masked by target_valid)
                                      + λ·Σ_k aux_weights[k]·aux_MAE_k(masked by valid_aux)
metrics                  : per-config test loss on deterministic first-100 shard (record-only; MASE deferred)
```

---

## 3. Module responsibilities (detail)

### `config.py`, `constants.py`, `backend.py`
- **config:** nested dataclasses; load/merge YAML via OmegaConf. Surfaces all D10 axes (`input_norm ∈ {anchored_arcsinh, zscore_arcsinh}`, `loss_target ∈ {locally_reanchored, global_norm_space}`, `loss_space ∈ {arcsinh, vol_units}`), D13 `dataset_weights {id: multiplier}` + `phase2_*`, D15 `loss_weighting`, the per-tier `aux_weights [6]` vector (replaces a single λ), and the `distributed` block (§9). **Every combination must run** (D10 mandate).
- **constants:** `PATCH=(4,8,16,64,256,512)`, enums `Role{CTX,QRY}` (D6/D9), `ContentState{OBSERVED,MASK,NA}` (D7), `RoleFT{FEATURE,TARGET}` (D11), `ChannelRole{FEATURE,TARGET,KFF}` (spec-level).
- **backend:** `attend(q,k,v, block_mask, score_mod)` dispatch. CUDA → `flex_attention` + compiled; else → `F.scaled_dot_product_attention` with materialized `[L,L]` bool mask. `make_mask(side_tensors)` returns the matching object for each backend. Single seam where compile/Flex is opt-in.

### `normalize.py` (D10)
- `compute_stats(x_ctx)` → `Stats{a: float, sigma_delta: float[6]}` (per-(sample,channel) constants). `a = median(last 32)` (window 8–16 tunable). `sigma_delta[0] = 1.4826·median|Δx|` (the base scale, fallback chain `mean|Δx| → IQR/1.349 → 1`, floor `≥1e-3·IQR/1.349`) used for the input `norm_values` and Stage-9 inversion; `sigma_delta[1..5]` are per-tier scales (robust σ of each tier's aggregated sequence) for aux targets + D4 receipts. `.sigma` property = `sigma_delta[0]`.
- `forward(x, a, σ)` → `arcsinh((x-a)/σ)`; `invert(z, a, σ)` → `sinh(z)·σ + a` (exact, denormalizes horizon preds; no leakage).
- Loss-target builders: `aux_target(t) = arcsinh((x[t+1..t+P] - level_t)/σ_tier)`; `horizon_target(y) = arcsinh((y - a)/σ)`. `loss_target=global_norm_space` swaps in global-z targets (one-line, per D10 hedge).
- Receipts (D4/D10): `(a, sigma_delta)` ride along on the segment (`stats_a`, `stats_sigma_delta`) and feed the variate scale-receipt MLP.

### `telescope.py` (D2/D12)
- `allocate(n_tokens, T_raw)` → per-tier counts via a **tunable ratio prior (front-loaded, coarse-reaching)** — config `model.tier_alloc_per_channel`, normalized and scaled by `n` (default reproduces the design-point `[8,8,8,8,8,4]` at n=44; literal-halving `[0.5,0.25,…]` is an available ablation) — then rebalance to `coverage ≈ T_raw` (walk coarse→fine); deterministic integer ops.
- `coverage(counts)` / `tokens_for_coverage(T)` — invert the ladder (used by D9.2: `n = ⌊(L−Q)/C⌋` → `T_cov`).
- `build_dispatch(counts, origin)` → per-tier raw-window slices + **channel-major** scatter positions `pos = base + c·n + t`. Optional per-channel `caps` is a convenience guard; real capacity is the model's `ENCODER_CAP` (Stage 7).

### `tokenize/` (walkthrough Stages 1–3)
- `spec.py`: `SegmentSpec` with per-channel `channel_roles[C] ∈ {FEATURE,TARGET,KFF}`; `S` computable from the spec alone; **no tokenization at packing time**.
- `window_sampler.py`: sample `q_tok`/variable-`p`, `Q/K/Q_total`, per-channel `n = ⌊(L−Q_total)/C⌋`, **uniform random origin**, counts via the ratio prior trimmed to history; build `channel_roles` (features-first; KFF-reveal among native features, D11 partial; demotion deferred). Stateless given an RNG; lives in the reservoir layer.
- `assemble.py`: given `(item, spec, P_out)`, normalize per channel (D10, base scale σΔ[0]), build the **flat per-segment `norm_values`/`observed` store** + per-token `raw_start` (the gather store), emit per-token `role`/`content_state`/`role_ft` (fully-missing patch → `content_state=NA`; **KFF = observed CTX feature token at t>0**, D11), `stats_a`/`stats_sigma_delta`, and **dense `horizon_target[S,P_out]`/`target_valid`** (real only at query slots). Also sets `valid_aux` (raw-time origin-crossing/history-edge/observed check). Pure — the heart of the collator.

### `packing/`
- `collator.py`: **stateless** `pack(buffers: list[list[AssembledSegment]], *, l_pack, p_out, num_buffers=None) -> Batch` (16 fields, §1). Pure tensor materialization of a **caller-provided** grouping (no best-fit packing here — that is the reservoir's job, S11): lays each segment channel-major into its buffer, pads the tail (`sample_id=-1`); concatenates per-segment `norm_values` into `[B,R]` and offsets each token's `raw_start` (and `variate_uid` → buffer-unique); broadcasts per-token `stats_a`/`stats_sigma`; scatters dense horizon GT into `[B,L,P_out]`. *Signature frozen* — both the trivial path (S6/S10) and the reservoir path (S11) call it unchanged. See the post-S6 reconciliation block up top.
- `reservoir.py`: `IterableDataset` wrapping a base loader: pulls items, runs `window_sampler`, keeps a reservoir of `K≈1000` specs (doubles as shuffle buffer), **best-fit-decreasing**, refills, yields pack-groups. *(Built in Stage 11, after the trivial path.)*
- `scheduler.py`: D9.4 cost = `Σ S_i²/2`; window of 64–256 buffers sorted by cost → each global step formed from similar-cost buffers (giants travel together). Shapes unchanged.

### `masks.py` (D9.1)
Truth table: `allow(q,k) = s[q]==s[k] AND ( (role[k]==ctx AND (role[q]==qry OR t[k]<=t[q])) OR (role[k]==qry AND role[q]==qry) )`. Pad (`s=-1`) gets a `q==k` self-attend exception (NaN guard; outputs discarded). Built once per batch, depth-invariant (v1). Two constructors: Flex `BlockMask` (`mask_mod`) and SDPA bool `[L,L]`; **equality-tested**. `score_mod` socket present, default `None` (A3 ladder hook).

### `model/`
- `variate_id.py` (D4): random orthonormal ID per `variate_uid` (resampled per sample; fixed seed at inference) `+` scale-receipt MLP over `(stats_a, stats_sigma_delta)`.
- `encoders.py` (D3/Stage 7): six encoders (Fourier number-features → MLP/conv) of native widths `P_k`. **Index-routed dispatch at static `ENCODER_CAP`**: for each tier, route slots with `content_state==OBSERVED` & `tier_id==k` (context + KFF; ≤CAP, sentinel-padded) → gather `[CAP, P_k, 2]` windows from `norm_values` via `raw_start` → `encoderₖ → [CAP, d]` → scatter into `content [B, L, d]`. `MASK`/`NA` slots get learned `[MASK]`/`[NA]` vectors. Pad token-count only, keep native `P_k`.
- `embeddings.py` (Stage 6): `e = content + time_emb(t_center) + span_emb(tier_id) + variate_emb(variate_uid) + role_ft_emb(role_ft)` (5 parts). `content` = encoder output | learned `[MASK]` | learned `[NA]`, selected by `content_state` (D7); CTX/QRY is structural (mask + content), not an added vector.
- `attention.py`: `attn(e, block_mask, score_mod=None)` → QKV proj `[B,H,L,d_h]` → backend `attend`. **`block_mask` is a parameter** (per-layer schedule = future arg, not refactor).
- `blocks.py` / `tetris.py`: pre-norm blocks; full forward assembles §2.1.
- `heads.py` (Stage 9, dense — no gather): horizon head runs over all `L`, `[B,L,d]→[B,L,P_out]`. **Six per-tier aux heads** each run dense `[B,L,d]→[B,L,P_k]`; each token tier-selects its own output. A tier-`k` aux target is the next `P_k` raw steps `[t+P_k, t+2P_k)` at tier-`k` granularity (collator-built). Selection/masking happen in the loss reduction, so shapes stay static.

### `losses.py` (D6/D10)
Horizon MAE (median-optimal) run dense `[B,L,P_out]` and masked by `target_valid` (NaN GT dropped, D7): `(|pred−tgt|·valid).sum()/valid.sum()`. `+` **per-tier-weighted** aux MAE (`aux_weights[6]` vector replaces a single λ): six dense tier-heads, each token picks its own tier's output, masked by `valid_aux`. **`valid_aux` is a precise raw-time check** (replacing D9.5's `span[p+1]==span[p]` proxy): the tier-`k` aux term is masked when its region `[t+P_k, t+2P_k)` **(a) crosses the origin** (horizon's job) or **(b) runs off history** (partial/unavailable), plus the D7 mostly-missing skip. `loss_space ∈ {arcsinh, vol_units}` toggle. Per-sample normalized influence (σ_Δ ≈ MASE-aligned).

> **Overlapping cross-resolution supervision (accepted, OA):** a coarse token's target region can overlap raw steps also supervised by finer, more-recent tokens. This is legitimate (predictions from different origins/resolutions), not double-counting a bug; the per-tier `aux_weights` vector damps it. Verified by `test_pack_invariance` (overlap must not depend on layout) and `test_aux_boundary`.

### `metrics.py` (D13)
**v1: test loss only**, on the **deterministic first-100-windows-per-config** shard, **record-only** (train-split validation windows are the decision signal). `seasonal_naive_denom` / true-formula **MASE is deferred** (O4): function stubs land now; seasonal-period detection + MASE tracking are wired in a later iteration.

### `data/`
- `contract.py`: `Item = Tuple[Tensor, int, int]`; `Protocol` for loaders; `EvalItem` superset (§6); `build_loader(cfg)` factory keyed by name (`standin_pretrain` | `gifteval_test` | real Pretrain name later).
- `standin_loader.py`: synthetic Pretrain stand-in (the **GiftEvalPretrain corpus is not downloadable**). `gifteval_download.py`: fetches the **real GIFT-Eval** (train+test) from the internet (O1) and is the source of the config table. `synthetic.py`, `eval_loader.py`: §5, §6.

---

## 4. Build order (each stage has a smoke test; **[GATE]** = unit test required *before any training*)

The four mandatory pre-training gates are **S1, S2, S5, S9**.

| Stage | Build | Smoke / **[GATE]** test |
|---|---|---|
| **S0** | `config`, `constants`, `backend` shell | Instantiate config from `shakedown.yaml`; `attend()` runs on random tensors on both backends (Flex on CUDA, SDPA on Mac). |
| **S1** | `normalize.py` | **[GATE] `test_normalize`**: `invert(forward(x))≈x` (tol) across 8 synthetic series incl. extremes; fallback chain triggers (constant→zeros, intermittent→mean\|Δx\|); per-tier σ; loss-target/inverse consistency. |
| **S2** | `telescope.py`, `encoders` dispatch index build | **[GATE] `test_telescope`**: allocation ≤ budget & coverage≈T_raw; channel-major positions collision-free; `scatter(gather(x))` round-trips; `cap_k` asserts. (`test_encoders` covers grouped encode dispatch.) |
| **S3** | `synthetic.py`, `standin_loader.py`, `contract.py` | `test_synthetic`: items honor exact `(float32 [n,t], int, int)` contract, features-first, NaNs present, `n`/`t` vary; shared-factor channels show measurable cross-correlation; lag-probe planted. |
| **S4** | `tokenize/` (spec, window_sampler, assemble) | Spec-predicted `S` equals assembled length exactly; origin/`p`/counts within bounds; 3-enum tagging correct (role CTX/QRY, content_state OBSERVED/MASK/NA, role_ft FEATURE/TARGET; KFF = observed CTX feature @ t>0); flat `norm_values`+`raw_start` gather in-bounds; dense `horizon_target`; capacity under `ENCODER_CAP`. |
| **S5** | `masks.py` | **[GATE] `test_masks`**: enumerate small grid; assert every `(q,k)` matches the boolean formula (ctx→ctx causal+channel-blind; qry→ctx always; qry→qry both ways; ctx→qry never; pad self-only). Flex `BlockMask` == SDPA bool. |
| **S6** | `packing/collator.py` (stateless `pack`; caller-provided grouping, no reservoir) | `test_collator_shapes`: materializes a pre-grouped `list[list[AssembledSegment]]` into `[B,L]` buffers; all 16 side tensors present; `raw_start`/`variate_uid` rebased buffer-global; tail pad `sample_id=-1`. |
| **S7** | `model/` (embeddings, variate_id, encoders, attention, blocks, heads, tetris) | `test_model_smoke`: forward on one packed buffer → finite dense `[B,L,P_out]` horizon + `[B,L,P_k]×6` aux; both backends; encoders at static `ENCODER_CAP`. |
| **S8** | `losses.py`, `metrics.py` | Loss finite & differentiable; NaN-GT and missing-patch positions contribute zero; **test loss** on a toy shard (MASE deferred, O4). **`test_aux_boundary`** (required): aux term zeroed exactly when the raw-time target region crosses the origin or runs off history; per-tier `aux_weights` applied. |
| **S9** | (integration of S6–S8) | **[GATE] `test_pack_invariance`**: identical samples packed together vs in solo buffers → **identical per-sample loss and per-token outputs** (permutation + packing invariance). Keystone guard against the D8 buffer-index-leakage bug; also confirms cross-resolution aux overlap (OA) is layout-independent. |
| **S10** | `train/step.py`, `train/shakedown.py` | Run `shakedown.yaml`: 50–200 steps on synthetic; loss decreases; **no recompiles** (assert via `torch._dynamo` recompile counter); CUDA path under `torch.compile`. |
| **S11** | `packing/reservoir.py`, `packing/scheduler.py` | Reservoir packs above the **unchanged** collator; tail waste low single-digit %; cost-bucketed steps group similar-cost buffers; train loop runs. |
| **S12** | `data/gifteval_download.py`, `data/eval_loader.py`, eval tracking in loop | `test_eval_loss`: **real GIFT-Eval** test split downloaded (O1); eval item carries held-out horizon; first-100 shard **test loss** computed (record-only); **training-loader contract unbroken** (shared collator consumes context-only). |
| **S13** | `train/distributed.py`; DDP/FSDP + (node,rank) sharding + cross-rank scheduler | **`test_distributed`**: single-process and 2-rank (gloo) give equal per-sample loss on a fixed seed; disjoint data shards per rank; cross-rank cost-bucketing forms balanced global steps; checkpoint saves/restores per-rank dataloader+reservoir state and re-shards at a different world size. |

**Why these four gates first:** they pin the invariants the whole design leans on — reversible normalization (D10), correct telescope dispatch (D2/D12/D14), the exact attention reachability (D9.1), and the no-leakage packing guarantee (D8). A regression in any of them produces silently-wrong training, not a crash. (`test_aux_boundary` is a required fifth check, gated before training once aux loss exists.)

> **Distributed is first-class (O6), not a bolt-on.** Although it is *wired and tested* at S13, the seams it needs are honored from S6 onward: the collator/model read no cross-rank or global state, packing/normalization/masks are per-buffer and rank-local, and `step.py`/`loop.py` are written DDP-aware. See §9.

---

## 5. Synthetic generator & stand-in loader spec

### 5.1 `synthetic.py` — generators (all raw, unnormalized, reproducible by seed)

**Univariate primitives** (compose with additive noise; expose knobs):
- **Trend** — linear / exponential growth (exercises D10 anchored-arcsinh log region).
- **Seasonal** — sum of sines, random periods/phases/amplitudes (within-channel seasonality as learned autocorrelation, D5).
- **AR(p)** — random stable coefficients.
- **Intermittent** — Croston-style sparse nonzero, random rate (exercises σ_Δ fallback `mean|Δx|`, D10).
- **Random walk / GBM** — non-stationary controls.
- **Level shifts / regime jumps** — the D10 stress case (recent-window resolution).
- **NaN injection** — random missing fraction up to a cap; fully-missing patches (D7 [NA]).

**Multivariate shared-factor (Mystic-B echo, D13-C)** — the primary cross-channel signal:
- Bank of `K_bank` latent factor processes (default 7) drawn from the primitives.
- Each channel = linear combo of `3–4` sampled factors + idiosyncratic noise + **per-channel scale/offset** (exercises D10 cross-channel scale receipts and D4 routing).
- Optional **lead-lag**: a channel reads a *lagged* factor — plants genuine lagged cross-channel dependence.
- Knobs: `bank_size`, `kernels_per_variate`, `overlap_degree`, `C_distribution`, `lag_distribution`.

**Lag-probe generator** (A3 diagnostic, D5): white-noise feature channel; target = feature lagged-`k` + noise. The channel-independent baseline *must* be beatable — used to detect cross-channel routing failure and trigger the A3 ladder.

### 5.2 Stand-in loaders (exact contract — zero collator change on swap)

- `StandInPretrainLoader` — `__iter__` (mirrors **GiftEvalPretrain**, which is *not* downloadable, so it stays synthetic). **GIFT-Eval itself is downloaded** (O1) via `gifteval_download.py` and wrapped by a thin map-style loader, so **only the Pretrain corpus is synthetic**. The stand-in yields exactly `(data_tensor: torch.float32 [n_i, t_i], num_features: int, num_targets: int)`, features-first, `n_i`/`t_i` varying, raw values, NaNs allowed.
- Config: `n_series`, `C` distribution, length distribution, frequency mix, NaN cap, synthetic-mixture weights (mirrors D13 `dataset_weights`).
- Registered in `build_loader(cfg)`; the real Pretrain loader drops in behind the same factory key. **Nothing downstream of the loader changes** — window sampler, reservoir, collator, model all consume only the item tuple.

---

## 6. Eval loader (GIFT-Eval test split, record-only — D13)

Goal: carry held-out ground truth for scoring **without** breaking the training-loader 3-tuple contract.

```python
class EvalItem(NamedTuple):
    data_tensor: torch.Tensor   # context only, [n_i, t_ctx], raw, may contain NaN   <-- fields 1..3
    num_features: int           #   identical to the training contract
    num_targets: int            #
    y_true: torch.Tensor        # held-out horizon, [p, num_targets]   (never enters context)
    naive_denom: Optional[torch.Tensor]  # seasonal-naive denominator; deferred (O4) — None in v1
    config_id: str              # which of the 97 GIFT-Eval configs
```

- **Contract preservation:** the first three fields *are* the training `Item`. A trivial adapter `to_train_item(e) = (e.data_tensor, e.num_features, e.num_targets)` (or duck-typed access) feeds the **shared, unchanged collator**. Held-out fields ride alongside and are stripped before packing → no leakage, no training-path fork.
- **Seasonal-naive denominators / MASE deferred** (O4): not computed in v1. The `naive_denom` field exists so wiring MASE later is purely additive.
- **Metrics (v1):** **test loss** on the **deterministic first-100-windows-per-config** shard. **Record-only**; decisions come from train-split validation windows. True-formula MASE deferred (O4).
- **97-config table + test data:** **downloaded from the real GIFT-Eval** (O1; GIFT-Eval — not the Pretrain corpus — is fetchable). The config table (horizons incl. ×10/×15, context lengths, freqs, `C`) comes from that download and drives D13 phase-2 `auto_from_test_configs`.

---

## 7. Minimal shakedown config (`configs/shakedown.yaml`)

Tiny model, tiny `L_pack`, full pipeline, full telescope vocabulary (counts tiny), exercising the **compiled train step** end-to-end.

```yaml
run: { name: shakedown, steps: 100, seed: 0 }
backend:
  device: auto            # cuda -> flex+compile ; mps/cpu -> sdpa+eager
  compile: true           # honored on cuda; no-op on mps/cpu
distributed:
  enabled: false          # smokes run single-process; S13 flips this on
  parallel: ddp           # ddp (default) | fsdp (scale-up switch)
  process_group: gloo      # gloo on cpu/mac; nccl on cuda
  shard_by: [node, rank]  # deterministic disjoint series shards
model:
  d_model: 32             # smokes much smaller than base (256); see §1
  n_layers: 2             # base = 8
  n_heads: 2
  patch_vocab: [4, 8, 16, 64, 256, 512]
  out_patch: 8
  tier_alloc_per_channel: [16, 12, 8, 6, 4, 2]   # ratio prior (scaled by n), NOT literal counts
  encoder_cap: 0                    # 0 -> L_pack; static ENCODER_CAP
data:
  loader: standin_pretrain
  C_distribution: [1, 8]            # univariate..small multivariate
  length_distribution: [64, 4096]
  nan_cap: 0.3
  synthetic_mix: { shared_factor: 0.5, univariate: 0.4, lag_probe: 0.1 }
packing:
  L_pack: 256
  buffers_per_step: 2               # B
  reservoir: false                  # Stage 10 trivial path; flip true at Stage 11
norm:                               # D10 axes — all must run
  input_norm: anchored_arcsinh
  loss_target: locally_reanchored
  loss_space: arcsinh
loss:
  aux_weights: [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]   # per-tier (D6 + raw-time note); coarse tiers down-weighted
eval:
  enabled: true
  loader: gifteval_test             # real GIFT-Eval download (O1)
  shard_windows: 16                 # tiny stand-in for "first 100 per config"; record-only test loss
checks:
  assert_no_recompile: true         # fail if torch._dynamo recompiles
  assert_pack_invariance: true      # run S9 invariant on a fixed mini-batch at startup
  assert_aux_boundary: true         # run test_aux_boundary on the live batch
```

Shakedown acceptance: (1) runs on Mac (SDPA/eager) and CUDA (Flex/compile); (2) zero recompiles; (3) loss trends down on synthetic; (4) `test_pack_invariance` and `test_aux_boundary` hold on the live batch; (5) test loss computes on the tiny eval shard.

---

## 8. Resolved this round + remaining open items

**Resolved** (folded into the relevant modules above):

- **O1 — eval data.** Download the **real GIFT-Eval** test set from the internet (the **Pretrain corpus is not downloadable** → stays synthetic). The 97-config table comes from that download. (`data/gifteval_download.py`.)
- **O2 — loader surface.** Confirmed: Pretrain `__iter__`, GiftEval map-style, rows **features-first contiguous**.
- **O3 — aux heads.** **Six per-tier aux heads**; targets defined in **raw time** at tier-`k` granularity (§heads / §losses; raw-time validity below).
- **O4 — MASE.** **Deferred.** v1 tracks **test loss only**; seasonal-naive period detection + true-formula MASE land in a later iteration (field stubs in place).
- **O5 — sizes.** `base.yaml`: `d_model=256`, `n_layers=8`. Smokes much smaller (`d_model≈32`, `n_layers=2`).
- **O6 — distributed.** **First-class, not deferred** — must work multi-rank **and** multi-node/distributed. See §9.

**New open item (raised by the aux-target redefinition):**

- **OA — overlapping cross-resolution aux supervision.** A coarse token's raw-time target region `[t+P_k, t+2P_k)` can overlap raw steps also supervised by finer, more-recent tokens (predictions from different origins/resolutions). **Accepted**, and damped by the per-tier `aux_weights` vector. **To verify before scaling:** confirm via `test_pack_invariance` (overlap must not depend on packing layout) and `test_aux_boundary`; revisit `aux_weights` defaults if coarse-tier aux destabilizes training.

---

## 9. Distributed training (multi-rank / multi-node) — O6

Designed in from the start; the seams below are honored from Stage 6 onward and wired + tested at Stage 13.

- **Data sharding.** `build_loader` shards source series **disjointly by `(node, rank)`**, deterministically from `world_size` / `rank` / global seed — no series double-counted, shards reproducible. The reservoir (`packing/reservoir.py`) is **per-rank**; shuffle seeds are rank-offset.
- **Cost-bucketed scheduling across ranks (D9.4).** Each global step is formed from **similar-cost buffers across all ranks** so giants travel together and no rank straggles. Mechanism: ranks exchange per-buffer costs (`Σ S_i²/2`) via a small all-gather, *or* follow a deterministic global cost-sorted schedule seeded identically on every rank; the scheduler assigns one similar-cost buffer per rank per step. Shapes stay static (one compile graph) — only buffer *contents* differ across ranks.
- **Parallelism.** **DDP** is the v1 default (modest model at `d=256/8L`); **FSDP** is a config switch for scale-up. `torch.compile` composes with both; compile happens **after** DDP/FSDP wrap.
- **No global-state assumptions.** Packing, normalization, and masks are per-buffer and rank-local; nothing in the collator or model reads cross-rank state ⇒ the single-process and multi-rank paths are **per-sample numerically identical** (`test_distributed`).
- **Checkpoint discipline (D13).** Save/restore **model + optimizer + per-rank dataloader/reservoir state**; the mandatory end-of-phase-1 checkpoint must restart at a **different world size** (re-shard deterministically; reservoir refills).

---

## Counts vs. capacity (resolved — do not re-flag)

D12's `[8,8,8,8,8,4]` was derived for the C=21 design point and never disambiguated between *per-channel* and *per-buffer*; those scale oppositely as C varies, so the literal vector cannot be a static cap. Resolution — decouple two concepts the config conflated:

- **Per-tier counts are dynamic per-segment data, not config constants.** Per-channel budget `n = ⌊(L_pack − Q)/C⌋` (D9). Distribute `n` across the 6 tiers by a **ratio prior** (config `model.tier_alloc_per_channel`, normalized), tuned so `n=44` reproduces `[8,8,8,8,8,4]`. Rebalance against `T_raw` (coarse→fine) so a long univariate crop piles tokens into the finer tiers it can fill and a short/high-C crop collapses to 1–2 tiers. Counts sum to ≤ `n` per channel and ≤ `L_pack` per buffer, and vary every buffer. Computed in `telescope.allocate` / `window_sampler`.
- **Dispatch capacity is static (`ENCODER_CAP` = L), values from the flat store, not persisted padded per-tier value tensors.** Persisting a `[B, cap_k, P_k, 2]` value tensor per tier would be ≈`L_pack`-sized per tier (worst case: a univariate buffer mostly one tier) → massive padding waste. Instead the buffer ships the **flat `norm_values[B,R]`/`observed[B,R]` store + per-token `raw_start`** (cheap, 1-D); Stage-7 routing picks slots with `role∈{CTX,KFF}` & `tier_id==k` (≤`ENCODER_CAP`, sentinel-padded) and **gathers `[CAP, P_k, 2]` transiently** from the store → `encoderₖ → [CAP,d]` → scatter. The transient gather is freed immediately; over-provisioning to `CAP=L` costs ~6% extra compute (walkthrough Stage 7). "Univariate fills the buffer" just works — no per-tier cap, no dropped tokens, no coverage capping.
- **Config rename:** `tier_caps` → `model.tier_alloc_per_channel` (ratio prior, scaled by `n` at runtime — *not* literal counts/caps); `n_ctx_cap` → `model.encoder_cap` (= `ENCODER_CAP`, `0` → `L_pack`; resolver `config.resolved_encoder_cap`).
- **`telescope.build_dispatch`:** the optional per-channel `counts[k] <= caps[k]` assert is a *convenience guard*; real enforcement is the static `ENCODER_CAP` in Stage-7 routing.
- **Verification:** `test_tokenize::test_capacity_univariate_and_21ch_same_encoder_cap` (a long univariate crop and a 21-channel crop both dispatch under one static `ENCODER_CAP`); dynamic-count layout-independence is the keystone `test_pack_invariance` (S9).

## Notes intentionally carried forward (do not re-flag)

- D5 per-layer mask schedule = **A3-ladder contingency**, not a v1 conflict with D9/D14's depth-invariant mask. Attention takes `block_mask` as a parameter so enabling it later is config + interface, not a refactor.
- D8 **no buffer-index positional encoding** — guarded by `test_pack_invariance` (S9). This is the single most likely implementation bug; the test exists to catch it on every change.
- D10 normalization is **two independent config axes + loss-space toggle**; all combinations must run from a one-line config change (no code edits).
- **Aux targets are raw-time, not "next token"** — each tier-`k` head predicts `[t+P_k, t+2P_k)` at tier-`k` granularity, built by the collator; validity is a precise raw-time check (origin-crossing → horizon's job; history-edge → unavailable), **not** the D9.5 `span` proxy. Overlapping cross-resolution supervision is accepted and damped by `aux_weights` (OA).
- **Distributed is first-class** (O6): disjoint `(node, rank)` shards, per-rank reservoir, cross-rank cost-bucketing, DDP→FSDP switch; single-process and multi-rank are per-sample identical.
