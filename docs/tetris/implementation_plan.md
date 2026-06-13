# TETRIS — Implementation Plan (v1)

**Status:** planning artifact. No model/training code yet. Source of truth: `docs/tetris/tetris_decision_log.html` (rev 17, D1–D15 + v2 backlog). This plan does **not** redesign any decided item; it sequences them into buildable modules with smoke/unit tests.

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
| `cap_k` | static per-buffer capacity of tier-`k` tokens | small | from D12 alloc bounds + headroom |
| `C_slot` | max variate slots per buffer | small | from packer |
| `Q_max` | max query tokens per buffer | small | `n_targets·⌈p/P_out⌉` bound |
| `A_max` | max aux source tokens per buffer | = ctx tokens | — |

**Variate slot (`vslot`)** = a buffer-local index unique per `(sample_id, channel)`. It is the binding handle for D4: the same random orthonormal ID and the same normalization stats attach to every tier token *and* every query token of one variate via `vslot`.

**Side tensors** (per token, batched `[B, L]` unless noted) — the *only* source of token geometry (D8 hard rule: **no buffer-index positional encoding anywhere**):

| Name | dtype | Meaning |
|---|---|---|
| `sample_id` | int32 | D9 packing segment id; `-1` = pad |
| `vslot` | int32 | variate slot (see above) |
| `channel_idx` | int32 | channel within its sample (features first, then targets) |
| `t_center` | float32 | continuous time vs forecast origin (negative = past, ≥0 = horizon) |
| `span` | int8 | tier index 0..5 (≡ log₂ patch via PATCH) |
| `role` | int8 | `{ctx, qry}` |
| `role_ft` | int8 | `{feature, target}` (D11) |
| `content_state` | int8 | `{observed, MASK, NA}` (D7) |
| `valid` | bool | boundary + loss-validity (D7/D9) |

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
    constants.py                 # PATCH vocab, ContentState/Role enums, dtypes
    backend.py                   # backend switch: flex+compile (CUDA) | sdpa+eager (Mac)

    normalize.py                 # D10: anchored-arcsinh, fallback chain, per-tier σ_Δ,
                                 #      exact inversion, local-reanchor loss-target builders
    telescope.py                 # D2/D12: allocation, coverage<->tokens, per-tier gather/scatter index build

    tokenize/
      spec.py                    # SegmentSpec (size computable from spec alone, no tokenization)
      window_sampler.py          # D9.2: origin/p sampling, per-tier counts, content-state assignment
      assemble.py                # raw->normalized patch windows + side tensors (pure, per spec)

    packing/
      collator.py                # STATELESS pack(items, specs) -> PackedBatch  (the "collator")
      reservoir.py               # streaming IterableDataset: reservoir + best-fit-decreasing (D9.3)
      scheduler.py               # D9.4 cost-bucketed step grouping (Σ S_i²/2), cross-rank (multi-node)

    masks.py                     # D9 truth table -> BlockMask (flex) / bool [L,L] (sdpa); equality-tested

    model/
      embeddings.py              # time/span/role/content-state embeddings (+ MASK/NA learned)
      variate_id.py              # D4: random orthonormal IDs per vslot + content-summary MLP
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
    test_collator_shapes.py     # PackedBatch shapes/capacities; specs predict sizes exactly
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
                            { origin, p, per_channel_tier_counts [C_i, 6], Q, S (segment length),
                              channel roles (role_ft), demotion/KFF flags (D11) }
                            S is exact from the spec alone (D9: packable without tokenizing)

reservoir IterableDataset -> List[(item, spec)]  (a chosen pack-group; ~B buffers' worth)

pack(items, specs)       -> PackedBatch:
  side tensors            : sample_id, vslot, channel_idx, t_center, span, role, role_ft,
                            content_state, valid                       each [B, L]
  per-tier encoder inputs : tier_input[k]   [B, cap_k, PATCH[k], 2]    (value, observed-indicator; D7)
                            tier_scatter[k] [B, cap_k]  (dest pos in [0,L))
                            tier_valid[k]   [B, cap_k]
  variate material        : var_id_table   [B, C_slot, d]   (orthonormal rows, resampled per sample; D4)
                            var_feats       [B, C_slot, F]    (content-summary stats; D4/D10 receipts)
  norm stats (inversion)  : norm_a [B, C_slot], norm_sigma [B, C_slot], norm_sigma_tier [B, C_slot, 6]
  loss indices            : qry_idx [B, Q_max], qry_valid [B, Q_max],
                            qry_target [B, Q_max, P_out], qry_target_valid [B, Q_max, P_out]
                            aux_target  per-tier raw-time region (tier-k granularity), aux_valid [B, L]
                              # tier-k token covering raw [t,t+P_k) -> target = next P_k raw steps
                              #   [t+P_k, t+2P_k) aggregated to tier-k granularity (built by collator)

model.forward(PackedBatch) :
  embeddings              : e [B, L, d]   (content_enc + time + span + variate + role + state)
  backbone (n_layers ×)   : attn(e, block_mask) + FFN -> e [B, L, d]
  horizon head            : gather qry_idx -> [B, Q_max, d] -> [B, Q_max, P_out]  (norm-increment space)
  aux heads (per tier)    : [B, L, d] -> per-tier raw-time region preds in norm-increment space

losses.compute           : scalar  = MAE(horizon, masked) + Σ_k aux_weights[k]·MAE(aux_k, masked)
metrics                  : per-config test loss on deterministic first-100 shard (record-only; MASE deferred)
```

---

## 3. Module responsibilities (detail)

### `config.py`, `constants.py`, `backend.py`
- **config:** nested dataclasses; load/merge YAML via OmegaConf. Surfaces all D10 axes (`input_norm ∈ {anchored_arcsinh, zscore_arcsinh}`, `loss_target ∈ {locally_reanchored, global_norm_space}`, `loss_space ∈ {arcsinh, vol_units}`), D13 `dataset_weights {id: multiplier}` + `phase2_*`, D15 `loss_weighting`, the per-tier `aux_weights [6]` vector (replaces a single λ), and the `distributed` block (§9). **Every combination must run** (D10 mandate).
- **constants:** `PATCH=(4,8,16,64,256,512)`, enums `ContentState{OBSERVED,MASK,NA}`, `Role{CTX,QRY}`, `RoleFT{FEATURE,TARGET}`.
- **backend:** `attend(q,k,v, block_mask, score_mod)` dispatch. CUDA → `flex_attention` + compiled; else → `F.scaled_dot_product_attention` with materialized `[L,L]` bool mask. `make_mask(side_tensors)` returns the matching object for each backend. Single seam where compile/Flex is opt-in.

### `normalize.py` (D10)
- `compute_stats(x_ctx)` → `(a, σ_Δ, σ_Δ_tier[6])`. `a = median(last 32)` (window size is a tuning knob, 8–16 for steep trends). `σ_Δ = 1.4826·median|Δx|` with fallback chain `mean|Δx| → IQR/1.349 → 1`, plus floor `σ_Δ ≥ 1e-3·IQR/1.349`. Per-tier σ_Δ from each tier's aggregated sequence.
- `forward(x, a, σ)` → `arcsinh((x-a)/σ)`; `invert(z, a, σ)` → `sinh(z)·σ + a` (exact, denormalizes horizon preds; no leakage).
- Loss-target builders: `aux_target(t) = arcsinh((x[t+1..t+P] - level_t)/σ_tier)`; `horizon_target(y) = arcsinh((y - a)/σ)`. `loss_target=global_norm_space` swaps in global-z targets (one-line, per D10 hedge).
- Cross-channel receipts (D10): emit `log σ_Δ`, level relations, fallback-tier flag into `var_feats`.

### `telescope.py` (D2/D12)
- `allocate(n_tokens, T_raw)` → per-tier counts via **halving prior** then rebalance to `coverage ≈ T_raw` (walk coarse→fine); deterministic integer ops.
- `coverage(counts)` / `tokens_for_coverage(T)` — invert the ladder (used by D9.2: `n = ⌊(L−Q)/C⌋` → `T_cov`).
- `build_dispatch(counts, origin)` → per-tier raw-window slices + **channel-major** scatter positions `pos = base + c·n + t`. Asserts `count_k ≤ cap_k`.

### `tokenize/` (D9.2)
- `spec.py`: `SegmentSpec` carries everything needed to compute `S` and to assemble later; **no tokenization at packing time**.
- `window_sampler.py`: sample `p ∈ [1, p_max]` (log-uniform, D13), `Q`, per-channel `n`, `T_cov`, **uniform random origin**; assign D11 role demotion / KFF reveal flags (crop-level prob `q≈0.2`). Stateless given an RNG; lives in the reservoir layer.
- `assemble.py`: given `(item, spec)`, normalize per channel (D10), build per-tier patch windows `[·, PATCH[k], 2]` (value, observed-indicator), set `content_state` (observed/NA per patch; MASK for queries), fill side tensors. **Also constructs the raw-time aux targets** per tier token — for a tier-`k` token covering raw `[t, t+P_k)`, the target is the next `P_k` raw steps `[t+P_k, t+2P_k)` aggregated to tier-`k` granularity — and the precise `aux_valid` mask (origin-crossing and history-edge cases, see `losses.py`). Pure function — the heart of the collator.

### `packing/`
- `collator.py`: **stateless** `pack(items, specs) -> PackedBatch`. Best-fit placement of segments into `B` buffers of length `L`, channel-major, pad tail (`sample_id=-1`). Emits all tensors in §2.1. *This signature is frozen* — both the trivial path and the reservoir path call it unchanged.
- `reservoir.py`: `IterableDataset` wrapping a base loader: pulls items, runs `window_sampler`, keeps a reservoir of `K≈1000` specs (doubles as shuffle buffer), **best-fit-decreasing**, refills, yields pack-groups. *(Built in Stage 11, after the trivial path.)*
- `scheduler.py`: D9.4 cost = `Σ S_i²/2`; window of 64–256 buffers sorted by cost → each global step formed from similar-cost buffers (giants travel together). Shapes unchanged.

### `masks.py` (D9.1)
Truth table: `allow(q,k) = s[q]==s[k] AND ( (role[k]==ctx AND (role[q]==qry OR t[k]<=t[q])) OR (role[k]==qry AND role[q]==qry) )`. Pad (`s=-1`) gets a `q==k` self-attend exception (NaN guard; outputs discarded). Built once per batch, depth-invariant (v1). Two constructors: Flex `BlockMask` (`mask_mod`) and SDPA bool `[L,L]`; **equality-tested**. `score_mod` socket present, default `None` (A3 ladder hook).

### `model/`
- `variate_id.py` (D4): sample `C_slot` rows of a random rotation (orthonormal → separation), resampled per sample (fixed seed at inference); `+` content-summary MLP over `var_feats`. Indexed by `vslot`.
- `encoders.py` (D3/D14): six encoders (Fourier number-features → MLP/conv). **Grouped dispatch**: for each tier, `gather → [cap_k, PATCH[k], 2] → encode → scatter` into `[B, L, d]`. Static shapes, no Python branching in-graph.
- `embeddings.py`: `e = content_enc + time_emb(t_center) + span_emb(span) + variate_emb(vslot) + role_emb(role_ft) + state_emb(content_state)`. MASK/NA are learned content embeddings (D7).
- `attention.py`: `attn(e, block_mask, score_mod=None)` → QKV proj `[B,H,L,d_h]` → backend `attend`. **`block_mask` is a parameter** (per-layer schedule = future arg, not refactor).
- `blocks.py` / `tetris.py`: pre-norm blocks; full forward assembles §2.1.
- `heads.py`: horizon head gathers `qry_idx` → `[B,Q_max,P_out]` (norm-increment space). **Six per-tier aux heads** (mirror encoders; grouped scatter). A tier-`k` aux head predicts the **next raw region in tier-time**: for a token covering raw `[t, t+P_k)`, the target is the next `P_k` raw steps `[t+P_k, t+2P_k)` aggregated to tier-`k` granularity (built by the collator from the raw series). Output width = the tier's own granularity → **no cross-tier width mismatch** (e.g. no patch-8→patch-4 conflict) and **no dependence on whatever token happens to follow** in channel-major order. This replaces the "next token" framing.

### `losses.py` (D6/D10)
MAE horizon (median-optimal) on `qry_target` masked by `qry_target_valid` (NaN GT dropped, D7). `+` **per-tier-weighted** aux MAE (`aux_weights[6]` config vector replaces a single λ) on the raw-time `aux_target` masked by `aux_valid`. **Aux validity is a precise raw-time check** (replacing D9.5's `span[p+1]==span[p]` proxy): the aux term for a tier-`k` token is masked when its target region `[t+P_k, t+2P_k)` **(a) crosses the forecast origin** (tier-0 edge — that region is the horizon task, owned by query tokens) or **(b) runs off available history** (oldest-tier edge — partial/unavailable patch), plus the D7 mostly-missing-patch skip. `loss_space ∈ {arcsinh, vol_units}` toggle. Per-sample normalized influence (σ_Δ scaling ≈ MASE-aligned).

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
| **S4** | `tokenize/` (spec, window_sampler, assemble) | Spec-predicted `S` equals assembled length exactly; origin/`p`/counts within bounds; content_state assignment correct. |
| **S5** | `masks.py` | **[GATE] `test_masks`**: enumerate small grid; assert every `(q,k)` matches the boolean formula (ctx→ctx causal+channel-blind; qry→ctx always; qry→qry both ways; ctx→qry never; pad self-only). Flex `BlockMask` == SDPA bool. |
| **S6** | `packing/collator.py` (trivial no-reservoir collate_fn) | `test_collator_shapes`: packs DataLoader's list into `[B,L]` buffers; all side tensors + capacities present; tail pad `sample_id=-1`. |
| **S7** | `model/` (embeddings, variate_id, encoders, attention, blocks, heads, tetris) | `test_model_smoke`: forward on one packed buffer → finite `[B,Q_max,P_out]` + aux; both backends. |
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
  tier_caps: [16, 12, 8, 6, 4, 2]   # cap_k, generous headroom over alloc
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

## Notes intentionally carried forward (do not re-flag)

- D5 per-layer mask schedule = **A3-ladder contingency**, not a v1 conflict with D9/D14's depth-invariant mask. Attention takes `block_mask` as a parameter so enabling it later is config + interface, not a refactor.
- D8 **no buffer-index positional encoding** — guarded by `test_pack_invariance` (S9). This is the single most likely implementation bug; the test exists to catch it on every change.
- D10 normalization is **two independent config axes + loss-space toggle**; all combinations must run from a one-line config change (no code edits).
- **Aux targets are raw-time, not "next token"** — each tier-`k` head predicts `[t+P_k, t+2P_k)` at tier-`k` granularity, built by the collator; validity is a precise raw-time check (origin-crossing → horizon's job; history-edge → unavailable), **not** the D9.5 `span` proxy. Overlapping cross-resolution supervision is accepted and damped by `aux_weights` (OA).
- **Distributed is first-class** (O6): disjoint `(node, rank)` shards, per-rank reservoir, cross-rank cost-bucketing, DDP→FSDP switch; single-process and multi-rank are per-sample identical.
