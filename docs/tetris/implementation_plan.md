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
