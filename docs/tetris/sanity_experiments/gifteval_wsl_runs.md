# TETRIS — GIFT-Eval WSL GPU runs (G3.1 + G4)

Real-GPU runs on the WSL box (RTX 3070, 8 GiB; Tailscale `ssh manoj@<TAILSCALE_IP>`), CUDA Flex +
`torch.compile`, model **d=224 / 6L / 4h, out_patch=16 → 10,555,980 params**. wandb project `tetris`.
Reproduction commands: `prompts/gifteval_plan.md` → *Reproduction commands*.

Leaderboard MASE = geo-mean across the 154 (config, term) cells; **skill** = model/seasonal-naive
(`snaive_MASE` ≈ 1.71); "finite/in-gmean/scored" counts non-NaN cells. Lower is better; skill < 1 beats
seasonal naive.

## Run 1 — G3.1 GIFT-Eval `test`-overfit (in-distribution capacity probe)

`configs/gifteval_test_overfit.yaml` — trains on the test split's own context windows, scored on their
held-out horizons (NOT zero-shot). Finishes the last G3.1 deliverable.

| run | steps | leaderboard MASE | skill | finite | time |
|---|---|---|---|---|---|
| 5k GPU sanity | 5000 | **209.65 → 8.14** | 122.47 → 4.75 | 154/154 | 16 min |
| 20k (full) | 20000 | — | — | — | *user will run later* |

The 5k GPU result **matches the Mac 5k CPU probe** (MASE 9.15, skill 4.78) — confirms the compiled CUDA +
leaderboard path is correct on GPU, finite throughout, 0 skipped. Run dir: `outputs/gifteval_test_overfit_20260615-195228/`
(model.pt, samples.png = 10 plots, train_log.txt). The full 20k is the user's to run (reproduction commands in the plan doc).

## Run 2 — G4 streaming corpus, **zero-shot** GIFT-Eval eval

`configs/streaming_run.yaml` — reservoir-train on the `streaming` loader over `corpus_mixed`
(**20,000 synthetic + 166,436 real GiftEvalPretrain = 186,436 series, 75 Arrow-IPC shards**), score/plot
**zero-shot** on real GIFT-Eval test (the model never sees the test split).

| run | steps | leaderboard MASE | skill | finite | time |
|---|---|---|---|---|---|
| 2k (end-to-end check) | 2000 | **209.65 → 3.78** | 122.47 → 2.21 | 154/154 | 9 min |
| 20k (longer) | 20000 | **209.65 → 3.41** | 122.47 → 1.99 | 154/154 | 1h24m |

20k run dir on the box: `outputs/streaming_run_20260615-211715/` (model.pt, samples.png, train_log.txt).
Training was stable end-to-end (loss bounded ~1.2–2.7, no divergence; all 154 cells finite at every eval).
The mid-train leaderboard MASE is **non-monotonic** (e.g. 3.78 @ step-eval early, 18.8 @ 5k, 3.41 final) —
the diverse 154-cell metric trades off across configs during training; the FINAL is the reported number.

The diverse synthetic+pretrain model generalizes **zero-shot** to real GIFT-Eval at **MASE 3.41 (skill 1.99)**
— better than the G3.1 5k test-*overfit* (8.14), a nice signal that the streamed corpus teaches transferable
structure. It does not yet beat seasonal naive (skill > 1); closing that is a scale/curriculum question for G5.

### Two stability fixes found by the first (diverging) 2k run
The first 2k diverged (loss 2.5→112, leaderboard NaN). Root causes, both fixed + regression-tested:
1. **`gen_exp_trend` fp32 overflow** (`735de55`): a fixed per-step exp rate exploded to ~1e29 on long
   (n=4096) series; the square overflows fp32 in the normalization variance → NaN. Bounded total growth to
   `exp(1)..exp(6)` (~3×..400×) regardless of length; corpus max now ~6e5.
2. **No gradient clipping**: a single high-loss batch spiked the loss and (un-clipped) exploded the weights.
   Added `RunCfg.grad_clip` (default 0.0 = off, so G3.1/sanity/shakedown are unchanged) threaded into
   `train_step` via `clip_grad_norm_`; `streaming_run.yaml` sets `1.0`. Post-fix: stable, all-finite.

## Run 3 — G5 curriculum (pretrain + train-split + curriculum) — *maintainer's to run*

`configs/gifteval_curriculum.yaml` — reservoir-train on the `curriculum` loader (decision-log **D13** two-phase
schedule) over **three sources**: varied synthetic (`corpus_synth`), the real ~1 GB GiftEvalPretrain slice
(`corpus_pretrain`), and the **live GIFT-Eval `train` split** (`gifteval_train`). Phase 1 is synthetic-heavy +
pretrain with the train split at natural weight; over `[phase2_start, 1]` synthetic fades, pretrain stays, the
**train split is upweighted**, and the crop sampler switches to the **test-matched horizon marginal**
(`auto_from_test_configs`). This trains on the train split + test-matched crops, so it is **in-distribution-ish,
NOT full zero-shot** — keep the framing honest.

| run | steps | leaderboard MASE | skill | finite | time |
|---|---|---|---|---|---|
| 2k (end-to-end check) | 2000 | — | — | — | *maintainer will run* |
| 20k (curriculum) | 20000 | — | — | — | *maintainer will run* |

Validated on Mac CPU through `uv run pytest` (the reservoir-path curriculum train smoke + crop-schedule switch run
green offline). The GPU run is the maintainer's, like G3.1/G4.

### One-time data prep on the box (TWO separate corpora so synthetic and pretrain weight apart)
```bash
ssh "$WSL" 'cd ~/tsfm-models && set -a && source .env && set +a && \
  ~/.local/bin/uv run --no-sync python -m tetris.data.materialize \
    --out outputs/corpus_synth --n-synthetic 20000'
ssh "$WSL" 'cd ~/tsfm-models && set -a && source .env && set +a && \
  ~/.local/bin/uv run --no-sync python -m tetris.data.materialize \
    --out outputs/corpus_pretrain --n-synthetic 0 --pretrain-root "$GIFT_EVAL_PRETRAIN"'
```
### Run (2k end-to-end check, then the 20k curriculum run)
```bash
ssh "$WSL" 'cd ~/tsfm-models && set -a && source .env && set +a && \
  nohup ~/.local/bin/uv run --no-sync python -m tetris.train.overfit_run \
    configs/gifteval_curriculum.yaml --steps 2000 --eval-every 2000 \
    --device cuda --n-plot 10 > /tmp/g5_2k.out 2>&1 & echo PID $!'
ssh "$WSL" 'cd ~/tsfm-models && set -a && source .env && set +a && \
  nohup ~/.local/bin/uv run --no-sync python -m tetris.train.overfit_run \
    configs/gifteval_curriculum.yaml --steps 20000 --eval-every 5000 \
    --device cuda --n-plot 10 > /tmp/g5_20k.out 2>&1 & echo PID $!'
```
**Tuning note:** `curriculum.total_items` is the schedule-progress **denominator** (items *pulled*, not steps).
If the 2k check shows phase 2 never engages (or engages too early), scale `total_items` so `phase2_start` (0.8)
lands in the last ~20% of the run — inspect the per-source mix in the train log against step count.
