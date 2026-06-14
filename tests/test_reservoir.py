"""S11 test_reservoir — streaming packer (D9.3) + cost-bucketed scheduler (D9.4).

Covers: the scheduler groups similar-cost buffers; the reservoir packs above the
**unchanged** collator with low single-digit tail waste, never overflowing
``pack``'s ``ΣS ≤ L`` contract; ``state_dict``/``load_state_dict`` resume exactly;
and the reservoir train loop runs end-to-end with finite losses.
"""

import os

import numpy as np
import torch

from tetris.config import load_config
from tetris.packing import scheduler as SCHED
from tetris.packing.reservoir import StreamingReservoir, packed_batches
from tetris.tokenize.assemble import AssembledSegment
from tetris.train.loop import run_training

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
P_OUT = 8


def _cfg():
    cfg = load_config(os.path.join(CONFIG_DIR, "shakedown.yaml"))
    cfg.packing.L_pack = 256
    cfg.packing.buffers_per_step = 2
    cfg.packing.reservoir = True
    cfg.packing.reservoir_k = 48
    cfg.packing.scheduler_window = 8
    cfg.packing.tail_tolerance = 0.05
    cfg.model.out_patch = P_OUT
    cfg.model.d_model = 16
    cfg.model.n_layers = 1
    cfg.model.n_heads = 2
    cfg.data.n_series = 400
    cfg.data.C_distribution = [1, 4]
    cfg.data.length_distribution = [64, 256]
    return cfg


def _reservoir(cfg):
    return StreamingReservoir.from_cfg(cfg)


# --- scheduler (D9.4) ---------------------------------------------------------

def test_scheduler_groups_similar_cost():
    # Buffers carry their own cost; identity cost_of makes the grouping checkable.
    costs = [100.0, 1.0, 50.0, 2.0, 99.0, 3.0, 51.0]
    steps = SCHED.cost_bucketed_steps(costs, buffers_per_step=2, cost_of=lambda c: c)
    flat = [c for step in steps for c in step]
    assert flat == sorted(costs)                       # cost-sorted order
    assert [len(s) for s in steps] == [2, 2, 2, 1]      # B-chunked, short tail
    # Each step is a contiguous chunk of the cost order: every buffer in a step
    # is ≤ every buffer in the next step (giants travel together; no interleave).
    for lo, hi in zip(steps, steps[1:]):
        assert max(lo) <= min(hi)


def test_buffer_cost_is_half_sum_of_squares():
    class S:
        def __init__(self, s):
            self.S = s

    assert SCHED.buffer_cost([S(4), S(6)]) == 0.5 * (16 + 36)


# --- reservoir packing (D9.3) -------------------------------------------------

def test_steps_respect_collator_contract():
    cfg = _cfg()
    res = _reservoir(cfg)
    L, B = cfg.packing.L_pack, cfg.packing.buffers_per_step
    seen = 0
    for step in res:
        assert isinstance(step, list) and 1 <= len(step) <= B
        for buf in step:
            assert all(isinstance(s, AssembledSegment) for s in buf)
            assert sum(s.S for s in buf) <= L        # never overflows pack's contract
        seen += 1
        if seen >= 12:
            break
    assert seen >= 1


def test_tail_waste_low_single_digit():
    cfg = _cfg()
    res = _reservoir(cfg)
    fills = []
    multi = 0
    for step_i, step in enumerate(res):
        for buf in step:
            used = sum(s.S for s in buf)
            fills.append(used / cfg.packing.L_pack)
            if len(buf) > 1:
                multi += 1
        if step_i >= 30:
            break
    # Short series → many small specs → buffers pack tight. Mean waste is low
    # single-digit %; assert a comfortable bound and that real packing happened.
    assert multi > 0                                   # buffers actually share
    assert np.mean(fills) > 0.90                       # < 10% mean tail waste


def test_state_dict_resume_is_exact():
    cfg = _cfg()
    r1 = _reservoir(cfg)
    it = iter(r1)
    next(it); next(it)                                 # consume two steps
    sd = r1.state_dict()
    assert sd["items_pulled"] > 0
    step_a = next(it)                                  # the next step from r1

    r2 = _reservoir(cfg)
    r2.load_state_dict(sd)
    step_b = next(iter(r2))                            # same step after resume

    def sig(step):
        return [sorted(s.S for s in buf) for buf in step]

    assert sig(step_a) == sig(step_b)


# --- train loop runs (S11 acceptance) -----------------------------------------

def test_reservoir_train_loop_runs():
    cfg = _cfg()
    torch.manual_seed(0)
    losses = run_training(cfg, steps=5, device="cpu")
    assert len(losses) == 5
    assert all(np.isfinite(losses))
