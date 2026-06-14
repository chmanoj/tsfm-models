"""Output heads — dense over all L, gather/mask at loss time (Stage 9).

Per the walkthrough's option-β, every head runs on the full ``[B, L, d]`` (no
token gather before the head) so shapes stay static; selection happens later in
the loss reduction (S8).

- Horizon head: ``[B, L, d] → [B, L, P_out]`` (real only at QRY slots).
- Six aux heads: ``[B, L, d] → [B, L, P_k]`` each; a token tier-selects its own
  head's output. Each predicts the next ``P_k`` raw steps ``[t+P_k, t+2P_k)`` (D6,
  raw-time aux); validity (``valid_aux``) and per-tier weights applied in the loss.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from ..constants import PATCH


class Heads(nn.Module):
    def __init__(self, d_model: int, out_patch: int):
        super().__init__()
        self.horizon = nn.Linear(d_model, out_patch)
        self.aux = nn.ModuleList(nn.Linear(d_model, p_k) for p_k in PATCH)

    def forward(self, hidden: torch.Tensor):
        """Returns ``(horizon [B,L,P_out], aux: list of 6 × [B,L,P_k])``."""
        horizon = self.horizon(hidden)
        aux: List[torch.Tensor] = [head(hidden) for head in self.aux]
        return horizon, aux
