"""
XTGAIA Stage 6, Tiers 1 and 2 (paper Algorithm 3, lines 6-9), on the temporal-attention TGN.

Tier 1 - PGExplainer (structural): an edge-mask network (Role 2's PGExplainerStructural) scores
every edge of the temporal 2-hop neighbourhood N2(v) from its endpoints' memory states. The mask
σ(M) acts on the TGN through the attention key mask, and the network is trained post hoc with the
TGN frozen to maximise the mutual information between the masked and the original prediction:
    M_sub = argmax_M MI(p_t(v), G_act[N2(v)] ⊙ σ(M))
    loss  = CE(p_masked, p_orig) + λ_size · mean(σ(M)) + λ_ent · H(σ(M))
Reported per explanation: top-k edges, fidelity+ = p(full) - p(without top-k),
fidelity- = p(full) - p(top-k only), sparsity = 1 - k / |E(N2(v))|.

Tier 2 - temporal attention rollout: T_trace = Π_l (0.5 · Ā^(l) + 0.5 · I) over the TGN's
attention layers (heads averaged; Role 2's AttentionRolloutTemporal). Row 0 gives each token's
contribution to v's representation; the one-hop tokens are v's interactions in time order, and
the attack onset is the earliest interaction carrying at least half of the peak contribution.
"""

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.role2_ai.xai_structural import PGExplainerStructural
from src.role2_ai.xai_temporal import AttentionRolloutTemporal


class TGNExplainer:
    def __init__(self, model, pg: Optional[PGExplainerStructural] = None, top_k: int = 5):
        self.model = model
        self.pg = pg or PGExplainerStructural(node_feat_dim=model.memory_dim)
        self.top_k = top_k
        self.rollout = AttentionRolloutTemporal(alpha=0.5)

    # ------------------------------------------------------------------ edge masks
    def edge_mask(self, n_id: torch.Tensor, nb: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(B, L) token mask: 1 for v itself, σ(M) for every N2(v) edge token, 0 for padding."""
        B, k1, k2 = n_id.numel(), self.model.k1, self.model.k2
        with torch.no_grad():
            mem_v, _ = self.model.memory(n_id)
            ids = torch.cat([nb["n1"].clamp(min=0).reshape(-1), nb["n2"].clamp(min=0).reshape(-1),
                             nb["parent2"].clamp(min=0).reshape(-1)])
            mem, _ = self.model.memory(ids)
        mem1 = mem[: B * k1].reshape(B, k1, -1)
        mem2 = mem[B * k1: B * k1 + B * k1 * k2].reshape(B, k1 * k2, -1)
        memp = mem[B * k1 + B * k1 * k2:].reshape(B, k1 * k2, -1)
        e1 = torch.cat([mem1, mem_v.unsqueeze(1).expand(-1, k1, -1)], -1)          # edge (u_i -> v)
        e2 = torch.cat([mem2, memp], -1)                                           # edge (w_ij -> u_i)
        m = self.pg.edge_mask_net(torch.cat([e1, e2], 1)).squeeze(-1)              # (B, k1 + k1*k2)
        valid = torch.cat([nb["valid1"], nb["valid2"]], 1).float()
        return torch.cat([torch.ones(B, 1), m * valid], 1)

    # ------------------------------------------------------------------ training (TGN frozen)
    def train_step(self, n_id: torch.Tensor, opt, size_coef: float = 0.005, ent_coef: float = 0.01) -> float:
        self.model.eval()
        with torch.no_grad():
            p_orig, _, _, nb = self.model.forward_full(n_id)
        mask = self.edge_mask(n_id, nb)
        p_mask, _, _, _ = self.model.forward_full(n_id, token_mask=mask)
        target = p_orig.detach().clamp(1e-6, 1 - 1e-6)
        pred = p_mask.clamp(1e-6, 1 - 1e-6)
        ce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred)).mean()
        valid = torch.cat([nb["valid1"], nb["valid2"]], 1).float()
        m = mask[:, 1:]
        n_valid = valid.sum().clamp(min=1)
        size = (m * valid).sum() / n_valid
        ent = (-(m * torch.log(m + 1e-6) + (1 - m) * torch.log(1 - m + 1e-6)) * valid).sum() / n_valid
        loss = ce + size_coef * size + ent_coef * ent
        opt.zero_grad()
        loss.backward()
        opt.step()
        return float(loss.item())

    def fit(self, batches: Iterable[torch.Tensor], lr: float = 3e-3, epochs: int = 1) -> List[float]:
        for p in self.model.parameters():
            p.requires_grad_(False)
        opt = torch.optim.Adam(self.pg.parameters(), lr=lr)
        history = []
        try:
            for _ in range(epochs):
                for n_id in batches:
                    history.append(self.train_step(n_id, opt))
        finally:
            for p in self.model.parameters():
                p.requires_grad_(True)
        return history

    # ------------------------------------------------------------------ explanation
    def _edges(self, b: int, nb: Dict[str, torch.Tensor], id_to_ip) -> List[Tuple[str, str, int, int]]:
        """Token index -> (src, dst, time, hop) for row b (token 0 = v itself is excluded)."""
        out = []
        for j in range(self.model.k1):
            if bool(nb["valid1"][b, j]):
                out.append((id_to_ip(int(nb["n1"][b, j])), None, int(nb["t1"][b, j]), 1))
            else:
                out.append(None)
        for j in range(self.model.k1 * self.model.k2):
            if bool(nb["valid2"][b, j]):
                out.append((id_to_ip(int(nb["n2"][b, j])), id_to_ip(int(nb["parent2"][b, j])), int(nb["t2"][b, j]), 2))
            else:
                out.append(None)
        return out

    def explain(self, node_id: int, host: str, id_to_ip, t0: float = 0.0) -> Dict[str, Any]:
        """Tier 1 + Tier 2 for one node; returns structural edges, fidelity, temporal trace, latency."""
        start = time.perf_counter()
        n_id = torch.tensor([node_id], dtype=torch.long)
        with torch.no_grad():
            p_full, _, weights, nb = self.model.forward_full(n_id)
            mask = self.edge_mask(n_id, nb)
            t_struct = time.perf_counter()
            edges = self._edges(0, nb, id_to_ip)
            scores = mask[0, 1:].numpy()
            valid_idx = [i for i, e in enumerate(edges) if e is not None]
            ranked = sorted(valid_idx, key=lambda i: scores[i], reverse=True)[: self.top_k]
            structural = []
            for i in ranked:
                src, parent, _, hop = edges[i]
                dst = host if hop == 1 else parent
                structural.append((src, dst, round(float(scores[i]), 4)))
            fid_plus = fid_minus = None
            if ranked:
                keep = torch.ones_like(mask)
                keep[0, [1 + i for i in ranked]] = 1e-6                      # remove top-k edges
                p_without, _, _, _ = self.model.forward_full(n_id, token_mask=keep)
                only = torch.full_like(mask, 1e-6)
                only[0, 0] = 1.0
                only[0, [1 + i for i in ranked]] = 1.0                       # keep top-k only
                p_only, _, _, _ = self.model.forward_full(n_id, token_mask=only)
                fid_plus = float(p_full - p_without)
                fid_minus = float(p_full - p_only)
            sparsity = 1.0 - len(ranked) / max(len(valid_idx), 1)
            t_temporal = time.perf_counter()
            mats = [w[0].mean(0, keepdim=True) for w in weights]              # heads averaged, (1, L, L)
            rollout = self.rollout(mats)                                      # Π (0.5 A + 0.5 I)
            contrib = rollout[0, 1:1 + self.model.k1].numpy()                 # one-hop interactions
            inter = [(edges[j][2], float(contrib[j])) for j in range(self.model.k1) if edges[j] is not None]
            inter.sort(key=lambda x: x[0])
        if inter:
            total = sum(c for _, c in inter) or 1.0
            trace = [c / total for _, c in inter]
            ts_ms = [(t - inter[0][0]) * 1000.0 for t, _ in inter]
            onset = self.rollout.identify_attack_onset(torch.tensor(trace), ts_ms, threshold=0.5 * max(trace))
            onset["trace_start_epoch"] = inter[0][0] + t0
        else:
            onset = {"onset_index": -1, "onset_timestamp_ms": 0.0, "attack_duration_ms": 0.0, "temporal_trace": []}
        end = time.perf_counter()
        return {"p_t": float(p_full), "structural_edges": structural, "temporal": onset,
                "fidelity_plus": fid_plus, "fidelity_minus": fid_minus, "sparsity": round(sparsity, 4),
                "n2_edges": len(valid_idx),
                "latency_ms": {"structural": (t_temporal - start) * 1000.0, "temporal": (end - t_temporal) * 1000.0,
                               "total": (end - start) * 1000.0}}
