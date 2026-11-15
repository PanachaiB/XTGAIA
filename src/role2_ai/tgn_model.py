import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
)

# นำเข้าโมดูล XAI Bundler สำหรับแพ็กหลักฐาน STIX 2.1
from src.role2_ai.xai_bundler import XAIBundler

class TemporalAttentionEmbedding(nn.Module):
    """
    Legacy embedding (XTGAIATGNLegacy): multi-head self-attention over a sequence of length 1.
    """
    def __init__(self, node_dim: int, num_heads: int = 8):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(memory_states, memory_states, memory_states)
        out = self.layer_norm(memory_states + attn_out)
        return out.squeeze(1)


class XTGAIATGNLegacy(nn.Module):
    """
    Original implementation kept to load older checkpoints: memory followed by multi-head
    attention over a sequence of length 1 (the node's own memory only).
    """
    def __init__(self, num_nodes: int = 10000, raw_msg_dim: int = 16, memory_dim: int = 128,
                 latent_z_dim: int = 8, num_heads: int = 8):
        super().__init__()
        self.memory_dim = memory_dim
        self.latent_z_dim = latent_z_dim
        self.memory = TGNMemory(num_nodes=num_nodes, raw_msg_dim=raw_msg_dim, memory_dim=memory_dim,
                                time_dim=memory_dim, message_module=IdentityMessage(raw_msg_dim, memory_dim, memory_dim),
                                aggregator_module=LastAggregator())
        self.attention_embedding = TemporalAttentionEmbedding(node_dim=memory_dim, num_heads=num_heads)
        self.z_projector = nn.Sequential(nn.Linear(memory_dim, 32), nn.ReLU(), nn.Linear(32, latent_z_dim))
        self.classifier = nn.Sequential(nn.Linear(latent_z_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, n_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem, _ = self.memory(n_id)
        h = self.attention_embedding(mem.unsqueeze(1))
        Z_t = self.z_projector(h)
        return self.classifier(Z_t), Z_t

    def update_memory_events(self, src, dst, t, raw_msg):
        self.memory.update_state(src, dst, t, raw_msg)

    def reset_state(self):
        self.memory.reset_state()


class TimeEncoder(nn.Module):
    """Learnable harmonic encoding of elapsed time, cos(w * log1p(Δt) + b) (TGN-style)."""
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(1, dim)

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        return torch.cos(self.lin(torch.log1p(dt.clamp(min=0).float()).unsqueeze(-1)))


class MaskedAttentionLayer(nn.Module):
    """Transformer encoder layer whose multi-head attention returns per-head weights and accepts a
    soft key mask m in [0, 1] (added as log m to the attention logits: attention ∝ m * exp(score)),
    which is how PGExplainer's edge mask σ(M) acts on the temporal neighbourhood."""
    def __init__(self, d: int, heads: int, ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.heads = heads
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad: torch.Tensor, key_mask: torch.Tensor = None):
        B, L, _ = x.shape
        bias = torch.zeros(B, L, device=x.device)
        bias = bias.masked_fill(pad, float("-inf"))
        if key_mask is not None:
            bias = bias + torch.log(key_mask.clamp(min=1e-6))
        attn_mask = bias[:, None, None, :].expand(B, self.heads, L, L).reshape(B * self.heads, L, L)
        out, w = self.attn(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        x = self.n1(x + self.drop(out))
        x = self.n2(x + self.drop(self.ff(x)))
        return x, w                                           # w: (B, heads, L, L)


class XTGAIATGN(nn.Module):
    """
    XTGAIA Temporal Threat Analyzer (paper Sec. III-B, Phase 3):
      * TGNMemory: 128-d GRU memory m_i(t) = GRU(m_i(t-), s̄_i(t))                     (Eq. 3)
      * 8-head temporal attention over the node's temporal neighbourhood N2(v): one token for
        v, K1 tokens for its most recent interactions and K2 tokens per neighbour for that
        neighbour's own most recent interactions; token = [memory || edge features || time(Δt)]
      * latent projection 128 -> 8 (Z_t) and node-level threat head p_t(v)               (Eq. 4)
    The attention weights feed temporal attention rollout (Tier 2) and the token mask is the
    edge mask PGExplainer optimises (Tier 1).
    """
    ARCHITECTURE = "temporal-attention-v2"

    def __init__(self, num_nodes: int = 10000, raw_msg_dim: int = 16, memory_dim: int = 128,
                 latent_z_dim: int = 8, num_heads: int = 8, k1: int = 10, k2: int = 3, layers: int = 2,
                 time_dim: int = 16, num_classes: int = 0, memory_skip: bool = True,
                 category_msg: bool = True):
        super().__init__()
        self._config = dict(num_nodes=num_nodes, raw_msg_dim=raw_msg_dim, memory_dim=memory_dim,
                            latent_z_dim=latent_z_dim, num_heads=num_heads, k1=k1, k2=k2, layers=layers,
                            time_dim=time_dim, num_classes=num_classes, memory_skip=memory_skip,
                            category_msg=category_msg)
        # Z_t from [attention output h || node memory m_v(t)]: the node's own GRU memory always reaches the
        # threat head directly, attention adds the temporal-neighbourhood evidence on top.
        self.memory_skip = memory_skip
        self.num_classes = num_classes
        self.memory_dim, self.latent_z_dim, self.raw_msg_dim = memory_dim, latent_z_dim, raw_msg_dim
        self.k1, self.k2 = k1, k2
        self.memory = TGNMemory(num_nodes=num_nodes, raw_msg_dim=raw_msg_dim, memory_dim=memory_dim,
                                time_dim=memory_dim, message_module=IdentityMessage(raw_msg_dim, memory_dim, memory_dim),
                                aggregator_module=LastAggregator())
        # temporal neighbour store (persisted with the checkpoint so live inference continues)
        self.register_buffer("nbr_id", torch.full((num_nodes, k1), -1, dtype=torch.long))
        self.register_buffer("nbr_t", torch.zeros((num_nodes, k1), dtype=torch.long))
        self.register_buffer("nbr_msg", torch.zeros((num_nodes, k1, raw_msg_dim)))
        self.register_buffer("nbr_ptr", torch.zeros(num_nodes, dtype=torch.long))
        self.time_enc = TimeEncoder(time_dim)
        self.hop_emb = nn.Embedding(3, memory_dim)            # 0: v itself, 1: one-hop, 2: two-hop
        self.token_proj = nn.Linear(memory_dim + raw_msg_dim + time_dim, memory_dim)
        self.layers = nn.ModuleList([MaskedAttentionLayer(memory_dim, num_heads) for _ in range(layers)])
        self.z_projector = nn.Sequential(nn.Linear(memory_dim * (2 if memory_skip else 1), 32), nn.ReLU(),
                                         nn.Linear(32, latent_z_dim))
        self.classifier = nn.Sequential(nn.Linear(latent_z_dim, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        # Auxiliary attack-category head (multi-class scoring): the category of a *detected event*. It reads the
        # 128-d node embedding h through a stop-gradient (so it never changes the representation behind
        # p_t / Z_t, the risk and response inputs) together with the triggering event's own flow message:
        # a victim node receives several attack families, so its embedding alone cannot name the family.
        self.category_msg = bool(category_msg)
        cat_in = memory_dim + (raw_msg_dim if self.category_msg else 0)
        if num_classes <= 0:
            self.category_head = None
        elif self.category_msg:
            self.category_head = nn.Sequential(nn.Linear(cat_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                               nn.Linear(64, num_classes))
        else:                                                 # node-embedding-only head (earlier checkpoints)
            self.category_head = nn.Sequential(nn.Linear(cat_in, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def last_message(self, nodes: torch.Tensor) -> torch.Tensor:
        """Most recent raw event message stored for each node (temporal neighbour store)."""
        slot = (self.nbr_ptr[nodes] - 1) % self.k1
        return self.nbr_msg[nodes, slot]

    def category_logits(self, h: torch.Tensor, msg: torch.Tensor = None) -> torch.Tensor:
        """h: node embedding from forward_full (nb["h"]), detached here by design; msg: the event's raw message."""
        if self.category_head is None:
            raise RuntimeError("model trained without the attack-category head")
        x = h.detach()
        if self.category_msg:
            if msg is None:
                msg = torch.zeros((x.shape[0], self.raw_msg_dim), dtype=x.dtype, device=x.device)
            x = torch.cat([x, msg.detach().to(x.dtype)], -1)
        return self.category_head(x)

    # ------------------------------------------------------------------ neighbourhood
    def _insert_loop(self, node: torch.Tensor, other: torch.Tensor, t: torch.Tensor, msg: torch.Tensor) -> None:
        """Reference (sequential) ring-buffer insert; _insert is the vectorised equivalent."""
        for i in range(node.numel()):
            v = int(node[i])
            slot = int(self.nbr_ptr[v] % self.k1)
            self.nbr_id[v, slot] = other[i]
            self.nbr_t[v, slot] = t[i]
            self.nbr_msg[v, slot] = msg[i]
            self.nbr_ptr[v] += 1

    def _insert(self, node: torch.Tensor, other: torch.Tensor, t: torch.Tensor, msg: torch.Tensor) -> None:
        """Vectorised per-node ring-buffer insert, identical to inserting the events one by one in order: the
        i-th occurrence of v in the batch goes to slot (ptr[v] + i) mod K1, and only a node's last K1 events in
        the batch are written (earlier ones would be overwritten anyway), so no two writes share a slot."""
        n = node.numel()
        if n == 0:
            return
        idx = torch.arange(n, device=node.device)
        order = torch.sort(node, stable=True).indices
        sn = node[order]
        first = torch.ones(n, dtype=torch.bool, device=node.device)
        first[1:] = sn[1:] != sn[:-1]
        start = torch.cummax(torch.where(first, idx, torch.zeros_like(idx)), 0).values
        rank = torch.empty_like(idx)
        rank[order] = idx - start                                   # occurrence index of each event within its node
        uniq, inv, counts = torch.unique(node, return_inverse=True, return_counts=True)
        keep = rank >= counts[inv] - self.k1
        v, r = node[keep], rank[keep]
        slot = (self.nbr_ptr[v] + r) % self.k1
        self.nbr_id[v, slot] = other[keep]
        self.nbr_t[v, slot] = t[keep]
        self.nbr_msg[v, slot] = msg[keep]
        self.nbr_ptr[uniq] += counts

    def neighbourhood(self, n_id: torch.Tensor):
        """Token layout for every node in n_id: [v] + K1 one-hop + K1*K2 two-hop entries."""
        B, k1, k2 = n_id.numel(), self.k1, self.k2
        order = torch.argsort(self.nbr_t[n_id], dim=1, descending=True)        # most recent first
        n1 = torch.gather(self.nbr_id[n_id], 1, order)                          # (B, k1)
        t1 = torch.gather(self.nbr_t[n_id], 1, order)
        m1 = torch.gather(self.nbr_msg[n_id], 1, order.unsqueeze(-1).expand(-1, -1, self.raw_msg_dim))
        valid1 = n1 >= 0
        n1c = n1.clamp(min=0)
        o2 = torch.argsort(self.nbr_t[n1c], dim=2, descending=True)[:, :, :k2]  # (B, k1, k2)
        n2 = torch.gather(self.nbr_id[n1c], 2, o2)
        t2 = torch.gather(self.nbr_t[n1c], 2, o2)
        m2 = torch.gather(self.nbr_msg[n1c], 2, o2.unsqueeze(-1).expand(-1, -1, -1, self.raw_msg_dim))
        valid2 = (n2 >= 0) & valid1.unsqueeze(-1)
        return {"n1": n1, "t1": t1, "m1": m1, "valid1": valid1, "n2": n2.reshape(B, k1 * k2),
                "t2": t2.reshape(B, k1 * k2), "m2": m2.reshape(B, k1 * k2, -1), "valid2": valid2.reshape(B, k1 * k2),
                "parent2": n1.unsqueeze(-1).expand(-1, -1, k2).reshape(B, k1 * k2)}

    # ------------------------------------------------------------------ forward
    def forward_full(self, n_id: torch.Tensor, token_mask: torch.Tensor = None):
        """-> p (B,1), Z (B,8), attention weights [layers x (B, heads, L, L)], neighbourhood dict."""
        mem_v, last_v = self.memory(n_id)
        nb = self.neighbourhood(n_id)
        B = n_id.numel()
        mem_all, _ = self.memory(torch.cat([nb["n1"].clamp(min=0).reshape(-1), nb["n2"].clamp(min=0).reshape(-1)]))
        if not self.training:
            # PyG folds pending messages into the memory buffer with autograd enabled when the module
            # switches to eval mode; memory is not trained in eval, so reads are detached.
            mem_v, mem_all = mem_v.detach(), mem_all.detach()
        mem1 = mem_all[: B * self.k1].reshape(B, self.k1, -1)
        mem2 = mem_all[B * self.k1:].reshape(B, self.k1 * self.k2, -1)
        now = last_v.unsqueeze(1)
        tok0 = torch.cat([mem_v, torch.zeros(B, self.raw_msg_dim, device=mem_v.device),
                          self.time_enc(torch.zeros(B, device=mem_v.device))], -1).unsqueeze(1)
        tok1 = torch.cat([mem1, nb["m1"], self.time_enc((now - nb["t1"]).float())], -1)
        tok2 = torch.cat([mem2, nb["m2"], self.time_enc((now - nb["t2"]).float())], -1)
        x = self.token_proj(torch.cat([tok0, tok1, tok2], 1))
        hops = torch.cat([torch.zeros(B, 1, dtype=torch.long), torch.ones(B, self.k1, dtype=torch.long),
                          torch.full((B, self.k1 * self.k2), 2, dtype=torch.long)], 1).to(x.device)
        x = x + self.hop_emb(hops)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~nb["valid1"], ~nb["valid2"]], 1)
        weights = []
        for layer in self.layers:
            x, w = layer(x, pad, token_mask)
            weights.append(w)
        h = x[:, 0]
        nb["h"] = h
        Z_t = self.z_projector(torch.cat([h, mem_v], -1) if self.memory_skip else h)
        return self.classifier(Z_t), Z_t, weights, nb

    def forward(self, n_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        p, z, _, _ = self.forward_full(n_id)
        return p, z

    def update_memory_events(self, src: torch.Tensor, dst: torch.Tensor, t: torch.Tensor, raw_msg: torch.Tensor):
        self.memory.update_state(src, dst, t, raw_msg)
        with torch.no_grad():
            self._insert(dst, src, t, raw_msg)
            self._insert(src, dst, t, raw_msg)

    def clone(self) -> "XTGAIATGN":
        """Independent copy (weights, memory, neighbour store) without autograd history."""
        other = XTGAIATGN(**self._config)
        other.load_state_dict(self.state_dict())
        return other

    def reset_state(self):
        self.memory.reset_state()
        self.nbr_id.fill_(-1)
        self.nbr_t.zero_()
        self.nbr_msg.zero_()
        self.nbr_ptr.zero_()


class MockTGNInference:
    """
    Mock inference interface required by unit tests (tests/role2_test.py).
    Produces contract-compliant p_t, Z_t, and mock XAI bundle.
    """
    def __init__(self, *args, **kwargs):
        self.bundler = XAIBundler()

    def predict(self, *args, **kwargs):
        p_t = 0.85
        z_t = torch.full((8,), 0.1, dtype=torch.float32)
        return p_t, z_t

    def predict_threat_and_explain(self, graph_tensors=None):
        """
        XTGAIA Person 2 Core Interface Contract:
        Returns: (p_t, Z_t, xai_bundle)
        """
        p_t = 0.85
        z_t = torch.full((8,), 0.1, dtype=torch.float32)
        
        # สร้าง STIX 2.1 XAI Bundle ผ่าน XAIBundler
        mock_structural = [(2, 0, 0.5458), (1, 2, 0.5098)]
        mock_temporal = {
            "onset_timestamp_ms": 187.5,
            "attack_duration_ms": 312.5,
            "temporal_trace": [0.0384, 0.0506, 0.079, 0.2719, 0.0924, 0.0564, 0.0413, 0.3701]
        }
        xai_bundle = self.bundler.build_stix_bundle(
            node_id="target_host_node",
            threat_prob=p_t,
            structural_edges=mock_structural,
            temporal_analysis=mock_temporal
        )
        return p_t, z_t, xai_bundle

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


# Module-level convenience function for end-to-end integration
_default_engine = MockTGNInference()

def predict_threat_and_explain(graph_tensors=None):
    """
    Public API callable for XTGAIA Pipeline (main.py and Person 3).
    Returns (p_t, Z_t, xai_bundle)

    Uses the trained live engine (src/role2_ai/inference.py) when the pipeline has
    loaded one; otherwise falls back to the mock for offline unit tests.
    """
    from src.role2_ai.inference import get_inference_engine  # lazy: inference imports this module
    engine = get_inference_engine()
    if engine is not None:
        return engine.predict_threat_and_explain(graph_tensors)
    return _default_engine.predict_threat_and_explain(graph_tensors)