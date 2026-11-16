"""
Role 2 live inference: the trained XTGAIATGN applied to Role 1's real event stream.

    engine = TGNInferenceEngine("data/models/tgn_best.pt")
    p_flow = engine.observe_batch(edges)          # TGN memory update, then p_t per flow
    p_t, Z_t = engine.score(host)                 # threat probability + 8-d latent of a host
    bundle = engine.explain(host, p_t, graph)     # STIX 2.1 detection-XAI bundle

Semantics match train_tgn.py exactly: identical feature transform (saved
means/stds/clip), node ids from the training ip_to_id map, timestamps rebased to
the training t0, micro-batched update_memory_events -> model(dst) -> p_t.
The checkpoint carries the TGN memory state from the end of training, so live
inference continues the learned temporal context instead of starting cold.

Detection XAI is derived from the model's own outputs (no fixed numbers):
  structural  2-hop neighbourhood of the host in Role 1's live graph; each edge is
              scored by the TGN threat scores of its endpoints, sqrt(p_u * p_v);
              the top-k edges are reported.
  temporal    the host's recent TGN threat-score trajectory; onset = first event at
              or above the calibrated decision threshold (AttentionRolloutTemporal
              .identify_attack_onset), duration = last event - onset.
"""

import math
import threading
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch

from src.role2_ai.tgn_model import XTGAIATGN, XTGAIATGNLegacy
from src.role2_ai.xai_bundler import XAIBundler
from src.role2_ai.xai_temporal import AttentionRolloutTemporal

STRUCTURAL_EXPLAINER = "TGN node-score 2-hop attribution"
TEMPORAL_EXPLAINER = "TGN threat-score trajectory"


class TGNInferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu", trace_len: int = 8, top_k: int = 5,
                 max_explain_edges: int = 400):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if not isinstance(ckpt, dict) or ckpt.get("format") != "xtgaia-tgn-v1":
            raise ValueError(f"{checkpoint_path} is not an xtgaia-tgn-v1 checkpoint; retrain with "
                             "src/role2_ai/train_tgn.py (it now saves weights + metadata together)")
        meta = ckpt.get("feature_meta")
        if not meta:
            raise ValueError("checkpoint has no feature metadata; run preprocess_cicids.py before training")
        self.meta = meta
        # Algorithm 1: a persisted 512-byte memory is only meaningful for the weights that wrote it
        import hashlib
        h = hashlib.sha1()
        for k in sorted(ckpt["state_dict"]):
            h.update(k.encode()); h.update(ckpt["state_dict"][k].detach().cpu().numpy().tobytes())
        self.fingerprint = h.hexdigest()[:16]
        self.num_nodes = int(ckpt["num_nodes"])
        self.architecture = ckpt.get("architecture") or "legacy"
        common = dict(num_nodes=self.num_nodes, raw_msg_dim=int(ckpt["raw_msg_dim"]),
                      memory_dim=int(ckpt["memory_dim"]), latent_z_dim=int(ckpt["latent_z_dim"]),
                      num_heads=int(ckpt["num_heads"]))
        if self.architecture == XTGAIATGN.ARCHITECTURE:
            self.model = XTGAIATGN(**common, k1=int(ckpt.get("k1") or 10), k2=int(ckpt.get("k2") or 3),
                                   num_classes=int(ckpt.get("num_classes") or 0),
                                   memory_skip=ckpt["state_dict"]["z_projector.0.weight"].shape[1]
                                   == 2 * int(ckpt["memory_dim"]),
                                   category_msg=("category_head.0.weight" in ckpt["state_dict"] and
                                                 ckpt["state_dict"]["category_head.0.weight"].shape[1]
                                                 > int(ckpt["memory_dim"])),
                                   flow_head="flow_head.0.weight" in ckpt["state_dict"]).to(self.device)
        else:
            self.model = XTGAIATGNLegacy(**common).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.classes = list(ckpt.get("classes") or [])
        self.last_category_probs = None              # per-flow category probabilities of the last batch
        self.last_flow_probs = None                  # per-flow scores of the optional flow head (design check B10)
        # Tier 1/2 explainers (paper Algorithm 3): PGExplainer + temporal attention rollout
        self.explainer = None
        if self.architecture == XTGAIATGN.ARCHITECTURE:
            from src.role2_ai.explainers import TGNExplainer
            from src.role2_ai.xai_structural import PGExplainerStructural
            pg = PGExplainerStructural(node_feat_dim=int(ckpt["memory_dim"]))
            if ckpt.get("pgexplainer"):
                pg.load_state_dict(ckpt["pgexplainer"])
            pg.eval()
            self.explainer = TGNExplainer(self.model, pg)
        self.threshold = float(ckpt["threshold"])
        self.flow_readout = ckpt.get("flow_readout") or "dst"     # flow projection of node p_t (B9 / B11)
        self.test_metrics = ckpt.get("test_metrics", {})

        self.features: List[str] = list(meta["features"])
        self._mean = torch.tensor([meta["means"][c] for c in self.features], dtype=torch.float32)
        self._std = torch.tensor([meta["stds"][c] for c in self.features], dtype=torch.float32)
        self._clip = float(meta.get("clip", 5.0))
        self._transform = meta.get("transform")
        self.t0 = float(meta["t0"])
        self.ip_to_id: Dict[str, int] = dict(meta["ip_to_id"])
        self.id_to_ip: Dict[int, str] = {v: k for k, v in self.ip_to_id.items()}
        self._next_free = int(meta["num_known_nodes"])
        self.overflow_id = self.num_nodes - 1
        self.unseen_hosts = 0

        self.trace_len, self.top_k, self.max_explain_edges = trace_len, top_k, max_explain_edges
        self.history: Dict[str, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=self.trace_len))
        self.bundler = XAIBundler()
        self.rollout = AttentionRolloutTemporal()
        self.lock = threading.RLock()
        self.events = 0
        # Algorithm 1: hosts whose latent memory m_v(t) is initialised in this process
        self.live: set = set()
        self.memory_dim = int(ckpt["memory_dim"])

    # ------------------------------------------------------------------ preprocessing
    def node_id(self, ip: str) -> int:
        nid = self.ip_to_id.get(ip)
        if nid is None:
            if self._next_free < self.overflow_id:
                nid = self._next_free
                self._next_free += 1
            else:
                nid = self.overflow_id          # reserve exhausted: shared overflow slot
            self.ip_to_id[ip] = nid
            self.id_to_ip.setdefault(nid, ip)
            self.unseen_hosts += 1
        return nid

    def encode_features(self, rows: Sequence[Sequence[float]]) -> torch.Tensor:
        x = torch.tensor(np.asarray(rows, dtype=np.float32).reshape(len(rows), len(self.features)))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if self._transform == "signed_log1p":          # heavy-tailed flow statistics (same as training)
            x = torch.sign(x) * torch.log1p(torch.abs(x))
        return torch.clamp((x - self._mean) / self._std, -self._clip, self._clip).to(self.device)

    def _t(self, timestamps: Sequence[float], ids: torch.Tensor) -> torch.Tensor:
        t = torch.tensor([max(0, int(ts - self.t0)) for ts in timestamps], dtype=torch.long, device=self.device)
        # TGNMemory expects non-decreasing time per node
        return torch.maximum(t, self.model.memory.last_update[ids])

    # ------------------------------------------------------------------ Algorithm 1 latent state
    def needs_hydration(self, hosts) -> List[str]:
        return [h for h in hosts if h not in self.live]

    def mark_live(self, hosts) -> None:
        self.live.update(hosts)

    def hydrate(self, states: Dict[str, Tuple[bytes, float]]) -> int:
        """Restore m_v(t-) from persisted 512-byte float32 arrays (Algorithm 1, lines 8-11).
        Hosts without a record keep their checkpoint memory (zero for never-seen hosts)."""
        n = 0
        with self.lock, torch.no_grad():
            for ip, (blob, ts) in states.items():
                vec = np.frombuffer(bytes(blob), dtype="<f4")
                if vec.size != self.memory_dim:
                    continue
                nid = self.node_id(ip)
                self.model.memory.memory[nid] = torch.from_numpy(vec.copy()).to(self.device)
                self.model.memory.last_update[nid] = max(0, int(float(ts or self.t0) - self.t0))
                n += 1
        return n

    def export_latent(self, hosts) -> List[Dict[str, Any]]:
        """Current m_v(t) of the given hosts as 512-byte float32 arrays for Neo4j."""
        rows = []
        with self.lock:
            for ip in hosts:
                nid = self.ip_to_id.get(ip)
                if nid is None or nid == self.overflow_id:
                    continue
                vec = self.model.memory.memory[nid].detach().cpu().numpy().astype("<f4")
                rows.append({"ip": ip, "state": vec.tobytes(),
                             "ts": float(self.model.memory.last_update[nid].item()) + self.t0})
        return rows

    # ------------------------------------------------------------------ inference
    def observe_batch(self, edges: Sequence[Dict[str, Any]]) -> List[float]:
        """Update TGN memory with a micro-batch of active edges (Role 1 contract) and
        return the threat probability of each flow (p_t of its destination node)."""
        if not edges:
            return []
        with self.lock, torch.no_grad():
            src = torch.tensor([self.node_id(e["source"]) for e in edges], dtype=torch.long, device=self.device)
            dst = torch.tensor([self.node_id(e["target"]) for e in edges], dtype=torch.long, device=self.device)
            msg = self.encode_features([e.get("features") or [0.0] * len(self.features) for e in edges])
            t = self._t([float(e["timestamp"]) for e in edges], dst)
            self.model.update_memory_events(src, dst, t, msg)
            has_cat = getattr(self.model, "category_head", None) is not None
            has_flow = getattr(self.model, "flow_head", None) is not None
            if has_cat or has_flow:
                p, _, _, nb = self.model.forward_full(dst)
                if has_cat:
                    self.last_category_probs = torch.softmax(self.model.category_logits(nb["h"], msg), -1).cpu().numpy()
                if has_flow:
                    nb_s = self.model.forward_full(src)[3]
                    self.last_flow_probs = torch.sigmoid(self.model.flow_logits(nb_s, nb, msg)).cpu().numpy()
            else:
                p, _ = self.model(dst)
            probs = p.squeeze(-1).cpu().numpy().astype(float).tolist()
            for e, pv in zip(edges, probs):
                ts = float(e["timestamp"])
                self.history[e["target"]].append((ts, pv))
                self.history[e["source"]].append((ts, pv))
            self.events += len(edges)
        return probs

    def score_nodes(self, hosts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        with self.lock, torch.no_grad():
            ids = torch.tensor([self.node_id(h) for h in hosts], dtype=torch.long, device=self.device)
            p, z = self.model(ids)
        return p.squeeze(-1).cpu().numpy(), z.cpu().numpy().astype(np.float32)

    def classify(self, host: str, p_t: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Hierarchical multi-class output (attack_classes): Benign below ω, else the most likely attack
        category from the Z_t head. None for checkpoints trained without the head."""
        if getattr(self.model, "category_head", None) is None or not self.classes:
            return None
        with self.lock, torch.no_grad():
            ids = torch.tensor([self.node_id(host)], dtype=torch.long, device=self.device)
            p, _, _, nb = self.model.forward_full(ids)
            q = torch.softmax(self.model.category_logits(nb["h"], self.model.last_message(ids)), -1)[0].cpu().numpy()
        p_t = float(p.item()) if p_t is None else float(p_t)
        if p_t < self.threshold:
            return {"category": "Benign", "confidence": round(1.0 - p_t, 4)}
        k = int(np.argmax(q[1:]) + 1)
        return {"category": self.classes[k], "confidence": round(float(q[k] / max(q[1:].sum(), 1e-9)), 4)}

    def score(self, host: str) -> Tuple[float, np.ndarray]:
        p, z = self.score_nodes([host])
        return float(p[0]), z[0]

    def latest_scores(self, hosts) -> Dict[str, float]:
        """p_t(v) for Algorithm 2: the most recent TGN threat score involving each node."""
        out = {}
        for h in hosts:
            hist = self.history.get(h)
            if hist:
                out[h] = float(hist[-1][1])
        return out

    def trend(self, host: str) -> Tuple[float, int]:
        """(mean, count) of the host's recent TGN threat scores (same trace the temporal XAI uses)."""
        hist = self.history.get(host)
        if not hist:
            return 0.0, 0
        return float(sum(p for _, p in hist) / len(hist)), len(hist)

    # ------------------------------------------------------------------ XAI
    def structural_attribution(self, host: str, graph: nx.DiGraph) -> List[Tuple[str, str, float]]:
        if graph is None or host not in graph:
            return []
        hood = nx.ego_graph(graph.to_undirected(as_view=True), host, radius=2)
        edges = [(u, v, d.get("timestamp", 0.0)) for u, v, d in graph.subgraph(hood.nodes).edges(data=True)]
        edges = sorted(edges, key=lambda e: e[2], reverse=True)[: self.max_explain_edges]
        if not edges:
            return []
        nodes = sorted({n for u, v, _ in edges for n in (u, v)})
        p, _ = self.score_nodes(nodes)
        score = dict(zip(nodes, p.tolist()))
        ranked = sorted(((u, v, math.sqrt(max(score[u], 0.0) * max(score[v], 0.0))) for u, v, _ in edges),
                        key=lambda e: e[2], reverse=True)
        return [(u, v, round(w, 4)) for u, v, w in ranked[: self.top_k]]

    def temporal_attribution(self, host: str) -> Dict[str, Any]:
        hist = list(self.history.get(host, []))
        if not hist:
            return {"onset_index": -1, "onset_timestamp_ms": 0.0, "attack_duration_ms": 0.0, "temporal_trace": []}
        t_first = hist[0][0]
        result = self.rollout.identify_attack_onset(torch.tensor([p for _, p in hist]),
                                                    timestamps_ms=[(ts - t_first) * 1000.0 for ts, _ in hist],
                                                    threshold=self.threshold)
        result["trace_start_epoch"] = t_first
        return result

    def explain_tiers(self, host: str) -> Optional[Dict[str, Any]]:
        """Paper Tier 1 (PGExplainer) + Tier 2 (attention rollout) for the paper architecture."""
        if self.explainer is None:
            return None
        with self.lock:
            return self.explainer.explain(self.node_id(host), host, lambda i: self.id_to_ip.get(i, f"node-{i}"),
                                          t0=self.t0)

    def explain(self, host: str, p_t: float, graph: Optional[nx.DiGraph] = None,
                trigger: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tiers = self.explain_tiers(host)
        self.last_tiers = tiers                       # raw Tier 1/2 output (explanation-stability evaluation)
        if tiers is not None:
            extra = {"decision_threshold": round(self.threshold, 4), "fidelity_plus": tiers["fidelity_plus"],
                     "fidelity_minus": tiers["fidelity_minus"], "sparsity": tiers["sparsity"],
                     "n2_edges": tiers["n2_edges"], "explanation_latency_ms": tiers["latency_ms"]}
            if trigger:
                extra["trigger_flow"] = {k: trigger.get(k) for k in ("source", "target", "port", "timestamp")}
            cat = self.classify(host, p_t)
            if cat:
                extra["attack_category"] = cat
            return self.bundler.build_stix_bundle(node_id=host, threat_prob=p_t,
                                                  structural_edges=tiers["structural_edges"],
                                                  temporal_analysis=tiers["temporal"], explainer_model="PGExplainer",
                                                  rollout_method="AttentionRollout", extra=extra)
        temporal = self.temporal_attribution(host)
        extra = {"decision_threshold": round(self.threshold, 4), "events_in_trace": len(temporal["temporal_trace"])}
        if trigger:
            extra["trigger_flow"] = {k: trigger.get(k) for k in ("source", "target", "port", "timestamp")}
        return self.bundler.build_stix_bundle(
            node_id=host, threat_prob=p_t, structural_edges=self.structural_attribution(host, graph),
            temporal_analysis=temporal, explainer_model=STRUCTURAL_EXPLAINER,
            rollout_method=TEMPORAL_EXPLAINER, extra=extra)

    # ------------------------------------------------------------------ Person 2 contract
    def predict_threat_and_explain(self, graph_tensors: Optional[Dict[str, Any]] = None):
        """(p_t, Z_t, xai_bundle) for graph_tensors = {"host": ip, "graph": live graph}."""
        if not graph_tensors or "host" not in graph_tensors:
            raise ValueError("graph_tensors must be {'host': <ip>, 'graph': <nx.DiGraph>}")
        host = graph_tensors["host"]
        p_t, z_t = self.score(host)
        return p_t, torch.as_tensor(z_t), self.explain(host, p_t, graph_tensors.get("graph"))


_engine: Optional[TGNInferenceEngine] = None


def set_inference_engine(engine: Optional[TGNInferenceEngine]) -> None:
    global _engine
    _engine = engine


def get_inference_engine() -> Optional[TGNInferenceEngine]:
    return _engine
