import torch
import torch.nn as nn

class PGExplainerStructural(nn.Module):
    """
    XTGAIA Stage 6: Structural XAI Core using PGExplainer edge masking.
    Identifies influential 2-hop graph structures associated with threat predictions.
    """
    def __init__(self, node_feat_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        # เครือข่ายคำนวณค่าน้ำหนักความสำคัญของแต่ละ Edge ใน Subgraph
        self.edge_mask_net = nn.Sequential(
            nn.Linear(node_feat_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        คำนวณ Mask ความสำคัญของเส้นเชื่อม (Edge Importance Mask)
        Args:
            node_embeddings: [N, node_feat_dim] เวกเตอร์ของแต่ละโหนด
            edge_index: [2, E] รายการเส้นเชื่อมใน Subgraph 2-hop
        Returns:
            edge_mask: [E] ค่าน้ำหนักความสำคัญใน [0.0, 1.0]
        """
        if edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.float32)

        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]

        # รวมคุณลักษณะของคู่โหนดต้นทางและปลายทาง
        edge_features = torch.cat([node_embeddings[src_nodes], node_embeddings[dst_nodes]], dim=-1)
        
        # ส่งผ่านเครือข่ายเพื่อสร้าง Mask [0, 1]
        edge_mask = self.edge_mask_net(edge_features).squeeze(-1)
        return edge_mask

    def extract_top_subgraph(
        self,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        top_k: int = 5
    ) -> list[tuple[int, int, float]]:
        """
        คัดเลือกเส้นทางเชื่อมต่อที่น่าสงสัยที่สุด top_k อันดับแรกสำหรับส่งต่อเป็นหลักฐาน
        """
        if edge_index.numel() == 0 or edge_mask.numel() == 0:
            return []

        k = min(top_k, edge_mask.size(0))
        top_values, top_indices = torch.topk(edge_mask, k=k)

        subgraph_edges = []
        for val, idx in zip(top_values, top_indices):
            u = edge_index[0, idx].item()
            v = edge_index[1, idx].item()
            subgraph_edges.append((u, v, round(val.item(), 4)))

        return subgraph_edges


if __name__ == "__main__":
    # การทดสอบการทำงานเบื้องต้น (Self-Test)
    explainer = PGExplainerStructural(node_feat_dim=128)
    dummy_emb = torch.randn(10, 128)
    dummy_edges = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=torch.long)
    
    mask = explainer(dummy_emb, dummy_edges)
    top_edges = explainer.extract_top_subgraph(dummy_edges, mask, top_k=2)
    print(f"[✓] Structural mask successfully generated: {mask.tolist()}")
    print(f"[✓] Top influential edges: {top_edges}")