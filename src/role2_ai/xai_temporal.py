import time
import torch
import torch.nn as nn
import numpy as np

class AttentionRolloutTemporal(nn.Module):
    """
    XTGAIA Stage 6: Temporal XAI Core using Attention Rollout.
    Implements: T_trace = \prod (0.5 * A^(l) + 0.5 * I)
    to trace historical attack onset and decay over time windows.
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, attention_matrices: list[torch.Tensor]) -> torch.Tensor:
        """
        คำนวณ Attention Rollout ข้ามเลเยอร์ของโมเดล
        Args:
            attention_matrices: List ของ Attention Weight Tensors แต่ละเลเยอร์ [batch_size, seq_len, seq_len]
        Returns:
            rollout: [seq_len, seq_len] เมทริกซ์สะสมการไหลของ Temporal Attention
        """
        if not attention_matrices:
            return torch.empty(0)

        # ใช้ Tensor ตัวแรกเพื่อเตรียมขนาดและ Identity Matrix
        device = attention_matrices[0].device
        seq_len = attention_matrices[0].size(-1)
        identity = torch.eye(seq_len, device=device)

        # เริ่มต้นที่ Identity Matrix
        rollout = identity.clone()

        for A in attention_matrices:
            # เฉลี่ยข้ามมิติ Heads หรือ Batch หากมีหลายมิติ
            if A.dim() == 4:  # [batch, heads, seq_len, seq_len]
                A = A.mean(dim=1).squeeze(0)
            elif A.dim() == 3:  # [batch, seq_len, seq_len]
                A = A.squeeze(0)

            # Normalization ค่าน้ำหนักแถว
            A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)

            # สูตร T_trace: (0.5 * A^(l) + 0.5 * I)
            A_aug = self.alpha * A + (1.0 - self.alpha) * identity
            rollout = torch.matmul(A_aug, rollout)

        return rollout

    def identify_attack_onset(
        self,
        temporal_trace: torch.Tensor,
        timestamps_ms: list[float] | None = None,
        threshold: float = 0.15
    ) -> dict:
        """
        วิเคราะห์หาช่วงเวลาที่เริ่มเกิดการโจมตี (Attack Onset Time)
        Args:
            temporal_trace: เวกเตอร์น้ำหนักความสำคัญเชิงเวลา [seq_len]
            timestamps_ms: รายการ Timestamps ของแต่ละ Event (มิลลิวินาที)
            threshold: ระดับความสำคัญขั้นต่ำที่นับว่าเริ่มมีความผิดปกติ
        Returns:
            dict สรุปข้อมูล Attack Onset และ Time-decay weight vector
        """
        trace_list = temporal_trace.tolist() if isinstance(temporal_trace, torch.Tensor) else list(temporal_trace)
        seq_len = len(trace_list)

        if seq_len == 0:
            return {"onset_index": -1, "onset_timestamp_ms": 0.0, "temporal_trace": []}

        if timestamps_ms is None or len(timestamps_ms) != seq_len:
            # จำลองไทม์สแตมป์ช่วง 500 ms inference window
            timestamps_ms = [i * (500.0 / max(1, seq_len)) for i in range(seq_len)]

        # ค้นหาจุดเวลาแรกที่ค่าน้ำหนัก Attention สูงเกินเกณฑ์ (Earliest Significant Onset)
        onset_idx = 0
        for i, weight in enumerate(trace_list):
            if weight >= threshold:
                onset_idx = i
                break

        onset_timestamp = timestamps_ms[onset_idx]
        duration_ms = timestamps_ms[-1] - onset_timestamp

        return {
            "onset_index": onset_idx,
            "onset_timestamp_ms": round(onset_timestamp, 2),
            "attack_duration_ms": round(duration_ms, 2),
            "temporal_trace": [round(w, 4) for w in trace_list]
        }


if __name__ == "__main__":
    # การทดสอบการทำงานเบื้องต้น (Self-Test พร้อมจับเวลา Execution Latency)
    start_time = time.perf_counter()
    rollout_engine = AttentionRolloutTemporal(alpha=0.5)

    # จำลอง Attention Weights 2 เลเยอร์ ขนาด 8x8 (8 time slots ใน 500ms window)
    seq_len = 8
    dummy_attn1 = torch.softmax(torch.randn(1, seq_len, seq_len), dim=-1)
    dummy_attn2 = torch.softmax(torch.randn(1, seq_len, seq_len), dim=-1)

    # 1. รัน Attention Rollout
    accumulated_rollout = rollout_engine([dummy_attn1, dummy_attn2])
    
    # ดึงเวกเตอร์ผลกระทบไปยังจังหวะเวลาปัจจุบัน (แถวสุดท้าย)
    final_temporal_weights = accumulated_rollout[-1]

    # 2. วิเคราะห์หาจังหวะ Attack Onset
    result = rollout_engine.identify_attack_onset(final_temporal_weights)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"[✓] Attention Rollout Matrix ({seq_len}x{seq_len}) computed.")
    print(f"[✓] Attack Onset Detected at: {result['onset_timestamp_ms']} ms (Index {result['onset_index']})")
    print(f"[✓] Temporal Trace Weights: {result['temporal_trace']}")
    print(f"[✓] Execution Time: {elapsed_ms:.2f} ms (Budget: <50.0 ms)")