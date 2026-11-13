"""
Corrected CICIDS2017 (Engelen et al., WTMC 2021 / Liu et al., CNS 2022; "CICIDS2017_improved")
loading and evaluation splits.

Dataset notes
  * Flows were regenerated from the original PCAPs with a fixed CICFlowMeter and relabelled;
    timestamps have microsecond resolution.
  * "Attempted" flows (Attempted Category != -1) are attack flows without malicious effect.
    Following the authors' recommendation they are relabelled BENIGN and never used as a class.

Protocols
  random   stratified random split over all days (comparable with most CICIDS2017 papers;
           optimistic because temporally adjacent flows land in train and test)
  heldout  train Monday-Thursday, test Friday (no temporal leakage; Bot, PortScan, DDoS)
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
DEFAULT_DIR = "data/raw/cicids2017_improved"

IDENTIFIERS = ["id", "Flow ID", "Src IP", "Src Port", "Dst IP", "Timestamp", "Label", "Attempted Category"]

# Role 1 FLOW_FEATURES (original CICIDS2017 names) -> corrected-dataset column names
FLOW_FEATURE_MAP = {
    "Flow Duration": "Flow Duration", "Total Fwd Packets": "Total Fwd Packet",
    "Total Backward Packets": "Total Bwd packets", "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet", "Fwd Packet Length Mean": "Fwd Packet Length Mean",
    "Bwd Packet Length Mean": "Bwd Packet Length Mean", "Flow Bytes/s": "Flow Bytes/s",
    "Flow Packets/s": "Flow Packets/s", "Flow IAT Mean": "Flow IAT Mean", "Fwd IAT Mean": "Fwd IAT Mean",
    "SYN Flag Count": "SYN Flag Count", "PSH Flag Count": "PSH Flag Count", "ACK Flag Count": "ACK Flag Count",
    "Init_Win_bytes_forward": "FWD Init Win Bytes", "Init_Win_bytes_backward": "Bwd Init Win Bytes",
}


def load_day(day: str, data_dir: str = DEFAULT_DIR) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, f"{day}.csv"), low_memory=False)
    df.columns = df.columns.str.strip()
    attempted = df["Attempted Category"].fillna(-1).astype(int) != -1
    label = df["Label"].astype(str).str.strip()
    label = label.where(~attempted, "BENIGN")
    out = pd.DataFrame({
        "src": df["Src IP"].astype(str), "dst": df["Dst IP"].astype(str),
        "src_port": df["Src Port"].astype(np.int64), "dst_port": df["Dst Port"].astype(np.int64),
        "ts": pd.to_datetime(df["Timestamp"], format="mixed").astype("int64") / 1e9,
        "label": label.values, "day": day,
    })
    out["y"] = (out["label"].str.upper() != "BENIGN").astype(np.int8)
    feats = df.drop(columns=[c for c in IDENTIFIERS if c in df.columns])
    feats = feats.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = feats.astype(np.float32)
    feats.columns = [f"f::{c}" for c in feats.columns]
    return pd.concat([out, feats.reset_index(drop=True)], axis=1)


def load(days: List[str] = DAYS, data_dir: str = DEFAULT_DIR) -> pd.DataFrame:
    df = pd.concat([load_day(d, data_dir) for d in days], ignore_index=True)
    return df.sort_values("ts", kind="mergesort").reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("f::")]


def split(df: pd.DataFrame, protocol: str, seed: int, test_size: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if protocol == "heldout":
        return df[df["day"] != "friday"], df[df["day"] == "friday"]
    if protocol == "random":
        rng = np.random.default_rng(seed)
        test_mask = np.zeros(len(df), dtype=bool)
        for _, idx in df.groupby("label").indices.items():       # stratified by attack class
            n_test = int(round(len(idx) * test_size))
            test_mask[rng.choice(idx, size=n_test, replace=False)] = True
        return df[~test_mask], df[test_mask]
    raise ValueError(f"unknown protocol {protocol!r}")


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most n rows, keeping every class's share (used only where memory requires it)."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("ts", kind="mergesort")
