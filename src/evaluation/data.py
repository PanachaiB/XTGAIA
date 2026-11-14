"""
Corrected CICIDS2017 (Engelen et al., WTMC 2021 / Liu et al., CNS 2022; "CICIDS2017_improved")
loading and evaluation splits.

Dataset notes
  * Flows were regenerated from the original PCAPs with a fixed CICFlowMeter and relabelled;
    timestamps have microsecond resolution.
  * "Attempted" flows (Attempted Category != -1) are attack flows without malicious effect.
    Following the authors' recommendation they are relabelled BENIGN and never used as a class.

Protocols (split3 returns train / validation / test; validation picks thresholds)
  random   stratified random 70/30 over all days (legacy, no validation set)
  pcic_a   P-CIC split A, parity with Vedesh et al. [3]: stratified random 80/20; 10% of the
           training part (stratified) is validation; benign flows are downsampled to the number of
           malicious flows in the training set only (validation and test keep the real distribution)
  chrono   P-CIC split B / CCSTGN [11] protocol: chronological 70/10/20 over all days, no shuffle
  heldout  train Monday-Thursday (last 15% of each training day is validation), test Friday
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


CIC2018_ZIP = "data/raw/cicids2018_improved/CSECICIDS2018_improved.zip"


def load_cicids2018_sampled(zip_path: str = CIC2018_ZIP, benign_frac: float = 0.005, attack_cap: int = 5000,
                            seed: int = 0, chunksize: int = 500000) -> pd.DataFrame:
    """CSE-CIC-IDS2018 improved (same regenerated CICFlowMeter schema as the 2017 corrected set), streamed
    from the authors' zip (~36 GB of CSV, never extracted). Benign flows are Bernoulli-sampled; each
    attack label keeps an approximately uniform reservoir of at most `attack_cap` flows per day, so
    attacks are drawn from the whole day rather than its beginning. Only the TGN columns are read."""
    import io
    import zipfile
    rng = np.random.default_rng(seed)
    feat_cols = list(FLOW_FEATURE_MAP.values())
    keep = ["Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Label", "Attempted Category"] + feat_cols
    days = []
    with zipfile.ZipFile(zip_path) as z:
        for name in sorted(n for n in z.namelist() if n.endswith(".csv")):
            benign_parts, reservoirs, seen = [], {}, {}
            with z.open(name) as f:
                for chunk in pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), usecols=keep, chunksize=chunksize,
                                         low_memory=False):
                    attempted = chunk["Attempted Category"].fillna(-1).astype(int) != -1
                    label = chunk["Label"].astype(str).str.strip().where(~attempted, "BENIGN")
                    chunk = chunk.assign(Label=label.values)
                    ben = chunk[label.str.upper() == "BENIGN"]
                    benign_parts.append(ben[rng.random(len(ben)) < benign_frac])
                    for lab, grp in chunk[label.str.upper() != "BENIGN"].groupby("Label"):
                        old = reservoirs.get(lab)
                        n_old, g = seen.get(lab, 0), len(grp)
                        total = n_old + g
                        size = min(attack_cap, total)
                        k_new = int(round(size * g / total))
                        new = grp.sample(min(k_new, g), random_state=int(rng.integers(1 << 31)))
                        if old is not None and size - len(new) > 0:
                            old = old.sample(min(size - len(new), len(old)), random_state=int(rng.integers(1 << 31)))
                            new = pd.concat([old, new])
                        reservoirs[lab], seen[lab] = new, total
            day = pd.concat(benign_parts + list(reservoirs.values()))
            day_name = name.rsplit("/", 1)[-1][:-4]
            out = pd.DataFrame({
                "src": day["Src IP"].astype(str).values, "dst": day["Dst IP"].astype(str).values,
                "src_port": day["Src Port"].astype(np.int64).values, "dst_port": day["Dst Port"].astype(np.int64).values,
                "ts": pd.to_datetime(day["Timestamp"], format="mixed").astype("int64").values / 1e9,
                "label": day["Label"].values, "day": day_name,
            })
            out["y"] = (out["label"].str.upper() != "BENIGN").astype(np.int8)
            fe = day[feat_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            for c in feat_cols:
                out[f"f::{c}"] = fe[c].astype(np.float32).values
            days.append(out)
            print(f"[2018] {day_name}: kept {len(out):,} flows ({int(out['y'].sum()):,} attacks)", flush=True)
    return pd.concat(days, ignore_index=True).sort_values("ts", kind="mergesort").reset_index(drop=True)


UNSW_PATH = "data/raw/nf_unsw_nb15_v2/data/NF-UNSW-NB15-v2.csv"


def load_nf_unsw(path: str = UNSW_PATH) -> pd.DataFrame:
    """NF-UNSW-NB15-v2 (Sarhan et al.; 2,390,275 NetFlow v2 flows, 3.98 % attack), the dataset of
    CCSTGN [11]. It has no timestamp column, so - as in [11] - the flow index is used as time."""
    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "src": df["IPV4_SRC_ADDR"].astype(str), "dst": df["IPV4_DST_ADDR"].astype(str),
        "src_port": df["L4_SRC_PORT"].astype(np.int64), "dst_port": df["L4_DST_PORT"].astype(np.int64),
        "ts": np.arange(len(df), dtype=np.float64), "label": df["Attack"].astype(str).str.strip().values,
        "day": "unsw",
    })
    out["y"] = df["Label"].astype(np.int8).values
    feats = df.drop(columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack"])
    feats = feats.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = feats.astype(np.float32)
    feats.columns = [f"f::{c}" for c in feats.columns]
    return pd.concat([out, feats.reset_index(drop=True)], axis=1)


def load_dataset(name: str) -> pd.DataFrame:
    if name == "nf_unsw_nb15_v2":
        return load_nf_unsw()
    if name == "cicids2017_improved":
        return load()
    raise ValueError(f"unknown dataset {name!r}")


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


def _stratified_mask(labels: pd.Series, frac: float, rng) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    for _, idx in labels.groupby(labels.values).indices.items():
        k = int(round(len(idx) * frac))
        if k:
            mask[rng.choice(idx, size=k, replace=False)] = True
    return mask


def split3(df: pd.DataFrame, protocol: str, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, validation, test) for the paper-comparison protocols."""
    rng = np.random.default_rng(seed)
    if protocol == "pcic_a":
        test_mask = _stratified_mask(df["label"].reset_index(drop=True), 0.20, rng)
        rest, test = df[~test_mask], df[test_mask]
        val_mask = _stratified_mask(rest["label"].reset_index(drop=True), 0.10, rng)
        train, val = rest[~val_mask], rest[val_mask]
        benign, attack = train[train["y"] == 0], train[train["y"] == 1]
        if len(benign) > len(attack):
            benign = benign.sample(len(attack), random_state=seed)
        train = pd.concat([benign, attack]).sort_values("ts", kind="mergesort")
        return train, val, test
    if protocol == "chrono":
        n = len(df)
        return df.iloc[: int(n * 0.70)], df.iloc[int(n * 0.70): int(n * 0.80)], df.iloc[int(n * 0.80):]
    if protocol == "heldout":
        tr = df[df["day"] != "friday"]
        val_mask = np.zeros(len(tr), dtype=bool)
        for _, idx in tr.reset_index(drop=True).groupby("day").indices.items():
            idx = np.sort(idx)
            val_mask[idx[int(len(idx) * 0.85):]] = True
        return tr[~val_mask], tr[val_mask], df[df["day"] == "friday"]
    raise ValueError(f"unknown protocol {protocol!r}")


def stratified_subsample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """At most n rows, keeping every class's share (used only where memory requires it)."""
    if len(df) <= n:
        return df
    frac = n / len(df)
    parts = [g.sample(max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby("label")]
    return pd.concat(parts).sort_values("ts", kind="mergesort")
