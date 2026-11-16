"""Validation-only tuning of the original TGN (node-only design), per docs/tuning_protocol.md (frozen).

Resumable state machine (runs/tune/state.json). Run inside xtgaia-core:
  MSYS_NO_PATHCONV=1 docker compose -f docker/docker-compose.yml run --rm --no-deps -T -e TZ=Etc/GMT-7 \
    -e PYTHONUNBUFFERED=1 -e XTGAIA_TORCH_THREADS=14 -e PYTHONPATH=/app:/app/data/pylib \
    xtgaia-core python scripts/tune_tgn.py > runs/tune/driver.log 2>&1
Restart after an interruption with -e XTGAIA_RESUME=1 (finished proxies are skipped; the proxy in progress
continues from its per-epoch checkpoint).

Stages: proxy table (UNSW without test rows) -> round 1 proxies (3 epochs, seed 0) -> selection -> optional
combination proxy -> final full runs via tgn_eval (UNSW chrono, CICIDS held-out; 10 epochs; the only test
access) -> evaluation report.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUT = os.getenv("TUNE_OUT", "runs/tune")
TABLE = "data/processed/eval/tgnv2_nf_un_chrono_train_s0.csv"
META = "data/processed/eval/tgnv2_nf_un_chrono_meta_s0.json"
PROXY = os.path.join(OUT, "unsw_proxy_no_test.csv")
STATE = os.path.join(OUT, "state.json")
DEVIATIONS = os.getenv("TUNE_DEVIATIONS", "docs/paper_deviations.md")
EPOCHS = int(os.getenv("TUNE_PROXY_EPOCHS", "3"))
FINAL_EPOCHS = int(os.getenv("TUNE_FINAL_EPOCHS", "10"))
LIMIT = int(os.getenv("TUNE_LIMIT", "0") or 0)              # smoke tests only: every k-th row
SKIP_FINALS = os.getenv("TUNE_SKIP_FINALS", "0") == "1"     # smoke tests only
KNOBS = ("XTGAIA_FLOW_READOUT", "XTGAIA_BATCH", "XTGAIA_K1", "XTGAIA_POS_WEIGHT_SCALE", "XTGAIA_LR")
ROUND1 = [("base", {}),
          ("readout_max", {"XTGAIA_FLOW_READOUT": "max"}),
          ("batch64", {"XTGAIA_BATCH": "64"}),
          ("k1_20", {"XTGAIA_K1": "20"}),
          ("pw_half", {"XTGAIA_POS_WEIGHT_SCALE": "0.5"}),
          ("lr_1e-4", {"XTGAIA_LR": "1e-4"})]


def load():
    if os.path.exists(STATE):
        if os.getenv("XTGAIA_RESUME", "0") != "1":
            sys.exit(f"[tune] {STATE} exists; restart with XTGAIA_RESUME=1")
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "proxies": {}, "log": []}


def save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, default=float)
    os.replace(tmp, STATE)


def log(st, title, text):
    entry = f"- **{time.strftime('%Y-%m-%d %H:%M')} - {title}.** {text}"
    st["log"].append(entry)
    save(st)
    print(f"[tune] {title}: {text}", flush=True)
    with open(DEVIATIONS, "a", encoding="utf-8") as f:
        f.write("\n" + entry + "\n")


def proxy_table():
    if os.path.exists(PROXY):
        return
    df = pd.read_csv(TABLE)
    df = df[df["split"] != "test"]                      # guard 1: no test rows anywhere in tuning
    if LIMIT and LIMIT < len(df):
        df = df.iloc[:: int(np.ceil(len(df) / LIMIT))]
    df.to_csv(PROXY, index=False)
    print(f"[tune] proxy table {len(df):,} rows (train + val)", flush=True)


def run_proxy(st, name, knobs):
    if st["proxies"].get(name, {}).get("done"):
        return st["proxies"][name]
    import torch
    from src.role2_ai.train_tgn import train_streaming
    import design_check_c3 as c3
    for k in KNOBS:
        os.environ.pop(k, None)
    os.environ.update(knobs)
    os.environ["XTGAIA_CATEGORY_HEAD"] = "1"
    os.environ.pop("XTGAIA_FLOW_HEAD", None)            # node-only design: no flow head
    torch.set_num_threads(int(os.getenv("XTGAIA_TORCH_THREADS", "14")))
    t0 = time.time()
    _, pr = train_streaming(PROXY, META, os.path.join(OUT, f"proxy_{name}.pt"), max_epochs=EPOCHS, patience=EPOCHS,
                            seed=0, return_predictions=True, fit_explainer=False)
    wall = time.time() - t0
    vm = pr["split"] == "val"
    y, s = pr["y"][vm], pr["preds"][vm]
    m = c3.view(y, s)
    m["boot_halfwidth"] = c3.bootstrap_halfwidth(y, s, m["threshold"])
    m["noise"] = max(m["boot_halfwidth"], 0.005)
    rec = {"done": True, "knobs": knobs, "val": m, "wall_s": wall, "epochs": EPOCHS}
    st["proxies"][name] = rec
    save(st)
    print(f"[tune] {name}: val F1 {m['f1']:.4f} AUC {m['auc']:.4f} TPR@0.1/1% {m['tpr_at_0.1pct_fpr']:.3f}/"
          f"{m['tpr_at_1pct_fpr']:.3f} ({wall / 60:.0f} min)", flush=True)
    return rec


def winners(st):
    b = st["proxies"]["base"]
    out = []
    for name, rec in st["proxies"].items():
        if name in ("base",) or name.startswith("combo") or not rec.get("done"):
            continue
        noise = max(rec["val"]["noise"], b["val"]["noise"])
        gain = rec["val"]["f1"] - b["val"]["f1"]
        ok = (gain > 2 * noise and rec["val"]["auc"] >= b["val"]["auc"] - 0.001
              and rec["wall_s"] <= 1.5 * b["wall_s"])
        rec["vs_base"] = {"gain": gain, "2x_noise": 2 * noise, "auc_delta": rec["val"]["auc"] - b["val"]["auc"],
                          "time_ratio": rec["wall_s"] / b["wall_s"], "wins": bool(ok)}
        if ok:
            out.append(name)
    save(st)
    return sorted(out, key=lambda n: -st["proxies"][n]["val"]["f1"])


def select(st):
    if "selected" in st:
        return st["selected"]
    win = winners(st)
    rows = "; ".join(f"{n} {r['val']['f1']:.4f} (gain {r.get('vs_base', {}).get('gain', 0):+.4f} vs 2x noise "
                     f"{r.get('vs_base', {}).get('2x_noise', 0):.4f}, AUC {r['val']['auc']:.4f}, "
                     f"{r['wall_s'] / 60:.0f} min)" for n, r in st["proxies"].items())
    log(st, "Round 1 (validation, 3 epochs)", f"base {st['proxies']['base']['val']['f1']:.4f}; {rows}. "
        f"Winners: {', '.join(win) if win else 'none'}")
    if not win:
        chosen, knobs = "base", {}
    elif len(win) == 1:
        chosen, knobs = win[0], st["proxies"][win[0]]["knobs"]
    else:
        knobs = {}
        for n in win:
            knobs.update(st["proxies"][n]["knobs"])
        combo = run_proxy(st, "combo", knobs)
        best = st["proxies"][win[0]]
        if combo["val"]["f1"] >= best["val"]["f1"] - max(best["val"]["noise"], combo["val"]["noise"]):
            chosen = "combo"
        else:
            chosen, knobs = win[0], best["knobs"]
        log(st, "Combination proxy", f"{'+'.join(win)}: val F1 {combo['val']['f1']:.4f} vs best single "
            f"{best['val']['f1']:.4f} -> {chosen}")
    st["selected"] = {"name": chosen, "knobs": knobs}
    log(st, "Selected configuration", f"{chosen} {knobs or '(defaults)'}")
    return st["selected"]


def final_runs(st, sel):
    st.setdefault("finals", {})
    jobs = [("cicids_heldout", ["--procedure", "streaming", "--protocol", "heldout"]),
            ("unsw_chrono", ["--procedure", "streaming", "--dataset", "nf_unsw_nb15_v2", "--protocol", "chrono"])]
    for name, args in jobs:
        if st["finals"].get(name, {}).get("done"):
            continue
        env = {k: v for k, v in os.environ.items() if k not in KNOBS + ("XTGAIA_FLOW_HEAD",)}
        env.update(sel["knobs"])
        env["XTGAIA_VARIANT"] = "evhead" + ("" if sel["name"] == "base" else "-tuned")
        if st["finals"].get(name, {}).get("started"):
            env["XTGAIA_RESUME"] = "1"
        else:
            env.pop("XTGAIA_RESUME", None)
        st["finals"][name] = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "knobs": sel["knobs"]}
        save(st)
        cmd = [sys.executable, "-m", "src.evaluation.tgn_eval", *args, "--seed", "0", "--epochs", str(FINAL_EPOCHS)]
        print(f"[tune] final {name}: {' '.join(cmd)} with {sel['knobs'] or 'defaults'}", flush=True)
        rc = subprocess.call(cmd, env=env)
        st["finals"][name].update({"done": rc == 0, "returncode": rc, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        save(st)
        rec = last_record(name, env["XTGAIA_VARIANT"], sel["knobs"]) if rc == 0 else None
        if rec:
            st["finals"][name]["test"] = {k: rec.get(k) for k in ("f1", "precision", "recall", "fpr", "roc_auc",
                                                                  "tpr_at_fpr_1pct", "tpr_at_fpr_0.1pct")}
            mc = rec.get("multiclass") or {}
            st["finals"][name]["test"]["multiclass_macro_f1"] = mc.get("macro_f1")
            save(st)
            t = st["finals"][name]["test"]
            log(st, f"Final {name} (test, touched once)", f"F1 {t['f1']:.4f}, AUC {t['roc_auc']:.4f}, TPR@1%FPR "
                f"{t['tpr_at_fpr_1pct']:.4f}, TPR@0.1%FPR {t['tpr_at_fpr_0.1pct']:.4f}, multi-class macro-F1 "
                f"{t['multiclass_macro_f1'] if t['multiclass_macro_f1'] is None else round(t['multiclass_macro_f1'], 4)}")


def last_record(name, variant, knobs):
    ds = "nf_unsw_nb15_v2" if name.startswith("unsw") else "cicids2017_improved"
    proto = "chrono" if name.startswith("unsw") else "heldout"
    hit = None
    with open("data/processed/eval/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("dataset") == ds and r.get("protocol") == proto and variant in r.get("model", ""):
                hit = r
    return hit


def summary(st):
    L = ["# TGN tuning summary (node-only design)", "", "Protocol: `docs/tuning_protocol.md`. Validation only "
         "except the final runs (test touched once per dataset).", "",
         "| Proxy | Knobs | Val F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | Noise | Gain vs base | 2×noise | Minutes | Wins |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, r in st["proxies"].items():
        if not r.get("done"):
            continue
        v, vb = r["val"], r.get("vs_base", {})
        L.append(f"| {n} | {r['knobs'] or 'defaults'} | {v['f1']:.4f} | {v['auc']:.4f} | {v['tpr_at_0.1pct_fpr']:.3f} | "
                 f"{v['tpr_at_1pct_fpr']:.3f} | {v['noise']:.4f} | {vb.get('gain', 0):+.4f} | {vb.get('2x_noise', 0):.4f} | "
                 f"{r['wall_s'] / 60:.0f} | {'yes' if vb.get('wins') else ('-' if n == 'base' else 'no')} |")
    if "selected" in st:
        L += ["", f"**Selected:** {st['selected']['name']} {st['selected']['knobs'] or '(defaults)'}", ""]
    for n, r in st.get("finals", {}).items():
        if r.get("test"):
            t = r["test"]
            L.append(f"- Final {n} (test, once): F1 {t['f1']:.4f}, AUC {t['roc_auc']:.4f}, TPR@1%FPR "
                     f"{t['tpr_at_fpr_1pct']:.4f}, TPR@0.1%FPR {t['tpr_at_fpr_0.1pct']:.4f}")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    st = load()
    save(st)
    proxy_table()
    for name, knobs in ROUND1:
        run_proxy(st, name, knobs)
        winners(st) if "base" in st["proxies"] else None
        summary(st)
    sel = select(st)
    summary(st)
    if SKIP_FINALS:
        return
    final_runs(st, sel)
    summary(st)
    subprocess.call([sys.executable, "-m", "src.evaluation.report", "--out", "docs/evaluation_results.md"])
    log(st, "Tuning finished", "report regenerated (docs/evaluation_results.md); summary in runs/tune/summary.md")


if __name__ == "__main__":
    main()
