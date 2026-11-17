# XTGAIA live demo: presenter guide

A live, hand-typed demo of the full system. You type network traffic at a prompt and every stage reacts on
screen:
- **Role 1:** STIX normalisation, the live security knowledge graph, ATT&CK mapping.
- **Role 2:** TGN threat score p_t, attack category, topology-aware risk.
- **Role 3:** zone Double DQN response.
- **The three XAI tiers.**
- **The SOAR dispatch:** Shuffle and a TheHive case.
- **The closed loop and the analyst rollback.**

The flows use the statistics of real CICIDS2017 rows (benign, port scan, DDoS, botnet). The pipeline code is
the same as the production run (`main.py`).

Run every command from the repository root. In PowerShell, use the commands as written. In Git Bash, prefix
each `docker compose ... run` command with `MSYS_NO_PATHCONV=1 `.

---

## 1. One-time setup

### TheHive analyst account (needed to see the cases)

The default `admin@thehive.local` account belongs to TheHive's *admin* organisation. That organisation
manages the platform but cannot see an organisation's cases. Create a user inside **XTGAIA**:

1. Log in at http://localhost:9000 as the admin account. The default credentials are in the TheHive
   documentation; this project's credentials are in `.env`, under `THEHIVE_ADMIN_USERNAME` /
   `THEHIVE_ADMIN_PASSWORD`.
2. Go to **Organizations**, then **XTGAIA**, then **Users**, then **+**. Use login e.g.
   `analyst@xtgaia.local` and profile **org-admin** (or **analyst**).
3. In the user's menu, choose **Set a new password**.
4. Log out and log back in as that user. **Cases** now shows the XTGAIA cases.

### Shuffle

The pipeline reaches Shuffle through the webhook in `.env`, so no login is needed for the demo. The UI
(http://localhost:3001) uses the account created when Shuffle was first set up.

### Demo flow presets

The first run extracts the preset flows from `data/raw/cicids2017_improved/friday.csv` into
`data/samples/demo_flow_presets.json`, which is not in git. To rebuild them:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -e PYTHONPATH=/app:/app/data/pylib xtgaia-core python scripts/live_demo.py --build-presets
```

---

## 2. Before the audience arrives (about 5 minutes)

1. Start **Docker Desktop** and wait until it reports *running*. If the machine has been busy for hours,
   restart Docker first (`wsl --shutdown`, then reopen Docker Desktop) so memory starts clean.
2. Start the SOC services:

   ```bash
   docker compose -f docker/docker-compose.yml up -d neo4j shuffle shuffle-backend shuffle-orborus shuffle-opensearch cassandra elasticsearch thehive
   ```

3. Wait about 2 minutes, then open these browser tabs:
   - TheHive: http://localhost:9000, logged in as the XTGAIA analyst
   - Shuffle: http://localhost:3001 (optional)
   - Neo4j Browser: http://localhost:7474 (optional)
4. Launch the demo:

   ```bash
   docker compose -f docker/docker-compose.yml run --rm -e PYTHONPATH=/app:/app/data/pylib xtgaia-core python scripts/live_demo.py
   ```

   On start it replays 30,000 real flows from the first hour of the Friday capture, which is all benign (about 30–40 s). This is the warm-up: it builds the
   live graph, the TGN memory and the burn-in alert threshold. Wait for the `xtgaia>` prompt.
5. **Warm Shuffle up.** The first Shuffle call can take a few seconds. Type one `scan` before the audience
   arrives, then `rollback last`, so the live run is quick.

---

## 3. The demo: commands and what to say

| # | Type | What appears | Talking point |
|---|---|---|---|
| 1 | `benign 40` | **[Role 1]** flows normalised to STIX, graph size, ATT&CK mapping. **[Role 2]** normal hosts *ok*; the busy web server may show *spike (not sustained)*. **[Role 3]** no host escalated. | "Normal office traffic. Every flow becomes a graph event and the TGN scores each host. A single high score is not enough: escalation needs sustained evidence, so nothing is escalated." |
| 2 | `status` | Live graph size, contained hosts, highest-risk hosts | "This is the live graph and the topology-aware risk. The web server 192.168.10.50 is a crown jewel." |
| 3 | `scan 172.16.0.1 192.168.10.50 60` | **[Role 2]** ALERT on 192.168.10.50 (p_t about 0.999, category PortScan, sustained). **[Risk]** high, next to the crown jewel. **[Role 3]** the zone Double DQN picks a playbook (RevokeSession or PortIsolation; the margin is shown). **[XAI]** Tier 1 top edges, Tier 2 onset, Tier 3 + bundle in about 35 ms. **[SOAR]** Shuffle dispatched, TheHive case id. **[Loop]** containment applied. **[Timing]** per stage. | "A port scan on the web server. The TGN flags it, the risk accounts for how close it is to the crown jewel, and the policy picks a graded playbook. We get a three-tier explanation in about 35 ms, Shuffle enforces it, TheHive logs it, and the graph is updated." |
| 4 | *switch to TheHive* | New case `[XTGAIA] <playbook> on 192.168.10.50`, with Role 1 / Role 2 / Role 3 evidence and the X_t bundle | "Every automated action comes with an auditable case and its explanation." |
| 5 | `scan 172.16.0.1 192.168.10.50 30` | Still ALERT, but *not dispatched: cooldown* (or *already_contained*; with PortIsolation, flows on that port are also *dropped by active containment*) | "The host is already handled, so XTGAIA does not flood the SOAR with duplicate actions." |
| 6 | `wait 900`, then `ddos 172.16.0.1 192.168.10.50 200` | The demo clock moves 15 minutes past the cooldown; the DDoS raises a new alert and the policy dispatches again (Shuffle and TheHive) | "Fifteen minutes later the attacker comes back with a flood, and the policy responds again." |
| 7 | `rollback last` | HTTP 200 SUCCESS, Shuffle revert True, enforcement lifted; penalty transition stored | "If the analyst disagrees, one rollback undoes the action in Shuffle. The policy learns from a penalty in its replay buffer." |
| 8 | `status` | No contained hosts | "Back to normal." |

**Extras** (if time allows):
- `bot 192.168.10.15 205.174.165.73 30`: botnet C2 beaconing from an infected workstation.
- `flow <src> <dst> <port> scan 20`: let the audience choose the addresses. Use 192.168.10.x hosts.
- In Neo4j Browser: `MATCH (h:Host)-[r:COMMUNICATED]->(b) RETURN h, r, b LIMIT 50` shows the persistent graph,
  and soft-deleted edges carry `active = false`.
- `help` lists all commands.

### All commands

| Command | Effect |
|---|---|
| `benign [n]` | n real benign flows (default 20) |
| `scan [attacker] [victim] [n]` | port scan to successive ports (default 172.16.0.1 → 192.168.10.50, 60 flows) |
| `ddos [attacker] [victim] [n]` | flood to port 80 (default 200 flows) |
| `bot [infected] [c2] [n]` | botnet beaconing (default 192.168.10.15 → 205.174.165.73, 30 flows) |
| `flow <src> <dst> <port> [kind] [n]` | custom flows with the statistics of `kind` (benign / scan / ddos / bot) |
| `status` | live graph, contained hosts, highest risks |
| `rollback [case_id\|last]` | analyst rollback (Algorithm 4 feedback) |
| `wait <seconds>` | advance the demo clock (cool-downs and windows are in event time) |
| `quit` | exit |

---

## 4. Tips and troubleshooting

| Situation | What to do |
|---|---|
| A repeated attack on the same host shows *already_contained* or *cooldown* | Intended (no duplicate dispatches for 15 minutes of demo time). `wait 900`, `rollback last`, or attack another host. |
| Why every host uses the same threshold | Demo setting: after the warm-up the demo judges every host against the global burn-in threshold. The busy web server's own baseline sits near saturation (about 0.9991), which made a live scan on it alert only some of the time. `--host-baselines` restores the pipeline's per-host thresholds (B12). |
| An address you typed gets odd scores | Use the CICIDS testbed addresses (192.168.10.x internal, 192.168.10.50 web server, 172.16.0.1 attacker). The TGN has node memory only for addresses it was trained on. |
| *Shuffle not reached* / *TheHive not reached* | The services are still starting. Wait a minute and repeat the command. The rest of the pipeline keeps working. |
| The SOC stack misbehaves | Run model-only (everything except the Shuffle/TheHive steps): `docker compose -f docker/docker-compose.yml run --rm --no-deps -e PYTHONPATH=/app:/app/data/pylib xtgaia-core python scripts/live_demo.py --no-neo4j --no-soar` |
| You want a hands-off run | `docker compose -f docker/docker-compose.yml run --rm -e PYTHONPATH=/app:/app/data/pylib xtgaia-core python scripts/live_demo.py --script scripts/demo_script.txt --pause 4` |
| No TheHive cases are visible | You are logged in as the admin account. Log in as the XTGAIA user (section 1). |
| Explanation over 200 ms on the first alert | First-call warm-up. The warm-up scan in section 2 avoids it. |

Model note: the demo uses the deployed TGN (`data/models/tgn_best.pt`, the pre-tuning destination read-out
model; see B11 in `docs/paper_deviations.md`). Do not switch it to `tgn_best.tuned_max.pt` for the demo:
that model does not escalate the scan in the live pipeline.

---

## 5. Afterwards

Type `quit`, then free the memory:

```bash
docker compose -f docker/docker-compose.yml stop thehive cassandra elasticsearch shuffle shuffle-backend shuffle-orborus shuffle-opensearch neo4j
```
