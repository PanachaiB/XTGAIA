"""Attack-category taxonomy shared by CIC-IDS2017, CSE-CIC-IDS2018 and NF-UNSW-NB15-v2.

Design (paper Eq. 4 + Sec. IV.C): the TGN's decision output is the *binary* node-level threat confidence
p_t(v) = Pr(malicious), which drives risk (Eq. 5) and response. Multi-class scoring is the attack
category of a detection: an auxiliary head on Z_t predicts the category, and the multi-class prediction
is hierarchical and consistent with the binary one -
        class(v) = Benign                     if p_t(v) <  ω
                 = argmax_{c != Benign} q_c   otherwise
so binary and multi-class results never disagree about whether something is an attack.
"""
import re
from typing import List

CLASSES: List[str] = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet", "Infiltration",
                      "Exploit", "Reconnaissance", "Fuzzers", "Backdoor", "Shellcode", "Worms", "Generic", "Analysis"]
BENIGN = 0

_RULES = [
    (r"benign|normal", "Benign"),
    (r"ddos|loic|hoic", "DDoS"),
    (r"dos|slowloris|slowhttp|hulk|goldeneye", "DoS"),
    (r"portscan|port scan", "PortScan"),
    (r"patator|brute ?force|bruteforce|ftp|ssh", "BruteForce"),
    (r"web|xss|sql", "WebAttack"),
    (r"bot", "Botnet"),
    (r"infilt", "Infiltration"),
    (r"heartbleed|exploit", "Exploit"),
    (r"recon", "Reconnaissance"),
    (r"fuzz", "Fuzzers"),
    (r"backdoor", "Backdoor"),
    (r"shellcode", "Shellcode"),
    (r"worm", "Worms"),
    (r"generic", "Generic"),
    (r"analysis", "Analysis"),
]


def attack_category(label: str) -> str:
    s = str(label).strip().lower()
    for pat, cat in _RULES:
        if re.search(pat, s):
            return cat
    return "Generic"                     # unknown attack label: still an attack


def category_index(label: str) -> int:
    return CLASSES.index(attack_category(label))
