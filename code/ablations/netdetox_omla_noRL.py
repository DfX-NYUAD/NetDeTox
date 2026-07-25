# llm_exo_norl.py
# LLM decides gate combo / k / nodes (only RL and RLK are disabled; all LLM calls are kept)
# Fixed allow/block lists are loaded at runtime (file/env vars); each round the LLM is
# explicitly told Allowed/Blocked and is forced to pick only from Allowed.

from __future__ import annotations
import os, re, json, time, math, random, tempfile, subprocess, argparse, hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set
import networkx as nx
# ===== JSON logger =====
import uuid
from datetime import datetime

def _iso_now():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

class ExperimentLogger:
    """
    Line-based JSON logging (JSONL), one record per line, convenient for later
    aggregation and safe for concurrent writes.
    Writes to work_dir/run_log.jsonl by default.
    """
    def __init__(self, path: Optional[str]):
        self.path = path
        self.run_id = f"run-{uuid.uuid4().hex[:8]}"  # short ID
        self._ok = bool(path)
        if self._ok:
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            except Exception:
                pass

    def log(self, event: str, payload: Dict[str, Any]):
        if not self._ok:
            return
        rec = {
            "ts": _iso_now(),
            "run_id": self.run_id,
            "event": event,
            **payload
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

def _env_snapshot():
    keys = [
        "SEC_WEIGHT","AREA_WEIGHT",
        "SEC_DROP_MIN_REL","SLOW_SWITCH_ROUNDS","STICKY_EMA_DS",
        "COMBO_EMA_BETA","SUCCESS_TOL",
        "K_SEC_PENALTY_LAMBDA","K_AREA_WEIGHT",
        "EARLY_STOP_FOR_OMLA","EARLY_STOP_FOR_GNN4IP","EARLY_STOP_FOR_GNNRE","EARLY_STOP_FOR_GTSAINT",
    ]
    return {k: os.getenv(k) for k in keys if k in os.environ}

# ================== Basic settings ==================
SEED = 42
random.seed(SEED)

# ===== RL settings (per bucket) =====
# Disable RL (keep only LLM decisions)
RL_ENABLE = False
RL_POLICY_PATH = "rl_bucket_policy.json"
RL_LR = 0.1
RL_TEMPERATURE = 1.0
RL_BASELINE_BETA = 0.1
RL_MAX_FEATS_SAMPLES = 20

# ===== Hop-size RL settings (adaptive k) =====
# Disable k-bandit RL (keep only LLM-selected k)
RLK_ENABLE = False
RLK_POLICY_PATH = "rl_k_policy.json"
K_MIN, K_MAX = 1, 20
K_ALPHA = 0.1
K_BASELINE_BETA = 0.1
K_TEMP = 1.0
K_SEC_PENALTY_LAMBDA = float(os.getenv("K_SEC_PENALTY_LAMBDA", "1.5"))
K_AREA_WEIGHT        = float(os.getenv("K_AREA_WEIGHT", "0.5"))

# ===== Gate-combo: disable combo-direction RL, use LLM decisions instead (no KL) =====
C_ENABLE = False  # do not use combo-direction RL

# ===== Run parameters =====
MAX_ITERS = 10
TRY_PER_ROUND = 5
POOL_MULT = 4
TOPK_FROM_LLM = 12

MAX_PER_BUCKET_IN_POOL  = max(2, TRY_PER_ROUND // 2)
MAX_PER_BUCKET_IN_PICKS = max(1, TRY_PER_ROUND // 3)
MAX_PER_GROUP_IN_POOL   = 1
MAX_PER_GROUP_IN_PICKS  = 1

EPISODE_STEPS = 10
SEC_DROP_MIN_REL = float(os.getenv("SEC_DROP_MIN_REL", "5e-4"))  # threshold for judging "not dropping fast enough"
AREA_UP_CAP_REL  = 0.01
SLOW_SWITCH_ROUNDS = int(os.getenv("SLOW_SWITCH_ROUNDS", "2"))   # how many consecutive "slow-drop" rounds before switching style
STICKY_EMA_DS = float(os.getenv("STICKY_EMA_DS", "5e-4"))        # stickiness threshold for a combo with a very strong history

INST_REGEX = re.compile(r"\bU(?:\d+|_OPT\d+(?:_\d+)*)\b")
IS_BASE = lambda s: bool(re.fullmatch(r"U\d+", s))
IS_OPT  = lambda s: bool(re.fullmatch(r"U_OPT\d+(?:_\d+)*", s))

# ===== Your environment =====
SUBCIRCUIT_OPT_CMD = ["python3", "subcircuit_opt.py"]
SUBCIRCUIT_OPT_COMMON_ARGS: Dict[str, str] = {
    "--k": "5",
    # "--liberty": "NangateOpenCellLibrary_typical.lib",
    "--use_gates": "INV,NAND,LOGIC"
    # --work_dir and --top are updated at runtime
}

# ===== Fixed Conda / evaluation-script configuration =====
FIXED_CONDA_SH = os.environ.get("NETDETOX_CONDA_SH", "/path/to/conda/etc/profile.d/conda.sh")

# --- OMLA ---
FIXED_OMLA_DIR     = "OMLA"
FIXED_OMLA_ENV     = "iplock"
FIXED_OMLA_SCRIPT  = "launch_omla_test_specify.py"
# ===== Early Stop (applies to OMLA only) =====
EARLY_STOP_FOR_OMLA = float(os.getenv("EARLY_STOP_FOR_OMLA", "0.5"))


# --- GNN4IP ---
FIXED_GNN4IP_DIR     = "GNN4IP/examples"
FIXED_GNN4IP_ENV     = "iplock"
FIXED_GNN4IP_SCRIPT  = "gnn4ip_test_specify.py"
EARLY_STOP_FOR_GNN4IP = float(os.getenv("EARLY_STOP_FOR_GNN4IP", "0.0"))

# --- GNNRE ---
FIXED_GNNRE_DIR     = "GNNRE/GraphSAINT"
FIXED_GNNRE_ENV     = "iplock"
FIXED_GNNRE_SCRIPT  = "attackgnnre_specify.py"
EARLY_STOP_FOR_GNNRE = float(os.getenv("EARLY_STOP_FOR_GNNRE", "0.25"))
GNNRE_UPDATE_EVERY = 10

# --- Trojan/GTSAINT ---
FIXED_GTSAINT_DIR     = "GTSAINT"
FIXED_GTSAINT_ENV     = "iplock"
FIXED_GTSAINT_SCRIPT  = "run_trojansaint_specify.py"
EARLY_STOP_FOR_GTSAINT = float(os.getenv("EARLY_STOP_FOR_GTSAINT", "0.5"))

# ================== Gate sets (include BUF to avoid occasional missing-gate errors) ==================
GATE_COMBOS: Dict[str, str] = {
    "C01_INV_NAND"            : "INV,NAND,BUF",
    "C02_INV_NOR"             : "INV,NOR,BUF",
    "C03_INV_NAND_LOGIC"      : "INV,NAND,LOGIC,BUF",
    "C04_INV_NAND_AND"        : "INV,NAND,AND,BUF",
    "C05_INV_NAND_OR"         : "INV,NAND,OR,BUF",
    "C06_INV_NAND_XOR"        : "INV,NAND,XOR,BUF",
    "C07_INV_NAND_XNOR"       : "INV,NAND,XNOR,BUF",
    "C08_INV_NOR_LOGIC"       : "INV,NOR,LOGIC,BUF",
    "C09_INV_NOR_AND"         : "INV,NOR,AND,BUF",
    "C10_INV_NOR_OR"          : "INV,NOR,OR,BUF",
    "C11_INV_NOR_XOR"         : "INV,NOR,XOR,BUF",
    "C12_INV_NOR_XNOR"        : "INV,NOR,XNOR,BUF",
    "C13_INV_AND_OR"          : "INV,AND,OR,BUF",
    "C14_INV_AND_OR_LOGIC"    : "INV,AND,OR,LOGIC,BUF",
    "C15_INV_AND_OR_XOR"      : "INV,AND,OR,XOR,BUF",
    "C16_INV_AND_OR_XNOR"     : "INV,AND,OR,XNOR,BUF",
    "C17_INV_NAND_LOGIC_XOR"  : "INV,NAND,LOGIC,XOR,BUF",
    "C18_INV_NAND_LOGIC_XNOR" : "INV,NAND,LOGIC,XNOR,BUF",
    "C19_INV_NOR_LOGIC_XOR"   : "INV,NOR,LOGIC,XOR,BUF",
    "C20_INV_NOR_LOGIC_XNOR"  : "INV,NOR,LOGIC,XNOR,BUF",
}
GATE_COMBO_NAMES: List[str] = list(GATE_COMBOS.keys())

GATE_COMBOS_AREA: Dict[str, str] = {
    # Basic cells (most area-efficient)
    "C01_INV_BUF"    : "INV,BUF",
    "C02_NAND_ONLY"  : "INV,NAND,BUF",
    "C03_NOR_ONLY"   : "INV,NOR,BUF",

    # NAND/NOR + simple AND/OR logic (still low area)
    "C04_INV_NAND_AND": "INV,NAND,AND,BUF",
    "C05_INV_NOR_OR" : "INV,NOR,OR,BUF",

    # Mixed NAND/NOR with limited AND/OR, still avoiding XOR/XNOR
    "C06_NAND_NOR"   : "INV,NAND,NOR,BUF",

    # Lightweight combo to add if more coverage is required
    "C07_AND_OR"     : "INV,AND,OR,BUF",
}

# ================== Data structures ==================
@dataclass
class QoR:
    area: float = 0.0

@dataclass
class Scores:
    security: float
    qor: QoR

@dataclass
class OptimizationResult:
    new_netlist: Any
    success: bool
    note: str = ""

# ================== U_OPT visit persistence & grouping ==================
VISITED_DB = "visited_opt.json"
VISITED_OPT: Dict[str, Dict[str, bool]] = {}
def _vk(work_dir: Optional[str], circuit: Optional[str]) -> str:
    return f"{work_dir or ''}|{circuit or ''}"
def load_visited_opt(work_dir: Optional[str], circuit: Optional[str]) -> None:
    global VISITED_OPT
    try:
        with open(VISITED_DB, "r") as f:
            VISITED_OPT = json.load(f)
    except Exception:
        VISITED_OPT = {}
    VISITED_OPT.setdefault(_vk(work_dir, circuit), {})
def save_visited_opt() -> None:
    try:
        with open(VISITED_DB, "w") as f:
            json.dump(VISITED_OPT, f, indent=2)
    except Exception:
        pass
def mark_visited_opt(insts: List[str], work_dir: Optional[str], circuit: Optional[str]) -> None:
    tbl = VISITED_OPT.setdefault(_vk(work_dir, circuit), {})
    for s in insts:
        if s.startswith("U_OPT"):
            tbl[s] = True
    save_visited_opt()
def is_visited_opt(inst: str, work_dir: Optional[str], circuit: Optional[str]) -> bool:
    return bool(VISITED_OPT.get(_vk(work_dir, circuit), {}).get(inst, False))
_OPT_GROUP_RE = re.compile(r"^U_OPT(\d+)")
def opt_group_id(inst: str) -> str:
    m = _OPT_GROUP_RE.match(inst)
    return m.group(1) if m else "NA"

# ================== LLM interface (JSON-only) ==================
import requests
def llm_call(prompt: str, system_prompt: str, temperature: float = 0.8) -> str:
    """
    When OPENAI_API_KEY is not set, fall back locally: pick from Candidates
    (already filtered to Allowed) ranked by a heuristic.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    # model = os.getenv("OPENAI_MODEL", "gpt-5")
    import requests
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": float(temperature),
        "max_tokens": 256,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    # print("response", r)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]
# def llm_call(prompt: str, system_prompt: str, temperature: float = 0.8) -> str:
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise RuntimeError("OPENAI_API_KEY not set")
    
#     base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").rstrip("/")
#     url = base + "/responses"
#     model = os.getenv("OPENAI_MODEL", "gpt-5")
    
#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }
    
#     payload = {
#         "model": model,
#         "input": [
#             {"role": "system", "content": system_prompt + " Respond in JSON format."},
#             {"role": "user", "content": prompt}
#         ],
#         "text": {"format": {"type": "json_object"}},
#         "reasoning": {"effort": "low"},
#         "max_output_tokens": 2048,
#         "temperature": float(temperature)
#     }
    
#     r = requests.post(url, headers=headers, json=payload, timeout=60)
#     r.raise_for_status()
    
#     data = r.json()
    
#     # Optional: print usage info (does not raise)
#     try:
#         usage = data.get('usage', {})
#         reasoning_tokens = usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
#         total_tokens = usage.get('output_tokens', 0)
#         print(f"[LLM usage] reasoning={reasoning_tokens}, output_tokens={total_tokens}")
#     except Exception:
#         pass
    
#     # Extract response content
#     if "output" in data and data["output"]:
#         for item in data["output"]:
#             if isinstance(item, dict) and item.get("type") == "message":
#                 content_list = item.get("content", [])
#                 if content_list and len(content_list) > 0:
#                     first_content = content_list[0]
#                     if isinstance(first_content, dict) and "text" in first_content:
#                         return first_content["text"]
#             elif isinstance(item, dict) and item.get("type") == "text" and "content" in item:
#                 return item["content"]
#     # Fallback
#     return json.dumps({"error": "No response content found"})

# ================== Graph & features ==================
def list_candidate_nodes(netlist_path: str) -> List[str]:
    out, seen = [], set()
    try:
        with open(netlist_path, "r", encoding="utf-8", errors="ignore") as f:
            for chunk in iter(lambda: f.read(1<<20), ""):
                for m in INST_REGEX.finditer(chunk):
                    s = m.group(0)
                    if s not in seen:
                        seen.add(s); out.append(s)
    except Exception:
        pass
    return out

def build_netlist_graph(netlist_path: str) -> nx.DiGraph:
    G = nx.DiGraph()
    with open(netlist_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s*(\w+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\);\s*$", line)
            if not m: continue
            gate, inst, conns = m.groups()
            ins, outs = [], []
            for conn in conns.split(","):
                conn = conn.strip()
                pm = re.match(r"\.(\w+)\((\w+)\)", conn)
                if not pm: continue
                pin, net = pm.groups()
                if pin.upper() in ("Z","ZN","Q","Y"): outs.append(net)
                else: ins.append(net)
            G.add_node(inst, gate=gate, inputs=ins, outputs=outs)
            for net in ins:  G.add_edge(net, inst)
            for net in outs: G.add_edge(inst, net)
    return G

def compute_logic_levels(G: nx.DiGraph) -> Dict[str,int]:
    levels = {}
    try:
        for n in nx.topological_sort(G):
            preds = list(G.predecessors(n))
            levels[n] = 0 if not preds else 1 + max(levels.get(p,0) for p in preds)
    except Exception:
        for n in G.nodes: levels[n] = 0
    return levels

def extract_node_features(netlist_path: str, inst_name: str) -> Dict[str,Any]:
    if not hasattr(extract_node_features, "_cache") or extract_node_features._cache_path != netlist_path:
        G = build_netlist_graph(netlist_path)
        levels = compute_logic_levels(G)
        extract_node_features._cache = (G, levels)
        extract_node_features._cache_path = netlist_path
    else:
        G, levels = extract_node_features._cache
    if inst_name not in G.nodes:
        return {"gate":"UNK","fanin":0,"fanout":0,"level":0}
    gate = G.nodes[inst_name].get("gate","UNK")
    fanin  = len([p for p in G.predecessors(inst_name) if p in G.nodes])
    fanout = len([s for s in G.successors(inst_name)  if s in G.nodes])
    level  = levels.get(inst_name, 0)
    return {"gate":gate, "fanin":fanin, "fanout":fanout, "level":level}

# ================== Bucketing (reused structure for RL; used for even quotas when RL is off) ==================
def _bucket_key(inst: str, feat: Dict[str,Any]) -> str:
    gate = feat.get("gate","UNK")
    lvl  = int(feat.get("level",0)) // 3
    fo   = int(feat.get("fanout",0)); fi = int(feat.get("fanin",0))
    fo_bin = 0 if fo==0 else (1 if fo<=2 else 2)
    fi_bin = 0 if fi<=1 else (1 if fi==2 else 2)
    return f"{gate}|L{lvl}|FO{fo_bin}|FI{fi_bin}"

def build_buckets(netlist_path: str, max_buckets=24) -> Tuple[Dict[str,str], Dict[str,List[str]]]:
    insts = list_candidate_nodes(netlist_path)
    inst2bucket, bucket2insts = {}, {}
    for inst in insts:
        bk = _bucket_key(inst, extract_node_features(netlist_path, inst))
        bucket2insts.setdefault(bk, []).append(inst)
        inst2bucket[inst] = bk
    keys_sorted = sorted(bucket2insts, key=lambda k: len(bucket2insts[k]), reverse=True)
    keep = set(keys_sorted[:max_buckets-1]) if len(keys_sorted)>max_buckets else set(keys_sorted)
    merged: Dict[str,List[str]] = {}
    for k in keys_sorted:
        tgt = k if k in keep else "OTHER"
        merged.setdefault(tgt, []).extend(bucket2insts[k])
    return inst2bucket, merged

# ================== RL: simple two-arm policy (class kept for compatibility; not instantiated/updated when RL_ENABLE=False) ==================
class RLBucketPolicy:
    def __init__(self, path=RL_POLICY_PATH, lr=RL_LR, temp=RL_TEMPERATURE, baseline_beta=RL_BASELINE_BETA, feat_dim=5):
        self.path = path; self.lr = lr; self.temp = temp; self.baseline_beta = baseline_beta
        self.theta = [0.0]*feat_dim; self.baseline = 0.0
        self.stats: Dict[str, Dict[str,Any]] = {}
        self._load()
    def _load(self):
        try:
            with open(self.path,"r") as f:
                o=json.load(f)
                self.theta = o.get("theta", self.theta)
                self.baseline = float(o.get("baseline",0.0))
                self.stats = o.get("stats",{})
        except Exception: pass
    def save(self):
        try:
            with open(self.path,"w") as f:
                json.dump({"theta":self.theta,"baseline":self.baseline,"stats":self.stats}, f, indent=2)
        except Exception: pass
    def _bucket_features(self, insts: List[str], netlist_path: str) -> List[float]:
        n=len(insts)
        if n==0: return [1.0,0.0,0.0,0.0,0.0]
        sample = insts[:min(n, RL_MAX_FEATS_SAMPLES)]
        lv_sum=fo_sum=fi_sum=0.0
        for inst in sample:
            f=extract_node_features(netlist_path, inst)
            lv_sum += float(f.get("level",0)); fo_sum += float(f.get("fanout",0)); fi_sum += float(f.get("fanin",0))
        m=float(len(sample))
        return [1.0, math.log1p(n), lv_sum/m, fo_sum/m, fi_sum/m]
    def _dot(self,a,b): return sum(ai*bi for ai,bi in zip(a,b))
    def _softmax(self, logits: Dict[str,float]) -> Dict[str,float]:
        if not logits: return {}
        mx=max(logits.values())
        exps={k: math.exp(v-mx) for k,v in logits.items()}
        Z=sum(exps.values()) or 1.0
        return {k: exps[k]/Z for k in logits}
    def plan(self, netlist_path: str, bucket2insts: Dict[str,List[str]], budget: int):
        feats={}; logits={}
        for bk, arr in bucket2insts.items():
            x=self._bucket_features(arr, netlist_path); feats[bk]=x; logits[bk]=self._dot(self.theta,x)
        probs=self._softmax(logits)
        raw={bk: probs[bk]*float(budget) for bk in probs}
        alloc={bk: int(math.floor(raw[bk])) for bk in probs}
        remain=budget - sum(alloc.values())
        if remain>0:
            frac=sorted(((raw[bk]-alloc[bk], bk) for bk in probs), reverse=True)
            for _,bk in frac[:remain]: alloc[bk]+=1
        alloc_list=[{"bucket":bk,"quota":q} for bk,q in alloc.items() if q>0]
        alloc_list.sort(key=lambda d: d["bucket"])
        return alloc_list, probs, feats
    def update(self, chosen_alloc, probs, feats, budget, reward):
        # Not called when RL is disabled
        y={bk:0 for bk in probs}
        for rec in chosen_alloc:
            bk=str(rec.get("bucket","")); q=int(rec.get("quota",0))
            if bk in y: y[bk]+=max(0,q)
        grad=[0.0]*len(self.theta)
        for bk,p in probs.items():
            x=feats.get(bk,[0.0]*len(self.theta))
            coeff=(y.get(bk,0) - budget*p)
            for i in range(len(grad)): grad[i]+=coeff*x[i]
        adv=reward - self.baseline
        for i in range(len(self.theta)): self.theta[i]+=RL_LR*adv*grad[i]
        self.baseline=(1.0-RL_BASELINE_BETA)*self.baseline + RL_BASELINE_BETA*reward

# ================== RL over hop size k (class kept; inactive when RLK_ENABLE=False) ==================
class RLKPolicy:
    def __init__(self, path=RLK_POLICY_PATH, alpha=K_ALPHA, baseline_beta=K_BASELINE_BETA):
        self.path = path
        self.alpha = alpha
        self.baseline_beta = baseline_beta
        self.h = [0.0] * (K_MAX - K_MIN + 1)
        self.baseline = 0.0
        self._load()
    def _load(self):
        try:
            with open(self.path, "r") as f:
                o = json.load(f)
                self.h = o.get("h", self.h)
                self.baseline = float(o.get("baseline", 0.0))
        except Exception:
            pass
    def save(self):
        try:
            with open(self.path, "w") as f:
                json.dump({"h": self.h, "baseline": self.baseline}, f, indent=2)
        except Exception:
            pass
    def _softmax_pi(self, temperature: float = 1.0) -> List[float]:
        if not self.h: return []
        mx = max(self.h)
        exps = [math.exp((v - mx) / max(1e-6, temperature)) for v in self.h]
        Z = sum(exps) or 1.0
        return [e / Z for e in exps]
    def select(self, temperature: float = K_TEMP) -> Tuple[int, List[float]]:
        pi = self._softmax_pi(temperature)
        r = random.random()
        acc = 0.0
        for i, p in enumerate(pi):
            acc += p
            if r <= acc:
                return K_MIN + i, pi
        return K_MIN + len(pi) - 1, pi
    def update(self, chosen_k: int, pi: List[float], reward: float):
        a = int(chosen_k) - K_MIN
        avg = self.baseline
        self.baseline = (1.0 - self.baseline_beta) * self.baseline + self.baseline_beta * reward
        for i in range(len(self.h)):
            grad = (1.0 - pi[i]) if i == a else (-pi[i])
            self.h[i] += self.alpha * (reward - avg) * grad

# ================== Candidate pool & LLM selection (nodes) ==================
def _heuristic_score(netlist: str, inst: str) -> float:
    f = extract_node_features(netlist, inst)
    return float(f.get("level",0)) + 2.0*float(f.get("fanout",0)) + 0.5*float(f.get("fanin",0))

def _fingerprint(strings: List[str]) -> str:
    s=",".join(strings[:500]); return hashlib.sha1(f"{len(strings)}|{s}".encode()).hexdigest()[:12]

def propose_pool_random(netlist_path: str,
                        budget_try: int,
                        *,
                        work_dir: Optional[str],
                        circuit_name: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, float], Dict[str, List[float]], bool]:
    """
    Build a candidate pool that does not depend on RL / bucketing:
    - If U\d+ (BASE) instances exist, prefer BASE (consistent with the original prefer_base logic)
    - Otherwise sample randomly from U_OPT*, avoiding too many from the same group (MAX_PER_GROUP_IN_POOL)
    - No bucket stats / softmax / allocation; alloc/probs/feats are returned empty
    """
    pool_size = max(budget_try * POOL_MULT, budget_try + 2)
    insts = list_candidate_nodes(netlist_path)

    # 1) Prefer BASE (keep behavior consistent with the original prefer_base)
    base = [x for x in insts if IS_BASE(x)]
    if base:
        base_sorted = sorted(base, key=lambda x: _heuristic_score(netlist_path, x), reverse=True)
        pool = base_sorted[:pool_size]
        prefer_base = True
        return pool, [], {}, {}, prefer_base

    # 2) Randomly draw from U_OPT, cap per group, prefer unvisited
    prefer_base = False
    opt = [x for x in insts if IS_OPT(x)]
    random.shuffle(opt)

    group_cnt: Dict[str, int] = {}
    pool: List[str] = []

    def can_take(inst: str) -> bool:
        gid = opt_group_id(inst)
        return group_cnt.get(gid, 0) < MAX_PER_GROUP_IN_POOL

    # 2a) Take "unvisited" ones first
    for x in opt:
        if len(pool) >= pool_size: break
        if is_visited_opt(x, work_dir, circuit_name): continue
        if can_take(x):
            pool.append(x)
            gid = opt_group_id(x)
            group_cnt[gid] = group_cnt.get(gid, 0) + 1

    # 2b) If not enough, fill with "already visited" ones
    if len(pool) < pool_size:
        for x in opt:
            if len(pool) >= pool_size: break
            if x in pool: continue
            if can_take(x):
                pool.append(x)
                gid = opt_group_id(x)
                group_cnt[gid] = group_cnt.get(gid, 0) + 1

    # Return empty RL-stat placeholders (keep the caller interface unchanged)
    alloc: List[Dict[str, Any]] = []
    probs: Dict[str, float] = {}
    feats: Dict[str, List[float]] = {}
    return pool[:pool_size], alloc, probs, feats, prefer_base


def rl_propose_pool(netlist_path: str,
                    bucket2insts: Dict[str, List[str]],
                    rl: Optional[RLBucketPolicy],
                    budget_try: int,
                    *,
                    work_dir: Optional[str],
                    circuit_name: Optional[str]) -> Tuple[List[str], List[Dict[str, Any]], Dict[str, float], Dict[str, List[float]], bool]:
    """
    When rl=None (the current default), assign an even quota per bucket; the rest of the
    logic is the same as before and can still trigger prefer_base.
    """
    pool_size = max(budget_try * POOL_MULT, budget_try + 2)
    has_base = any(IS_BASE(x) for arr in bucket2insts.values() for x in arr)
    prefer_base = has_base
    if rl is None:
        alloc = [{"bucket":bk, "quota": max(1, pool_size // max(1, len(bucket2insts)))} for bk in bucket2insts]
        probs, feats = {}, {}
    else:
        alloc, probs, feats = rl.plan(netlist_path, bucket2insts, pool_size)
    def depth_rank(bk: str) -> int:
        m = re.search(r"\bL(\d+)\b", bk); return int(m.group(1)) if m else 0
    alloc_sorted = sorted(alloc, key=lambda r: (-depth_rank(r["bucket"]), -r["quota"]))
    for rec in alloc_sorted:
        rec["quota"] = min(rec["quota"], MAX_PER_BUCKET_IN_POOL)
    pool: List[str] = []
    if prefer_base:
        for rec in alloc_sorted:
            bk, q = rec["bucket"], int(rec["quota"])
            cand = [x for x in bucket2insts.get(bk, []) if IS_BASE(x) and x not in pool]
            cand.sort(key=lambda x: _heuristic_score(netlist_path, x), reverse=True)
            pool.extend(cand[:q])
            if len(pool) >= pool_size: break
        if len(pool) < pool_size:
            rest = [x for arr in bucket2insts.values() for x in arr if IS_BASE(x) and x not in pool]
            rest.sort(key=lambda x: _heuristic_score(netlist_path, x), reverse=True)
            pool.extend(rest[:(pool_size - len(pool))])
        return pool[:pool_size], alloc, probs, feats, True
    group_cnt: Dict[str,int] = {}
    def can_take(inst: str, cap: int) -> bool:
        gid = opt_group_id(inst); return group_cnt.get(gid,0) < cap
    for rec in alloc_sorted:
        bk, q = rec["bucket"], int(rec["quota"])
        if q<=0: continue
        raw = bucket2insts.get(bk, [])
        fresh = [x for x in raw if IS_OPT(x) and not is_visited_opt(x, work_dir, circuit_name) and x not in pool]
        seen  = [x for x in raw if IS_OPT(x) and     is_visited_opt(x, work_dir, circuit_name) and x not in pool]
        fresh.sort(key=lambda x: _heuristic_score(netlist_path,x), reverse=True)
        seen .sort(key=lambda x: _heuristic_score(netlist_path,x), reverse=True)
        take=[]
        for x in fresh:
            if can_take(x, MAX_PER_GROUP_IN_POOL):
                take.append(x); group_cnt[opt_group_id(x)] = group_cnt.get(opt_group_id(x),0)+1
            if len(take)>=q: break
        if len(take)<q:
            for x in seen:
                if can_take(x, MAX_PER_GROUP_IN_POOL):
                    take.append(x); group_cnt[opt_group_id(x)] = group_cnt.get(opt_group_id(x),0)+1
                if len(take)>=q: break
        pool.extend(take)
        if len(pool)>=pool_size: break
    if len(pool) < pool_size:
        all_fresh = [x for arr in bucket2insts.values() for x in arr
                     if IS_OPT(x) and not is_visited_opt(x, work_dir, circuit_name) and x not in pool]
        all_seen  = [x for arr in bucket2insts.values() for x in arr
                     if IS_OPT(x) and     is_visited_opt(x, work_dir, circuit_name) and x not in pool]
        all_fresh.sort(key=lambda x: _heuristic_score(netlist_path,x), reverse=True)
        all_seen .sort(key=lambda x: _heuristic_score(netlist_path,x), reverse=True)
        for lst in (all_fresh, all_seen):
            for x in lst:
                if len(pool)>=pool_size: break
                if can_take(x, MAX_PER_GROUP_IN_POOL):
                    pool.append(x); group_cnt[opt_group_id(x)] = group_cnt.get(opt_group_id(x),0)+1
    return pool[:pool_size], alloc, probs, feats, False

def llm_choose_from_pool(netlist: str, pool: List[str], try_per_round: int,
                         security: float, area: float, prefer_base: bool,
                         *, work_dir: Optional[str], circuit_name: Optional[str]) -> List[str]:
    valid = list(dict.fromkeys(pool))
    if not valid:
        return []
    # Warm the feature cache for later combo/k decisions (callers may re-run extract; not required to return)
    _ = [extract_node_features(netlist, inst) for inst in valid[:1]]  # trigger cache

    cand = [{"node_id": inst, "features": extract_node_features(netlist, inst)}
            for inst in valid[: max(try_per_round * 8, 100)]]

    system_prompt = (
        "You are a gate-level instance selection agent.\n"
        "Return ONLY compact valid JSON, no commentary, no markdown.\n"
        "Schema: {\"picks\": [{\"node_id\": \"U123\"}, ...]}\n"
        "Rules:\n"
        f"- Select up to {try_per_round} instances.\n"
        "- Pick ONLY from provided node_id set (do NOT invent ids).\n"
        "- Prefer higher impact by features: higher level, higher fanout, moderate fanin.\n"
        "- Avoid duplicates; order does not matter.\n"
        "Context:\n"
        f"- security={security:.6f}\n"
        f"- area={area:.6f}\n"
        f"- try_per_round={try_per_round}\n"
        f"- Pool_size={len(valid)}\n"
        f"- Pool_fingerprint={_fingerprint(sorted(list(set(valid))))}\n"
        "Candidates:\n" + json.dumps(cand, ensure_ascii=False)
    )
    prompt = "Pick gate-level instances now and return JSON: {\"picks\": [{\"node_id\": \"...\"}, ...]}"

    picks: List[str] = []
    try:
        data = json.loads(llm_call(prompt, system_prompt, temperature=0.2))
        vset = set(valid)
        for p in data.get("picks", []):
            if isinstance(p, dict):
                nid = str(p.get("node_id","")).strip()
                if nid in vset and nid not in picks:
                    picks.append(nid)
            if len(picks) >= try_per_round: break
    except Exception:
        pass

    if len(picks) < try_per_round:
        left = [x for x in valid if x not in picks]
        left.sort(key=lambda x: _heuristic_score(netlist, x), reverse=True)
        for x in left:
            picks.append(x)
            if len(picks) >= try_per_round: break

    return picks[:try_per_round]


# ================== Fixed/history file paths and loading (runtime) ==================
def _path_in_wd(name: str, work_dir: Optional[str]) -> str:
    if work_dir:
        try: os.makedirs(work_dir, exist_ok=True)
        except Exception: pass
        return os.path.join(work_dir, name)
    return name

def _combo_perf_path(work_dir: Optional[str]) -> str:
    return _path_in_wd("combo_perf.json", work_dir)

def _allowlist_path(work_dir: Optional[str]) -> str:
    return _path_in_wd("combo_allowlist.json", work_dir)

def _blocklist_path(work_dir: Optional[str]) -> str:
    return _path_in_wd("combo_blocklist.json", work_dir)

def _read_json_list(path: str) -> List[str]:
    try:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    return []

def _from_env_csv(key: str) -> List[str]:
    return [s.strip() for s in os.getenv(key, "").split(",") if s.strip()]

def _load_fixed_lists(work_dir: Optional[str]) -> Tuple[Set[str], Set[str]]:
    allow = set(_read_json_list(_allowlist_path(work_dir))) or set(_from_env_csv("COMBO_ALLOWLIST"))
    block = set(_read_json_list(_blocklist_path(work_dir))) or set(_from_env_csv("COMBO_BLOCKLIST"))
    return allow, block

# ================== combo history-performance persistence (no KL, purely history-driven) ==================
_COMBO_PERF: Dict[str, Dict[str, float]] = {}
_COMBO_PERF_BETA = float(os.getenv("COMBO_EMA_BETA","0.2"))  # EMA smoothing
SUCCESS_TOL = float(os.getenv("SUCCESS_TOL","1e-4"))         # ds_rel threshold considered "effective"

def _load_combo_perf(work_dir: Optional[str]):
    global _COMBO_PERF
    path = _combo_perf_path(work_dir)
    try:
        with open(path, "r") as f:
            _COMBO_PERF = json.load(f)
    except Exception:
        _COMBO_PERF = {}

def _save_combo_perf(work_dir: Optional[str]):
    path = _combo_perf_path(work_dir)
    try:
        with open(path, "w") as f:
            json.dump(_COMBO_PERF, f, indent=2)
    except Exception:
        pass

def _update_combo_perf(name: str, ds_rel: float, da_rel: float, work_dir: Optional[str]):
    rec = _COMBO_PERF.setdefault(name, {
        "n":0, "ema_ds":0.0, "ema_da":0.0, "last_ds":0.0, "last_da":0.0,
        "succ":0, "streak":0, "best_ds":0.0
    })
    beta = _COMBO_PERF_BETA
    if rec["n"] == 0:
        rec["ema_ds"] = ds_rel
        rec["ema_da"] = da_rel
    else:
        rec["ema_ds"] = (1.0 - beta) * rec["ema_ds"] + beta * ds_rel
        rec["ema_da"] = (1.0 - beta) * rec["ema_da"] + beta * da_rel
    rec["last_ds"] = ds_rel
    rec["last_da"] = da_rel
    rec["n"] += 1
    if ds_rel > SUCCESS_TOL:
        rec["succ"] += 1
        rec["streak"] = max(1, rec.get("streak",0)+1)
    else:
        rec["streak"] = 0
    rec["best_ds"] = max(rec.get("best_ds",0.0), ds_rel)
    _save_combo_perf(work_dir)

def _best_combo_by_history(fallback="C14_INV_AND_OR_LOGIC") -> str:
    if not _COMBO_PERF:
        return fallback
    ranked = sorted(_COMBO_PERF.items(),
                    key=lambda kv: (kv[1].get("ema_ds",0.0),
                                    kv[1].get("ema_da",0.0),
                                    kv[1].get("succ",0),
                                    kv[1].get("best_ds",0.0)),
                    reverse=True)
    return ranked[0][0] if ranked else fallback

def _bad_combo(name: str, min_trials: int = int(os.getenv("BAD_MIN_TRIALS","2")),
               tol: float = float(os.getenv("BAD_TOL","1e-4"))) -> bool:
    rec = _COMBO_PERF.get(name)
    if not rec: return False
    return rec.get("n",0) >= min_trials and rec.get("ema_ds",0.0) <= tol

# ================== LLM decides gate combo (history only + fixed allow/block lists) ==================
def llm_choose_combo_by_history(
    security: float,
    area: float,
    *,
    last_combo: Optional[str],
    need_switch: bool,
    work_dir: Optional[str],
    last_ds_rel: Optional[float] = None,
    last_da_rel: Optional[float] = None,
    recent_ds_rel: Optional[List[float]] = None,
    recent_da_rel: Optional[List[float]] = None,
    picked_nodes: Optional[List[str]] = None,          # <<< added
    pool_digest: Optional[Dict[str, Any]] = None       # <<< added (e.g. {"size":N,"fingerprint": "..."})
) -> str:
    # Reuse / fallback thresholds
    SEC_FOCUS_THRESHOLD = float(os.getenv("SEC_FOCUS_THRESHOLD", os.getenv("EARLY_STOP_FOR_OMLA", "0.5")))
    SMALL_AREA_DROP = float(os.getenv("SMALL_AREA_DROP", "0.02"))  # upper bound for "slight area decrease", ~2%
    AREA_TOL_UP     = float(os.getenv("AREA_TOL_UP",   "0.01"))    # small area increase allowed when balancing, <=1%
    SEC_BLOWUP_ABS  = float(os.getenv("SEC_BLOWUP_ABS","0.05"))    # threshold for "security rose a lot" (|ds_rel|>=5% and negative)

    # ============== Allow/block lists & candidate assembly (same as your original logic) ==============
    allow_set, block_set = _load_fixed_lists(work_dir)
    allowed_names = set(GATE_COMBO_NAMES) if not allow_set else (set(GATE_COMBO_NAMES) & allow_set)
    allowed_names = [n for n in sorted(list(allowed_names)) if n not in block_set]
    blocked_names = sorted(list(block_set))
    if not allowed_names:
        allowed_names = [n for n in GATE_COMBOS.keys() if n not in block_set] or GATE_COMBO_NAMES[:]

    cand = []
    for name in allowed_names:
        hist = _COMBO_PERF.get(name, {})
        cand.append({
            "combo": name,
            "use_gates": GATE_COMBOS.get(name, ""),
            "trials": int(hist.get("n",0)),
            "ema_ds": float(hist.get("ema_ds",0.0)),  # +: security drops (larger is better)
            "ema_da": float(hist.get("ema_da",0.0)),  # +: area drops (larger is better)
            "last_ds": float(hist.get("last_ds",0.0)),
            "last_da": float(hist.get("last_da",0.0)),
            "succ": int(hist.get("succ",0)),
            "streak": int(hist.get("streak",0)),
            "best_ds": float(hist.get("best_ds",0.0)),
            "bad": _bad_combo(name)
        })
    cand.sort(key=lambda x: (not x["bad"], x["ema_ds"], x["ema_da"], x["succ"], x["best_ds"], -x["trials"]), reverse=True)

    prefer_rule = (
        "Security dropped too slowly last round. DO NOT pick '{last}'. "
        "Choose ONLY from Allowed; never from Blocked. Prefer higher ema_ds."
    ) if (need_switch and last_combo) else (
        "Choose ONLY from Allowed; never from Blocked. Prefer higher ema_ds (more security drop)."
    )

    # ============== Pack "history" into the system prompt: recent_* and their EMAs ==============
    def _ema(xs: List[float], beta: float = 0.3) -> float:
        if not xs: return 0.0
        m = xs[0]
        for x in xs[1:]:
            m = (1.0 - beta) * m + beta * x
        return m

    recent_ds_rel = list(recent_ds_rel or ([] if last_ds_rel is None else [last_ds_rel]))
    recent_da_rel = list(recent_da_rel or ([] if last_da_rel is None else [last_da_rel]))

    ema_ds_rel = _ema(recent_ds_rel) if recent_ds_rel else 0.0
    ema_da_rel = _ema(recent_da_rel) if recent_da_rel else 0.0

    # Whether to trigger the "last round area dropped but security rose a lot" balancing strategy
    trigger_balance = (last_da_rel is not None and last_ds_rel is not None
                       and last_da_rel > 0.0 and last_ds_rel < -SEC_BLOWUP_ABS)

    # ======= System prompt: rules + current values + recent history (array, old->new) =======
    # Append picked-node and pool info to the Context in system_prompt
    ctx_extra = {
        "picked_nodes": list(picked_nodes or [])[:32],  # cap length
        "pool": {
            "size": int(pool_digest.get("size", 0)) if pool_digest else None,
            "fingerprint": str(pool_digest.get("fingerprint", "")) if pool_digest else None
        }
    }

    system_prompt = (
        "You are a gate-combo selection agent for securing design.\n"
        "Output ONLY compact valid JSON with schema: {\"combo\": \"NAME\"}. No commentary, no markdown.\n"
        "Rules:\n"
        f"- {prefer_rule.replace('{last}', last_combo or '')}\n"
        "- Do not invent names; pick exactly one from Allowed.\n"
        "- If multiple are similar, prefer higher ema_ds; tie-breaker: higher ema_da, then higher succ, then higher best_ds.\n"
        "- Avoid items marked as bad=true when a good option exists.\n"
        f"- When security <= {SEC_FOCUS_THRESHOLD:.3f}: prefer slight area decrease (ema_da>0, ideally <= {SMALL_AREA_DROP:.3f}).\n"
        "Context:\n"
        f"- current_security={security:.6f}\n"
        f"- current_area={area:.6f}\n"
        "- recent_relative_changes (old->new):\n"
        + json.dumps(
            [{"ds_rel": float(ds), "da_rel": float(da)}
             for ds, da in zip((recent_ds_rel or [])[-8:], (recent_da_rel or [])[-8:])],
            ensure_ascii=False
        ) + "\n"
        + "Extra:\n" + json.dumps(ctx_extra, ensure_ascii=False) + "\n"
        "Allowed (with stats):\n"
        + json.dumps(
            [{"combo": c["combo"], "use_gates": c["use_gates"], "ema_ds": c["ema_ds"], "ema_da": c["ema_da"],
              "trials": c["trials"], "bad": c["bad"]} for c in cand],
            ensure_ascii=False
        )
        + "\nBlocked:\n"
        + json.dumps(blocked_names, ensure_ascii=False)
        + "\nCandidates:\n"
        + json.dumps(cand, ensure_ascii=False)
    )

    prompt = "Pick ONE combo from Allowed and return JSON: {\"combo\": \"NAME\"}"
    print(f"[combo-guard] Allowed={ [c['combo'] for c in cand] }  Blocked={ blocked_names }")

    # ===== Call the LLM =====
    choice = None
    try:
        data = json.loads(llm_call(prompt, system_prompt=system_prompt, temperature=0.8))
        c = str(data.get("combo","")).strip()
        allowed_set = set([x["combo"] for x in cand])
        if c and c in allowed_set:
            choice = c
    except Exception:
        choice = None

    # ===== Fallback / correction =====
    if choice is None:
        choice = cand[0]["combo"] if cand else "C14_INV_AND_OR_LOGIC"
    if need_switch and last_combo and choice == last_combo:
        for it in cand:
            if it["combo"] != last_combo and not it["bad"]:
                choice = it["combo"]; break
    if choice in block_set:
        for it in cand:
            if it["combo"] not in block_set and not it["bad"]:
                choice = it["combo"]; break

    # A) When security is below the threshold, prefer a "slight area decrease"
    if security <= SEC_FOCUS_THRESHOLD:
        good = [x for x in cand if (x["combo"] not in block_set) and (not x["bad"]) and (x["ema_da"] > 0.0)]
        if good:
            good.sort(key=lambda x: (
                0 if (0.0 < x["ema_da"] <= SMALL_AREA_DROP) else 1,
                x["ema_da"],         # smaller is more "slight"
                -x["ema_ds"],        # then consider the security drop
                -x["ema_da"]
            ))
            cur = next((z for z in cand if z["combo"] == choice), None)
            need_replace = (cur is None) or (cur["ema_da"] <= 0.0) or (cur["bad"]) or (choice in block_set)
            if not need_replace and (cur["ema_da"] > SMALL_AREA_DROP):
                need_replace = True
            if need_replace:
                choice = good[0]["combo"]

    # B) Last round area dropped but security rose a lot -> balancing strategy
    if (last_da_rel is not None and last_ds_rel is not None
        and last_da_rel > 0.0 and last_ds_rel < -SEC_BLOWUP_ABS):
        bal = [x for x in cand if (x["combo"] not in block_set) and (not x["bad"]) and (x["ema_da"] >= -AREA_TOL_UP)]
        if bal:
            bal.sort(key=lambda x: (
                0 if (x["ema_da"] > 0.0 and x["ema_da"] <= SMALL_AREA_DROP) else
                (1 if (x["ema_da"] >= -AREA_TOL_UP and x["ema_da"] <= 0.0) else 2),
                -x["ema_ds"],
                abs(x["ema_da"]),
                -x["ema_da"]
            ))
            cur = next((z for z in cand if z["combo"] == choice), None)
            need_replace = (
                cur is None or cur["bad"] or (choice in block_set) or
                (cur["ema_da"] < -AREA_TOL_UP)
            )
            if not need_replace and bal and (cur["ema_ds"] < bal[0]["ema_ds"] * 0.9):
                need_replace = True
            if need_replace:
                choice = bal[0]["combo"]

    return choice

def llm_choose_combo_area_priority(
    security: float,
    area: float,
    *,
    last_combo: Optional[str],
    need_switch: bool,
    work_dir: Optional[str],
    last_ds_rel: Optional[float] = None,
    last_da_rel: Optional[float] = None,
    recent_ds_rel: Optional[List[float]] = None,
    recent_da_rel: Optional[List[float]] = None,
    picked_nodes: Optional[List[str]] = None,
    pool_digest: Optional[Dict[str, Any]] = None,
    llm_temperature: float = 0.6,
) -> str:
    """
    LLM area-first: prefer combos that reduce area, under the constraint of "not worsening security".
    """
    import json

    SEC_TOL  = float(os.getenv("SEC_NO_WORSE_TOL", "5e-4"))   # tiny negative security fluctuation allowed
    AREA_MIN = float(os.getenv("AREA_IMPROVE_MIN", "5e-3"))   # prioritize a significant area drop
    SMALL_AREA_DROP = float(os.getenv("SMALL_AREA_DROP", "0.02"))

    combo_table = globals().get("GATE_COMBOS_AREA", None)
    if not isinstance(combo_table, dict) or not combo_table:
        combo_table = GATE_COMBOS
    combo_names = list(combo_table.keys())

    # ---------- Allow/block lists ----------
    allow_set, block_set = _load_fixed_lists(work_dir)
    allowed_names = set(combo_names) if not allow_set else (set(combo_names) & allow_set)
    allowed_names = [n for n in sorted(list(allowed_names)) if n not in block_set]
    if not allowed_names:
        allowed_names = [n for n in combo_names if n not in block_set] or combo_names[:]
    blocked_names = sorted(list(block_set))

    # ---------- Candidates and history ----------
    cand: List[Dict[str, Any]] = []
    for name in allowed_names:
        hist = _COMBO_PERF.get(name, {})
        cand.append({
            "combo": name,
            "use_gates": combo_table.get(name, ""),
            "trials": int(hist.get("n",0)),
            "ema_ds": float(hist.get("ema_ds",0.0)),  # +: security drops
            "ema_da": float(hist.get("ema_da",0.0)),  # +: area drops
            "last_ds": float(hist.get("last_ds",0.0)),
            "last_da": float(hist.get("last_da",0.0)),
            "succ": int(hist.get("succ",0)),
            "streak": int(hist.get("streak",0)),
            "best_ds": float(hist.get("best_ds",0.0)),
            "bad": _bad_combo(name)
        })

    ctx = {
        "current_security": float(security),
        "current_area": float(area),
        "recent_changes_old_to_new": [
            {"ds_rel": float(ds), "da_rel": float(da)}
            for ds, da in zip((recent_ds_rel or [])[-8:], (recent_da_rel or [])[-8:])
        ],
        "last_combo": last_combo, "need_switch": bool(need_switch),
        "picked_nodes": list(picked_nodes or [])[:32],
        "pool": {
            "size": int(pool_digest.get("size",0)) if pool_digest else None,
            "fingerprint": str(pool_digest.get("fingerprint","")) if pool_digest else None
        },
        "constraints": {
            "security_no_worse_tol": SEC_TOL,
            "prefer_area_drop_min": AREA_MIN
        },
        "allowed": [
            {"combo": c["combo"], "use_gates": c["use_gates"],
             "ema_ds": c["ema_ds"], "ema_da": c["ema_da"],
             "trials": c["trials"], "bad": c["bad"]}
            for c in cand
        ],
        "blocked": blocked_names
    }

    system_prompt = (
        "You are a gate-combo selection agent for the AREA-MINIMIZATION phase.\n"
        "Goal: MINIMIZE AREA while NOT WORSENING SECURITY.\n"
        "Hard constraints:\n"
        f"- Prefer combos with ema_da > 0 (area goes down), ideally <= {SMALL_AREA_DROP:.3f} for stability.\n"
        f"- Do NOT pick combos whose ema_ds < {-float(SEC_TOL):.6f} (security tends to worsen).\n"
        "- Avoid items marked as bad=true when a good option exists.\n"
        "- Pick exactly ONE combo from Allowed; never from Blocked.\n"
        "- If need_switch=true, avoid repeating last_combo when a good alternative exists.\n"
        "Tie-breakers: larger ema_da, then higher ema_ds, then more trials.\n"
        "Return ONLY JSON: {\"combo\": \"NAME\"}\n"
        "Context:\n" + json.dumps(ctx, ensure_ascii=False)
    )
    prompt = 'Pick ONE combo for area minimization under security non-degradation. Return {"combo": "NAME"}.'

    print(f"[combo-area] Allowed={ [c['combo'] for c in cand] }  Blocked={ blocked_names }")

    # ---------- LLM ----------
    choice = None
    try:
        data = json.loads(llm_call(prompt, system_prompt=system_prompt, temperature=float(llm_temperature)))
        c = str(data.get("combo","")).strip()
        allowed_set = {x["combo"] for x in cand}
        if c and (c in allowed_set) and (c not in block_set):
            choice = c
    except Exception:
        choice = None

    # ---------- Local fallback (area-first + security floor) ----------
    if choice is None:
        good = [x for x in cand if (x["ema_ds"] >= -SEC_TOL) and (x["ema_da"] > 0.0) and (not x["bad"]) and (x["combo"] not in block_set)]
        if good:
            good.sort(key=lambda x: (x["ema_da"], x["ema_ds"], x["trials"]), reverse=True)
            choice = good[0]["combo"]
        else:
            ok_sec = [x for x in cand if (x["ema_ds"] >= -SEC_TOL) and (not x["bad"]) and (x["combo"] not in block_set)]
            if ok_sec:
                ok_sec.sort(key=lambda x: (x["ema_da"], x["ema_ds"], x["trials"]), reverse=True)
                choice = ok_sec[0]["combo"]
            else:
                choice = cand[0]["combo"] if cand else (list(combo_table.keys())[0] if combo_table else "C01_INV_NAND")

    if need_switch and last_combo and choice == last_combo:
        for it in cand:
            if it["combo"] != last_combo and not it["bad"] and it["combo"] not in block_set:
                choice = it["combo"]; break

    return str(choice)


def llm_choose_k_area_priority(
    *,
    security: float,
    area: float,
    recent_ds_rel: Optional[List[float]] = None,
    recent_da_rel: Optional[List[float]] = None,
    combo_name: Optional[str] = None,
    use_gates: Optional[str] = None,
    prefer_small_k: bool = True,
    k_min: int = 1,
    k_max: int = 10,
    last_k: Optional[int] = None,
    picked_nodes: Optional[List[str]] = None,
    pool_digest: Optional[Dict[str, Any]] = None,
    llm_temperature: float = 0.6,
) -> int:
    """
    LLM area-first: choose k to drive area down without worsening security.
    """
    SEC_TOL  = float(os.getenv("SEC_NO_WORSE_TOL", "5e-4"))
    AREA_TOL = float(os.getenv("AREA_IMPROVE_MIN", "5e-3"))

    ds_hist = list(recent_ds_rel or [])
    da_hist = list(recent_da_rel or [])
    ema_ds = _ema_list(ds_hist) if ds_hist else 0.0
    ema_da = _ema_list(da_hist) if ds_hist else 0.0

    allowed_k = list(range(int(k_min), int(k_max)+1))
    if not allowed_k:
        allowed_k = list(range(1, 11))

    ctx = {
        "current_security": float(security),
        "current_area": float(area),
        "recent_changes_old_to_new": [
            {"ds_rel": float(ds), "da_rel": float(da)}
            for ds, da in zip((recent_ds_rel or [])[-8:], (recent_da_rel or [])[-8:])
        ],
        "ema_ds_rel": float(ema_ds),
        "ema_da_rel": float(ema_da),
        "combo": {"name": combo_name, "use_gates": use_gates},
        "Allowed_k": allowed_k,
        "prefer_small_k": bool(prefer_small_k),
        "last_k": int(last_k) if last_k is not None else None,
        "picked_nodes": list(picked_nodes or [])[:32],
        "pool": {"size": int(pool_digest.get("size",0)) if pool_digest else None,
                 "fingerprint": str(pool_digest.get("fingerprint","")) if pool_digest else None},
        "constraints": {"security_no_worse_tol": SEC_TOL, "prefer_area_drop_min": AREA_TOL}
    }

    system_prompt = (
        "You are a hop-size k selection agent for AREA-MINIMIZATION phase.\n"
        "Goal: choose k that helps DECREASE AREA while NOT WORSENING SECURITY.\n"
        "Guidelines:\n"
        f"- If recent area drop is small (ema_da <= {AREA_TOL:.4f}), increase k moderately.\n"
        f"- If recent security shows degradation (ema_ds < {-float(SEC_TOL):.6f}), avoid large k; prefer small-to-mid k for safer edits.\n"
        "- If area is clearly dropping and security is stable or improving, prefer smaller k.\n"
        "- Pick exactly ONE INTEGER k from Allowed_k.\n"
        'Return ONLY JSON: {"k": <int>}.\n'
        "Context:\n" + json.dumps(ctx, ensure_ascii=False)
    )
    prompt = 'Pick ONE integer k for area-first optimization under security non-degradation. Return {"k": <int>}.'

    # ---------- LLM ----------
    k_from_llm: Optional[int] = None
    try:
        data = json.loads(llm_call(prompt, system_prompt=system_prompt, temperature=float(llm_temperature)))
        val = data.get("k", None)
        if isinstance(val, (int, float)):
            k_candidate = int(round(float(val)))
            if k_candidate in allowed_k:
                k_from_llm = k_candidate
    except Exception:
        k_from_llm = None

    if k_from_llm is not None:
        return int(k_from_llm)

    # ---------- Local fallback (heuristic consistent with the prompt) ----------
    k = 4 if prefer_small_k else 5
    if ema_ds < -SEC_TOL:
        k += 0  # avoid large k; keep it small-to-mid
    elif ema_da <= AREA_TOL:
        k += 1  # area has not dropped noticeably -> increase moderately
    else:
        k -= 1  # area is dropping and security is stable -> smaller k

    if last_k is not None:
        k = int(round(0.6*last_k + 0.4*k))
    return int(max(k_min, min(k_max, k)))

def _ema_list(xs: List[float], beta: float = 0.3) -> float:
    if not xs: return 0.0
    m = xs[0]
    for x in xs[1:]:
        m = (1.0 - beta) * m + beta * x
    return m

def llm_choose_k(
    *,
    security: float,
    area: float,
    recent_ds_rel: Optional[List[float]] = None,
    recent_da_rel: Optional[List[float]] = None,
    combo_name: Optional[str] = None,
    use_gates: Optional[str] = None,
    prefer_small_k: bool = True,
    k_min: int = 1,
    k_max: int = 10,
    last_k: Optional[int] = None,
    picked_nodes: Optional[List[str]] = None,      # <<< added
    pool_digest: Optional[Dict[str, Any]] = None   # <<< added
) -> int:
    """
    The LLM picks an integer k in [k_min, k_max] (default 1..10).
    """
    ds_hist = list(recent_ds_rel or [])
    da_hist = list(recent_da_rel or [])
    ema_ds = _ema_list(ds_hist) if ds_hist else 0.0
    ema_da = _ema_list(da_hist) if ds_hist else 0.0

    SEC_SLOW_THR = max(1e-6, float(os.getenv("SEC_DROP_MIN_REL", "5e-4")))
    tail = ds_hist[-4:] if len(ds_hist) >= 4 else ds_hist
    plateau_rounds = sum(1 for x in tail if abs(float(x)) < SEC_SLOW_THR) if tail else 0

    # Historical performance of this combo (if any)
    hist = _COMBO_PERF.get(combo_name or "", {}) if "_COMBO_PERF" in globals() else {}
    combo_hist = {
        "trials": int(hist.get("n",0)),
        "ema_ds": float(hist.get("ema_ds",0.0)),
        "ema_da": float(hist.get("ema_da",0.0)),
        "last_ds": float(hist.get("last_ds",0.0)),
        "last_da": float(hist.get("last_da",0.0)),
        "succ": int(hist.get("succ",0)),
        "best_ds": float(hist.get("best_ds",0.0)),
    }

    allowed_k = list(range(int(k_min), int(k_max) + 1))
    ctx = {
        "current_security": float(security),
        "current_area": float(area),
        "recent_changes_old_to_new": [
            {"ds_rel": float(ds), "da_rel": float(da)}
            for ds, da in zip((recent_ds_rel or [])[-8:], (recent_da_rel or [])[-8:])
        ],
        "ema_ds_rel": float(_ema_list(recent_ds_rel or [])) if (recent_ds_rel) else 0.0,
        "ema_da_rel": float(_ema_list(recent_da_rel or [])) if (recent_da_rel) else 0.0,
        "combo": {"name": combo_name, "use_gates": use_gates, "hist": combo_hist},
        "Allowed_k": allowed_k,
        "prefer_small_k": bool(prefer_small_k),
        "last_k": int(last_k) if last_k is not None else None,
        "picked_nodes": list(picked_nodes or [])[:32],
        "pool": {
            "size": int(pool_digest.get("size", 0)) if pool_digest else None,
            "fingerprint": str(pool_digest.get("fingerprint", "")) if pool_digest else None
        }
    }
    system_prompt = (
        "You are a hop-size (k) selection agent for gate-level subcircuit optimization.\n"
        "- Goal: ensure the security score reducing while maintain the area drops\n"
        "- Consider the selected gate combo and its historical performance when deciding k.\n"
        "- If recent security drop plateaued or already low, please reconsider k .\n"
        "- the number of k does not have the exact correlation with the security and area changes, it is determined by multiple factors including the gate combo, the circuit structure, and the recent optimization history.\n"
        "- Pick exactly one INTEGER k from Allowed_k.\n"
        "- Please make good use of k range, not always pick the smaller one.\n"
        'Return ONLY JSON: {"k": <int>} with no extra text.'
        "\nContext:\n" + json.dumps(ctx, ensure_ascii=False)
    )
    prompt = 'Pick ONE integer k from Allowed_k (1..20). Return {"k": <int>}.'

    # ======== LLM ========
    k_from_llm: Optional[int] = None
    try:
        data = json.loads(llm_call(prompt, system_prompt=system_prompt, temperature=0.8))
        val = data.get("k", None)
        if isinstance(val, (int, float)):
            k_candidate = int(round(float(val)))
            if k_candidate in allowed_k:
                k_from_llm = k_candidate
    except Exception:
        k_from_llm = None

    if k_from_llm is not None:
        return int(k_from_llm)

    # ======== Local fallback (heuristic consistent with the prompt) =======
    k = 5
    if ema_ds <= 0.0:
        k += 2
    elif ema_ds < 2.0 * SEC_SLOW_THR:
        k += 1
    else:
        if ema_da > 0.0:
            k -= 1
    if last_k is not None:
        k = int(round(0.5 * last_k + 0.5 * k))
    return int(max(k_min, min(k_max, k)))

# ================== Optimizer calls (serial, cumulative) ==================
def _netlist_has_inst_fast(netlist_path: str, inst_name: str) -> bool:
    pat = re.compile(rf"\b{re.escape(inst_name)}\b\s*\(")
    try:
        with open(netlist_path, "r", encoding="utf-8", errors="ignore") as f:
            for chunk in iter(lambda: f.read(1<<20), ""):
                if pat.search(chunk): return True
    except Exception:
        pass
    return False

def _call_optimizer_once(in_netlist: str, root_inst: str, iter_index: Optional[int] = None,
                         override_k: Optional[int] = None,
                         override_use_gates: Optional[str] = None, eval_backend: Optional[str] = None) -> OptimizationResult:
    if not _netlist_has_inst_fast(in_netlist, root_inst):
        return OptimizationResult(in_netlist, True, note=f"[skip] {root_inst} not in current netlist")
    cmd = SUBCIRCUIT_OPT_CMD + ["--netlist", in_netlist]
    for k, v in SUBCIRCUIT_OPT_COMMON_ARGS.items():
        if override_k is not None and k == "--k":
            continue
        if override_use_gates is not None and k == "--use_gates":
            continue
        cmd += [k, v]
    if override_k is not None:
        cmd += ["--k", str(int(override_k))]
    if override_use_gates is not None:
        cmd += ["--use_gates", override_use_gates]
    
    # <<< NEW: append flag for gnnre / trojan
    if eval_backend == "gnnre":
        cmd += ["--gnnre"]
    if eval_backend == "trojan":
        cmd += ["--trojan"]
        if "rs232" in in_netlist.lower():
            cmd += ["--liberty",  "trojansaint.lib"]
        else:
            cmd += ["--liberty", "saed90nm_max_lt.lib"]
    else:
        cmd += ["--liberty", "NangateOpenCellLibrary_typical.lib"]

    cmd += ["--root_inst", str(root_inst)]
    work_dir = SUBCIRCUIT_OPT_COMMON_ARGS.get("--work_dir","tmp")
    expected_post=None
    if iter_index is not None:
        cmd += ["--iter", str(iter_index)]
        expected_post = os.path.join(work_dir, f"iter_{int(iter_index)}", f"netlist_spliced_post_{int(iter_index)}.v")
        os.makedirs(os.path.dirname(expected_post), exist_ok=True)
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3600)
        raw = (cp.stdout or "") + (cp.stderr or "")
        if ("not found" in raw.lower()) and ("instance" in raw.lower()):
            return OptimizationResult(in_netlist, True, note="[skip] optimizer says instance not found")
        if iter_index is not None:
            ok = (cp.returncode==0) and expected_post and os.path.isfile(expected_post) and os.path.getsize(expected_post)>0
            if not ok and (cp.returncode==0):
                return OptimizationResult(in_netlist, True, note="[warn] optimizer emitted no new netlist; keep original")
            return OptimizationResult(expected_post if ok else in_netlist, ok, note=raw[-2000:])
        else:
            return OptimizationResult(in_netlist, (cp.returncode==0), note=raw[-2000:])
    except Exception as e:
        return OptimizationResult(in_netlist, False, note=f"EXCEPTION: {e}")

def optimize_subcircuit_batch(netlist_path: str, inst_names: List[str], iter_index: Optional[int] = None,
                              override_k: Optional[int] = None,
                              override_use_gates: Optional[str] = None,
                              eval_backend: Optional[str] = None) -> OptimizationResult:
    curr = str(netlist_path)
    notes = []
    for inst in inst_names:
        try:
            res = _call_optimizer_once(
                curr, inst,
                iter_index=iter_index,
                override_k=override_k,
                override_use_gates=override_use_gates,
                eval_backend=eval_backend
            )
            if res.success:
                curr = res.new_netlist
                notes.append(f"[{inst}] success")
            else:
                # Error: record it, keep curr, do not abort
                notes.append(f"[{inst}] failed, keep current netlist ({curr})")
        except Exception as e:
            # Exception: record it, keep curr, do not abort
            notes.append(f"[{inst}] EXCEPTION {e}, keep current netlist ({curr})")

    return OptimizationResult(curr, True, "\n".join(notes))


# ================== Evaluation backends (OMLA & GNN4IP & GNNRE & Trojan) ==================
def _bash_run_in_conda_env(wd: str, conda_env: str, bash_body: str, timeout: int=3600) -> str:
    chain = f'source "{FIXED_CONDA_SH}" && conda activate "{conda_env}" && {bash_body} && conda deactivate'
    cp = subprocess.run(["bash","-lc",chain], cwd=wd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    return cp.stdout or ""

def _parse_security_from_text_omla(text: str) -> float:
    m = re.search(r"SECURITY_SCORE\s*:\s*([-\d\.eE]+)", text)
    if m: return float(m.group(1))
    m2 = re.search(r"\baccuracy\s*=\s*([-\d\.eE]+)", text, re.I)
    if m2: return float(m2.group(1))
    for ln in reversed([ln.strip() for ln in text.strip().splitlines()[-50:]]):
        m3 = re.search(r"([-\d\.eE]+)$", ln)
        if m3:
            try: return float(m3.group(1))
            except: pass
    return 0.0

def _parse_security_from_text_gnn4ip(text: str) -> float:
    if not text:
        return 0.0
    m = re.search(r"\[RESULT\].*?\bsimilarity\s*[:=]\s*([-\d\.eE]+)", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except:
            pass
    m2 = re.search(r"\bsimilarity\s*[:=]\s*([-\d\.eE]+)", text, re.I)
    if m2:
        try:
            return float(m2.group(1))
        except:
            pass
    try:
        for ln in reversed([ln.strip() for ln in text.strip().splitlines()[-50:]]):
            if "similarity" in ln.lower():
                m3 = re.search(r"([-\d\.eE]+)$", ln)
                if m3:
                    return float(m3.group(1))
    except:
        pass
    return 0.0

def _parse_security_from_text_gnnre(text: str) -> float:
    """
    Parse F1_Micro from GNNRE/GraphSAINT-style logs.
    """
    if not text:
        return 0.0
    m = re.search(
        r"\[RESULT\].*?\b(F1[\s_-]*Micro|micro[\s_-]*F1|micro[\s_-]*f1)\b\s*[:=]\s*([-\d\.eE]+)",
        text, re.I
    )
    if m:
        try:
            return float(m.group(2))
        except:
            pass

    patterns = [
        r"\bF1[\s_-]*Micro\b\s*=\s*([-\d\.eE]+)",
        r"\bmicro[\s_-]*F1\b\s*=\s*([-\d\.eE]+)",
        r"\bF1[\s_-]*Micro\b\s*:\s*([-\d\.eE]+)",
        r"\bmicro[\s_-]*F1\b\s*:\s*([-\d\.eE]+)",
        r"\bmicro[_\s-]*f1\b\s*[:=]\s*([-\d\.eE]+)",
    ]
    for pat in patterns:
        m2 = re.search(pat, text, re.I)
        if m2:
            try:
                return float(m2.group(1))
            except:
                pass

    block = re.search(r"Full\s+test\s+stats\s*:?(.*?)(?:\n\s*\n|$)", text, re.I | re.S)
    if block:
        for pat in patterns:
            m3 = re.search(pat, block.group(1), re.I)
            if m3:
                try:
                    return float(m3.group(1))
                except:
                    pass

    try:
        for ln in reversed([ln.strip() for ln in text.strip().splitlines()[-100:]]):
            if "micro" in ln.lower():
                m4 = re.search(r"([-\d\.eE]+)\s*$", ln)
                if m4:
                    return float(m4.group(1))
    except:
        pass

    return 0.0

def _parse_security_from_text_trojan(text: str) -> float:
    """
    Parse the metric value from the [RESULT] line.
    For example:
      [RESULT] iter_001 metric=0.617050
    Returns a float, or raises if parsing fails.
    """
    if not text:
        return 0.0
    m = re.search(r"\[RESULT\].*?\bmetric\s*=\s*([-\d\.eE]+)", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    # Fallback: look for a number on the last line
    last = text.strip().splitlines()[-1] if text.strip().splitlines() else ""
    m2 = re.search(r"([-\d\.eE]+)\s*$", last)
    if m2:
        return float(m2.group(1))
    raise RuntimeError("Could not parse metric from [RESULT] line.\n--- RAW OUTPUT ---\n" + text)

def score_security(netlist: Any, *, iter_i: Optional[int]=None, work_dir: Optional[str]=None,
                   circuit_name: Optional[str]=None, batch_size: int=64, timeout: int=3600,
                   eval_backend: str="omla") -> float:
    if eval_backend not in ("omla","gnn4ip", "gnnre", "trojan"):
        raise ValueError("eval_backend must be 'omla', 'gnn4ip', 'gnnre' or 'trojan")
    if iter_i is None:
        if not netlist: raise ValueError("score_security: netlist required when iter_i is None")
    if eval_backend == "omla":
        if iter_i is None:
            bash_body = (f'python "{FIXED_OMLA_SCRIPT}" '
                         f'--work_dir "{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--netlist "../{netlist}" --batch_size {batch_size}')
        else:
            bash_body = (f'python "{FIXED_OMLA_SCRIPT}" '
                         f'--work_dir "{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--iter {int(iter_i)} --batch_size {batch_size} --verbose')
        try:
            out = _bash_run_in_conda_env(wd=FIXED_OMLA_DIR, conda_env=FIXED_OMLA_ENV, bash_body=bash_body, timeout=timeout)
            sc = _parse_security_from_text_omla(out)
            if sc == 0.0:
                print("[SEC WARN OMLA] parsed 0.0; RAW:\n", out[:1000])
            return sc
        except subprocess.TimeoutExpired:
            print("[SEC ERROR OMLA] Timeout"); return 1.0
        except Exception as e:
            print("[SEC ERROR OMLA]", e); return 1.0
    elif eval_backend == "gnn4ip":
        if iter_i is None:
            bash_body = (f'python "{FIXED_GNN4IP_SCRIPT}" '
                         f'--work_dir "../{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--netlist "../../{netlist}" --verbose')
        else:
            bash_body = (f'python "{FIXED_GNN4IP_SCRIPT}" '
                         f'--work_dir "../{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--iter {int(iter_i)} --verbose')
        try:
            out = _bash_run_in_conda_env(wd=FIXED_GNN4IP_DIR, conda_env=FIXED_GNN4IP_ENV, bash_body=bash_body, timeout=timeout)
            sc = _parse_security_from_text_gnn4ip(out)
            if sc == 0.0:
                print("[SEC WARN GNN4IP] parsed 0.0; RAW:\n", out[:1000])
            return sc
        except subprocess.TimeoutExpired:
            print("[SEC ERROR GNN4IP] Timeout"); return 1.0
        except Exception as e:
            print("[SEC ERROR GNN4IP]", e); return 1.0
    
    elif eval_backend == "gnnre":
        if iter_i is None:
            bash_body = (f'python "{FIXED_GNNRE_SCRIPT}" '
                         f'--work_dir "../{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--netlist "../../{netlist}" --verbose')
        else:
            bash_body = (f'python "{FIXED_GNNRE_SCRIPT}" '
                         f'--work_dir "../{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--iter {int(iter_i)} --verbose')
        try:
            out = _bash_run_in_conda_env(wd=FIXED_GNNRE_DIR, conda_env=FIXED_GNNRE_ENV, bash_body=bash_body, timeout=timeout)
            sc = _parse_security_from_text_gnnre(out)
            if sc == 0.0:
                print("[SEC WARN GNNRE] parsed 0.0; RAW:\n", out[:1000])
            return sc
        except subprocess.TimeoutExpired:
            print("[SEC ERROR GNNRE] Timeout"); return 1.0
        except Exception as e:
            print("[SEC ERROR GNNRE]", e); return 1.0
    
    elif eval_backend == "trojan":
        if iter_i is None:
            bash_body = (f'python "{FIXED_GTSAINT_SCRIPT}" '
                         f'--work_dir "{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--netlist "../{netlist}" --verbose')
        else:
            bash_body = (f'python "{FIXED_GTSAINT_SCRIPT}" '
                         f'--work_dir "{work_dir}" '
                         f'--circuit_name "{circuit_name}" '
                         f'--iter {int(iter_i)} --verbose')
        try:
            out = _bash_run_in_conda_env(wd=FIXED_GTSAINT_DIR, conda_env=FIXED_GTSAINT_ENV, bash_body=bash_body, timeout=timeout)
            sc = _parse_security_from_text_trojan(out)
            if sc == 0.0:
                print("[SEC WARN GTSAINT] parsed 0.0; RAW:\n", out[:1000])
            return sc
        except subprocess.TimeoutExpired:
            print("[SEC ERROR GTSAINT] Timeout"); return 1.0
        except Exception as e:
            print("[SEC ERROR GTSAINT]", e); return 1.0


# ================== Yosys area ==================
def score_qor(netlist: Any, *, work_dir: Optional[str]=None, iter_i: Optional[int]=None) -> QoR:
    netlist_path = str(netlist)
    if "rs232" in netlist_path.lower():
        # RS232 circuits use a smaller technology library
        SUBCIRCUIT_OPT_COMMON_ARGS["--liberty"] = "trojansaint.lib"
    elif "s15850" in netlist_path.lower() or "s35932" in netlist_path.lower() or "s38417" in netlist_path.lower() or "s38584" in netlist_path.lower():
        # S15850/S35932/S38417/S38584 circuits use a larger technology library
        SUBCIRCUIT_OPT_COMMON_ARGS["--liberty"] = "saed90nm_max_lt.lib"
    else:
        # Other circuits use the default technology library
        SUBCIRCUIT_OPT_COMMON_ARGS["--liberty"] = "NangateOpenCellLibrary_typical.lib"
    liberty = SUBCIRCUIT_OPT_COMMON_ARGS.get("--liberty")
    top     = SUBCIRCUIT_OPT_COMMON_ARGS.get("--top")
    if not liberty or not os.path.exists(liberty) or not top:
        print("[QOR] missing liberty/top:", liberty, top); return QoR(area=0.0)

    techlib_path=None
    wd_for_logs = SUBCIRCUIT_OPT_COMMON_ARGS.get("--work_dir")
    if wd_for_logs and iter_i is not None:
        cand = os.path.join(wd_for_logs, f"iter_{int(iter_i)}", "cec", "techlib_auto.v")
        if os.path.isfile(cand) and os.path.getsize(cand)>0: techlib_path=cand

    ys = [f'read_liberty -lib "{liberty}"']
    if techlib_path: ys.append(f'read_verilog -lib "{techlib_path}"')
    ys += [
        f'read_verilog "{netlist_path}"',
        f"hierarchy -check -top {top}",
        f'stat -liberty "{liberty}"'
    ]
    script="\n".join(ys)+"\n"

    area=0.0; scr=None
    try:
        fd, scr = tempfile.mkstemp(suffix=".ys", prefix="qor_")
        os.write(fd, script.encode()); os.close(fd)
        cp = subprocess.run(["yosys","-s",scr], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        out = cp.stdout or ""
        if wd_for_logs and iter_i is not None:
            try:
                os.makedirs(os.path.join(wd_for_logs, f"iter_{int(iter_i)}"), exist_ok=True)
                with open(os.path.join(wd_for_logs, f"iter_{int(iter_i)}","qor_yosys_area.log"),"w") as f: f.write(out)
            except Exception: pass
        for pat in [
            r"Chip\s+area\s+for\s+module\s+'[^']+'\s*:\s*([-\d\.eE\+]+)",
            r"Total\s+cell\s+area\s*:\s*([-\d\.eE\+]+)",
            r"Total\s+area\s*:\s*([-\d\.eE\+]+)",
            r"Design\s+area\s*:\s*([-\d\.eE\+]+)",
        ]:
            m=re.search(pat,out,flags=re.I)
            if m:
                try: area=float(m.group(1)); break
                except: pass
    except Exception as e:
        print("[QOR] yosys failed:", e)
    finally:
        try:
            if scr: os.remove(scr)
        except Exception: pass
    return QoR(area=area)

# Unified tunable reward weights (used only by node-bucket RL and the k-bandit; RL is currently off, functions kept for compatibility)
SEC_WEIGHT  = float(os.getenv("SEC_WEIGHT",  "1.0"))
AREA_WEIGHT = float(os.getenv("AREA_WEIGHT", "0.5"))

# ================== Reward (node-bucket RL) ==================
_REWARD_BASELINE = 0.0
_REWARD_VAR_EMA  = 1e-6
def rl_reward(before: Scores, after: Scores,
              w_s: float = SEC_WEIGHT,
              w_a: float = AREA_WEIGHT,
              gamma: float = 1.5) -> float:
    import math
    eps=1e-9
    ds=(before.security-after.security)/ (abs(before.security)+eps)
    da=(before.qor.area -after.qor.area)/ (before.qor.area+eps)
    ds=max(-1.0,min(1.0,ds)); da=max(-1.0,min(1.0,da))
    r_raw = w_s*ds + w_a*da
    r = math.tanh(gamma*r_raw)
    global _REWARD_BASELINE, _REWARD_VAR_EMA
    beta=0.1
    _REWARD_BASELINE=(1.0-beta)*_REWARD_BASELINE + beta*r
    centered=r-_REWARD_BASELINE
    _REWARD_VAR_EMA=(1.0-beta)*_REWARD_VAR_EMA + beta*(centered*centered)
    return centered/max(1e-6, math.sqrt(_REWARD_VAR_EMA))

# ================== Score aggregation ==================
def scores_of(netlist: Any, **eval_kwargs) -> Scores:
    sec = score_security(netlist, **eval_kwargs)
    qor = score_qor(netlist, work_dir=eval_kwargs.get("work_dir"), iter_i=eval_kwargs.get("iter_i"))
    print(f"  -> security={sec:.6f}, area={qor.area:.6f}")
    return Scores(security=sec, qor=qor)

def _log_iter(logger: Optional[ExperimentLogger], *, phase: str, iter_idx: int,
              timing_s: float,
              eval_backend: str,
              netlist_before: str,
              netlist_after: str,
              before: Scores,
              after: Scores,
              ds_rel: float,
              da_rel: float,
              picks: List[str],
              chosen_combo_name: str,
              chosen_use_gates: str,
              chosen_k: Optional[int],
              pool_digest: Optional[Dict[str, Any]],
              rl_alloc: Optional[List[Dict[str,Any]]],
              rl_probs: Optional[Dict[str,float]],
              rl_feats: Optional[Dict[str,List[float]]],
              reward_node: Optional[float],
              optim_notes: Optional[str],
              best_scores: Optional[Scores],
              early_stop: Optional[str]=None):
    if logger is None:
        return
    combo_hist = _COMBO_PERF.get(chosen_combo_name, {}).copy() if chosen_combo_name else {}
    payload = {
        "phase": phase,                  # "security" or "area"
        "iter": iter_idx,
        "eval_backend": eval_backend,
        "timing_s": round(float(timing_s), 3),
        "netlist_before": netlist_before,
        "netlist_after": netlist_after,
        "security": {
            "before": float(before.security),
            "after":  float(after.security),
            "delta":  float(after.security - before.security),
            "rel":    float(ds_rel),
        },
        "area": {
            "before": float(before.qor.area),
            "after":  float(after.qor.area),
            "delta":  float(after.qor.area - before.qor.area),
            "rel":    float(da_rel),
        },
        "selection": {
            "combo": {
                "name":  chosen_combo_name,
                "use_gates": chosen_use_gates,
                "history_snapshot": combo_hist
            },
            "nodes": picks,
            "k": chosen_k,
            "pool": pool_digest or {}
        },
        "rl": {
            "bucket_alloc": rl_alloc or [],
            "bucket_probs": rl_probs or {},
            "bucket_feats": rl_feats or {}
        },
        "reward_nodeRL": reward_node,
        "optimizer_notes_tail": (optim_notes[-1000:] if isinstance(optim_notes, str) else None),
        "best_scores_snapshot": (
            {"security": float(best_scores.security), "area": float(best_scores.qor.area)}
            if best_scores is not None else None
        ),
        "early_stop": early_stop
    }
    logger.log("iteration", payload)



_iter_dir_pat  = re.compile(r"[\\/](?:iter_)(\d+)(?:[\\/]|$)")
_iter_file_pat = re.compile(r"(?:netlist_spliced_post_)(\d+)\.v\b")
def _extract_iter_from_path(p: Any) -> Optional[int]:
    s=str(p)
    m=_iter_dir_pat.search(s)
    if m: return int(m.group(1))
    m2=_iter_file_pat.search(s)
    if m2: return int(m2.group(1))
    return None

from collections import deque

def run(initial_netlist: Any, max_iters: int=MAX_ITERS, *,
        work_dir: Optional[str]=None, circuit_name: Optional[str]=None,
        batch_size: int=64, eval_backend: str="omla", logger: Optional[ExperimentLogger]=None) -> Any:
    random.seed(SEED)
    rl  = RLBucketPolicy() if RL_ENABLE else None
    krl = RLKPolicy() if RLK_ENABLE else None
    load_visited_opt(work_dir, circuit_name)
    _load_combo_perf(work_dir)

    curr = str(initial_netlist)
    best_netlist = curr
    best_scores  = None

    base_evalkw = dict(
        work_dir=(os.path.join("..", work_dir) if work_dir else None),
        circuit_name=circuit_name,
        batch_size=batch_size,
        eval_backend=eval_backend
    )

    slow_sec_count = 0
    last_combo_name: Optional[str] = None

    ds_hist = deque(maxlen=8)
    da_hist = deque(maxlen=8)
    ds_rel_prev = None
    da_rel_prev = None
    last_k_used = None

    # === gnnre only: refresh security every 10 rounds ===

    def _gnnre_should_refresh(iter_idx: int) -> bool:
        return (iter_idx % GNNRE_UPDATE_EVERY) == 0

    # Record the security from the most recent "real evaluation" (used to hold it on non-refresh rounds)
    last_full_security: Optional[float] = None

    for t in range(max_iters):
        start = time.time()
        before_iter = _extract_iter_from_path(curr)

        # ---------- BEFORE scoring ----------
        before = scores_of(curr, **(base_evalkw | {"iter_i": before_iter}))

        if eval_backend == "gnnre":
            if _gnnre_should_refresh(t):
                # Refresh round: accept the real security and update the cache
                last_full_security = before.security
                print(f"[iter {t}] GNNRE security REFRESH (before) = {before.security:.6f}")
            else:
                # BEFORE (gnnre non-refresh round)
                if last_full_security is not None:
                    print(f"[iter {t}] GNNRE security HELD (before) = {last_full_security:.6f}")
                    before = Scores(security=last_full_security, qor=before.qor)

        if best_scores is None or before.security < best_scores.security:
            best_netlist, best_scores = curr, before

        # ---------- Early stop (only effective for the matching backend; no undefined vars referenced here) ----------
        if eval_backend == "omla" and before.security <= EARLY_STOP_FOR_OMLA:
            print(f"[early-stop] OMLA security {before.security:.6f} <= {EARLY_STOP_FOR_OMLA:.3f} at iter {t} (before); stopping.")
            return best_netlist
        if eval_backend == "gnn4ip" and before.security <= EARLY_STOP_FOR_GNN4IP:
            print(f"[iter {t}] Early stop (GNN4IP before {before.security:.6f} <= {EARLY_STOP_FOR_GNN4IP})")
            return best_netlist
        if eval_backend == "gnnre" and _gnnre_should_refresh(t) and before.security <= EARLY_STOP_FOR_GNNRE:
            print(f"[iter {t}] Early stop (GNNRE before {before.security:.6f} <= {EARLY_STOP_FOR_GNNRE})")
            return best_netlist
        if eval_backend == "trojan" and before.security < EARLY_STOP_FOR_GTSAINT:
            print(f"[iter {t}] Early stop (Trojan before {before.security:.6f} <= {EARLY_STOP_FOR_GTSAINT})")
            return best_netlist

        # ---------- Select nodes (from RL pool; step 1) ----------
        # inst2bucket, bucket2insts = build_buckets(curr)
        # pool, alloc, probs, feats, prefer_base = rl_propose_pool(
        #     netlist_path=curr, bucket2insts=bucket2insts, rl=rl, budget_try=TRY_PER_ROUND,
        #     work_dir=work_dir, circuit_name=circuit_name
        # )
        # pool_digest = {"size": len(pool), "fingerprint": _fingerprint(sorted(list(set(pool))))}

        # ---------- Select nodes (purely random pool; no RL / no bucketing) ----------
        pool, alloc, probs, feats, prefer_base = propose_pool_random(
            netlist_path=curr, budget_try=TRY_PER_ROUND,
            work_dir=work_dir, circuit_name=circuit_name
        )
        pool_digest = {"size": len(pool), "fingerprint": _fingerprint(sorted(list(set(pool))))}

        picks = llm_choose_from_pool(
            netlist=curr, pool=pool, try_per_round=TRY_PER_ROUND,
            security=before.security, area=before.qor.area, prefer_base=prefer_base,
            work_dir=work_dir, circuit_name=circuit_name
        )
        if not picks:
            picks = pool[:TRY_PER_ROUND]
        print(f"[iter {t}] prefer_base={prefer_base} pool={len(pool)} picks={picks}")

        # ---------- Select gate combo (step 2; pass picks / pool_digest) ----------
        need_switch = (slow_sec_count >= SLOW_SWITCH_ROUNDS)
        if last_combo_name and _COMBO_PERF.get(last_combo_name, {}).get("ema_ds", 0.0) > STICKY_EMA_DS:
            need_switch = False

        chosen_combo_name = llm_choose_combo_by_history(
            before.security, before.qor.area,
            last_combo=last_combo_name, need_switch=need_switch,
            work_dir=SUBCIRCUIT_OPT_COMMON_ARGS.get("--work_dir"),
            last_ds_rel=ds_rel_prev, last_da_rel=da_rel_prev,
            recent_ds_rel=list(ds_hist), recent_da_rel=list(da_hist),
            picked_nodes=picks, pool_digest=pool_digest
        )
        chosen_use_gates = GATE_COMBOS[chosen_combo_name]
        print(f"[iter {t}] combo={chosen_combo_name} use_gates=[{chosen_use_gates}]")

        # ---------- Select k (step 3; pass picks / pool_digest / combo) ----------
        chosen_k = llm_choose_k(
            security=before.security,
            area=before.qor.area,
            recent_ds_rel=list(ds_hist),
            recent_da_rel=list(da_hist),
            combo_name=chosen_combo_name,
            use_gates=chosen_use_gates,
            prefer_small_k=True,
            k_min=3, k_max=10,
            last_k=last_k_used,
            picked_nodes=picks, pool_digest=pool_digest
        )
        last_k_used = chosen_k
        print(f"[iter {t}] hop-size k (LLM) = {chosen_k}")

        # ---------- Run optimization ----------
        res = optimize_subcircuit_batch(curr, picks, iter_index=t+1,
                                        override_k=chosen_k,
                                        override_use_gates=chosen_use_gates,
                                        eval_backend=eval_backend)
        
        if not res.success:
            print(f"[iter {t}] optimizer failed; stop.\n{res.note[-400:]}")
            break
        candidate_net = res.new_netlist

        # ---------- AFTER scoring ----------
        after = scores_of(candidate_net, **(base_evalkw | {"iter_i": t+1}))
        if eval_backend == "gnnre":
            if _gnnre_should_refresh(t):
                last_full_security = after.security
                print(f"[iter {t}] GNNRE security REFRESH (after) = {after.security:.6f}")
            else:
                # Non-refresh round: hold security, keep it consistent with before
                if last_full_security is not None:
                    print(f"[iter {t}] GNNRE security HELD (after) = {before.security:.6f}")
                    after = Scores(security=before.security, qor=after.qor)     

        curr = candidate_net
        mark_visited_opt(picks, work_dir, circuit_name)

        # ---------- Early stop (after; gnnre checks only on refresh rounds) ----------
        if eval_backend == "omla" and after.security < EARLY_STOP_FOR_OMLA:
            if (best_scores is None) or (after.security < best_scores.security):
                best_netlist, best_scores = curr, after
            print(f"[early-stop] OMLA security {after.security:.6f} < {EARLY_STOP_FOR_OMLA:.3f} at iter {t}; stopping.")
            return best_netlist

        if eval_backend == "gnn4ip" and after.security < EARLY_STOP_FOR_GNN4IP:
            if (best_scores is None) or (after.security < best_scores.security):
                best_netlist, best_scores = curr, after
            print(f"[early-stop] GNN4IP security {after.security:.6f} < {EARLY_STOP_FOR_GNN4IP:.3f} at iter {t}; stopping.")

        if eval_backend == "gnnre" and _gnnre_should_refresh(t) and after.security < EARLY_STOP_FOR_GNNRE:
            if (best_scores is None) or (after.security < best_scores.security):
                best_netlist, best_scores = curr, after
            print(f"[early-stop] GNNRE security {after.security:.6f} < {EARLY_STOP_FOR_GNNRE:.3f} at iter {t}; stopping.")


        if eval_backend == "trojan" and after.security < EARLY_STOP_FOR_GTSAINT:
            if (best_scores is None) or (after.security < best_scores.security):
                best_netlist, best_scores = curr, after
            print(f"[early-stop] GTSAINT security {after.security:.6f} < {EARLY_STOP_FOR_GTSAINT:.3f} at iter {t}; stopping.")

        # ---------- RL (node buckets) ----------
        reward_node = rl_reward(before, after)
        # RL_ENABLE=False -> no update, no save

        # ---------- k-bandit ----------
        eps = 1e-9
        da_rel = (before.qor.area - after.qor.area) / (before.qor.area + eps)
        ds_rel = (before.security - after.security) / (abs(before.security) + eps)
        # On non-refresh rounds ds_rel=0 (security is held), so only area drives the update
        ds_hist.append(ds_rel)
        da_hist.append(da_rel)
        ds_rel_prev = ds_rel
        da_rel_prev = da_rel
        # RLK_ENABLE=False -> policy is not updated

        # ---------- Slow-drop tracking + history ----------
        if ds_rel < SEC_DROP_MIN_REL:
            slow_sec_count += 1
        else:
            slow_sec_count = 0
        last_combo_name = chosen_combo_name
        _update_combo_perf(chosen_combo_name, ds_rel, da_rel, work_dir)

        dt = time.time() - start
        print(f"[iter {t}] Δsec={after.security-before.security:+.6f} Δarea={after.qor.area-before.qor.area:+.3f} "
              f"reward(nodeRL)={reward_node:+.3f}")
        print(f"[iter {t}] took {dt:.2f}s; best_security={(best_scores.security if best_scores else before.security):.6f}", flush=True)
        # Log this round
        _log_iter(
            logger,
            phase="security",
            iter_idx=t,
            timing_s=(time.time() - start),
            eval_backend=eval_backend,
            netlist_before=str(before_iter),
            netlist_after=str(curr),
            before=before,
            after=after,
            ds_rel=ds_rel,
            da_rel=da_rel,
            picks=picks,
            chosen_combo_name=chosen_combo_name,
            chosen_use_gates=chosen_use_gates,
            chosen_k=chosen_k,
            pool_digest=pool_digest,
            rl_alloc=alloc,
            rl_probs=probs,
            rl_feats=feats,
            reward_node=reward_node,
            optim_notes=res.note,
            best_scores=best_scores,
            early_stop=None
        )

    return best_netlist


def run_area(initial_netlist: Any, max_iters: int=20, *,
             work_dir: Optional[str]=None, circuit_name: Optional[str]=None,
             batch_size: int=64, eval_backend: str="omla", logger: Optional[ExperimentLogger]=None) -> Any:
    """
    Stage two: area optimization (uses LLM calls).
    Goal: reduce area without letting security get worse.
    """

    SEC_TOL  = float(os.getenv("SEC_NO_WORSE_TOL", "5e-4"))
    AREA_TOL = float(os.getenv("AREA_IMPROVE_MIN", "5e-3"))
    PLATEAU_ROUNDS = int(os.getenv("AREA_PLATEAU_ROUNDS", "10"))

    curr = str(initial_netlist)
    best_netlist = curr
    best_scores  = None

    base_evalkw = dict(
        work_dir=(os.path.join("..", work_dir) if work_dir else None),
        circuit_name=circuit_name,
        batch_size=batch_size,
        eval_backend=eval_backend
    )

    last_combo_name=None
    last_k_used=None
    plateau = 0
    from collections import deque
    ds_hist = deque(maxlen=8)
    da_hist = deque(maxlen=8)

    last_full_security: Optional[float] = None

    def _gnnre_should_refresh(iter_idx: int) -> bool:
        return (iter_idx % GNNRE_UPDATE_EVERY) == 0

    for t in range(max_iters):
        before_iter = _extract_iter_from_path(curr)
        before = scores_of(curr, **(base_evalkw | {"iter_i": before_iter}))

        # gnnre: refresh / hold security
        if eval_backend == "gnnre":
            if _gnnre_should_refresh(t):
                last_full_security = before.security
            else:
                if last_full_security is not None:
                    before = Scores(security=last_full_security, qor=before.qor)

        # ---------- LLM node selection ----------
        inst2bucket, bucket2insts = build_buckets(curr)
        pool, alloc, probs, feats, prefer_base = rl_propose_pool(
            netlist_path=curr, bucket2insts=bucket2insts, rl=None,
            budget_try=TRY_PER_ROUND, work_dir=work_dir, circuit_name=circuit_name
        )
        pool_digest = {"size": len(pool), "fingerprint": _fingerprint(sorted(list(set(pool))))}

        picks = llm_choose_from_pool(
            netlist=curr, pool=pool, try_per_round=TRY_PER_ROUND,
            security=before.security, area=before.qor.area,
            prefer_base=prefer_base,
            work_dir=work_dir, circuit_name=circuit_name
        )
        if not picks:
            picks = pool[:TRY_PER_ROUND]
        print(f"[AREA iter {t}] picks={picks}")

        # ---------- LLM combo selection ----------
        chosen_combo_name = llm_choose_combo_area_priority(
            before.security, before.qor.area,
            last_combo=last_combo_name, need_switch=False,
            work_dir=work_dir,
            last_ds_rel=(ds_hist[-1] if ds_hist else None),
            last_da_rel=(da_hist[-1] if da_hist else None),
            recent_ds_rel=list(ds_hist), recent_da_rel=list(da_hist),
            picked_nodes=picks, pool_digest=pool_digest
        )
        # chosen_use_gates = GATE_COMBOS[chosen_combo_name]
        chosen_use_gates = GATE_COMBOS_AREA[chosen_combo_name]
        print(f"[AREA iter {t}] combo={chosen_combo_name} use_gates=[{chosen_use_gates}]")

        # ---------- LLM k selection ----------
        chosen_k = llm_choose_k_area_priority(
            security=before.security,
            area=before.qor.area,
            recent_ds_rel=list(ds_hist),
            recent_da_rel=list(da_hist),
            combo_name=chosen_combo_name,
            use_gates=chosen_use_gates,
            prefer_small_k=True,   # bias toward smaller, less complexity
            k_min=1, k_max=3,
            last_k=last_k_used,
            picked_nodes=picks, pool_digest=pool_digest
        )
        last_k_used = chosen_k
        print(f"[AREA iter {t}] hop-size k (LLM area-phase) = {chosen_k}")

        # ---------- Run optimization ----------
        res = optimize_subcircuit_batch(curr, picks, iter_index=t+1,
                                        override_k=chosen_k,
                                        override_use_gates=chosen_use_gates,
                                        eval_backend=eval_backend)
        if not res.success:
            print(f"[AREA iter {t}] optimizer failed; stop.")
            break
        candidate_net = res.new_netlist

        # ---------- AFTER scoring ----------
        after = scores_of(candidate_net, **(base_evalkw | {"iter_i": t+1}))
        if eval_backend == "gnnre":
            if _gnnre_should_refresh(t):
                last_full_security = after.security
            else:
                if last_full_security is not None:
                    after = Scores(security=before.security, qor=after.qor)

        # ---------- Hard constraints ----------
        eps = 1e-9
        ds_rel = (before.security - after.security) / (abs(before.security) + eps)
        da_rel = (before.qor.area - after.qor.area) / (before.qor.area + eps)
        if (ds_rel < -SEC_TOL) or (da_rel <= AREA_TOL):
            print(f"[AREA iter {t}] REJECT: Δsec={ds_rel:+.6f}, Δarea={da_rel:+.6f}")
            _log_iter(
                logger,
                phase="area",
                iter_idx=t,
                timing_s=0.0,
                eval_backend=eval_backend,
                netlist_before=str(curr),
                netlist_after=str(curr),    # rejected, do not update
                before=before,
                after=after,
                ds_rel=ds_rel,
                da_rel=da_rel,
                picks=picks,
                chosen_combo_name=chosen_combo_name,
                chosen_use_gates=chosen_use_gates,
                chosen_k=chosen_k,
                pool_digest=pool_digest,
                rl_alloc=None, rl_probs=None, rl_feats=None,
                reward_node=None,
                optim_notes=res.note,
                best_scores=best_scores,
                early_stop="area_reject"
            )

            plateau += 1
            if plateau >= PLATEAU_ROUNDS:
                print(f"[AREA] Early stop by plateau ({PLATEAU_ROUNDS}).")
                break
            continue

        # ---------- Accept ----------
        plateau = 0
        curr = candidate_net
        _log_iter(
            logger,
            phase="area",
            iter_idx=t,
            timing_s=0.0,
            eval_backend=eval_backend,
            netlist_before=str(before_iter),
            netlist_after=str(curr),
            before=before, after=after,
            ds_rel=ds_rel, da_rel=da_rel,
            picks=picks,
            chosen_combo_name=chosen_combo_name,
            chosen_use_gates=chosen_use_gates,
            chosen_k=chosen_k,
            pool_digest=pool_digest,
            rl_alloc=None, rl_probs=None, rl_feats=None,
            reward_node=None,
            optim_notes=res.note,
            best_scores=best_scores,
            early_stop=None
        )

        mark_visited_opt(picks, work_dir, circuit_name)
        _update_combo_perf(chosen_combo_name, ds_rel, da_rel, work_dir)

        ds_hist.append(ds_rel)
        da_hist.append(da_rel)
        last_combo_name = chosen_combo_name

        # Update best
        if (best_scores is None) or \
           (after.security < best_scores.security - SEC_TOL) or \
           (abs(after.security - best_scores.security) <= SEC_TOL and after.qor.area < best_scores.qor.area):
            best_netlist, best_scores = curr, after

        print(f"[AREA iter {t}] ACCEPT Δsec={ds_rel:+.6f}, Δarea={da_rel:+.6f}")

    return best_netlist



# ================== CLI ==================
if __name__ == "__main__":
    # Set your OpenAI API key via the OPENAI_API_KEY environment variable before running.
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

    parser = argparse.ArgumentParser(description="LLM-driven combo (history-only, fixed allow/block lists); RL disabled.")
    parser.add_argument("--netlist", type=str, default="locked_c1355_0_1_0_0_0_1_flat.v")
    parser.add_argument("--work_dir", type=str, default=os.environ.get("WORK_DIR","tmp"),
                        help="working directory for subcircuit_opt (e.g. tmp) - the evaluator uses ../{work_dir} internally")
    parser.add_argument("--circuit_name", type=str, required=True,
                        help="dataset circuit name (e.g. c1355 / gnn4ip_c432_BE280); top is set to locked_<name> (OMLA) or <name> (GNN4IP)")
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("OMLA_BATCH","64")),
                        help="batch size for the OMLA attack (ignored for GNN4IP/others)")
    parser.add_argument("--eval_backend", type=str, choices=["omla","gnn4ip", "gnnre", "trojan"], default=os.environ.get("EVAL_BACKEND","omla"),
                        help="choose the security evaluation backend: omla / gnn4ip / gnnre / trojan")
    parser.add_argument("--max_iters", type=int, default=MAX_ITERS, help="maximum number of iterations")
    args = parser.parse_args()

    # ============== Instantiate the JSON logger ==============
    log_path = os.path.join(args.work_dir, "run_log.jsonl") if args.work_dir else "run_log.jsonl"
    LOGGER = ExperimentLogger(log_path)
    LOGGER.log("run_start", {
        "script": "llm_exo_norl.py",
        "args": {
            "netlist": args.netlist,
            "work_dir": args.work_dir,
            "circuit_name": args.circuit_name,
            "batch_size": args.batch_size,
            "eval_backend": args.eval_backend,
            "max_iters": args.max_iters
        },
        "env": _env_snapshot(),
        "seed": SEED
    })

    # Update the parameters subcircuit_opt needs
    SUBCIRCUIT_OPT_COMMON_ARGS["--work_dir"] = args.work_dir
    if args.eval_backend == "omla":
        SUBCIRCUIT_OPT_COMMON_ARGS["--top"] = "locked_" + args.circuit_name
    else:
        SUBCIRCUIT_OPT_COMMON_ARGS["--top"] = args.circuit_name

    # === Stage 1: reduce security ===
    print("=== Stage 1: Security reduction (LLM-driven; RL disabled) ===")
    final_netlist = run(
        args.netlist,
        max_iters=args.max_iters,
        work_dir=args.work_dir,
        circuit_name=args.circuit_name,
        batch_size=args.batch_size,
        eval_backend=args.eval_backend,
        logger=LOGGER,
    )
    print("[Stage 1 DONE] Netlist after security reduction:", final_netlist)

    # For the stage-two area optimization, uncomment the block below
    # print("\n=== Stage 2: Area reduction (LLM-driven; RL disabled) ===")
    # final_netlist_area = run_area(
    #     final_netlist,
    #     max_iters=args.max_iters,
    #     work_dir=args.work_dir,
    #     circuit_name=args.circuit_name,
    #     batch_size=args.batch_size,
    #     eval_backend=args.eval_backend,
    #     logger=LOGGER,
    # )
    # print("[Stage 2 DONE] Netlist after area reduction:", final_netlist_area)

    LOGGER.log("run_end", {
        "final_netlist_security": str(final_netlist),
        # "final_netlist_area": str(final_netlist_area)
    })

    print("DONE:", final_netlist)
