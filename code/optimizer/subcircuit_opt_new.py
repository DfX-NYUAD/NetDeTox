#!/usr/bin/env python3
import argparse, os, random, json, subprocess, sys, copy
from collections import defaultdict, deque
from pathlib import Path

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def q(p: str | Path) -> str:
    # Safe shell quoting
    return shlex.quote(str(p))
# -----------------------------
# Utils
# -----------------------------
# def run(cmd):
#     p = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
#     print(p.stdout)
#     if p.returncode != 0:
#         raise RuntimeError(f"CMD FAIL: {cmd}")

def run(cmd, quiet=True, cwd=None, log_path=None, tail_lines=200):
    """
    Run a command:
      - quiet=True: print nothing on success; on failure print the last tail_lines lines and raise
      - quiet=False: also print the full output on success
      - log_path: write the full output to this file regardless of success or failure
      - return value: always returns the full stdout string
    """
    p = subprocess.run(
        cmd, shell=True, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    out = p.stdout or ""

    # Save the log (if specified)
    if log_path:
        try:
            Path(log_path).write_text(out)
        except Exception:
            pass

    if p.returncode != 0:
        # Only print the tail if the log is long
        tail = "\n".join(out.splitlines()[-tail_lines:]) if tail_lines else out
        print(tail)
        raise RuntimeError(f"CMD FAIL: {cmd}")

    if not quiet:
        print(out)
    return out

def uniq(name, used):
    base = name
    i = 0
    while name in used:
        i += 1
        name = f"{base}_{i}"
    used.add(name)
    return name

# -----------------------------
# Parse gate-level Verilog -> graph using pyverilog
# -----------------------------
from pyverilog.vparser.parser import parse
from pyverilog.vparser.ast import (
    ModuleDef, InstanceList, Instance, PortArg, Decl,
    Wire, Input, Output, Identifier, Pointer, Partselect, IntConst, Concat
)

def _expr_to_basename(expr):
    """Convert a port expression into a base sanitized signal name string; return None for constants/concatenations."""
    if expr is None:
        return None
    # Named signal
    if isinstance(expr, Identifier):
        return expr.name
    # Bit select foo[i]
    if isinstance(expr, Pointer):
        if isinstance(expr.var, Identifier):
            return expr.var.name
        return _expr_to_basename(expr.var)
    # Part select foo[msb:lsb]
    if isinstance(expr, Partselect):
        if isinstance(expr.var, Identifier):
            return expr.var.name
        return _expr_to_basename(expr.var)
    # Concatenation {a,b} cannot be uniquely resolved -> skip
    if isinstance(expr, Concat):
        return None
    # Constant / other
    if isinstance(expr, IntConst):
        return None
    # Fallback: try hasattr .name
    return getattr(expr, 'name', None)

class NetlistGraph:
    def __init__(self, topmod:str):
        self.top = topmod
        self.instances = {}          # inst_name -> {'cell': celltype, 'pins': {pinname: netname}}
        self.net_drivers = defaultdict(list) # net -> [(inst, pin)]
        self.net_loads   = defaultdict(list) # net -> [(inst, pin)]
        self.primary_inputs  = set()
        self.primary_outputs = set()

    def from_ast(self, ast):
        # Find top
        tops = [d for d in ast.description.definitions
                if isinstance(d, ModuleDef) and d.name == self.top]
        if not tops:
            raise ValueError(f"Top {self.top} not found")
        m = tops[0]

        # Record PIs/POs
        for item in m.items:
            if isinstance(item, Decl):
                for obj in item.list:
                    if isinstance(obj, Input):
                        self.primary_inputs.add(obj.name)
                    elif isinstance(obj, Output):
                        self.primary_outputs.add(obj.name)

        # Instance parsing (handles named/positional ports & various expressions)
        for item in m.items:
            if not isinstance(item, InstanceList):
                continue
            cell = item.module
            for inst in item.instances:
                assert isinstance(inst, Instance)
                iname = inst.name
                pins = {}
                # Some netlists use positional ports (portname=None); give them sequential names
                pos_idx = 0
                for pa in inst.portlist:
                    assert isinstance(pa, PortArg)
                    # Get the expression object: handle different pyverilog versions
                    arg_expr = getattr(pa, 'arg', None)
                    if arg_expr is None:
                        arg_expr = getattr(pa, 'argname', None)
                    net = _expr_to_basename(arg_expr)
                    if net is None:
                        # Constant/concatenation etc., skip
                        pos_idx += 1
                        continue
                    # Port name (named port preferred; otherwise use _p{idx})
                    if pa.portname is None:
                        pname = f"_p{pos_idx}"
                    else:
                        pname = pa.portname
                    pins[pname] = net
                    pos_idx += 1
                self.instances[iname] = {'cell': cell, 'pins': pins}

        # Build driver/load tables (output-pin heuristic + full load registration)
        out_hint = {'Z','ZN','Y','Q','QN','OUT','O','S','CO'}
        for iname, data in self.instances.items():
            pins = data['pins']
            # Record everything as a load first, then distinguish via the driver table
            for p, n in pins.items():
                self.net_loads[n].append((iname, p))
            # Mark drivers using the pin-name heuristic
            for p, n in pins.items():
                if p is not None and p.upper() in out_hint:
                    self.net_drivers[n].append((iname, p))
        # For nets still without a driver, do not force inference; boundary detection later treats them as PIs or external signals

    def build_graph(self):
        # instance-level graph edges via nets: driver inst -> load inst
        g_succ = defaultdict(set)
        g_pred = defaultdict(set)
        for net, drivers in self.net_drivers.items():
            for (dinst, dpin) in drivers:
                for (linst, lpin) in self.net_loads.get(net, []):
                    if linst != dinst:
                        g_succ[dinst].add(linst)
                        g_pred[linst].add(dinst)
        return g_pred, g_succ

# -----------------------------
# Extract k-hop fanin cone (by instances)
# -----------------------------

# ==== NEW: instance category detection ====
def inst_category(name: str) -> str | None:
    n = name.lower()
    if n.startswith("add"):       return "adder"
    if n.startswith("mul"):  return "multiplier"
    if n.startswith("sub"):  return "subtractor"
    if n.startswith("comp"):  return "comparator"
    if name.startswith("U"):        return "U"
    return None

# ==== NEW: k-hop fanin that only backtracks within allow_set ====
def k_hop_fanin_cone_filtered(root_inst, g_pred, k, allow_set: set[str] | None):
    S = {root_inst}
    frontier = {root_inst}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in g_pred.get(u, []):
                if (allow_set is None) or (v in allow_set):
                    nxt.add(v)
        S |= nxt
        frontier = nxt
    return S


def k_hop_fanin_cone(root_inst, g_pred, k):
    S = {root_inst}
    frontier = {root_inst}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in g_pred.get(u, []):
                nxt.add(v)
        S |= nxt
        frontier = nxt
    return S

def cone_boundary(graph: NetlistGraph, cone_insts:set, root_inst:str):
    """Return:
       - boundary_nets_in : nets entering the cone from outside or PIs
       - cone_out_net     : the net driven by root_inst (assume single-output)
    """
    # find root output net (heuristic via driver table)
    root_outs = []
    for net, drivers in graph.net_drivers.items():
        for (inst, pin) in drivers:
            if inst == root_inst:
                root_outs.append(net)
    if not root_outs:
        raise ValueError(f"Root {root_inst} seems output-less (driver not found). Please set output pin hints or choose another root.")
    cone_out_net = root_outs[0]  # assume single-output cone

    # boundary inputs: any net that feeds a pin of an inst in cone, but whose driver is outside cone or net is a PI
    boundary_in = set()
    for inst in cone_insts:
        pins = graph.instances[inst]['pins']
        for p,n in pins.items():
            # if this net has a driver outside cone, and this pin is not the driver pin
            drivers = graph.net_drivers.get(n, [])
            if not drivers:
                # no explicit driver (could be PI or constant)
                if n in graph.primary_inputs:
                    boundary_in.add(n)
                continue
            for (dinst, dpin) in drivers:
                if dinst not in cone_insts:
                    boundary_in.add(n)
    return sorted(boundary_in), cone_out_net

# -----------------------------
# Emit subcircuit Verilog (structural) copied from original cone
# -----------------------------
# def emit_subcone_verilog(
#     graph: NetlistGraph,
#     cone_insts: set,
#     boundary_in: list,
#     cone_out_net: str,
#     sub_name: str = "subcone",
#     out_path: str = "subcone_raw.v"
# ):
#     _OUTPIN_RE = re.compile(
#         r'^(?:Z|ZN|Z\d+|ZN\d+|Y|Y\d+|Q|QN|QB|QBAR|O\d*|S|CO|SUM|OUT)$',
#         re.I
#     )
#     def _is_out_pin(pin_name: str) -> bool:
#         return bool(_OUTPIN_RE.match(pin_name))

#     # 1) net->drivers/users: prefer the graph's own tables, fall back to heuristics if missing
#     if hasattr(graph, "net_drivers") and hasattr(graph, "net_users"):
#         net_drivers = graph.net_drivers
#         net_users   = graph.net_users
#     else:
#         net_drivers, net_users = {}, {}
#         for inst, info in graph.instances.items():
#             for p, n in info["pins"].items():
#                 (net_drivers if _is_out_pin(p) else net_users).setdefault(n, []).append((inst, p))

#     # 2) Extra boundary outputs
#     extra_boundary_outs = set()
#     for inst in cone_insts:
#         for p, n in graph.instances[inst]["pins"].items():
#             if not _is_out_pin(p):
#                 continue
#             users = net_users.get(n, [])
#             used_outside = any(u_inst not in cone_insts for (u_inst, _up) in users)
#             if used_outside and (n != cone_out_net) and (n not in boundary_in):
#                 extra_boundary_outs.add(n)

#     # 3) Ports (inputs first, outputs after)
#     port_inputs  = list(dict.fromkeys(boundary_in))  # dedup while preserving order
#     port_outputs = [cone_out_net] + sorted(extra_boundary_outs)
#     port_names   = set(port_inputs) | set(port_outputs)

#     netmap: dict[str, str] = {}
#     for n in port_inputs + port_outputs:
#         netmap[n] = n

#     # 3.5) Fill gaps: nets on some instances' "input pins" inside the cone that have no driver inside the cone -> must be inputs
#     missing_inputs = set()
#     for inst in cone_insts:
#         for p, n in graph.instances[inst]["pins"].items():
#             if _is_out_pin(p):
#                 continue
#             drivers = net_drivers.get(n, [])
#             has_internal_driver = any(d_inst in cone_insts for (d_inst, _dp) in drivers)
#             if (not has_internal_driver) and (n not in port_names):
#                 missing_inputs.add(n)
#     if missing_inputs:
#         for n in sorted(missing_inputs):
#             port_inputs.append(n)
#             port_names.add(n)
#             netmap[n] = n
#         # No need for internal_wires -= missing_inputs -- step 4 won't add them again anyway

#     # 4) Wires used only internally
#     internal_wires = set()
#     for inst in cone_insts:
#         for p, n in graph.instances[inst]['pins'].items():
#             if (n not in netmap) and (n not in port_names):
#                 internal_wires.add(n)
#                 netmap[n] = n

#     # 5) Write Verilog
#     with open(out_path, "w") as f:
#         f.write(f"module {sub_name} (\n")
#         port_lines = [f"  input {n}" for n in port_inputs] + [f"  output {n}" for n in port_outputs]
#         f.write(",\n".join(port_lines) + "\n")
#         f.write(");\n")
#         if internal_wires:
#             f.write("  wire " + ", ".join(sorted(internal_wires)) + ";\n")
#         for inst in cone_insts:
#             data = graph.instances[inst]
#             cell = data['cell']
#             pins = data['pins']
#             plist = [f".{p}({netmap.get(n, n)})" for p, n in pins.items()]
#             f.write(f"  {cell} {inst} ( " + ", ".join(plist) + " );\n")
#         f.write("endmodule\n")
    
#     # print("emit_info_port_inputs", port_inputs)
#     return {
#         "module": sub_name,
#         "inputs": port_inputs,
#         "outputs": port_outputs,
#         "netmap": netmap,
#         "boundary_extra_outputs": sorted(extra_boundary_outs),
#     }
def emit_whole_as_submodule(
    original_v: str,
    top: str,
    sub_name: str,
    out_path: str
) -> dict:
    """Copy the entire top module's instance block into a standalone submodule (I/O kept identical to top)"""
    in_names, out_names = _get_top_ios(original_v, top)
    inst_block = _extract_all_instance_lines(original_v, sub_top_hint=top)

    with open(out_path, "w") as f:
        f.write(f"module {sub_name} (\n")
        port_lines = [f"  input {n}" for n in in_names] + [f"  output {n}" for n in out_names]
        f.write(",\n".join(port_lines) + "\n")
        f.write(");\n")
        if inst_block.strip():
            f.write(inst_block + "\n")
        f.write("endmodule\n")

    return {
        "module": sub_name,
        "inputs": in_names,
        "outputs": out_names,
        "netmap": {n: n for n in (in_names + out_names)},
        "boundary_extra_outputs": []
    }


def emit_subcone_verilog(
    graph: NetlistGraph,
    cone_insts: set,
    boundary_in: list,
    cone_out_net: str,
    sub_name: str = "subcone",
    out_path: str = "subcone_raw.v"
):
    """
    Extract structural Verilog from a cone:
      - Ports: inputs first, outputs after; cone_out_net goes first, remaining overflow outputs follow
      - MUX's S is treated as an input; Adder's S/SUM/CO are treated as outputs; other cells use the generic output names
      - Automatically fill in missing inputs (used inside the cone but with no driver inside the cone and non-constant), rewriting the file once if needed
    The returned info includes inputs/outputs/netmap etc., for subsequent port rewriting.
    """
    # ---------- Helpers ----------
    _OUTPIN_RE_GENERIC = re.compile(
        r'^(?:Z|ZN|Z\d+|ZN\d+|Y|Y\d+|Q|QN|QB|QBAR|O\d*|CO|SUM|OUT)$', re.I
    )

    def _is_out_pin(cell_name: str, pin_name: str) -> bool:
        Uc = (cell_name or "").upper()
        Up = (pin_name or "").upper()
        # MUX: S is the select input, outputs are usually Z/ZN/Y/Q/O*
        if "MUX" in Uc:
            return Up in ("Z", "ZN", "Y", "Q", "O", "O1", "O2")
        # Adder: S/SUM and CO are outputs
        if re.search(r"(ADDF|ADDH|FA|HA)", Uc):
            return Up in ("S", "SUM", "CO")
        # Other cells use the generic set (excluding a bare S)
        return bool(_OUTPIN_RE_GENERIC.match(Up))

    def _is_const_net(n: str) -> bool:
        # Filter out constant literals like 1'b0/1/binary/hex/x/z
        return isinstance(n, str) and bool(re.match(r"^1'[bhod][0-9a-fxz]+$", n, re.I))

    # ---------- 1) Get net->drivers / net->users ----------
    if hasattr(graph, "net_drivers") and hasattr(graph, "net_users"):
        net_drivers = graph.net_drivers   # net -> [(inst, pin)]
        net_users   = graph.net_users     # net -> [(inst, pin)]
    else:
        net_drivers, net_users = {}, {}
        for inst, info in graph.instances.items():
            cell = info.get("cell", "")
            for p, n in info.get("pins", {}).items():
                (net_drivers if _is_out_pin(cell, p) else net_users).setdefault(n, []).append((inst, p))

    # ---------- 2) Extra boundary outputs: driven inside the cone but still used outside ----------
    extra_boundary_outs = set()
    for inst in cone_insts:
        info = graph.instances[inst]
        cell = info.get("cell", "")
        for p, n in info.get("pins", {}).items():
            if not _is_out_pin(cell, p):
                continue
            users = net_users.get(n, [])
            used_outside = any(u_inst not in cone_insts for (u_inst, _up) in users)
            if used_outside and (n != cone_out_net) and (n not in boundary_in):
                extra_boundary_outs.add(n)

    # ---------- 3) Initial ports ----------
    port_inputs  = list(dict.fromkeys(boundary_in))           # dedup while preserving order
    port_outputs = [cone_out_net] + sorted(extra_boundary_outs)
    port_names   = set(port_inputs) | set(port_outputs)
    netmap: dict[str, str] = {n: n for n in (port_inputs + port_outputs)}

    # ---------- 3.5) One-pass gap fill: nets "used but with no internal driver" inside the cone are treated as inputs ----------
    # driven_in_cone: prefer to decide via net_drivers
    if net_drivers:
        driven_in_cone = {
            n for n, ds in net_drivers.items()
            if any(d_inst in cone_insts for (d_inst, _dp) in ds)
        }
    else:
        driven_in_cone = set()
        for inst in cone_insts:
            info = graph.instances[inst]
            cell = info.get("cell", "")
            for p, n in info.get("pins", {}).items():
                if _is_out_pin(cell, p):
                    driven_in_cone.add(n)

    # used_in_cone: all net names referenced by instances inside the cone (in first-appearance order)
    used_in_cone_order, used_seen = [], set()
    for inst in cone_insts:
        for _p, n in graph.instances[inst].get("pins", {}).items():
            if n not in used_seen:
                used_in_cone_order.append(n)
                used_seen.add(n)

    missing_inputs = []
    for n in used_in_cone_order:
        if _is_const_net(n) or (n in port_names) or (n in driven_in_cone):
            continue
        missing_inputs.append(n)

    if missing_inputs:
        for n in missing_inputs:
            port_inputs.append(n)
            port_names.add(n)
            netmap[n] = n

    # ---------- 4) internal wires: appear inside the cone, not a port, and non-constant ----------
    internal_wires = set()
    for inst in cone_insts:
        for _p, n in graph.instances[inst].get("pins", {}).items():
            if (n not in port_names) and (not _is_const_net(n)):
                internal_wires.add(n)
                netmap[n] = n

    # ---------- 5) Write once ----------
    def _write_once():
        with open(out_path, "w") as f:
            f.write(f"module {sub_name} (\n")
            port_lines = [f"  input {n}" for n in port_inputs] + [f"  output {n}" for n in port_outputs]
            f.write(",\n".join(port_lines) + "\n")
            f.write(");\n")
            if internal_wires:
                f.write("  wire " + ", ".join(sorted(internal_wires)) + ";\n")
            for inst in cone_insts:
                data = graph.instances[inst]
                cell = data.get('cell', '')
                pins = data.get('pins', {})
                plist = [f".{p}({netmap.get(n, n)})" for p, n in pins.items()]
                f.write(f"  {cell} {inst} ( " + ", ".join(plist) + " );\n")
            f.write("endmodule\n")

    _write_once()

    # ---------- 6) Second check: still missing inputs? Add them and rewrite if so ----------
    still_missing = []
    for n in used_in_cone_order:
        if _is_const_net(n) or (n in port_names) or (n in driven_in_cone):
            continue
        still_missing.append(n)

    if still_missing:
        for n in still_missing:
            if n not in port_names:
                port_inputs.append(n)
                port_names.add(n)
                netmap[n] = n
        # Since the port set changed, internal_wires needs to be rebuilt
        internal_wires.clear()
        for inst in cone_insts:
            for _p, n in graph.instances[inst].get("pins", {}).items():
                if (n not in port_names) and (not _is_const_net(n)):
                    internal_wires.add(n)
                    netmap[n] = n
        _write_once()

    return {
        "module": sub_name,
        "inputs": port_inputs,
        "outputs": port_outputs,
        "netmap": netmap,
        "boundary_extra_outputs": sorted(extra_boundary_outs),
    }




from pathlib import Path
import re

def _detect_out(pin_names):
    order = ['ZN','Z','Y','Q','QN','OUT','O','CO','S']
    up = [p.upper() for p in pin_names]
    for p in order:
        if p in up: return pin_names[up.index(p)]
    return pin_names[-1]

def _autogen_techlib(sub_v="subcone_raw.v", outlib="techlib_auto.v"):
    txt = Path(sub_v).read_text()
    inst_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s+[A-Za-z0-9_]+\s*\((.*?)\);\s*$',
                         re.M|re.S)
    port_re = re.compile(r'\.\s*([A-Za-z0-9_]+)\s*\(')
    cells = {}
    for m in inst_re.finditer(txt):
        cell, ports = m.group(1), m.group(2)
        pins = port_re.findall(ports)
        if pins and cell not in cells:
            cells[cell] = pins
    lines = []
    for cell, pins in cells.items():
        y  = _detect_out(pins)
        xs = [p for p in pins if p != y]
        U  = cell.upper()
        if   "INV"  in U: expr = f"~{xs[0]}"
        elif "BUF"  in U: expr = f"{xs[0]}"
        elif "NAND" in U: expr = "~(" + " & ".join(xs) + ")"
        elif "NOR"  in U: expr = "~(" + " | ".join(xs) + ")"
        elif "XNOR" in U: expr = f"~({xs[0]} ^ {xs[1]})"
        elif "XOR"  in U: expr = f"{xs[0]} ^ {xs[1]}"
        elif re.search(r'\bAND\b', U): expr = " & ".join(xs)
        elif re.search(r'\bOR\b',  U): expr = " | ".join(xs)
        else: expr = xs[-1]
        plist = ", ".join([*xs, y])
        decl  = "\n".join([*(f"  input {i};" for i in xs), f"  output {y};"])
        lines.append(f"module {cell}({plist});\n{decl}\n  assign {y} = {expr};\nendmodule\n")
    Path(outlib).write_text("\n\n".join(lines))
    return outlib

from pathlib import Path
from typing import Dict, Optional

def _ys_escape(name: str) -> str:
    # Simple escaping for Yosys selectors/renames: prefix a backslash to non-standard identifiers
    return name if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) else "\\" + name

# Matches a Verilog "escaped identifier" (\name<whitespace>) or a plain identifier
_ESC_ID_TERM = r'(?=[\s,();\[\].]|$)'

def _rewrite_escaped_identifier(text: str, old: str, new: str) -> str:
    # 1) Escaped form: \old<whitespace/separator/end-of-line>
    pat_escaped = re.compile(r'\\' + re.escape(old) + _ESC_ID_TERM)
    text = pat_escaped.sub(new, text)
    # 2) Plain form: the whole identifier equals old (avoid replacing substrings)
    pat_plain = re.compile(r'(?<!\\)\b' + re.escape(old) + r'\b')
    text = pat_plain.sub(new, text)
    return text

def _rewrite_module_name(text: str, new_module: Optional[str]) -> str:
    if not new_module:
        return text
    # module <name> ( ... );
    return re.sub(
        r'(^\s*module\s+)(\\?[^\s(]+)',
        r'\1' + new_module,
        text,
        count=1,
        flags=re.MULTILINE
    )

def rewrite_verilog_ports(
    verilog_in: str,
    verilog_out: str,
    port_map: Dict[str, str],
    new_module_name: Optional[str] = None
):
    txt = Path(verilog_in).read_text()
    txt = _rewrite_module_name(txt, new_module_name)
    for cur, tgt in port_map.items():
        print(f"[rewrite] {cur} -> {tgt}")
        txt = _rewrite_escaped_identifier(txt, cur, tgt)
    Path(verilog_out).write_text(txt)

def _liberty_cell_names(lib_path):
    txt = Path(lib_path).read_text()
    return re.findall(r"cell\s*\(\s*([A-Za-z0-9_]+)\s*\)", txt)

def liberty_dont_use_by_prefixes(lib_path, prefixes=("AOI","OAI","MUX")):
    """Original blacklist mode: disable all cells whose names start with prefixes."""
    names = _liberty_cell_names(lib_path)
    banned = [n for n in names if any(n.startswith(p) for p in prefixes)]
    return "" if not banned else " " + " ".join(f"-dont_use {n}" for n in banned)

def liberty_dont_use_except_prefixes(lib_path, allow_prefixes):
    """
    Whitelist mode: only allow cells whose names start with allow_prefixes; -dont_use all others.
    Example: allow_prefixes = ["INV","BUF","AND","NAND","NOR","OR","XOR","XNOR"]
    """
    allow_prefixes = tuple(allow_prefixes or [])
    names = _liberty_cell_names(lib_path)
    banned = [n for n in names if not any(n.startswith(p) for p in allow_prefixes)]
    return "" if not banned else " " + " ".join(f"-dont_use {n}" for n in banned)


def liberty_dont_use_flags(lib_path, prefixes=("AOI", "OAI")):
    """
    Grab cell names from Liberty that start with prefixes and generate a
    ' -dont_use <cellA> -dont_use <cellB> ...' string.
    This avoids hand-writing every variant like AOI21_X1/AOI22_X4.
    """
    txt = Path(lib_path).read_text()
    # Grab all cell names: cell (NAME) { ... }
    names = re.findall(r"cell\s*\(\s*([A-Za-z0-9_]+)\s*\)", txt)
    banned = [n for n in names if any(n.startswith(p) for p in prefixes)]
    if not banned:
        print("[warn] No AOI/OAI cells found in Liberty; please confirm the library file contains these gates.")
        return ""
    # print("[info] Disabling these cells:", banned)
    return " " + " ".join(f"-dont_use {n}" for n in banned)

def _strip_comments_and_attrs(s: str) -> str:
    # Remove block & line comments
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//.*?$", "", s, flags=re.M)
    # Remove Yosys/Verilog attributes: (* ... *)
    s = re.sub(r"\(\*.*?\*\)", "", s, flags=re.S)
    return s

def _scan_balanced(text: str, start_idx: int, open_ch: str = "(", close_ch: str = ")") -> tuple[str, int]:
    """Starting from the '(' at start_idx, return the matched substring content (without the parentheses) and the end index (index of the closing paren)."""
    assert text[start_idx] == open_ch, "scan_balanced: not at open paren"
    depth = 0
    i = start_idx
    n = len(text)
    i += 1  # Skip the first '('
    start_content = i
    while i < n:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            if depth == 0:
                return text[start_content:i], i
            depth -= 1
        i += 1
    raise RuntimeError("scan_balanced: unbalanced parentheses")

def _get_top_ios(verilog_path: str, top: str) -> tuple[list[str], list[str]]:
    """Parse top's input/output names from the netlist (keeping the header order as much as possible)"""
    raw = Path(verilog_path).read_text()
    text = _strip_comments_and_attrs(raw)

    # Find the top module block
    m_mod = re.search(rf"\bmodule\b\s+(?:\\\S+|{re.escape(top)})\b", text)
    if not m_mod:
        raise RuntimeError(f"[whole] Module {top} not found")
    i = m_mod.end()
    # Find '('
    j = text.find('(', i)
    if j == -1: raise RuntimeError("[whole] Could not locate the port list '('")
    plist, r = _scan_balanced(text, j, "(", ")")
    header = plist

    # Port name order (preserve order as much as possible)
    raw_ports = [p.strip() for p in header.replace("\n"," ").split(",") if p.strip()]

    def _last_ident(tok: str) -> str:
        tok = re.sub(r"\[[^]]+\]", " ", tok)
        toks = re.findall(r"[A-Za-z_]\w*", tok)
        return toks[-1] if toks else tok

    raw_ports = [_last_ident(p) for p in raw_ports]

    # Direction sets
    def collect(dir_kw):
        names = []
        for mm in re.finditer(rf"\b{dir_kw}\b\s*(?:\[[^]]+\]\s*)?([^;]+);", text):
            chunk = mm.group(1)
            chunk = re.sub(r"\[[^]]+\]", " ", chunk)
            names += [x.strip() for x in chunk.replace("\n"," ").split(",") if x.strip()]
        norm = set()
        for name in names:
            norm.update(re.findall(r"[A-Za-z_]\w*", name))
        return norm

    in_set  = collect("input")
    out_set = collect("output")

    cur_inputs  = [p for p in raw_ports if p in in_set  and p not in out_set]
    cur_outputs = [p for p in raw_ports if p in out_set and p not in in_set]
    # Fault tolerance: if anything remains, assign it once more via in_set/out_set
    remain = [p for p in raw_ports if p not in cur_inputs and p not in cur_outputs]
    for p in remain:
        if p in in_set: cur_inputs.append(p)
        elif p in out_set: cur_outputs.append(p)

    return cur_inputs, cur_outputs


def build_port_map_from_verilog(verilog_path, tgt_inputs, tgt_outputs):
    raw = Path(verilog_path).read_text()
    text = _strip_comments_and_attrs(raw)

    # --- Find the first module declaration: supports escaped module names e.g. \subcone_opt.aag ---
    m_mod = re.search(
        r"\bmodule\b\s+(?P<mname>(?:\\\S+|[A-Za-z_]\w*))",
        text
    )
    if not m_mod:
        raise RuntimeError("module declaration not found (still failed after stripping comments/attributes)")
    i = m_mod.end()  # Current position is right after the module name

    # Skip whitespace
    n = len(text)
    while i < n and text[i].isspace():
        i += 1

    # Optional parameter block: #(...)
    if i < n and text[i] == '#':
        i += 1
        while i < n and text[i].isspace():
            i += 1
        if i >= n or text[i] != '(':
            raise RuntimeError("Failed to parse parameterized module header: no '(' after '#'")
        _, end_paren = _scan_balanced(text, i, "(", ")")
        i = end_paren + 1

    # Handle more whitespace/newlines: don't require '(' immediately after, search for the next '('
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != '(':
        # Just find the first '(' from the current position
        j = text.find("(", i)
        if j == -1:
            ctx = text[max(0, i-80):min(n, i+80)]
            raise RuntimeError(f"Could not locate the port list '(' , context: >>>{ctx}<<<")
        i = j

    # Port list
    plist, right_paren = _scan_balanced(text, i, "(", ")")

    # Split port names (keep the previous handling)
    raw_ports = [p.strip() for p in plist.replace("\n", " ").split(",") if p.strip()]

    def _last_ident(tok: str) -> str:
        tok = re.sub(r"\[[^]]+\]", " ", tok)  # Strip bit width
        toks = re.findall(r"[A-Za-z_]\w*", tok)
        return toks[-1] if toks else tok

    raw_ports = [_last_ident(p) for p in raw_ports]

    # Collect directions
    def collect(dir_kw):
        names = []
        for mm in re.finditer(rf"\b{dir_kw}\b\s*(?:\[[^]]+\]\s*)?([^;]+);", text):
            chunk = mm.group(1)
            chunk = re.sub(r"\[[^]]+\]", " ", chunk)
            names += [x.strip() for x in chunk.replace("\n"," ").split(",") if x.strip()]
        norm = set()
        for name in names:
            ids = re.findall(r"[A-Za-z_]\w*", name)
            norm.update(ids)
        return norm

    in_set  = collect("input")
    out_set = collect("output")

    cur_inputs  = [p for p in raw_ports if p in in_set and p not in out_set]
    cur_outputs = [p for p in raw_ports if p in out_set and p not in in_set]

    if len(cur_inputs) + len(cur_outputs) != len(raw_ports):
        remaining = [p for p in raw_ports if p not in cur_inputs and p not in cur_outputs]
        for p in remaining:
            if p in in_set:   cur_inputs.append(p)
            elif p in out_set: cur_outputs.append(p)

    if len(cur_inputs) != len(tgt_inputs) or len(cur_outputs) != len(tgt_outputs):
        raise RuntimeError(
            f"Port count mismatch: cur_in={len(cur_inputs)} tgt_in={len(tgt_inputs)}; "
            f"cur_out={len(cur_outputs)} tgt_out={len(tgt_outputs)}; "
            f"raw_ports={raw_ports}"
        )

    pm = {}
    for c, t in zip(cur_inputs, tgt_inputs):
        pm[c] = t
    for c, t in zip(cur_outputs, tgt_outputs):
        pm[c] = t
    return pm

def rewrite_verilog_ports_v2(
    verilog_in: str,
    verilog_out: str,
    port_map: dict[str, str],
    new_module_name: str | None = None,
):
    """
    More robust port renaming tool:
    1) Remove redundant wire declarations that share a name with a port
    2) Whole-word replace port names (including the module header, port declarations, and references in the module body)
    3) Optionally rename the module
    """
    text = Path(verilog_in).read_text()

    # ---- 0) Safe prep: take out the set of ports to change (only rename these names)
    old_ports = set(port_map.keys())
    # If the map has identity mappings (a->a), drop them to avoid pointless replacement
    old_ports = {p for p in old_ports if port_map.get(p) != p}

    # ---- 1) Optional: rename the module (only the first module declaration)
    if new_module_name:
        text = re.sub(
            r'(\bmodule\s+)([A-Za-z_]\w*)',
            lambda m: m.group(1) + new_module_name,
            text,
            count=1
        )

    # ---- 2) Remove redundant wire declarations that share a name with a port
    # Supports "wire a;" or multiple on one line "wire a, b, c;"
    def strip_redundant_wires(m):
        decl = m.group(0)
        names = [s.strip() for s in m.group("names").split(",")]
        # Filter out those sharing a name with a port
        kept = [n for n in names if n not in old_ports]
        if not kept:
            return ""  # Delete the whole line
        return re.sub(r'(?<=wire)\s+[^;]+;', " " + ", ".join(kept) + ";", decl)

    text = re.sub(
        r'(^\s*wire\s+(?P<names>[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*;\s*$)',
        strip_redundant_wires,
        text,
        flags=re.M
    )

    # ---- 3) Whole-word replace port names (affects header, input/output declarations, and references in the module body)
    # To avoid prefix/suffix collisions like "_02_" vs "_02__tmp", use word boundaries + escaping
    # Also sort by descending length so replacing "_0_" first doesn't partially match "_02_"
    for old in sorted(old_ports, key=len, reverse=True):
        new = port_map[old]
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)

    Path(verilog_out).write_text(text)

def _kill_assigns_with_buf(
    v_in: str,
    v_out: str,
    cell: str = "BUF_X1",
    in_pin: str = "A",
    out_pin: str = "Z",
    inst_prefix: str = "U_BUF"
):
    """
    Rewrite continuous assignments in Verilog:
        assign LHS = RHS;
    into buffer instances:
        BUF_X1 U_BUF# ( .A(RHS), .Z(LHS) );
    Rules:
      - LHS/RHS support: identifier (including $ and \escaped) + optional bit/part select [i] or [msb:lsb]
      - RHS can also be a constant (e.g. 1'b0/1'b1/...)
      - If LHS/RHS are both part selects with the same width, automatically expand into per-bit BUFs
      - Only single-line assigns are handled; complex expressions are left as-is
    Returns:
      {"replaced": N, "leftover": M}
    """
    text = Path(v_in).read_text()

    # Simply remove block comments to avoid false matches (line/column counts don't matter)
    text_wo_block = re.sub(r"/\*.*?\*/", lambda m: " " * (m.end() - m.start()), text, flags=re.S)

    # Token patterns
    IDENT_BASE = r'(?:\\[^ \t\r\n]+|[A-Za-z_]\w*|\$[A-Za-z_]\w*)'
    INDEX      = r'(?:\[\s*\d+\s*(?::\s*\d+\s*)?\])'   # [i] or [msb:lsb]
    IDENT_FULL = rf'{IDENT_BASE}(?:\s*{INDEX})?'       # Allow optional bit/part select
    CONST      = r"(?:\d*'s?[bhod][0-9a-fxz_]+|\d+)"   # Handles 1'b0 / 8'hFF / 0 etc.

    ASSIGN_RE = re.compile(
        rf'^\s*assign\s+(?P<lhs>{IDENT_FULL})\s*=\s*(?P<rhs>{IDENT_FULL}|{CONST})\s*;\s*$',
        re.I
    )

    # Helpers to parse bit/part selects
    _range_re = re.compile(rf'^(?P<base>{IDENT_BASE})\s*\[\s*(?P<a>\d+)\s*:\s*(?P<b>\d+)\s*\]\s*$')
    _bit_re   = re.compile(rf'^(?P<base>{IDENT_BASE})\s*\[\s*(?P<i>\d+)\s*\]\s*$')

    def _parse_slice(expr: str):
        """Return ('bit', base, i) or ('range', base, msb, lsb) or ('id', expr)"""
        m = _bit_re.match(expr)
        if m:
            return ('bit', m.group('base'), int(m.group('i')))
        m = _range_re.match(expr)
        if m:
            a, b = int(m.group('a')), int(m.group('b'))
            return ('range', m.group('base'), a, b)
        # Plain identifier or constant
        return ('id', expr.strip())

    out_lines = []
    replaced = 0
    leftover = 0
    inst_idx = 0

    # Process line by line, preserving // comments
    for raw_line in text_wo_block.splitlines(keepends=True):
        code, sep, comment = raw_line.partition("//")
        m = ASSIGN_RE.match(code)
        if not m:
            out_lines.append(raw_line)
            continue

        lhs_raw = m.group('lhs').strip()
        rhs_raw = m.group('rhs').strip()

        # Parse both sides
        lt = _parse_slice(lhs_raw)
        rt = _parse_slice(rhs_raw)

        def emit_buf(lhs_expr: str, rhs_expr: str):
            nonlocal inst_idx, replaced
            inst_idx += 1
            line = f"  {cell} {inst_prefix}{inst_idx} ( .{in_pin}({rhs_expr}), .{out_pin}({lhs_expr}) );"
            if comment:
                line += "  // " + comment.strip()
            out_lines.append(line + "\n")
            replaced += 1

        # Case classification
        if lt[0] == 'range' and rt[0] == 'range' and lt[2] - lt[3] == rt[2] - rt[3]:
            # Same-width part selects, expand bit by bit
            lmsb, llsb = lt[2], lt[3]
            rmsb, rlsb = rt[2], rt[3]
            lstep = 1 if lmsb >= llsb else -1
            rstep = 1 if rmsb >= rlsb else -1
            for li, ri in zip(range(lmsb, llsb - lstep, -lstep),
                              range(rmsb, rlsb - rstep, -rstep)):
                emit_buf(f"{lt[1]}[{li}]", f"{rt[1]}[{ri}]")
        elif lt[0] in ('id','bit') and rt[0] in ('id','bit') or (rt[0] == 'id' and re.match(rf'^{CONST}$', rhs_raw, re.I)):
            # Scalar or single bit: just one BUF
            emit_buf(lhs_raw, rhs_raw)
        else:
            # Unsupported complex expression or width mismatch: leave as-is and count as leftover
            # Still write back the original line (with comments)
            out_lines.append(raw_line)
            leftover += 1

    Path(v_out).write_text("".join(out_lines))
    return {"replaced": replaced, "leftover": leftover}



def optimize_sub_aig(
    in_v="subcone_raw.v",
    top="subcone",
    out_v="subcone_opt.v",
    liberty="NangateOpenCellLibrary_typical.lib",
    stdcell_func_v=None,
    boundary_in=None,
    cone_out=None,
    emit_info=None,
    work_dir="tmp",
    allow_gates=None,                 # Whitelist (takes precedence)
    ban_prefixes=("AOI","OAI","MUX"), # Blacklist (used when allow_gates is not provided)
    abc_recipe=None,                  # New: AIG optimization sequence (None uses the default)
    map_extra="",                     # New: extra abc args for the mapping stage (e.g. " -fast" / " -D 1000")
):
    W = ensure_dir(work_dir)

    # Function library: prefer the provided .v, otherwise auto-generate a simplified techlib by guessing expressions from instances
    techlib_v = stdcell_func_v if (stdcell_func_v and Path(stdcell_func_v).exists()) \
                else _autogen_techlib_from_many([in_v], W / "techlib_auto_sub.v")

    # 1) Yosys -> AIG / AAG
    ys1_path   = W / "to_aig_sub.ys"
    aag_path   = W / "subcone.aag"
    ys1 = f"""
read_liberty -lib {liberty}
read_verilog {q(techlib_v)}
read_verilog {q(in_v)}
hierarchy -check -top {top}
flatten
proc; opt; fsm; opt; memory; opt
techmap; opt
aigmap
opt_clean
write_aiger {q(aag_path)}
"""
    Path(ys1_path).write_text(ys1)
    try:
        run(f"yosys -q -s {q(ys1_path)}")
        use_blif = False
    except Exception as e:
        print("[warn] write_aiger failed, fallback to BLIF path:", e)
        use_blif = True

    if use_blif:
        ys1b_path = W / "to_blif_sub.ys"
        blif_path = W / "subcone.blif"
        ys1b = f"""
read_liberty -lib {liberty}
read_verilog {q(techlib_v)}
read_verilog {q(in_v)}
hierarchy -check -top {top}
flatten
proc; opt; fsm; opt; memory; opt
techmap; opt
aigmap
opt_clean
write_blif {q(blif_path)}
"""
        Path(ys1b_path).write_text(ys1b)
        run(f"yosys -q -s {q(ys1b_path)}")

    # 2) ABC: AIG-level optimization
    if not use_blif:
        abc_in  = f"read_aiger {q(aag_path)}"
        orig    = aag_path
    else:
        abc_in  = f"read_blif {q(W / 'subcone.blif')}"
        orig    = W / "subcone.blif"

    aag_opt_path = W / "subcone_opt.aag"
    seq = abc_recipe or "strash; dch; dc2; rewrite -z; refactor -z; resub -K 6; balance"
    abc_cmd = (
        f"{abc_in}; "
        f"{('source ' + os.environ['NETDETOX_ABC_RC'] + '; ') if os.environ.get('NETDETOX_ABC_RC') else ''}"
        f"{seq}; "
        f"write_aiger {q(aag_opt_path)}; "
        f"cec {q(orig)} {q(aag_opt_path)}"
    )
    print(f"[ABC] Running: {abc_cmd}")
    run(f"abc -c {q(abc_cmd)}")

    # 3) Probe ports (AIG->JSON)
    probe_ys = W / "probe_ports.ys"
    probe_json = W / "sub_probe.json"
    Path(probe_ys).write_text(f"""
read_aiger {q(aag_opt_path)}
hierarchy -auto-top
write_json {q(probe_json)}
""")
    run(f"yosys -q -s {q(probe_ys)}")
    design = json.loads(Path(probe_json).read_text())
    mods = design.get("modules", {})
    if not mods:
        raise RuntimeError("No modules after read_aiger subcone_opt.aag")
    top_mod = next(iter(mods.keys()))
    ports = mods[top_mod].get("ports", {})

    cur_inputs, cur_outputs = [], []
    for pname, pobj in ports.items():
        if pobj.get("direction") == "input":
            cur_inputs.append((pname, pobj["bits"]))
        elif pobj.get("direction") == "output":
            cur_outputs.append((pname, pobj["bits"]))
    cur_inputs.sort(key=lambda x: x[1]); cur_outputs.sort(key=lambda x: x[1])
    cur_in_names  = [name for (name, _) in cur_inputs]
    cur_out_names = [name for (name, _) in cur_outputs]

    # 4) Target port names: use emit_info's I/O (works for both whole and subcone)
    if emit_info and emit_info.get("inputs"):
        tgt_in_names = list(dict.fromkeys(emit_info["inputs"]))
    else:
        tgt_in_names = list(dict.fromkeys(boundary_in or []))
    if emit_info and "outputs" in emit_info and emit_info["outputs"]:
        outs = emit_info["outputs"]
        tgt_out_names = [cone_out] + [o for o in outs if o != cone_out]
    else:
        tgt_out_names = [cone_out]

    # 5) Map AIG back to Verilog (random gate policy + extra mapping args)
    if allow_gates:
        ban_flags = liberty_dont_use_except_prefixes(liberty, allow_gates)
    else:
        ban_flags = liberty_dont_use_by_prefixes(liberty, ban_prefixes)
    tmp_v = W / "subcone_opt.tmp.v"
    ys2_path = W / "aig2v_tmp.ys"
    ys2 = f"""
read_liberty -lib {liberty}
read_aiger {q(aag_opt_path)}
abc -liberty {liberty}{ban_flags}{map_extra}
clean
write_verilog {q(tmp_v)}
"""
    Path(ys2_path).write_text(ys2)
    run(f"yosys -q -s {q(ys2_path)}")

    port_map = build_port_map_from_verilog(tmp_v, tgt_in_names, tgt_out_names)
    rewrite_verilog_ports(
        verilog_in=tmp_v,
        verilog_out=out_v,
        port_map=port_map,
        new_module_name="subcone_opt"
    )
    stats = _kill_assigns_with_buf(
        v_in=out_v, v_out=out_v,
        cell="BUF_X1", in_pin="A", out_pin="Z", inst_prefix="U_BUF"
    )
    print("assign->BUF:", stats)




# -----------------------------
# Splice optimized subcircuit back into original netlist
#   - Remove cone instances from top
#   - Declare wires/ports as needed
#   - Instantiate optimized submodule with same boundary nets / output net
# -----------------------------
from pyverilog.ast_code_generator.codegen import ASTCodeGenerator
from copy import deepcopy

def _normalize_inst_block(txt: str) -> str:
    """
    Normalize instance text into a "one instance per line" netlist style (safe rebuild version):
      - Split into instance blocks by paren depth: a block ends at ';' when depth is 0
      - For each block, extract the header (cell + instance name) and pin-list, then rebuild into:
        CELL INST ( .PIN(net), .PIN(net) );
      - Strictly ensure each block ends with a single ');'
    """
    s = txt.strip()
    if not s:
        return ""

    # 1) Split into instance blocks
    blocks, par, buf = [], 0, []
    for ch in s:
        buf.append(ch)
        if ch == '(':
            par += 1
        elif ch == ')':
            if par > 0:
                par -= 1
        if ch == ';' and par == 0:
            blk = ''.join(buf).strip()
            if blk:
                blocks.append(blk)
            buf = []
    tail = ''.join(buf).strip()
    if tail:
        blocks.append(tail)

    out_lines = []
    # 2) Parse and rebuild block by block
    for blk in blocks:
        b = blk.strip()

        # Remove extra newlines to ease parsing
        b = re.sub(r'\s+', ' ', b).strip()

        # Find the first "header(" pair; header is "CELL INST" (INST may be an escaped name)
        m = re.match(r'^([A-Za-z_]\w*)\s+([A-Za-z_\\][^(\s]*)\s*\(', b)
        if not m:
            # Not a regular instance format; fallback: return the original block but ensure it ends with ');'
            b = re.sub(r'\)+\s*;\s*$', ');', b)
            out_lines.append(b)
            continue

        cell, inst = m.group(1), m.group(2)
        head_end = m.end()  # Points right after '('

        # Find the last ')' matching the instance-level '(' (before ';')
        # Use rfind to locate the last ')'; if not found, set pinlist empty
        semi = b.rfind(';')
        if semi == -1:
            semi = len(b)
        close = b.rfind(')', 0, semi)
        pin_blob = b[head_end:close] if close != -1 else ''

        # Split the pinlist by comma (at outer depth 0)
        pins = []
        cur, depth = [], 0
        for ch in pin_blob:
            if ch == '(':
                depth += 1
            elif ch == ')':
                if depth > 0:
                    depth -= 1
            if ch == ',' and depth == 0:
                token = ''.join(cur).strip()
                if token:
                    pins.append(token)
                cur = []
            else:
                cur.append(ch)
        last = ''.join(cur).strip()
        if last:
            pins.append(last)

        # Normalize each ".PIN(arg)" (remove extra whitespace, enforce format)
        norm_pins = []
        for t in pins:
            mm = re.match(r'^\.\s*([A-Za-z_]\w*)\s*\(\s*([^)]+?)\s*\)$', t.strip())
            if mm:
                pin, arg = mm.group(1), mm.group(2)
                norm_pins.append(f'.{pin}({arg})')
            else:
                # If not standard, try to repair: grab pin and arg
                # e.g. ".A1 ( net )" or ".ZN ( \escaped )"
                mm2 = re.search(r'\.\s*([A-Za-z_]\w*)\s*\(\s*([^)]+?)\s*\)', t)
                if mm2:
                    pin, arg = mm2.group(1), mm2.group(2)
                    norm_pins.append(f'.{pin}({arg})')
                else:
                    # As a last resort, strip whitespace and put it back as-is (rare)
                    norm_pins.append(re.sub(r'\s+', ' ', t.strip()))

        # Rebuild one line: CELL INST ( .PIN(arg), ... );
        line = f"{cell} {inst} ( " + ", ".join(norm_pins) + " );"
        out_lines.append(line)

    return "\n".join(out_lines)

def _extract_cone_instance_block(sub_v: str, sub_top_hint: str|None,
                                 cone_out_net: str, boundary_in: list[str]) -> str:
    """
    Extract from sub_v: the instance text for the entire fanin cone that directly/indirectly drives cone_out_net (multiple blocks).
    - Identify module inputs (from 'input ...;' and the ANSI header).
    - Candidate instance output pin names: Z, ZN, Q, QN, Y, O, S, CO (extensible).
    Returns a multi-line string assembled in dependency order (without extra blank lines).
    """
    with open(sub_v, "r") as f:
        txt = f.read()

    # -------- Find module blocks (supports escaped names \foo.bar) --------
    mod_blocks = []
    for m in re.finditer(r'(^\s*module\s+(?P<name>\\\S+|[A-Za-z_]\w*)\b.*?^\s*endmodule\s*)',
                         txt, flags=re.S|re.M):
        mod_blocks.append((m.group('name').strip(), m.start(), m.end()))
    if not mod_blocks:
        raise RuntimeError(f"[extract] No module in {sub_v}")
    block = None
    if sub_top_hint:
        for name, s, e in mod_blocks:
            if name == sub_top_hint:
                block = txt[s:e]
                break
    if block is None:
        block = txt[mod_blocks[0][1]:mod_blocks[0][2]]

    # -------- Find module inputs (header and in-body declarations) --------
    # ANSI header: module X (input a, output b, ...);
    head_m = re.search(r'\bmodule\b\s+(?:\\\S+|[A-Za-z_]\w*)\s*\((?P<head>.*?)\)\s*;', block, flags=re.S)
    head_inputs = set()
    if head_m:
        head = head_m.group('head')
        for mm in re.finditer(r'\binput\b\s*(?:\[[^]]+\]\s*)?([^,);]+(?:\s*,\s*[^,);]+)*)', head):
            names = re.sub(r'\[[^]]+\]', ' ', mm.group(1))
            head_inputs.update(n.strip() for n in names.replace("\n"," ").split(",") if n.strip())

    body_inputs = set()
    for mm in re.finditer(r'^\s*input\b\s*(?:\[[^]]+\]\s*)?([^;]+);', block, flags=re.M):
        names = re.sub(r'\[[^]]+\]', ' ', mm.group(1))
        body_inputs.update(n.strip() for n in names.replace("\n"," ").split(",") if n.strip())

    prim_inputs = set(head_inputs) | set(body_inputs) | set(boundary_in)

    # -------- Split out all instance blocks --------
    lines = block.splitlines(keepends=True)
    inst_blocks = []  # [(start,end, text)]
    N = len(lines)
    i = 0
    while i < N:
        l = lines[i]
        # Skip declarations/endmodule etc.
        if re.match(r'^\s*(module|endmodule|input|output|inout|wire|assign|parameter|localparam)\b', l):
            i += 1
            continue
        if "(" not in l:
            i += 1
            continue
        # Try to grab a segment up to ');' with balanced parentheses
        start, par, j = i, 0, i
        touched = False
        buf = []
        while j < N:
            s = lines[j]
            par += s.count("(")
            par -= s.count(")")
            buf.append(s)
            if par <= 0 and ");" in s:
                inst_blocks.append((start, j, "".join(buf)))
                touched = True
                break
            j += 1
        i = j + 1 if touched else i + 1

    if not inst_blocks:
        raise RuntimeError("[extract] No instances found inside the submodule")

    # -------- Identify output net/input nets for each instance --------
    out_pins = ("Z","ZN","Q","QN","Y","O","S","CO")
    def parse_ios(text_block: str):
        # Output: the first matching output pin; inputs: net names of all .A/.A1/.B/...
        out_net = None
        for pin in out_pins:
            m = re.search(rf'\.\s*{pin}\s*\(\s*([A-Za-z_]\w*|\\\S+)\s*\)', text_block)
            if m:
                out_net = m.group(1)
                break
        in_nets = set()
        for m in re.finditer(r'\.\s*(?:A\d*|B\d*|C\d*|D\d*|I\d*|IN\d*|S|SI|SE|D|A|B|C|IN|I)\s*\(\s*([A-Za-z_]\w*|\\\S+)\s*\)', text_block):
            in_nets.add(m.group(1))
        return out_net, in_nets

    blocks = []
    net_to_block_idx = {}
    for (s,e,t) in inst_blocks:
        out_net, in_nets = parse_ios(t)
        blocks.append({"text": t, "out": out_net, "ins": in_nets})
        if out_net:
            net_to_block_idx[out_net] = len(blocks)-1

    # -------- Backtrack from cone_out_net to collect the whole cone --------
    need = []
    seen_blocks = set()
    work = [cone_out_net]
    seen_nets = set(work)
    while work:
        net = work.pop()
        bi = net_to_block_idx.get(net)
        if bi is None:
            # net may be a PI/constant/assign, no need to backtrack further
            continue
        if bi in seen_blocks:
            continue
        seen_blocks.add(bi)
        need.append(bi)
        # Add its inputs to the queue (ignoring primary inputs)
        for inn in blocks[bi]["ins"]:
            if inn not in prim_inputs and inn not in seen_nets:
                seen_nets.add(inn)
                work.append(inn)

    if not need:
        raise RuntimeError(f"[extract] No instance driving {cone_out_net} found; please check the output pin names or submodule content")

    # -------- Output in dependency (topological) order (fanin first, target last) --------
    # Simply use reversed DFS order (backtracking goes output->input, so reversing works)
    need = need[::-1]
    # Dedup while preserving order
    ordered = []
    seen = set()
    for bi in need:
        if bi not in seen:
            seen.add(bi)
            ordered.append(bi)

    text = "\n".join(_normalize_inst_block(blocks[bi]["text"]) for bi in ordered)
    return _normalize_inst_block(text)


def _strip_comments_and_attrs(s: str) -> str:
    # Remove //... and /* ... */ as well as (* ... *) attribute blocks
    s = re.sub(r'//.*', '', s)
    s = re.sub(r'/\*.*?\*/', '', s, flags=re.S)
    s = re.sub(r'\(\*.*?\*\)', '', s, flags=re.S)
    return s

def _find_module_region(src: str, top_hint: str|None):
    """
    In the comment-stripped text, find the module body of the target module and return (body_text, module_name).
    The module body starts after the header semicolon ';' and ends before 'endmodule'.
    Supports escaped module names: \subcone_opt.aag
    """
    def _after_module_header(i: int) -> int:
        """Given position i after 'module <name>', skip the parameter/port list and return the position after the header ';'."""
        n = len(src)
        # Skip whitespace
        while i < n and src[i].isspace(): i += 1
        # Optional #(...)
        if i < n and src[i] == '#':
            i += 1
            while i < n and src[i].isspace(): i += 1
            if i >= n or src[i] != '(':
                raise RuntimeError("Failed to parse parameterized module header: no '(' after '#'")
            # Scan balanced parentheses
            depth = 0
            while i < n:
                if src[i] == '(':
                    depth += 1
                elif src[i] == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            while i < n and src[i].isspace(): i += 1
        # Port list '(' ... ')'
        if i >= n or src[i] != '(':
            # Some netlists may omit the port list, going straight to ';'
            pass
        else:
            depth = 0
            while i < n:
                if src[i] == '(':
                    depth += 1
                elif src[i] == ')':
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            while i < n and src[i].isspace(): i += 1
        # Find the semicolon ';'
        while i < n and src[i].isspace(): i += 1
        if i >= n or src[i] != ';':
            # Fallback: search for the next ';'
            j = src.find(';', i)
            if j == -1:
                raise RuntimeError("End semicolon ';' not found in the module header")
            i = j
        return i + 1  # One character after the semicolon

    # Collect all modules
    mods = []
    for m in re.finditer(r'\bmodule\b\s+((?:\\\S+)|([A-Za-z_]\w*))', src):
        name = m.group(1)
        header_end = _after_module_header(m.end())
        # Find the matching endmodule (simple sequential search)
        tail = src[header_end:]
        m_end = re.search(r'\bendmodule\b', tail)
        if not m_end:
            continue
        endpos = header_end + m_end.start()
        # Only add the module body (excluding header/footer)
        mods.append((name, header_end, endpos))

    if not mods:
        raise RuntimeError("No module declaration found")

    if top_hint:
        for name, lo, hi in mods:
            plain = name.lstrip('\\')
            if plain == top_hint or name == top_hint:
                return src[lo:hi], name

    # Default to the first one
    name, lo, hi = mods[0]
    return src[lo:hi], name

def _scan_instances(src: str):
    """
    Scan structural instances:
      return list[ {cell, inst, ports(dict pin->net), text(str)} ]
    Supports multi-line; matches up to the ';' after ')'
    """
    insts = []
    i, n = 0, len(src)
    while i < n:
        m = re.search(r'\b([A-Za-z_]\w*)\s+([A-Za-z_.$\\][\w.$\\]*)\s*\(', src[i:])
        if not m:
            break
        cell, inst = m.group(1), m.group(2)
        s = i + m.end() - 1  # Points to '('
        # Balanced search for ')'
        depth, j = 0, s
        while j < n:
            if src[j] == '(':
                depth += 1
            elif src[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if j >= n:
            # Unbalanced parentheses, skip
            i = i + m.end()
            continue
        # Skip whitespace to find ';'
        k = j + 1
        while k < n and src[k].isspace():
            k += 1
        if k >= n or src[k] != ';':
            # Not a standard instance, continue
            i = j + 1
            continue

        full_txt = src[i + m.start(): k + 1]
        ports_chunk = src[s+1:j]

        # Rough check: if it looks like a declaration statement (wire/reg/input/output/assign), skip
        if re.search(r'\b(?:input|output|inout|wire|reg|logic|assign)\b', full_txt):
            i = k + 1
            continue

        # Parse named ports (not all are required)
        ports = {}
        for pm in re.finditer(r'\.\s*([A-Za-z_]\w*)\s*\(\s*([^)]+?)\s*\)', ports_chunk):
            pin = pm.group(1)
            net = pm.group(2).strip()
            net = re.sub(r'\s+', '', net)
            ports[pin] = net

        insts.append({
            "cell": cell,
            "inst": inst,
            "ports": ports,
            "text": full_txt.strip()
        })
        i = k + 1
    return insts

def _one_line_instance(txt: str) -> str:
    """
    Compress instance text into a single line, normalizing spaces and commas:
      CELL INST ( .A(n1), .B(n2), .ZN(n3) );
    """
    t = re.sub(r'\s+', ' ', txt.strip())
    # Normalize whitespace around parentheses and commas
    t = t.replace(' (', ' (').replace('( ', '(').replace(' )', ')')
    t = re.sub(r'\s*,\s*', ', ', t)
    return t

def _normalize_inst_block(txt: str) -> str:
    """
    Light cleanup for a whole block of instance text:
      - Remove leading/trailing blank lines
      - Collapse extra blank lines (at most one)
      - Strip trailing whitespace on each line
    """
    lines = txt.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    cleaned, prev_blank = [], False
    for l in lines:
        if not l.strip():
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(l.rstrip())
            prev_blank = False
    return "\n".join(cleaned)

def _extract_all_instance_lines(sub_v: str, sub_top_hint: str|None = None) -> str:
    """
    Extract all instance lines from the specified module (or the first module) in sub_v,
    without cone filtering; each line normalized to a single-line instance.
    """
    raw = Path(sub_v).read_text()
    text = _strip_comments_and_attrs(raw)

    body, _mname = _find_module_region(text, sub_top_hint)
    insts = _scan_instances(body)

    if not insts:
        # Allow returning an empty string, but a hint makes debugging easier
        # raise RuntimeError("No instances scanned inside the module body")
        return ""

    one_liners = [_one_line_instance(x["text"]) for x in insts]
    return _normalize_inst_block("\n".join(one_liners))


# -------- Main function: text-replacement version (main's call stays unchanged) --------
def splice_back(original_v, topmod, cone_insts:set, boundary_in:list, cone_out_net:str,
                sub_v="subcone_opt.v", sub_top="subcone", out_v="netlist_spliced.v",
                new_inst_text: str | None = None,
                work_dir="tmp"
                ):
    W = ensure_dir(work_dir)

    # Read the original file
    with open(original_v, "r") as f:
        src = f.read()
    lines = src.splitlines(keepends=True)

    # 2) Locate the instance segments to remove (by instance name, in the original netlist from the line containing the instance name to its ');')
    spans = []  # [(start_idx, end_idx)]
    inst_set = set(cone_insts)
    if not inst_set:
        raise ValueError("cone_insts is empty, cannot replace")

    # Word-boundary protection to avoid accidentally matching signal names
    name_patterns = {name: re.compile(rf'(^|[^A-Za-z0-9_]){re.escape(name)}([^A-Za-z0-9_]|$)') for name in inst_set}

    N = len(lines)
    i = 0
    first_indent = ""
    while i < N and name_patterns:
        line = lines[i]
        hit = None
        for name, pat in list(name_patterns.items()):
            if pat.search(line):
                hit = name
                break
        if hit is None:
            i += 1
            continue

        # Record the first segment's indentation for aligning the new text
        if not first_indent:
            m_ind = re.match(r"(\s*)", line)
            first_indent = m_ind.group(1) if m_ind else ""

        # Accumulate downward from the current line until the instance ends
        start = i
        paren = 0
        j = i
        touched = False
        while j < N:
            l = lines[j]
            paren += l.count("(")
            paren -= l.count(")")
            if ");" in l and paren <= 0:
                spans.append((start, j))
                name_patterns.pop(hit, None)
                touched = True
                break
            j += 1
        i = j + 1
        if not touched:
            # Unbalanced; remove the name anyway to avoid an infinite loop
            name_patterns.pop(hit, None)

    if not spans:
        raise RuntimeError(f"No instances to replace found in {original_v}: {sorted(list(cone_insts))}")

    # 3) Generate/get the new instance text to insert
    if new_inst_text is None:
        # Automatically extract all instances from the submodule (keeps the existing approach)
        new_inst_text = _extract_all_instance_lines(
            sub_v=sub_v,
            sub_top_hint=sub_top
        )

    # Clean up extra blank lines/trailing whitespace
    new_inst_text = _normalize_inst_block(new_inst_text)
    # Ensure a trailing newline
    if not new_inst_text.endswith("\n"):
        new_inst_text += "\n"

    # ================== New: uniformly rename to <prefix>_OPT# ==================
    # a) Category detection (consistent with run_one_iter)
    def _inst_category(name: str) -> str | None:
        n = name.lower()
        if n.startswith("add"):      return "adder"
        if n.startswith("mul"): return "multiplier"
        if n.startswith("sub"): return "subtractor"
        if n.startswith("comp"): return "comparator"
        if name.startswith("U"):       return "U"
        return None

    # Desired prefix
    wanted_prefix = "U_OPT"
    try:
        any_inst = next(iter(cone_insts))
        cat = _inst_category(any_inst)
        if   cat in ("adder","multiplier","subtractor","comparator"):
            wanted_prefix = f"{cat}_OPT"
        elif cat == "U":
            wanted_prefix = "U_OPT"
    except Exception:
        pass

    # b) Collect existing instance names in the original top (to avoid name collisions)
    def _collect_existing_inst_names(full_text: str) -> set[str]:
        pat = re.compile(r'(?m)^[ \t]*(?:\(\*.*?\*\)[ \t]*)*([A-Za-z_]\w*)[ \t]+([A-Za-z_.$\\][\w.$\\]*)[ \t]*\(',
                         flags=re.S)
        return set(m.group(2) for m in pat.finditer(full_text))
    existing_names = _collect_existing_inst_names(src)

    # c) Uniformly rename all instances in the inserted block to <prefix>_OPT#, continuing the numbering from the existing max
    def _rename_block_instances(block_text: str, wanted_prefix: str, existing: set[str]) -> str:
        pat = re.compile(r'(?m)^([ \t]*(?:\(\*.*?\*\)[ \t]*)*([A-Za-z_]\w*)[ \t]+)([A-Za-z_.$\\][\w.$\\]*)([ \t]*\()')
        pref_re = re.compile(rf'^{re.escape(wanted_prefix)}(\d+)$')
        # Find the max existing number for wanted_prefix
        max_id = 0
        for name in existing:
            m = pref_re.match(name)
            if m:
                try: max_id = max(max_id, int(m.group(1)))
                except: pass
        counter = max_id

        def gen_name():
            nonlocal counter
            while True:
                counter += 1
                cand = f"{wanted_prefix}{counter}"
                if cand not in existing:
                    existing.add(cand)
                    return cand

        def _repl(m: re.Match) -> str:
            head = m.group(1)  # Includes attributes/CELL/whitespace
            # m.group(3) is the original instance name, unused; generate a new name directly
            tail = m.group(4)
            newname = gen_name()
            return f"{head}{newname}{tail}"

        return pat.sub(_repl, block_text)

    new_inst_text = _rename_block_instances(new_inst_text, wanted_prefix, existing_names)
    # ================== End of new section ==================

    # 4) ... (keeps the indentation alignment + insertion logic)
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    insert_at = merged[0][0]
    # Delete from back to front
    for s, e in reversed(merged):
        del lines[s:e+1]

    # Indentation alignment
    new_block_aligned = "".join(first_indent + ln if ln.strip() else ln
                                for ln in new_inst_text.splitlines(True))
    lines.insert(insert_at, new_block_aligned)

    # 5) Write back the final result
    with open(out_v, "w") as f:
        f.writelines(lines)

    # 6) Report -> put in work_dir
    with open(W / "report.json", "w") as f:
        json.dump({
            "mode": "text_replace",
            "removed_instances": sorted(list(cone_insts)),
            "insert_pos_line": insert_at + 1,  # 1-based
            "top": topmod,
            "inserted_from": sub_v,
            "renamed_prefix": wanted_prefix
        }, f, indent=2)


from collections import defaultdict

IDENT = r'(?:\\\S+|[A-Za-z_]\w*)'   # Supports escaped names \foo.bar as well as plain identifiers

def format_wire_decl(names, per_line=10, indent="  "):
    """
    Format ['n1','n2',...] into a multi-line wire declaration
    """
    chunks = [names[i:i+per_line] for i in range(0, len(names), per_line)]
    lines = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            lines.append(f"{indent}wire {', '.join(chunk)},")
        elif i == len(chunks)-1:
            lines.append(f"{indent}{' ' * 5}{', '.join(chunk)};")
        else:
            lines.append(f"{indent}{' ' * 5}{', '.join(chunk)},")
    return "\n".join(lines) + "\n"

def _parse_wire_names_from_chunk(chunk: str) -> list[str]:
    # Strip bit widths, attributes, and runs of whitespace
    chunk = re.sub(r'\(\*.*?\*\)', '', chunk, flags=re.S)     # (* ... *)
    chunk = re.sub(r'\[[^]]+\]', ' ', chunk)                  # [msb:lsb]
    chunk = re.sub(r'\s+', ' ', chunk)
    # Grab the names part after the wire keyword up to the semicolon
    m = re.search(r'\bwire\b([^;]*);', chunk)
    if not m:
        return []
    names_part = m.group(1)
    # Split by comma, drop empty items
    raw = [x.strip() for x in names_part.split(',') if x.strip()]
    return raw

def _scan_multiline_wire_decls(lines: list[str]) -> list[tuple[int,int,set[str]]]:
    """
    Return all wire declaration blocks: [(start_idx, end_idx, {names...}), ...]
    Supports multi-line declarations (wire ... ,\n    ..., ... ;)
    """
    decls = []
    i, n = 0, len(lines)
    while i < n:
        if re.match(r'^\s*wire\b', lines[i]):
            start = i
            buf = [lines[i]]
            i += 1
            # Aggregate until a line containing ';'
            while i < n and ';' not in lines[i]:
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                end = i
            else:
                # Unbalanced, conservatively treat as a single line
                end = start
            names = set(_parse_wire_names_from_chunk(''.join(buf)))
            decls.append((start, end, names))
            i += 1
        else:
            i += 1
    return decls

# ===== Random strategy tools (used by Whole Variants) =====
import random as _random

def _rand_gate_policy(rng:_random.Random) -> dict:
    """
    Return {"allow": list[str] | None, "ban": list[str] | None}
    - 55% chance to use a whitelist (always includes INV/BUF, plus a random 2~6 common families)
    - 45% chance to use a blacklist (always bans AOI/OAI/MUX, plus a random 0~2 extras)
    """
    BASE_ALLOW = ["INV","BUF","AND","NAND","NOR","OR","XOR","XNOR","MUX"]
    BASE_BAN   = ["AOI","OAI","MUX"]
    if rng.random() < 0.55:
        extras = [x for x in BASE_ALLOW if x not in ("INV","BUF")]
        take = rng.sample(extras, k=rng.randint(2, min(6, len(extras))))
        return {"allow": ["INV","BUF"] + take, "ban": None}
    else:
        pool = ["XOR","XNOR","AND","OR"]
        extra = rng.sample(pool, k=rng.randint(0,2))
        return {"allow": None, "ban": BASE_BAN + extra}

def _rand_abc_recipe(rng:_random.Random) -> str:
    """
    Generate an AIG-level optimization sequence (semicolon-separated, without write/cec)
    """
    pool = ["dch", "dc2", "rewrite", "rewrite -z", "refactor", "refactor -z",
            "resub -K 4", "resub -K 6", "resub -K 8", "balance", "fraig"]
    core = rng.sample(pool, k=rng.randint(4, 7))
    if "balance" not in core:
        core.append("balance")
    return "; ".join(["strash"] + core)

def _rand_map_extra(rng:_random.Random) -> str:
    """
    Extra abc args for the Yosys mapping stage: randomly empty / -fast / -D <ns>
    """
    opts = ["", " -fast", f" -D {rng.choice([500,800,1000,1200,1500,2000,2500,3000])}"]
    return rng.choice(opts)

def postprocess_netlist(in_v: str, out_v: str, topmod: str,
                        inst_prefix: str = "U_OPT",
                        net_prefix: str = "W_OPT",
                        work_dir: str = "tmp"):
    """
    Final unified cleanup for the top module:
      1) Rename "non-standard instance names" (^_[0-9]+_$) -> U_OPT1..N (configurable prefix)
      2) (New) Uniformly rename U_BUF1/2/... into a sequence matching inst_prefix, continuing existing inst_prefix* numbering
      3) Rename "temporary net names" (^_[0-9]+_$) -> W_OPTx (configurable prefix), with safe replacement in the module body
      4) Auto-add missing wires; remove unused wires; break wire declarations into fixed-count lines
    Only the top module text is modified; other modules are left as-is
    """
    W = ensure_dir(work_dir)
    with open(in_v, "r") as f:
        text = f.read()

    # --- Locate the top module ---
    top_pat = re.compile(rf'(^\s*module\s+{re.escape(topmod)}\b.*?^\s*endmodule\s*)',
                         flags=re.S | re.M)
    m = top_pat.search(text)
    if not m:
        raise RuntimeError(f"[post] Module {topmod} not found")
    pre, block, post = text[:m.start()], m.group(1), text[m.end():]

    # --- Parse the module header, collect port names ---
    mhead = re.match(rf'^\s*module\s+{re.escape(topmod)}\s*\((.*?)\)\s*;\s*',
                     block, flags=re.S | re.M)
    if not mhead:
        raise RuntimeError(f"[post] Could not parse the module header of {topmod}")
    head_span_end = mhead.end()
    header_inside = mhead.group(1)
    ports = set()
    # ANSI ports
    for mm in re.finditer(rf'\b(?:input|output|inout)\b[^;()]*?\b({IDENT})(?=[^;()]*?(?:,|\)|$))', header_inside):
        ports.add(mm.group(1))
    # Fallback: bare identifiers within the parentheses
    for mm in re.finditer(rf'\b{IDENT}\b', header_inside):
        ports.add(mm.group(0))

    body = block[head_span_end:]

    # --- Collect wire declarations (only within this module body) ---
    def _split_wire_names(s):
        s = re.sub(r'\[[^]]+\]', ' ', s)  # Strip bit width
        return [x.strip() for x in s.split(',') if x.strip()]

    lines = body.splitlines(keepends=True)
    wire_decl_locs = []  # (idx, set(names))
    declared_wires = set()
    for i, ln in enumerate(lines):
        m_wire = re.match(r'^\s*wire\b([^;]*);', ln)
        if m_wire:
            names = _split_wire_names(m_wire.group(1))
            name_set = set(names)
            declared_wires |= name_set
            wire_decl_locs.append((i, name_set))

    # --- Helper: collect "used nets" (instance .PIN(net) and assign) ---
    def collect_used_nets(text_blob: str):
        used = set()
        for ln in text_blob.splitlines():
            for mm in re.finditer(rf'\.\s*[A-Za-z_]\w*\s*\(\s*({IDENT}|1\'[bhod][0-9A-Fa-f]+)\s*\)', ln):
                net = mm.group(1)
                if re.match(r"1'[bhod]", net):  # Constant
                    continue
                used.add(net)
        for ln in text_blob.splitlines():
            if re.search(r'^\s*assign\b', ln):
                for mm in re.finditer(rf'\b{IDENT}\b', ln):
                    tok = mm.group(0)
                    if tok != "assign":
                        used.add(tok)
        return used

    used_nets = collect_used_nets(body)

    # ============ A) Rename "temporary instance names" (^_[0-9]+_$) ============
    # Supports an attribute block (* ... *) before the instance, and allows multi-line
    inst_name_pat = re.compile(
        rf'(?P<prefix>^\s*(?:\(\*.*?\*\)\s*)*)'   # Optional attributes (greedy but controlled by S/M)
        rf'(?P<cell>[A-Za-z_]\w*)\s+'             # CELL
        rf'(?P<inst>{IDENT})\s*\(',               # Instance name (supports escaping)
        flags=re.M | re.S
    )

    def _need_inst_rename_initial(inst: str) -> bool:
        return bool(re.match(r'^_[0-9]+_$', inst))

    # First get the existing instance set (used to generate collision-free unique names)
    existing_inst_names = [m.group('inst') for m in inst_name_pat.finditer(body)]
    existing_inst_set = set(existing_inst_names)

    def _gen_unique_names(prefix, existing):
        k = 1
        while True:
            cand = f"{prefix}{k}"
            if cand not in existing:
                yield cand
            k += 1

    inst_gen = _gen_unique_names(inst_prefix, set(existing_inst_set))

    def _repl_inst_initial(m: re.Match) -> str:
        prefix_txt = m.group('prefix') or ""
        cell = m.group('cell')
        inst = m.group('inst')
        if not _need_inst_rename_initial(inst):
            return f"{prefix_txt}{cell} {inst} ("
        new_name = next(inst_gen)
        existing_inst_set.add(new_name)
        return f"{prefix_txt}{cell} {new_name} ("

    body = inst_name_pat.sub(_repl_inst_initial, body)

    # ============ A2) Unify U_BUF\d+ into the inst_prefix sequence (continue numbering, no collision with existing) ============
    # Count all instance names now (after step A), find the max existing inst_prefix number
    cur_inst_names = [m.group('inst') for m in inst_name_pat.finditer(body)]
    cur_set = set(cur_inst_names)

    # Support a trailing underscore in the prefix: e.g. U_OPT3_7 still yields 7
    # Rule: match inst_prefix + optional non-alphanumeric-underscore separator + digits
    #   e.g. inst_prefix='U_OPT'  matches U_OPT7
    #        inst_prefix='U_OPT3_' matches U_OPT3_7
    # Note: only take the "last digit string" as the number
    sep = r'(?:[_\.]*)'  # A bit lenient (usually an underscore)
    inst_prefix_num_pat = re.compile(rf'^{re.escape(inst_prefix)}{sep}(\d+)$')

    def _max_index_of(prefix_pat, names: set[str]) -> int:
        mx = 0
        for n in names:
            m = prefix_pat.match(n)
            if m:
                try:
                    mx = max(mx, int(m.group(1)))
                except ValueError:
                    pass
        return mx

    start_idx = _max_index_of(inst_prefix_num_pat, cur_set) + 1

    def _gen_from(start: int, prefix: str, existing: set[str]):
        k = start
        while True:
            cand = f"{prefix}{k}"
            if cand not in existing:
                yield cand
            k += 1

    inst_gen_buf = _gen_from(start_idx, inst_prefix, set(cur_set))

    # Uniformly rename U_BUF<digits> instances to inst_prefix<new_id>
    def _repl_unify_buf(m: re.Match) -> str:
        prefix_txt = m.group('prefix') or ""
        cell = m.group('cell')
        inst = m.group('inst')
        if re.match(r'^U_BUF\d+$', inst):
            new_name = next(inst_gen_buf)
            cur_set.add(new_name)
            return f"{prefix_txt}{cell} {new_name} ("
        return f"{prefix_txt}{cell} {inst} ("

    body = inst_name_pat.sub(_repl_unify_buf, body)

    # ============ B) Rename "temporary wires" (^_[0-9]+_$) ============
    temp_wires = set()
    temp_pat = re.compile(r'^_[0-9]+_$')

    # Get the "existing" identifiers again (including the just-unified instance names)
    inst_names_after = set(m.group('inst') for m in inst_name_pat.finditer(body))
    existing_identifiers = set(ports) | set(declared_wires) | set(collect_used_nets(body)) | set(inst_names_after)

    for w in declared_wires | used_nets:
        if w in ports:
            continue
        if temp_pat.match(w):
            temp_wires.add(w)

    net_gen = _gen_unique_names(net_prefix, set(existing_identifiers))
    rename_map = {}
    for old in sorted(temp_wires):  # Fixed order for stable output
        newname = next(net_gen)
        rename_map[old] = newname
        existing_identifiers.add(newname)

    if rename_map:
        key_alt = "|".join(sorted(map(re.escape, rename_map.keys()), key=len, reverse=True))
        full_re = re.compile(rf'\b(?:{key_alt})\b')
        body = full_re.sub(lambda m: rename_map[m.group(0)], body)
        lines = body.splitlines(keepends=True)
        declared_wires = {rename_map.get(w, w) for w in declared_wires}
        used_nets = collect_used_nets(body)

    # ============ C) Re-add/remove wires and format line breaks ============
    lines = body.splitlines(keepends=True)
    decl_blocks = _scan_multiline_wire_decls(lines)

    declared_wires = set()
    first_wire_idx = None
    for s, e, names in decl_blocks:
        declared_wires |= names
        if first_wire_idx is None:
            first_wire_idx = s

    used_nets = collect_used_nets(body)
    used_signal_candidates = used_nets - ports

    # After removing all wire lines, search the pure body to see if it is still used
    body_no_wire = ''.join(
        ln for idx, ln in enumerate(lines)
        if not (decl_blocks and any(s <= idx <= e for (s, e, _) in decl_blocks))
    )
    unused = set()
    for w in declared_wires:
        if w in ports:
            continue
        pat = (r'\b' + re.escape(w) + r'\b') if not w.startswith('\\') else re.escape(w)
        if not re.search(pat, body_no_wire):
            unused.add(w)

    keep = declared_wires - unused
    missing = used_signal_candidates - keep
    final_wires = sorted(keep | missing)

    # 1) Remove all old wire declaration blocks
    new_lines = lines[:]
    for s, e, _ in reversed(decl_blocks):
        del new_lines[s:e+1]

    # 2) Insert the new wire declaration block (if any)
    if final_wires:
        add_decl = format_wire_decl(final_wires)  # May still be multi-line
        insert_at = first_wire_idx if first_wire_idx is not None else 0
        new_lines.insert(insert_at, add_decl)

    new_body = ''.join(new_lines)
    new_block = block[:head_span_end] + new_body

    with open(out_v, "w") as f:
        f.write(pre + new_block + post)


import subprocess, textwrap, os, shlex

# ----------------- helpers -----------------
def _sh(cmd: str):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate()
    return p.returncode, out

def _detect_out(pins):
    order = ["ZN","Z","Y","QN","Q","CO","S","O","O1","O2","O3","O4"]
    U = [p.upper() for p in pins]
    for k in order:
        if k in U: return pins[U.index(k)]
    return pins[-1] if pins else "Y"


def _guess_expr(cell_upper: str, xs: list[str]) -> str:
    """
    Decide polarity based solely on the gate family:
      INV / NAND* / NOR* / XNOR*      -> inverting output
      BUF / AND* / OR* / XOR* / MUX*  -> non-inverting output
    Completely ignore whether the output pin name ends in N (ZN/QN etc.).
    """
    U = cell_upper

    # Detect the family prefix, avoiding \b issues (supports OR2_X1, NAND3_X1, etc.)
    m = re.match(r'(INV|BUF|XNOR\d*|XOR\d*|NAND\d*|NOR\d*|AND\d*|OR\d*|MUX\d*)', U)
    family = m.group(1) if m else U
    fam_letters = re.match(r'[A-Z]+', family).group(0) if family else U

    # Write the "positive-logic" core expression first (without inversion), then invert based on whether the family is natively inverting
    if fam_letters == 'INV':
        core = xs[0] if xs else "1'b0"; native_inv = True
    elif fam_letters == 'BUF':
        core = xs[0] if xs else "1'b0"; native_inv = False
    elif fam_letters == 'XNOR':
        core = f"({xs[0]} ^ {xs[1]})" if len(xs) >= 2 else (xs[0] if xs else "1'b0"); native_inv = True
    elif fam_letters == 'XOR':
        core = f"({xs[0]} ^ {xs[1]})" if len(xs) >= 2 else (xs[0] if xs else "1'b0"); native_inv = False
    elif fam_letters == 'NAND':
        core = "(" + " & ".join(xs) + ")" if xs else "1'b1"; native_inv = True
    elif fam_letters == 'NOR':
        core = "(" + " | ".join(xs) + ")" if xs else "1'b0"; native_inv = True
    elif fam_letters == 'AND':
        core = " & ".join(xs) if xs else "1'b1"; native_inv = False
    elif fam_letters == 'OR':
        core = " | ".join(xs) if xs else "1'b0"; native_inv = False
    elif fam_letters == 'MUX':
        # Simple MUX2 fallback: S, A, B (guess from pin names where possible)
        s = next((p for p in xs if p.upper() in ('S','S0','SEL')), (xs[2] if len(xs)>=3 else (xs[0] if xs else "1'b0")))
        a = next((p for p in xs if p.upper() in ('A','A0','I0')), (xs[0] if xs else "1'b0"))
        b = next((p for p in xs if p.upper() in ('B','A1','I1')), (xs[1] if len(xs)>=2 else a))
        core = f"(({s}) ? ({b}) : ({a}))"; native_inv = False
    else:
        core = xs[-1] if xs else "1'b0"; native_inv = False

    return f"~({core})" if native_inv else core


def _autogen_techlib_from_many(files, outlib="techlib_auto.v"):
    txt = "\n".join(Path(f).read_text() for f in files)
    inst_re = re.compile(r'^\s*([A-Za-z0-9_]+)\s+[A-Za-z0-9_\\]+\s*\((.*?)\);\s*$',
                         re.M|re.S)
    port_re = re.compile(r'\.\s*([A-Za-z0-9_]+)\s*\(')
    cells = {}
    for m in inst_re.finditer(txt):
        cell, ports = m.group(1), m.group(2)
        pins = port_re.findall(ports)
        if pins and cell not in cells:
            cells[cell] = pins

    lines = []
    for cell, pins in cells.items():
        # Output-pin priority order (only used to determine which pin is the output, no longer decides polarity)
        order = ["ZN","Z","Y","QN","Q","CO","S","O","O1","O2","O3","O4"]
        U = [p.upper() for p in pins]
        y = None
        for k in order:
            if k in U:
                y = pins[U.index(k)]
                break
        if y is None:
            y = pins[-1]
        xs = [p for p in pins if p != y]

        expr = _guess_expr(cell.upper(), xs)

        plist = ", ".join([*xs, y]) if xs else y
        decl_in  = "\n".join([f"  input {i};" for i in xs])
        decl_out = f"  output {y};"
        body = f"  assign {y} = {expr};"
        lines.append(f"module {cell}({plist});\n{decl_in}\n{decl_out}\n{body}\nendmodule\n")

    Path(outlib).write_text("\n\n".join(lines) if lines else "// empty techlib\n")
    return outlib


# ----------------- main: only CEC -----------------
def check_equivalence_with_abc(
    orig_v: str = "locked_c1355_0_1_0_0_0_1_flat.v",
    new_v:  str = "netlist_spliced_post.v",
    top:    str = "locked_c1355",
    liberty: str = "NangateOpenCellLibrary_typical.lib",
    work_dir: str = "tmp"
):
    """
    Only run CEC, but guarantee readable gate-level cells:
      1) Prefer reading a real stdcell function library; if none is found, auto-generate techlib_auto.v
      2) Yosys: read_liberty -lib + read_verilog(function library) + read_verilog(design)
         -> setundef -undriven -zero -> clean -purge -> aigmap
         -> write_aiger gold.aig / opt.aig
      3) ABC: abc -c "cec gold.aig opt.aig"
    """
    # --- Absolutize all paths first ---
    # orig_v  = os.path.abspath(orig_v)
    new_v   = os.path.abspath(new_v)
    # liberty = os.path.abspath(liberty)
    work_dir = os.path.abspath(work_dir)
    cwd = os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    try:
        # 1) Function library
        stdcell_v = None#_find_real_stdcell_v_from_liberty(liberty)
        if stdcell_v is None:
            stdcell_v = "techlib_auto.v"
            _autogen_techlib_from_many([cwd + "/" + orig_v, new_v], outlib=stdcell_v)

        common_pass = """
            flatten
            proc; opt; memory; opt
            techmap; opt
            setundef -undriven -zero
            clean -purge
            aigmap
            opt_clean
        """.strip()

        gold_ys = textwrap.dedent(f"""
            read_liberty -lib {shlex.quote(cwd + "/" + liberty)}
            read_verilog {shlex.quote(stdcell_v)}
            read_verilog {shlex.quote(cwd + "/" + orig_v)}
            hierarchy -check -top {shlex.quote(top)}
            {common_pass}
            write_aiger gold.aig
        """).strip()

        opt_ys = textwrap.dedent(f"""
            read_liberty -lib {shlex.quote(cwd + "/" + liberty)}
            read_verilog {shlex.quote(stdcell_v)}
            read_verilog {shlex.quote(new_v)}
            hierarchy -check -top {shlex.quote(top)}
            {common_pass}
            write_aiger opt.aig
        """).strip()

        Path("gold.ys").write_text(gold_ys + "\n")
        Path("opt.ys").write_text(opt_ys + "\n")

        rc1, y1 = _sh("yosys -q -s gold.ys"); Path("yosys_gold.log").write_text(y1)
        if rc1 != 0 or not Path("gold.aig").exists():
            return {"result":"error", "stdout":"[yosys] Failed to generate gold.aig\n"+y1}

        rc2, y2 = _sh("yosys -q -s opt.ys");  Path("yosys_opt.log").write_text(y2)
        if rc2 != 0 or not Path("opt.aig").exists():
            return {"result":"error", "stdout":"[yosys] Failed to generate opt.aig\n"+y2}

        # 2) Only run CEC (single command)
        rc3, a3 = _sh('abc -c "cec gold.aig opt.aig"')
        Path("abc_cec.log").write_text(a3)

        low = a3.lower()
        if "are equivalent" in low:
            return {"result":"equivalent", "stdout":a3}
        if "are not equivalent" in low or "no proof of equivalence" in low:
            return {"result":"inequivalent", "stdout":a3}
        if rc3 != 0:
            return {"result":"error", "stdout":a3}
        return {"result":"error", "stdout":a3}
    finally:
        os.chdir(cwd)



# # -----------------------------
# # Main
# # -----------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--netlist", required=True)
#     ap.add_argument("--top", required=True)
#     ap.add_argument("--k", type=int, default=2, choices=[1,2,3])
#     ap.add_argument("--work_dir", default="tmp", help="work directionary")
#     ap.add_argument("--liberty", required=True, help="liberty file for synthesis")
#     ap.add_argument("--root_inst", default=None, help="root instance name; if not set, pick random")
#     args = ap.parse_args()
    
    
#     ast, _ = parse([args.netlist])
#     ng = NetlistGraph(args.top)
#     ng.from_ast(ast)
#     g_pred, g_succ = ng.build_graph()

#     # Pick root: must be an instance that drives at least one net
#     driver_insts = set(inst for net, drivers in ng.net_drivers.items() for inst,_ in drivers)
#     cands = sorted(driver_insts) if driver_insts else list(ng.instances.keys())
#     if args.root_inst is None:
#         random.seed(42)
#         root_inst = random.choice(cands)
#     else:
#         root_inst = args.root_inst
#         if root_inst not in ng.instances:
#             raise ValueError(f"Instance {root_inst} not found")

#     # extract cone
#     cone = k_hop_fanin_cone(root_inst, g_pred, args.k)
#     boundary_in, cone_out = cone_boundary(ng, cone, root_inst)
#     print(f"[cone] root={root_inst}, k={args.k}, |cone|={len(cone)}, boundary_in={len(boundary_in)}, out={cone_out}")
#     print("work_dir", args.work_dir)
#     # emit subcircuit verilog
#     meta = emit_subcone_verilog(ng, cone, boundary_in, cone_out, sub_name="subcone", out_path=f"{args.work_dir}/subcone_raw.v")
    
#     liberty = args.liberty
#     # optimize sub via AIG (keep the original I/O names)
#     optimize_sub_aig(
#         in_v="subcone_raw.v",
#         top="subcone",
#         out_v="subcone_opt.v",
#         liberty=liberty,
#         boundary_in=boundary_in,
#         cone_out=cone_out,
#         emit_info=meta,
#         work_dir=args.work_dir
#     )

#     # splice back
#     splice_back(args.netlist, args.top, cone, boundary_in, cone_out, "subcone_opt.v", "subcone", "netlist_spliced.v", work_dir=args.work_dir)
#     print("[DONE] Wrote subcone_raw.v, subcone_opt.v, netlist_spliced.v, report.json")
    
#     postprocess_netlist("netlist_spliced.v", "netlist_spliced_post.v", topmod=args.top, inst_prefix="U_OPT", net_prefix="W_OPT", work_dir=args.work_dir)

# -----------------------------
# Main (loop 5 roots)
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--netlist", required=True)
    ap.add_argument("--top", required=True)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--work_dir", default="tmp", help="work directory")
    ap.add_argument("--liberty", required=True, help="liberty file for synthesis")
    ap.add_argument("--root_inst", default=None, help="root instance name; if set, only optimize this one")
    ap.add_argument("--num_roots", type=int, default=5, help="number of roots to optimize when --root_inst is not set")
    ap.add_argument("--iter", type=int, default=None, help="explicit iteration index when --root_inst is set (default=1)")
    ap.add_argument("--use_gates", default=None,
                    help="Comma-separated gate prefixes to ALLOW (others will be -dont_use). "
                         "Example: INV,BUF,AND,NAND,NOR,OR,XOR,XNOR")
    ap.add_argument("--ban_prefixes", default="AOI,OAI,MUX",
                    help="Comma-separated cell prefixes to ban (ignored if --use_gates is set).")
    ap.add_argument("--gnnre", action="store_true",
                    help="GNNRE mode: extract a subcircuit of the same prefix category as root (adder/multiplier/subtractor/comparator/U); splice back with unified naming <prefix>_OPT#")
    ap.add_argument("--whole", action="store_true",
                    help="Whole-circuit mode: extract the entire top as a subcircuit, optimize, then splice back.")

    # ===== New: batch of global variants =====
    ap.add_argument("--whole_variants", type=int, default=0,
                    help="Batch-generate N global optimization variants (each with a random strategy set)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--variant_retries", type=int, default=1,
                    help="Extra retries when a single variant fails (switching random strategy)")

    args = ap.parse_args()
    # Prepare the top-level work directory
    os.makedirs(args.work_dir, exist_ok=True)
    
    def _to_list(s):
        if s is None: return None
        return [x.strip() for x in s.split(",") if x.strip()]
    use_gates_list = _to_list(args.use_gates)
    ban_prefixes_list = _to_list(args.ban_prefixes) or ["AOI","OAI","MUX"]

    # The "current netlist" is updated each iteration
    current_netlist = args.netlist

    def run_one_iter(iter_idx: int, root_inst_name: str) -> tuple[bool, str]:
        """
        Run one iteration: use root_inst_name as the root, outputs go to work_dir/iter_{iter_idx}/...
        Returns (success, new_current_netlist)
        """
        iter_tag = f"iter_{iter_idx}"
        Wi = os.path.join(args.work_dir, iter_tag)
        os.makedirs(Wi, exist_ok=True)

        # --- Parse the current netlist, build the graph ---
        ast, _ = parse([current_netlist])
        ng = NetlistGraph(args.top)
        ng.from_ast(ast)
        g_pred, g_succ = ng.build_graph()

        # --- Verify the root exists; error if not ---
        if root_inst_name not in ng.instances:
            raise ValueError(f"[{iter_tag}] Instance {root_inst_name} not found in current netlist")

        # --- Extract the fanin cone ---
        # cone = k_hop_fanin_cone(root_inst_name, g_pred, args.k)
        root_cat = inst_category(root_inst_name)
        if args.gnnre and root_cat is not None:
            # Allow set: instances of the same category as root
            same_cat = {iname for iname in ng.instances.keys() if inst_category(iname) == root_cat}
            cone = k_hop_fanin_cone_filtered(root_inst_name, g_pred, args.k, same_cat)
        else:
            cone = k_hop_fanin_cone(root_inst_name, g_pred, args.k)
        boundary_in, cone_out = cone_boundary(ng, cone, root_inst_name)
        print(f"[{iter_tag}][cone] root={root_inst_name}, k={args.k}, |cone|={len(cone)}, |boundary_in|={len(boundary_in)}, out={cone_out}")

        # --- Export the subcircuit ---
        sub_raw_v = os.path.join(Wi, f"subcone_raw_{iter_idx}.v")
        meta = emit_subcone_verilog(
            ng, cone, boundary_in, cone_out,
            sub_name=f"subcone_{iter_idx}",
            out_path=sub_raw_v
        )

        # --- Optimize the subcircuit ---
        sub_opt_v = os.path.join(Wi, f"subcone_opt_{iter_idx}.v")
        optimize_sub_aig(
            in_v=sub_raw_v,
            top=f"subcone_{iter_idx}",
            out_v=sub_opt_v,
            liberty=args.liberty,
            boundary_in=boundary_in,
            cone_out=cone_out,
            emit_info=meta,
            work_dir=Wi,
            allow_gates=use_gates_list,
            ban_prefixes=tuple(ban_prefixes_list)
        )

        # --- Splice back into the "current netlist" ---
        spliced_v = os.path.join(Wi, f"netlist_spliced_{iter_idx}.v")
        splice_back(
            original_v=current_netlist,
            topmod=args.top,
            cone_insts=cone,
            boundary_in=boundary_in,
            cone_out_net=cone_out,
            sub_v=sub_opt_v,
            sub_top=f"subcone_{iter_idx}",
            out_v=spliced_v,
            work_dir=Wi
        )
        print(f"[{iter_tag}] spliced -> {spliced_v}")

        # --- Cleanup & normalization ---
        post_v = os.path.join(Wi, f"netlist_spliced_post_{iter_idx}.v")
        postprocess_netlist(
            in_v=spliced_v,
            out_v=post_v,
            topmod=args.top,
            inst_prefix=f"U_OPT{iter_idx}_",
            net_prefix=f"W_OPT{iter_idx}_"
        )
        print(f"[{iter_tag}] post -> {post_v}")

        # === Embedded CEC: accept this iteration only if it passes; roll back and stop on failure ===
        prev_netlist = current_netlist
        cec_dir = os.path.join(Wi, "cec")
        os.makedirs(cec_dir, exist_ok=True)

        try:
            res = check_equivalence_with_abc(
                orig_v=prev_netlist,
                new_v=post_v,
                top=args.top,
                liberty=args.liberty,
                work_dir=cec_dir
            )
        except Exception as e:
            print(f"[{iter_tag}][CEC] ERROR: {e}")
            print(f"[{iter_tag}] Reverting to previous netlist: {prev_netlist}")
            return (False, prev_netlist)

        if res.get("result") == "equivalent":
            print(f"[{iter_tag}][CEC] equivalent ✅")
            return (True, post_v)
        else:
            print(f"[{iter_tag}][CEC] NON-EQUIVALENT ❌ — aborting.")
            try:
                from pathlib import Path
                Path(os.path.join(cec_dir, "abc_stdout.log")).write_text(res.get("stdout", ""))
            except Exception:
                pass
            return (False, prev_netlist)
    
    def run_whole_once(netlist: str, top: str, liberty: str, work_dir: str,
                    allow_gates: list[str] | None, ban_prefixes: list[str] | None,
                    abc_recipe: str | None = None, map_extra: str = "") -> str:
        """
        Whole-circuit mode:
        1) Copy the entire top into a submodule (I/O fully identical)
        2) AIG optimization (per abc_recipe) + write back Verilog (with map_extra during mapping)
        3) Remove all instances in top and replace them wholesale with the optimized instance block
        4) postprocess + CEC
        Returns: the path of the netlist that passed CEC (work_dir/whole_post.v)
        """
        W = ensure_dir(work_dir)
        sub_name = "sub_whole"
        sub_raw_v = str(W / "whole_raw.v")
        meta = emit_whole_as_submodule(netlist, top, sub_name, sub_raw_v)

        tgt_inputs  = meta["inputs"]
        tgt_outputs = meta["outputs"]
        if not tgt_outputs:
            raise RuntimeError("[whole] top has no outputs?")
        cone_out = tgt_outputs[0]

        sub_opt_v = str(W / "whole_opt.v")
        optimize_sub_aig(
            in_v=sub_raw_v,
            top=sub_name,
            out_v=sub_opt_v,
            liberty=liberty,
            boundary_in=tgt_inputs,
            cone_out=cone_out,
            emit_info=meta,
            work_dir=str(W),
            allow_gates=allow_gates,
            ban_prefixes=tuple(ban_prefixes or ["AOI","OAI","MUX"]),
            abc_recipe=abc_recipe,
            map_extra=map_extra
        )

        # Replace with the optimized "all instances block"
        ast, _ = parse([netlist])
        ng = NetlistGraph(top); ng.from_ast(ast)
        cone_insts = set(ng.instances.keys())
        if not cone_insts:
            raise RuntimeError("[whole] No instances to replace inside top")

        new_inst_text = _extract_all_instance_lines(sub_opt_v, sub_top_hint=sub_name)
        spliced_v = str(W / "whole_spliced.v")
        splice_back(
            original_v=netlist,
            topmod=top,
            cone_insts=cone_insts,
            boundary_in=tgt_inputs,
            cone_out_net=cone_out,
            sub_v=sub_opt_v,
            sub_top=sub_name,
            out_v=spliced_v,
            new_inst_text=new_inst_text,
            work_dir=str(W)
        )

        post_v = str(W / "whole_post.v")
        postprocess_netlist(
            in_v=spliced_v,
            out_v=post_v,
            topmod=top,
            inst_prefix="U_OPT",
            net_prefix="W_OPT",
            work_dir=str(W)
        )

        # CEC
        res = check_equivalence_with_abc(
            orig_v=netlist,
            new_v=post_v,
            top=top,
            liberty=liberty,
            work_dir=str(W / "cec")
        )
        if res.get("result") != "equivalent":
            print("[whole][CEC] ❌\n", res.get("stdout",""))
            raise RuntimeError("[whole] CEC did not pass")
        print("[whole][CEC] ✅ equivalent")
        return post_v
    
    def run_whole_variants(
        num_variants: int,
        netlist: str,
        top: str,
        liberty: str,
        work_dir: str,
        seed: int = 0,
        retries: int = 1
    ):
        """
        Generate num_variants "global optimization" variants.
        - Each variant in its own directory: work_dir/var_XXX/
        - Each variant randomizes: gate policy, ABC recipe, extra mapping args
        - Runs CEC by default; on failure, retry up to 'retries' times (with a new random strategy)
        - Output manifest: work_dir/summary.json
        """
        W = ensure_dir(work_dir)
        rng = _random.Random(seed)
        summary = []

        for i in range(1, num_variants + 1):
            var_dir = W / f"var_{i:03d}"
            ensure_dir(var_dir)
            success = False
            trial_logs = []
            for t in range(retries + 1):
                gp = _rand_gate_policy(rng)
                recipe = _rand_abc_recipe(rng)
                mextra = _rand_map_extra(rng)
                meta = {
                    "trial": t,
                    "allow_gates": gp["allow"],
                    "ban_prefixes": gp["ban"],
                    "abc_recipe": recipe,
                    "map_extra": mextra
                }
                try:
                    out_v = run_whole_once(
                        netlist=netlist,
                        top=top,
                        liberty=liberty,
                        work_dir=str(var_dir),
                        allow_gates=gp["allow"],
                        ban_prefixes=gp["ban"],
                        abc_recipe=recipe,
                        map_extra=mextra
                    )
                    summary.append({
                        "variant": i,
                        "status": "ok",
                        "out_netlist": str(out_v),
                        **meta
                    })
                    success = True
                    break
                except Exception as e:
                    trial_logs.append({"meta": meta, "error": str(e)})
                    continue

            if not success:
                summary.append({
                    "variant": i,
                    "status": "failed",
                    "out_netlist": None,
                    "trials": trial_logs
                })

        Path(W / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[SUMMARY] {len([s for s in summary if s['status']=='ok'])}/{num_variants} succeed")



    # ===== New branch: whole_variants =====
    if args.whole_variants and args.whole_variants > 0:
        out_root = os.path.join(args.work_dir, "whole_variants")
        run_whole_variants(
            num_variants=args.whole_variants,
            netlist=args.netlist,
            top=args.top,
            liberty=args.liberty,
            work_dir=out_root,
            seed=args.seed,
            retries=args.variant_retries
        )
        print(f"[RESULT] Variants in: {out_root}")
        return

    # ===== Original single WHOLE run =====
    if args.whole:
        final_v = run_whole_once(
            netlist=args.netlist,
            top=args.top,
            liberty=args.liberty,
            work_dir=os.path.join(args.work_dir, "whole"),
            allow_gates=use_gates_list,
            ban_prefixes=ban_prefixes_list
        )
        print(f"[RESULT] Full-circuit optimized netlist -> {final_v}")
        return

    # ===== Original local/multi-iteration logic (unchanged) =====
    current_netlist = args.netlist

    def run_one_iter(iter_idx: int, root_inst_name: str) -> tuple[bool, str]:
        # Reuse the original implementation (omitted here; keep the existing run_one_iter definition in your script)
        raise NotImplementedError("Keep your existing run_one_iter implementation")

    if args.root_inst:
        iter_idx = int(args.iter) if args.iter is not None else 1
        ok, new_netlist = run_one_iter(iter_idx, args.root_inst)
        current_netlist = new_netlist
    else:
        for i in range(1, args.num_roots + 1):
            # Reuse the original multi-iteration logic (keep the existing code)
            pass

    print(f"[RESULT] Final netlist: {current_netlist}")

if __name__ == "__main__":
    main()
