#!/usr/bin/env python3
import argparse, os, random, json, subprocess, sys, copy
from collections import defaultdict, deque
from pathlib import Path

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def q(p: str | Path) -> str:
    # shell-safe quoting
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
      - quiet=True: print nothing on success; on failure, print the last tail_lines lines and raise
      - quiet=False: print the full output on success too
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
        # Print only the tail if the log is long
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
    """Convert a port expression to a cleaned base signal-name string; return None for constants/concatenations."""
    if expr is None:
        return None
    # Named signal
    if isinstance(expr, Identifier):
        return expr.name
    # Bit-select foo[i]
    if isinstance(expr, Pointer):
        if isinstance(expr.var, Identifier):
            return expr.var.name
        return _expr_to_basename(expr.var)
    # Part-select foo[msb:lsb]
    if isinstance(expr, Partselect):
        if isinstance(expr.var, Identifier):
            return expr.var.name
        return _expr_to_basename(expr.var)
    # Concatenation {a,b} cannot be uniquely resolved -> skip
    if isinstance(expr, Concat):
        return None
    # Constant / other
    # if isinstance(expr, IntConst):
    #     return None
    if isinstance(expr, IntConst):
        return expr.value  # e.g. "1'b0"
    # Fallback: try hasattr .name
    return getattr(expr, 'name', None)

import re
_CONST_RE = re.compile(r"^\d*'s?[bhod][0-9a-fxz_]+$|^[01]$", re.I)

def _is_const_literal(s: str) -> bool:
    return isinstance(s, str) and bool(_CONST_RE.match(s))



def inst_has_const_pin(inst_info: dict) -> bool:
    """Check whether the instance has any constant-connected pin."""
    for pin, net in inst_info.get("pins", {}).items():
        if isinstance(net, str) and (net.startswith("1'b") or net.startswith("1’b") or net in ("1'b0","1'b1","0","1")):
            return True
    return False


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

        # Record PI/PO
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
                    # Get the expression object: compatible with different pyverilog versions
                    arg_expr = getattr(pa, 'arg', None)
                    if arg_expr is None:
                        arg_expr = getattr(pa, 'argname', None)
                    net = _expr_to_basename(arg_expr)
                    if net is None:
                        # Constant/concatenation etc., skip
                        pos_idx += 1
                        continue
                    # Port name (named ports preferred; otherwise use _p{idx})
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
                if _is_const_literal(n):        # A constant is not a net, skip
                    continue
                self.net_loads[n].append((iname, p))
            # Use pin-name heuristics to mark the driver
            for p, n in pins.items():
                if _is_const_literal(n):        # A constant is not a net, skip
                    continue
                if p is not None and p.upper() in out_hint:
                    self.net_drivers[n].append((iname, p))
        # For nets that still have no driver, do not force an inference; boundary detection later treats them as PIs or external signals

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
def inst_has_const_pin(inst_info: dict) -> bool:
    for _pin, net in inst_info.get("pins", {}).items():
        if _is_const_literal(net):
            return True
    return False

# ==== NEW: instance category detection ====
def inst_category(name: str) -> str | None:
    n = name.lower()
    if n.startswith("add"):       return "adder"
    if n.startswith("mul"):  return "multiplier"
    if n.startswith("sub"):  return "subtractor"
    if n.startswith("comp"):  return "comparator"
    if n.startswith("trojan"): return "Trojan"
    # if name.startswith("U"):        return "U"
    return "U"

# Allowed "normal gate family" prefixes
_NORMAL_FAMILIES = ("AND", "NAND", "OR", "NOR", "XOR", "XNOR", "INV", "BUF")

def is_normal_gate_family(cellname: str) -> bool:   # <<< NEW
    if not cellname:
        return False
    U = cellname.upper()
    return any(U.startswith(pref) for pref in _NORMAL_FAMILIES)

# ==== NEW: k-hop fanin that only traverses within allow_set ====
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

#     # 1) net->drivers/users: prefer those from the graph; fall back to the heuristic if missing
#     if hasattr(graph, "net_drivers") and hasattr(graph, "net_users"):
#         net_drivers = graph.net_drivers
#         net_users   = graph.net_users
#     else:
#         net_drivers, net_users = {}, {}
#         for inst, info in graph.instances.items():
#             for p, n in info["pins"].items():
#                 (net_drivers if _is_out_pin(p) else net_users).setdefault(n, []).append((inst, p))

#     # 2) extra boundary outputs
#     extra_boundary_outs = set()
#     for inst in cone_insts:
#         for p, n in graph.instances[inst]["pins"].items():
#             if not _is_out_pin(p):
#                 continue
#             users = net_users.get(n, [])
#             used_outside = any(u_inst not in cone_insts for (u_inst, _up) in users)
#             if used_outside and (n != cone_out_net) and (n not in boundary_in):
#                 extra_boundary_outs.add(n)

#     # 3) ports (inputs first, outputs last)
#     port_inputs  = list(dict.fromkeys(boundary_in))  # dedup, keep order
#     port_outputs = [cone_out_net] + sorted(extra_boundary_outs)
#     port_names   = set(port_inputs) | set(port_outputs)

#     netmap: dict[str, str] = {}
#     for n in port_inputs + port_outputs:
#         netmap[n] = n

#     # 3.5) fill-in: nets on some instances' "input pins" in the cone that have no in-cone driver -> must be inputs
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
#         # No need for internal_wires -= missing_inputs -- step 4 will not add them anyway

#     # 4) wires used only internally
#     internal_wires = set()
#     for inst in cone_insts:
#         for p, n in graph.instances[inst]['pins'].items():
#             if (n not in netmap) and (n not in port_names):
#                 internal_wires.add(n)
#                 netmap[n] = n

#     # 5) write Verilog
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

def emit_subcone_verilog(
    graph: NetlistGraph,
    cone_insts: set,
    boundary_in: list,
    cone_out_net: str,
    sub_name: str = "subcone",
    out_path: str = "subcone_raw.v"
):
    """
    Extract structural Verilog from the cone:
      - Ports: inputs first, outputs last; cone_out_net goes first, other spill-over outputs follow
      - MUX's S is treated as an input; Adder's S/SUM/CO are treated as outputs; others use generic output names
      - Automatically fill in missing inputs (used inside the cone but with no in-cone driver and non-constant), rewriting the file once if needed
    The returned info includes inputs/outputs/netmap etc., for later port rewriting.
    """
    # ---------- helpers ----------
    _OUTPIN_RE_GENERIC = re.compile(
        r'^(?:Z|ZN|Z\d+|ZN\d+|Y|Y\d+|Q|QN|QB|QBAR|O\d*|CO|SUM|OUT)$', re.I
    )

    def _is_out_pin(cell_name: str, pin_name: str) -> bool:
        Uc = (cell_name or "").upper()
        Up = (pin_name or "").upper()
        # MUX: S is the select input; outputs are usually Z/ZN/Y/Q/O*
        if "MUX" in Uc:
            return Up in ("Z", "ZN", "Y", "Q", "O", "O1", "O2")
        # Adder: S/SUM and CO are outputs
        if re.search(r"(ADDF|ADDH|FA|HA)", Uc):
            return Up in ("S", "SUM", "CO")
        # Other cells use the generic set (excluding bare S)
        return bool(_OUTPIN_RE_GENERIC.match(Up))

    def _is_const_net(n: str) -> bool:
        # Filter out constant forms like 1'b0/1/binary/hex/x/z
        return isinstance(n, str) and bool(re.match(r"^1'[bhod][0-9a-fxz]+$", n, re.I))

    # ---------- 1) obtain net->drivers / net->users ----------
    if hasattr(graph, "net_drivers") and hasattr(graph, "net_users"):
        net_drivers = graph.net_drivers   # net -> [(inst, pin)]
        net_users   = graph.net_users     # net -> [(inst, pin)]
    else:
        net_drivers, net_users = {}, {}
        for inst, info in graph.instances.items():
            cell = info.get("cell", "")
            for p, n in info.get("pins", {}).items():
                (net_drivers if _is_out_pin(cell, p) else net_users).setdefault(n, []).append((inst, p))

    # ---------- 2) extra boundary outputs: driven inside the cone but still used outside ----------
    extra_boundary_outs = set()
    for inst in cone_insts:
        info = graph.instances[inst]
        if inst_has_const_pin(info):
            continue# skip this instance
        cell = info.get("cell", "")
        for p, n in info.get("pins", {}).items():
            if not _is_out_pin(cell, p):
                continue
            users = net_users.get(n, [])
            used_outside = any(u_inst not in cone_insts for (u_inst, _up) in users)
            if used_outside and (n != cone_out_net) and (n not in boundary_in):
                extra_boundary_outs.add(n)

    # ---------- 3) initial ports ----------
    port_inputs  = list(dict.fromkeys(boundary_in))           # dedup, keep order
    port_outputs = [cone_out_net] + sorted(extra_boundary_outs)
    port_names   = set(port_inputs) | set(port_outputs)
    netmap: dict[str, str] = {n: n for n in (port_inputs + port_outputs)}

    # ---------- 3.5) one fill-in pass: nets "used but with no internal driver" in the cone are treated as inputs ----------
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

    # used_in_cone: all nets referenced by instances in the cone (in first-seen order)
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

    # ---------- 4) internal wires: appear in the cone, are not ports, and are non-constant ----------
    internal_wires = set()
    for inst in cone_insts:
        for _p, n in graph.instances[inst].get("pins", {}).items():
            if (n not in port_names) and (not _is_const_net(n)):
                internal_wires.add(n)
                netmap[n] = n

    # ---------- 5) write out once ----------
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

    # ---------- 6) second check: any inputs still missing? if so, add them and rewrite ----------
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
        # Since the port set changed, internal_wires must be rebuilt
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
    # Simple escaping for Yosys selectors/renaming: prefix a backslash for non-standard identifiers
    return name if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name) else "\\" + name

# Used to match Verilog "escaped identifiers" (\name<whitespace>) or plain identifiers
_ESC_ID_TERM = r'(?=[\s,();\[\].]|$)'

def _rewrite_escaped_identifier(text: str, old: str, new: str) -> str:
    # 1) Escaped form: \old<whitespace/delimiter/end-of-line>
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
    Whitelist mode: only allow cells starting with allow_prefixes; -dont_use all others.
    Example: allow_prefixes = ["INV","BUF","AND","NAND","NOR","OR","XOR","XNOR"]
    """
    allow_prefixes = tuple(allow_prefixes or [])
    names = _liberty_cell_names(lib_path)
    banned = [n for n in names if not any(n.startswith(p) for p in allow_prefixes)]
    return "" if not banned else " " + " ".join(f"-dont_use {n}" for n in banned)


def _liberty_parse_cells_with_pins(lib_path: str):
    """
    Roughly parse a liberty file, returning {cell_name: {"pins": [pin1, pin2, ...]}}.
    Only pin names matter; input/output is not distinguished here and is judged later during filtering.
    """
    txt = open(lib_path, "r").read()
    cells = {}

    # Find each cell block
    cell_re = re.compile(r'cell\s*\(\s*([A-Za-z0-9_]+)\s*\)\s*{([^}]*)}', re.S)
    pin_re  = re.compile(r'pin\s*\(\s*([A-Za-z0-9_]+)\s*\)')

    for m in cell_re.finditer(txt):
        cell_name, body = m.group(1), m.group(2)
        pins = pin_re.findall(body)
        cells[cell_name] = {"pins": pins}

    return cells

def liberty_dont_use_except_prefixes_trojan(lib_path, allow_prefixes):
    """
    Whitelist mode:
      - Always allow BUF, INV
      - Other prefixes are allowed if and only if the input pin count == 2
      - -dont_use everything else
    """
    allow_prefixes = tuple(allow_prefixes or [])
    cells = _liberty_parse_cells_with_pins(lib_path)
    # Assumes it returns {cell_name: {"pins": ["A","B","Y",...]}}

    allowed = []
    for cell, info in cells.items():
        pins = [p for p in info["pins"] if p.upper() not in {"Z","ZN","Y","Q","QN","O","O1","O2"}]
        # Filter out output pins
        if cell.startswith("INV") or cell.startswith("BUF"):
            allowed.append(cell)
        elif any(cell.startswith(p) for p in allow_prefixes):
            if len(pins) == 2:
                allowed.append(cell)

    banned = [c for c in cells if c not in allowed]
    return "" if not banned else " " + " ".join(f"-dont_use {c}" for c in banned)

def liberty_dont_use_flags(lib_path, prefixes=("AOI", "OAI")):
    """
    Grab cell names starting with prefixes from the Liberty file and produce a
    ' -dont_use <cellA> -dont_use <cellB> ...' string.
    This avoids having to hand-write every variant like AOI21_X1/AOI22_X4.
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
    """Starting at the '(' at start_idx, return the matched substring content (excluding the parentheses themselves) and the end index (index of the closing paren)."""
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



def build_port_map_from_verilog(verilog_path, tgt_inputs, tgt_outputs):
    raw = Path(verilog_path).read_text()
    text = _strip_comments_and_attrs(raw)

    # --- Find the first module declaration: supports escaped module names, e.g. \subcone_opt.aag ---
    m_mod = re.search(
        r"\bmodule\b\s+(?P<mname>(?:\\\S+|[A-Za-z_]\w*))",
        text
    )
    if not m_mod:
        raise RuntimeError("module declaration not found (still failing after stripping comments/attributes)")
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
            raise RuntimeError("failed to parse parameterized module header: no '(' after '#'")
        _, end_paren = _scan_balanced(text, i, "(", ")")
        i = end_paren + 1

    # Tolerate more whitespace/newlines: do not require '(' immediately after; search for the next '('
    while i < n and text[i].isspace():
        i += 1
    if i >= n or text[i] != '(':
        # Directly find the first '(' from the current position
        j = text.find("(", i)
        if j == -1:
            ctx = text[max(0, i-80):min(n, i+80)]
            raise RuntimeError(f"cannot locate the port list '(' , context: >>>{ctx}<<<")
        i = j

    # Port list
    plist, right_paren = _scan_balanced(text, i, "(", ")")

    # Split port names (keeping the earlier handling)
    raw_ports = [p.strip() for p in plist.replace("\n", " ").split(",") if p.strip()]

    def _last_ident(tok: str) -> str:
        tok = re.sub(r"\[[^]]+\]", " ", tok)  # strip bit-width
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
            f"port count mismatch: cur_in={len(cur_inputs)} tgt_in={len(tgt_inputs)}; "
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
    A more robust port-renaming tool:
    1) Remove redundant wire declarations that share a name with a port
    2) Whole-word replacement of port names (module header, port declarations, references in the module body)
    3) Optionally rename the module name
    """
    text = Path(verilog_in).read_text()

    # ---- 0) Prepare safely: collect the port set to change (only rename these names)
    old_ports = set(port_map.keys())
    # If the map has an identity mapping (a->a), drop it to avoid a meaningless replacement
    old_ports = {p for p in old_ports if port_map.get(p) != p}

    # ---- 1) Optional: rename the module (only the first module declaration)
    if new_module_name:
        text = re.sub(
            r'(\bmodule\s+)([A-Za-z_]\w*)',
            lambda m: m.group(1) + new_module_name,
            text,
            count=1
        )

    # ---- 2) Remove redundant wire declarations sharing a name with a port
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

    # ---- 3) Whole-word replacement of port names (affects header, input/output declarations, and body references)
    # To avoid prefix/suffix clashes like "_02_" vs "_02__tmp", use word boundaries + escaping
    # Also sort by descending length so changing "_0_" first doesn't partially match "_02_"
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
      - LHS/RHS support: identifier (including $ and \escaped) + optional bit/part-select [i] or [msb:lsb]
      - RHS may also be a constant (e.g. 1'b0/1'b1/...)
      - If LHS/RHS are both part-selects of equal width, automatically expand into multiple per-bit BUFs
      - Only **single-line** assigns are handled; complex expressions are left as-is
    Returns:
      {"replaced": N, "leftover": M}
    """
    text = Path(v_in).read_text()

    # Simply remove block comments to avoid mismatches (line/column counts do not matter here)
    text_wo_block = re.sub(r"/\*.*?\*/", lambda m: " " * (m.end() - m.start()), text, flags=re.S)

    # Token patterns
    IDENT_BASE = r'(?:\\[^ \t\r\n]+|[A-Za-z_]\w*|\$[A-Za-z_]\w*)'
    INDEX      = r'(?:\[\s*\d+\s*(?::\s*\d+\s*)?\])'   # [i] or [msb:lsb]
    IDENT_FULL = rf'{IDENT_BASE}(?:\s*{INDEX})?'       # allow optional bit/part-select
    CONST      = r"(?:\d*'s?[bhod][0-9a-fxz_]+|\d+)"   # accepts 1'b0 / 8'hFF / 0 etc.

    ASSIGN_RE = re.compile(
        rf'^\s*assign\s+(?P<lhs>{IDENT_FULL})\s*=\s*(?P<rhs>{IDENT_FULL}|{CONST})\s*;\s*$',
        re.I
    )

    # Helpers for parsing bit/part-selects
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
            # Equal-width part-selects, expand per bit
            lmsb, llsb = lt[2], lt[3]
            rmsb, rlsb = rt[2], rt[3]
            lstep = 1 if lmsb >= llsb else -1
            rstep = 1 if rmsb >= rlsb else -1
            for li, ri in zip(range(lmsb, llsb - lstep, -lstep),
                              range(rmsb, rlsb - rstep, -rstep)):
                emit_buf(f"{lt[1]}[{li}]", f"{rt[1]}[{ri}]")
        elif lt[0] in ('id','bit') and rt[0] in ('id','bit') or (rt[0] == 'id' and re.match(rf'^{CONST}$', rhs_raw, re.I)):
            # Scalar or single-bit: emit a single BUF
            emit_buf(lhs_raw, rhs_raw)
        else:
            # Unsupported complex expression or width mismatch: keep as-is and count as leftover
            # Still write the original line (with comment) back
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
    allow_gates=None,                 # NEW: whitelist (higher priority)
    ban_prefixes=("AOI","OAI","MUX"), # NEW: blacklist (used when allow_gates is not specified)
):
    # Prepare the working directory
    W = ensure_dir(work_dir)

    # Choose the functional library: prefer a real functional .v, otherwise auto-generate a minimal techlib -- also written into work_dir
    techlib_v = stdcell_func_v if (stdcell_func_v and Path(stdcell_func_v).exists()) \
                else _autogen_techlib_from_many([in_v], W / "techlib_auto_sub.v")

    # --- Path 1: force AIG mapping then write AAG (everything into work_dir) ---
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

    # --- If writing AIG failed, fall back to BLIF for ABC to read ---
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

    # --- ABC: AIG-level optimization (everything in work_dir) ---
    if not use_blif:
        abc_in  = f"read_aiger {q(aag_path)}"
        orig    = aag_path
    else:
        abc_in  = f"read_blif {q(W / 'subcone.blif')}"
        orig    = W / "subcone.blif"

    aag_opt_path = W / "subcone_opt.aag"
    abc_cmd = (
        f"{abc_in}; "
        f"{('source ' + os.environ['NETDETOX_ABC_RC'] + '; ') if os.environ.get('NETDETOX_ABC_RC') else ''}"
        f"strash; dch; dc2; "
        f"rewrite -z; refactor -z; "
        f"resub -K 6; balance; "
        f"write_aiger {q(aag_opt_path)}; "
        f"cec {q(orig)} {q(aag_opt_path)}"
    )
    print(f"[ABC] Running: {abc_cmd}")
    run(f"abc -c {q(abc_cmd)}")

    # --- 4) Probe ports (artifacts all in work_dir) ---
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
    
    print("cur_inputs", cur_inputs)
    print("cur_outputs", cur_outputs)
    cur_inputs.sort(key=lambda x: x[1])
    cur_outputs.sort(key=lambda x: x[1])

    cur_in_names  = [name for (name, _) in cur_inputs]
    cur_out_names = [name for (name, _) in cur_outputs]

    # --- Target input names: prefer emit_info["inputs"], otherwise fall back to boundary_in ---
    if emit_info and emit_info.get("inputs"):
        tgt_in_names = list(dict.fromkeys(emit_info["inputs"]))   # dedup, keep order
    else:
        tgt_in_names = list(dict.fromkeys(boundary_in or []))
    
    if emit_info and "outputs" in emit_info and emit_info["outputs"]:
        outs = emit_info["outputs"]
        tgt_out_names = [cone_out] + [o for o in outs if o != cone_out]
    else:
        tgt_out_names = [cone_out]

    # --- 5) Map AIG back to Verilog (intermediate tmp written to work_dir; final out_v per your argument) ---
    # ban_flags = liberty_dont_use_flags(liberty, prefixes=("AOI", "OAI", "MUX"))
    # --- 5) Map AIG back to Verilog (intermediate tmp written to work_dir; final out_v per your argument) ---
    if "Nangate" in liberty:
        if allow_gates:  # whitelist takes priority
            ban_flags = liberty_dont_use_except_prefixes(liberty, allow_gates)
        else:
            ban_flags = liberty_dont_use_by_prefixes(liberty, ban_prefixes)
    else:
        if allow_gates:  # whitelist takes priority
            ban_flags = liberty_dont_use_except_prefixes_trojan(liberty, allow_gates)
        else:
            ban_flags = liberty_dont_use_flags(liberty, prefixes=ban_prefixes)
    tmp_v = W / "subcone_opt.tmp.v"
    ys2_path = W / "aig2v_tmp.ys"
    ys2 = f"""
read_liberty -lib {liberty}
read_aiger {q(aag_opt_path)}
abc -liberty {liberty}{ban_flags}
clean
write_verilog {q(tmp_v)}
"""
    Path(ys2_path).write_text(ys2)
    run(f"yosys -q -s {q(ys2_path)}")
    
    print("tgt_in_names:", tgt_in_names)
    # print("tgt_out_names:", tgt_out_names)
    print(emit_info)

    port_map = build_port_map_from_verilog(tmp_v, tgt_in_names, tgt_out_names)

    rewrite_verilog_ports(
        verilog_in=tmp_v,
        verilog_out=out_v,          # final file: keep the original location
        port_map=port_map,
        new_module_name="subcone_opt"
    )

    stats = _kill_assigns_with_buf(
        v_in=out_v,          # the optimized netlist
        v_out=out_v,         # overwrite in place
        cell="BUF_X1",
        in_pin="A",
        out_pin="Z",
        inst_prefix="U_BUF"
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
      - For each block, extract the header (cell + instance name) and pin-list, then **rebuild** into:
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

        # Remove extra newlines for easier parsing
        b = re.sub(r'\s+', ' ', b).strip()

        # Find the first "header(", where header is "CELL INST" (INST may be an escaped name)
        m = re.match(r'^([A-Za-z_]\w*)\s+([A-Za-z_\\][^(\s]*)\s*\(', b)
        if not m:
            # Not a standard instance format; fallback: return the original block but ensure it ends with ');'
            b = re.sub(r'\)+\s*;\s*$', ');', b)
            out_lines.append(b)
            continue

        cell, inst = m.group(1), m.group(2)
        head_end = m.end()  # points past '('

        # Find the last ')' matching the instance-level '(' (before ';')
        # Here rfind locates the last ')'; if not found, set the pinlist to empty
        semi = b.rfind(';')
        if semi == -1:
            semi = len(b)
        close = b.rfind(')', 0, semi)
        pin_blob = b[head_end:close] if close != -1 else ''

        # Split the pinlist by commas (at outer depth 0)
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
                # Non-standard, but repair as best as possible: grab pin and arg
                # e.g. ".A1 ( net )" or ".ZN ( \escaped )"
                mm2 = re.search(r'\.\s*([A-Za-z_]\w*)\s*\(\s*([^)]+?)\s*\)', t)
                if mm2:
                    pin, arg = mm2.group(1), mm2.group(2)
                    norm_pins.append(f'.{pin}({arg})')
                else:
                    # As a last resort, collapse whitespace and put it back as-is (very rare)
                    norm_pins.append(re.sub(r'\s+', ' ', t.strip()))

        # Rebuild one line: CELL INST ( .PIN(arg), ... );
        line = f"{cell} {inst} ( " + ", ".join(norm_pins) + " );"
        out_lines.append(line)

    return "\n".join(out_lines)

def _extract_cone_instance_block(sub_v: str, sub_top_hint: str|None,
                                 cone_out_net: str, boundary_in: list[str]) -> str:
    """
    Extract from sub_v: the **entire fanin cone** instance text (multiple blocks) that directly/indirectly drives cone_out_net.
    - Identify module inputs (from 'input ...;' and the ANSI header).
    - Candidate instance output pin names: Z, ZN, Q, QN, Y, O, S, CO (extensible).
    Return a multi-line string assembled in dependency order (without extra blank lines).
    """
    with open(sub_v, "r") as f:
        txt = f.read()

    # -------- Find module blocks (supports escaped names \foo.bar) --------
    mod_blocks = []
    for m in re.finditer(r'(^\s*module\s+(?P<name>\\\S+|[A-Za-z_]\w*)\b.*?^\s*endmodule\s*)',
                         txt, flags=re.S|re.M):
        mod_blocks.append((m.group('name').strip(), m.start(), m.end()))
    if not mod_blocks:
        raise RuntimeError(f"[extract] no module in {sub_v}")
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
        # Try to grab a segment until ');' with balanced parentheses
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
        raise RuntimeError("[extract] no instance found inside the submodule")

    # -------- Identify output/input nets for each instance --------
    out_pins = ("Z","ZN","Q","QN","Y","O","S","CO")
    def parse_ios(text_block: str):
        # Output: the first matching output pin; inputs: all .A/.A1/.B/... net names
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

    # -------- Traverse back from cone_out_net to collect the whole cone --------
    need = []
    seen_blocks = set()
    work = [cone_out_net]
    seen_nets = set(work)
    while work:
        net = work.pop()
        bi = net_to_block_idx.get(net)
        if bi is None:
            # net may be a PI/constant/assign; no need to keep traversing
            continue
        if bi in seen_blocks:
            continue
        seen_blocks.add(bi)
        need.append(bi)
        # Enqueue its inputs (ignoring primary inputs)
        for inn in blocks[bi]["ins"]:
            if inn not in prim_inputs and inn not in seen_nets:
                seen_nets.add(inn)
                work.append(inn)

    if not need:
        raise RuntimeError(f"[extract] no instance driving {cone_out_net} found; please check output pin names or submodule content")

    # -------- Emit in dependency-topological order (fanin first, target last) --------
    # Here we simply reverse the DFS order (traversal goes from output to input; reversing gives the right order)
    need = need[::-1]
    # Dedup while keeping order
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
    In the comment-stripped text, find the target module's **body** region and return (body_text, module_name).
    The module body starts after the header semicolon ';' and ends before 'endmodule'.
    Supports escaped module names: \subcone_opt.aag
    """
    def _after_module_header(i: int) -> int:
        """Given position i right after 'module <name>', skip the parameter/port list and return the position after the header ';'."""
        n = len(src)
        # Skip whitespace
        while i < n and src[i].isspace(): i += 1
        # Optional #(...)
        if i < n and src[i] == '#':
            i += 1
            while i < n and src[i].isspace(): i += 1
            if i >= n or src[i] != '(':
                raise RuntimeError("failed to parse parameterized module header: no '(' after '#'")
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
            # Some netlists may omit the port list and go straight to ';'
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
                raise RuntimeError("closing semicolon ';' not found in module header")
            i = j
        return i + 1  # one character past the semicolon

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
        # Add only the **module body** (excluding header/footer)
        mods.append((name, header_end, endpos))

    if not mods:
        raise RuntimeError("no module declaration found")

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
      Return list[ {cell, inst, ports(dict pin->net), text(str)} ]
    Supports multi-line; matches up to the ';' after ')'
    """
    insts = []
    i, n = 0, len(src)
    while i < n:
        m = re.search(r'\b([A-Za-z_]\w*)\s+([A-Za-z_.$\\][\w.$\\]*)\s*\(', src[i:])
        if not m:
            break
        cell, inst = m.group(1), m.group(2)
        s = i + m.end() - 1  # points to '('
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

        # Rough check: if it looks like a declaration (wire/reg/input/output/assign), skip
        if re.search(r'\b(?:input|output|inout|wire|reg|logic|assign)\b', full_txt):
            i = k + 1
            continue

        # Named-port parsing (not all are required)
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
    Collapse instance text into a single line, normalizing spaces and commas:
      CELL INST ( .A(n1), .B(n2), .ZN(n3) );
    """
    t = re.sub(r'\s+', ' ', txt.strip())
    # Normalize spaces around parentheses and commas
    t = t.replace(' (', ' (').replace('( ', '(').replace(' )', ')')
    t = re.sub(r'\s*,\s*', ', ', t)
    return t

def _normalize_inst_block(txt: str) -> str:
    """
    Lightweight cleanup for a whole block of instance text:
      - Remove leading/trailing blank lines
      - Collapse extra blank lines (at most one)
      - Trim trailing spaces on each line
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
    Extract **all instance lines** from the specified module (or the first module) in sub_v,
    without cone filtering; each line is normalized into a single-line instance.
    """
    raw = Path(sub_v).read_text()
    text = _strip_comments_and_attrs(raw)

    body, _mname = _find_module_region(text, sub_top_hint)
    insts = _scan_instances(body)

    if not insts:
        # Returning an empty string is allowed, but a hint makes debugging easier
        # raise RuntimeError("no instance scanned inside the module body")
        return ""

    one_liners = [_one_line_instance(x["text"]) for x in insts]
    return _normalize_inst_block("\n".join(one_liners))


# -------- Main function: text-replacement version (the call in main stays unchanged) --------
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

    # 2) Locate the instance segments to delete (by instance name, from the line containing the name to its ');' in the original netlist)
    spans = []  # [(start_idx, end_idx)]
    inst_set = set(cone_insts)
    if not inst_set:
        raise ValueError("cone_insts is empty; nothing to replace")

    # Word-boundary protection to avoid matching signal names by mistake
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

        # Record the first indentation for aligning the new text
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
            # Could not balance; remove this name too to avoid an infinite loop
            name_patterns.pop(hit, None)

    if not spans:
        raise RuntimeError(f"could not find the instances to replace in {original_v}: {sorted(list(cone_insts))}")

    # 3) Generate/obtain the new instance text to insert
    if new_inst_text is None:
        # Automatically extract all instances from the submodule (keeping your existing approach)
        new_inst_text = _extract_all_instance_lines(
            sub_v=sub_v,
            sub_top_hint=sub_top
        )

    # Clean up extra blank lines/trailing spaces
    new_inst_text = _normalize_inst_block(new_inst_text)
    # Ensure a trailing newline
    if not new_inst_text.endswith("\n"):
        new_inst_text += "\n"

    # ================== NEW: rename everything to <prefix>_OPT# ==================
    # a) Category detection (consistent with run_one_iter)
    def _inst_category(name: str) -> str | None:
        n = name.lower()
        if n.startswith("add"):      return "adder"
        if n.startswith("mul"): return "multiplier"
        if n.startswith("sub"): return "subtractor"
        if n.startswith("comp"): return "comparator"
        if n.startswith("trojan"):   return "Trojan"
        # if name.startswith("U"):       return "U"
        return "U"

    # Desired prefix
    wanted_prefix = "U_OPT"
    try:
        any_inst = next(iter(cone_insts))
        cat = _inst_category(any_inst)
        if   cat in ("adder","multiplier","subtractor","comparator","Trojan"):
            wanted_prefix = f"{cat}_OPT"
        elif cat == "U":
            wanted_prefix = "U_OPT"
    except Exception:
        pass

    # b) Collect existing instance names in the original top (to avoid name clashes)
    def _collect_existing_inst_names(full_text: str) -> set[str]:
        pat = re.compile(r'(?m)^[ \t]*(?:\(\*.*?\*\)[ \t]*)*([A-Za-z_]\w*)[ \t]+([A-Za-z_.$\\][\w.$\\]*)[ \t]*\(',
                         flags=re.S)
        return set(m.group(2) for m in pat.finditer(full_text))
    existing_names = _collect_existing_inst_names(src)

    # c) Rename all instances in the inserted block to <prefix>_OPT#, continuing numbering from the existing max
    def _rename_block_instances(block_text: str, wanted_prefix: str, existing: set[str]) -> str:
        pat = re.compile(r'(?m)^([ \t]*(?:\(\*.*?\*\)[ \t]*)*([A-Za-z_]\w*)[ \t]+)([A-Za-z_.$\\][\w.$\\]*)([ \t]*\()')
        pref_re = re.compile(rf'^{re.escape(wanted_prefix)}(\d+)$')
        # Find the existing max index for wanted_prefix
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
            head = m.group(1)  # includes attributes/CELL/whitespace
            # m.group(3) is the original instance name, unused; generate a new name directly
            tail = m.group(4)
            newname = gen_name()
            return f"{head}{newname}{tail}"

        return pat.sub(_repl, block_text)

    new_inst_text = _rename_block_instances(new_inst_text, wanted_prefix, existing_names)
    # ================== end of NEW ==================

    # 4) ... (keeping your indentation alignment + insertion logic)
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

    # 6) Report -> place in work_dir
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
    # Strip bit-widths, attributes, and runs of whitespace
    chunk = re.sub(r'\(\*.*?\*\)', '', chunk, flags=re.S)     # (* ... *)
    chunk = re.sub(r'\[[^]]+\]', ' ', chunk)                  # [msb:lsb]
    chunk = re.sub(r'\s+', ' ', chunk)
    # Grab the name portion after the wire keyword up to the semicolon
    m = re.search(r'\bwire\b([^;]*);', chunk)
    if not m:
        return []
    names_part = m.group(1)
    # Split by comma, dropping empty items
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
            # Aggregate until a line containing ';' is found
            while i < n and ';' not in lines[i]:
                buf.append(lines[i])
                i += 1
            if i < n:
                buf.append(lines[i])
                end = i
            else:
                # Unbalanced; conservatively treat as a single line
                end = start
            names = set(_parse_wire_names_from_chunk(''.join(buf)))
            decls.append((start, end, names))
            i += 1
        else:
            i += 1
    return decls


def postprocess_netlist(in_v: str, out_v: str, topmod: str,
                        inst_prefix: str = "U_OPT",
                        net_prefix: str = "W_OPT",
                        work_dir: str = "tmp"):
    """
    Apply a uniform finishing pass to the top module:
      1) Rename "non-standard instance names" (^_[0-9]+_$) -> U_OPT1..N (configurable prefix)
      2) (NEW) Unify U_BUF1/2/... into a sequence consistent with inst_prefix, continuing numbering from existing inst_prefix*
      3) Rename "temporary net names" (^_[0-9]+_$) -> W_OPTx (configurable prefix), and do a safe replacement in the module body
      4) Automatically add missing wires; remove unused wires; wrap wire declarations at a fixed count
    Only the top module text is modified; other modules are kept as-is
    """
    W = ensure_dir(work_dir)
    with open(in_v, "r") as f:
        text = f.read()

    # --- Locate the top module ---
    top_pat = re.compile(rf'(^\s*module\s+{re.escape(topmod)}\b.*?^\s*endmodule\s*)',
                         flags=re.S | re.M)
    m = top_pat.search(text)
    if not m:
        raise RuntimeError(f"[post] module {topmod} not found")
    pre, block, post = text[:m.start()], m.group(1), text[m.end():]

    # --- Parse the module header, collect port names ---
    mhead = re.match(rf'^\s*module\s+{re.escape(topmod)}\s*\((.*?)\)\s*;\s*',
                     block, flags=re.S | re.M)
    if not mhead:
        raise RuntimeError(f"[post] cannot parse the module header of {topmod}")
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
        s = re.sub(r'\[[^]]+\]', ' ', s)  # strip bit-width
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
                if re.match(r"1'[bhod]", net):  # constant
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
    # Supports a pre-instance attribute block (* ... *) and allows spanning lines
    inst_name_pat = re.compile(
        rf'(?P<prefix>^\s*(?:\(\*.*?\*\)\s*)*)'   # optional attributes (greedy but controlled by S/M)
        rf'(?P<cell>[A-Za-z_]\w*)\s+'             # CELL
        rf'(?P<inst>{IDENT})\s*\(',               # instance name (supports escaping)
        flags=re.M | re.S
    )

    def _need_inst_rename_initial(inst: str) -> bool:
        return bool(re.match(r'^_[0-9]+_$', inst))

    # First grab the existing instance set (used to generate unique names that avoid clashes)
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

    # ============ A2) Unify U_BUF\d+ into the inst_prefix sequence (continue numbering, no clash with existing) ============
    # Count all instance names now (after step A) and find the existing max index for inst_prefix
    cur_inst_names = [m.group('inst') for m in inst_name_pat.finditer(body)]
    cur_set = set(cur_inst_names)

    # Support a trailing underscore in the prefix: e.g. U_OPT3_7 still yields 7
    # Rule: match inst_prefix + optional non-alphanumeric-underscore separator + digits
    #   e.g. inst_prefix='U_OPT'  matches U_OPT7
    #        inst_prefix='U_OPT3_' matches U_OPT3_7
    # Note: take only the "last digit string" as the index
    sep = r'(?:[_\.]*)'  # a bit lenient (usually an underscore)
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

    # Rename instances named U_BUF<digits> to inst_prefix<new_id>
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

    # Re-collect the "existing" identifiers (including the just-unified instance names)
    inst_names_after = set(m.group('inst') for m in inst_name_pat.finditer(body))
    existing_identifiers = set(ports) | set(declared_wires) | set(collect_used_nets(body)) | set(inst_names_after)

    for w in declared_wires | used_nets:
        if w in ports:
            continue
        if temp_pat.match(w):
            temp_wires.add(w)

    net_gen = _gen_unique_names(net_prefix, set(existing_identifiers))
    rename_map = {}
    for old in sorted(temp_wires):  # fixed order for stable output
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

    # ============ C) Re-add/remove wires and format line-wrapping ============
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

    # After removing all wire lines, search the pure body to see whether it is still used
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

    # 1) Delete all old wire declaration blocks
    new_lines = lines[:]
    for s, e, _ in reversed(decl_blocks):
        del new_lines[s:e+1]

    # 2) Insert the new wire declaration block (if any)
    if final_wires:
        add_decl = format_wire_decl(final_wires)  # may still be multi-line
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


# def _guess_expr(cell_upper: str, xs: list[str]) -> str:
#     """
#     Determine polarity by gate family only:
#       INV / NAND* / NOR* / XNOR*      -> inverting output
#       BUF / AND* / OR* / XOR* / MUX*  -> non-inverting output
#     Completely ignores whether the output pin name ends in N (ZN/QN etc.).
#     """
#     U = cell_upper

#     # Identify the family prefix, avoiding \b issues (supports OR2_X1, NAND3_X1, etc.)
#     m = re.match(r'(INV|BUF|XNOR\d*|XOR\d*|NAND\d*|NOR\d*|AND\d*|OR\d*|MUX\d*)', U)
#     family = m.group(1) if m else U
#     fam_letters = re.match(r'[A-Z]+', family).group(0) if family else U

#     # First write the "positive-logic" core expression (without inversion), then invert based on whether the family is natively inverting
#     if fam_letters == 'INV':
#         core = xs[0] if xs else "1'b0"; native_inv = True
#     elif fam_letters == 'BUF':
#         core = xs[0] if xs else "1'b0"; native_inv = False
#     elif fam_letters == 'XNOR':
#         core = f"({xs[0]} ^ {xs[1]})" if len(xs) >= 2 else (xs[0] if xs else "1'b0"); native_inv = True
#     elif fam_letters == 'XOR':
#         core = f"({xs[0]} ^ {xs[1]})" if len(xs) >= 2 else (xs[0] if xs else "1'b0"); native_inv = False
#     elif fam_letters == 'NAND':
#         core = "(" + " & ".join(xs) + ")" if xs else "1'b1"; native_inv = True
#     elif fam_letters == 'NOR':
#         core = "(" + " | ".join(xs) + ")" if xs else "1'b0"; native_inv = True
#     elif fam_letters == 'AND':
#         core = " & ".join(xs) if xs else "1'b1"; native_inv = False
#     elif fam_letters == 'OR':
#         core = " | ".join(xs) if xs else "1'b0"; native_inv = False
#     elif fam_letters == 'MUX':
#         # Simple MUX2 fallback: S, A, B (guess from pin names as much as possible)
#         s = next((p for p in xs if p.upper() in ('S','S0','SEL')), (xs[2] if len(xs)>=3 else (xs[0] if xs else "1'b0")))
#         a = next((p for p in xs if p.upper() in ('A','A0','I0')), (xs[0] if xs else "1'b0"))
#         b = next((p for p in xs if p.upper() in ('B','A1','I1')), (xs[1] if len(xs)>=2 else a))
#         core = f"(({s}) ? ({b}) : ({a}))"; native_inv = False
#     else:
#         core = xs[-1] if xs else "1'b0"; native_inv = False

#     return f"~({core})" if native_inv else core

# import re
import re
from pathlib import Path
from typing import List, Dict, Optional

# =========================
# Helper: family classification
# =========================
def _classify_family(cell_upper: str) -> str:
    U = cell_upper.upper()
    if re.match(r'(MUX\d*|MX\d*)', U):         return 'MUX'
    if re.match(r'(AOI\d*)', U):               return 'AOI'
    if re.match(r'(OAI\d*)', U):               return 'OAI'
    if re.match(r'(HA|HAX\d*|ADDH\w*)', U):    return 'HA'
    if re.match(r'(FA|FAX\d*|ADDF\w*)', U):    return 'FA'
    if re.match(r'(DFF\w*|SDFF\w*|DFFR\w*|DLH\w*|LATCH\w*)', U):
        return 'SEQ'
    if re.match(r'(INV|BUF|XNOR\d*|XOR\d*|NAND\d*|NOR\d*|AND\d*|OR\d*)', U):
        return 'LOGIC'
    return 'GEN'  # other generic


# =========================
# Helper: output pin selection (family-aware)
# =========================
def _pick_outputs(cell_upper: str, pins: List[str]) -> List[str]:
    fam = _classify_family(cell_upper)
    U = [p.upper() for p in pins]

    if fam == 'MUX':
        # Explicitly exclude the select pin S; outputs are commonly Z/Y/O/Q...
        mux_out_cands = ('ZN','Z','Y','O','Q','Y0','Y1','Z0','Z1','O1','O2','Q1','Q2','OUT')
        outs = [pins[i] for i, u in enumerate(U) if u in mux_out_cands]
        if not outs and pins:
            outs = [pins[-1]]
        return outs

    if fam in ('HA','FA'):
        add_out_cands = ('S','SUM','CO','COUT','CARRY')
        outs = [pins[i] for i, u in enumerate(U) if u in add_out_cands]
        if not outs and pins:
            outs = [pins[-1]]
        return outs

    if fam == 'SEQ':
        # Sequential cells commonly use Q/QN etc. as outputs; still use the generic candidates here (keeping Q/QN)
        gen_out_cands = ('ZN','Z','Y','QB','QBAR','QN','Q','CO','COUT','OUT','O','O1','O2','O3','O4')
        outs = [pins[i] for i, u in enumerate(U) if u in gen_out_cands]
        if not outs and pins:
            outs = [pins[-1]]
        return outs

    # Other gates: generic candidates (excluding 'S' to avoid misjudging a MUX select pin or other inputs)
    gen_out_cands = ('ZN','Z','Y','QB','QBAR','QN','Q','CO','COUT','OUT','O','O1','O2','O3','O4')
    outs = [pins[i] for i, u in enumerate(U) if u in gen_out_cands]
    if not outs and pins:
        outs = [pins[-1]]
    return outs


# =========================
# Expression generation (including AOI/OAI/MUX)
# =========================
def _guess_expr(cell_upper: str, xs: List[str]) -> str:
    """
    Determine polarity by gate family only (independent of whether the output pin name has an N).
    - Explicit fix: parenthesize each AOI/OAI group's expression, and combine groups in first-seen order from xs.
    - Covers MUX/MX, AOI/OAI, and regular logic gate families.
    """
    U = cell_upper.upper()
    XU = [p.upper() for p in xs]

    def and_reduce(v: List[str]) -> str:
        # For the non-AOI/OAI case: keep it simple, but give the identity element for an empty set
        return " & ".join(v) if v else "1'b1"

    def or_reduce(v: List[str]) -> str:
        # For the non-AOI/OAI case: keep it simple, but give the identity element for an empty set
        return " | ".join(v) if v else "1'b0"

    def reduce_group_with_parens(v: List[str], op: str) -> str:
        # AOI/OAI-specific: always wrap the group expression in parentheses to avoid precedence ambiguity
        if not v:
            return "(1'b1)" if op == '&' else "(1'b0)"
        if len(v) == 1:
            return f"({v[0]})"
        joiner = f" {op} "
        return "(" + joiner.join(v) + ")"

    # --- MUX / MX2 ---
    if re.match(r'(MUX\d*|MX\d*)', U):
        # Select pin
        s = None
        for cand in ('S','S0','SEL','SE','S1'):
            if cand in XU:
                s = xs[XU.index(cand)]
                break
        # Data pins: prefer I0/I1, then D0/D1, then A/B order
        d0 = d1 = None
        if 'I0' in XU and 'I1' in XU:
            d0, d1 = xs[XU.index('I0')], xs[XU.index('I1')]
        elif 'D0' in XU and 'D1' in XU:
            d0, d1 = xs[XU.index('D0')], xs[XU.index('D1')]
        elif 'A' in XU and ('B' in XU or 'A1' in XU):
            d0 = xs[XU.index('A')]
            d1 = xs[XU.index('B')] if 'B' in XU else xs[XU.index('A1')]
        # Positional fallback
        if not s and len(xs) >= 1: s = xs[0]
        if not d0 and len(xs) >= 2: d0 = xs[1]
        if not d1 and len(xs) >= 3: d1 = xs[2]
        if not (s and d0 and d1):
            return or_reduce(xs) if xs else "1'b0"
        return f"(({s}) ? ({d1}) : ({d0}))"

    # --- AOI / OAI (explicit parentheses + group order taken from xs) ---
    if U.startswith('AOI') or U.startswith('OAI'):
        # Group by prefix: A/B/C/D/E
        groups: Dict[str, List[str]] = {}
        first_seen_order: List[str] = []  # group appearance order, from scanning xs
        for p in xs:
            m = re.match(r'^([A-Z]+)\d*$', p.upper())
            key_full = m.group(1) if m else p.upper()
            key = key_full if key_full in ('A','B','C','D','E') else None
            if not key:
                continue
            groups.setdefault(key, []).append(p)
            if key not in first_seen_order:
                first_seen_order.append(key)

        # Use an intermediate variable here to avoid a backslash-containing literal inside an f-string expression
        if not groups:
            if U.startswith('AOI'):
                inner = or_reduce(xs) if xs else "1'b0"
                return f"~({inner})"
            else:
                inner = and_reduce(xs) if xs else "1'b1"
                return f"~({inner})"

        if U.startswith('AOI'):
            parts = [reduce_group_with_parens(groups[g], '&') for g in first_seen_order]
            inner = " | ".join(parts) if parts else "1'b0"
            return f"~({inner})"
        else:
            parts = [reduce_group_with_parens(groups[g], '|') for g in first_seen_order]
            inner = " & ".join(parts) if parts else "1'b1"
            return f"~({inner})"


    # --- Regular gate families ---
    m = re.match(r'(INV|BUF|XNOR\d*|XOR\d*|NAND\d*|NOR\d*|AND\d*|OR\d*)', U)
    family = m.group(1) if m else U
    fam_letters = re.match(r'[A-Z]+', family).group(0) if family else U

    if fam_letters == 'INV':
        a = xs[0] if xs else "1'b0"; return f"~({a})"
    if fam_letters == 'BUF':
        a = xs[0] if xs else "1'b0"; return f"{a}"
    if fam_letters == 'XNOR':
        if len(xs) >= 2: return f"~({xs[0]} ^ {xs[1]})"
        return xs[0] if xs else "1'b0"
    if fam_letters == 'XOR':
        if len(xs) >= 2: return f"({xs[0]} ^ {xs[1]})"
        return xs[0] if xs else "1'b0"
    if fam_letters == 'NAND':
        return f"~(({and_reduce(xs)}))" if xs else "1'b1"
    if fam_letters == 'NOR':
        return f"~(({or_reduce(xs)}))" if xs else "1'b0"
    if fam_letters == 'AND':
        return and_reduce(xs) if xs else "1'b1"
    if fam_letters == 'OR':
        return or_reduce(xs) if xs else "1'b0"

    # Truly cannot guess: OR chain or single-input pass-through
    if not xs:     return "1'b0"
    if len(xs) == 1: return xs[0]
    return "(" + " | ".join(xs) + ")"


# =========================
# HA / FA specific assignments
# =========================
def _detect_ha_fa_family(cell_upper: str) -> Optional[str]:
    U = cell_upper.upper()
    if re.match(r'(HA|HAX\d*|ADDH\w*)', U):  return 'HA'
    if re.match(r'(FA|FAX\d*|ADDF\w*)', U):  return 'FA'
    return None

def _find_pin(pins: List[str], *names: str) -> Optional[str]:
    U = [p.upper() for p in pins]
    for nm in names:
        if nm in U:
            return pins[U.index(nm)]
    return None

def _emit_assigns_for_ha_fa(family: str, ins: List[str], outs: List[str]) -> Optional[List[str]]:
    """
    Recognize A/B/(CI|CIN|C) and outputs S|SUM and CO|COUT|CARRY.
    Return HA/FA-specific assigns; if recognition fails, return None so the caller uses the old logic.
    """
    if not outs:
        return None

    A  = _find_pin(ins, 'A', 'A0', 'IN1')
    B  = _find_pin(ins, 'B', 'A1', 'IN2')
    CI = _find_pin(ins, 'CI', 'CIN', 'C')

    S  = _find_pin(outs, 'S', 'SUM')
    CO = _find_pin(outs, 'CO', 'COUT', 'CARRY')

    if not (A and B):
        return None
    if family == 'FA' and not CI:
        return None

    if not S:
        S = outs[0]
    if not CO and len(outs) >= 2:
        CO = outs[1]

    assigns: List[str] = []
    if family == 'HA':
        sum_expr   = f"({A} ^ {B})"
        carry_expr = f"({A} & {B})"
    else:  # FA
        sum_expr   = f"({A} ^ {B} ^ {CI})"
        carry_expr = f"(({A} & {B}) | ({B} & {CI}) | ({A} & {CI}))"

    if S:
        assigns.append(f"  assign {S} = {sum_expr};")
    if CO:
        assigns.append(f"  assign {CO} = {carry_expr};")
    return assigns if assigns else None


# =========================
# Main flow: aggregate instances -> generate techlib
# =========================
def _autogen_techlib_from_many(files: List[str], outlib: str = "techlib_auto.v",
                               blackbox_seq: bool = False) -> str:
    """
    Collect instantiation statements from several Verilog sources and infer each cell's pin set (deduplicated union, in appearance order).
    Generate a simplified functional module (combinational equivalent/placeholder) for each cell and write it to outlib.
    """
    # Read the text
    txt = "\n".join(Path(f).read_text(errors="ignore") for f in files)

    # Rough regex: match "CELL INST ( .PIN(net), ... );"
    inst_re = re.compile(
        r'^\s*([A-Za-z0-9_]+)\s+[A-Za-z0-9_\\]+\s*\((.*?)\);\s*$',
        re.M | re.S
    )
    port_re = re.compile(r'\.\s*([A-Za-z0-9_]+)\s*\(')

    # Aggregate pins
    cells: Dict[str, List[str]] = {}
    for m in inst_re.finditer(txt):
        cell, ports_blob = m.group(1), m.group(2)
        pins = port_re.findall(ports_blob)
        if not pins:
            continue
        lst = cells.setdefault(cell, [])
        for p in pins:
            if p not in lst:
                lst.append(p)

    lines: List[str] = []

    for cell, pins in sorted(cells.items()):
        if not pins:
            continue

        fam = _classify_family(cell.upper())

        # Sequential cells: optional blackbox (no functionality generated)
        if fam == 'SEQ' and blackbox_seq:
            plist = ", ".join(pins)
            outs = _pick_outputs(cell.upper(), pins)
            ins  = [p for p in pins if p not in outs]
            decls = []
            decls.extend(f"  input {i};" for i in ins)
            decls.extend(f"  output {o};" for o in outs)
            mod = (
                f"(* blackbox *)\n"
                f"module {cell}({plist});\n" +
                ("\n".join(decls) + "\n" if decls else "") +
                "endmodule\n"
            )
            lines.append(mod)
            continue

        # Family-aware output selection
        outs = _pick_outputs(cell.upper(), pins)
        ins  = [p for p in pins if p not in outs]

        # Shell declaration
        plist = ", ".join([*ins, *outs]) if ins or outs else ", ".join(pins)
        decl_in  = "\n".join(f"  input {i};" for i in ins)
        decl_out = "\n".join(f"  output {o};" for o in outs)

        # Assignment body
        assigns: List[str] = []

        # HA/FA prefer the specialized logic
        ha_fa = _detect_ha_fa_family(cell.upper())
        if ha_fa:
            a2 = _emit_assigns_for_ha_fa(ha_fa, ins, outs)
            if a2:
                assigns.extend(a2)

        # If nothing was generated above, use generic inference + mirror/inversion
        if not assigns:
            expr0 = _guess_expr(cell.upper(), ins) if ins else "1'b0"
            if outs:
                assigns.append(f"  assign {outs[0]} = {expr0};")
                for o in outs[1:]:
                    ou = o.upper()
                    if ou in ("QN","ZN","QB","QBAR"):
                        assigns.append(f"  assign {o} = ~({outs[0]});")
                    else:
                        assigns.append(f"  assign {o} = {outs[0]};")

        body = "\n".join(assigns)

        mod = (
            f"module {cell}({plist});\n"
            f"{decl_in}\n{decl_out}\n{body}\nendmodule\n"
        )
        lines.append(mod)

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
    Only run CEC, but ensure readable gate-level cells:
      1) Prefer reading a real stdcell functional library; if not found, auto-generate techlib_auto.v
      2) Yosys: read_liberty -lib + read_verilog(functional library) + read_verilog(design)
         -> setundef -undriven -zero -> clean -purge -> aigmap
         -> write_aiger gold.aig / opt.aig
      3) ABC: abc -c "cec gold.aig opt.aig"
    """
    # --- First make all paths absolute ---
    # orig_v  = os.path.abspath(orig_v)
    new_v   = os.path.abspath(new_v)
    # liberty = os.path.abspath(liberty)
    work_dir = os.path.abspath(work_dir)
    cwd = os.getcwd()
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    try:
        # 1) Functional library
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
            return {"result":"error", "stdout":"[yosys] failed to generate gold.aig\n"+y1}

        rc2, y2 = _sh("yosys -q -s opt.ys");  Path("yosys_opt.log").write_text(y2)
        if rc2 != 0 or not Path("opt.aig").exists():
            return {"result":"error", "stdout":"[yosys] failed to generate opt.aig\n"+y2}

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
#     # optimize sub via AIG (keep original I/O names)
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
    # NEW: when root_inst is provided, use this iteration index to name the output directory; defaults to 1 if not provided
    ap.add_argument("--iter", type=int, default=None, help="explicit iteration index for output naming when --root_inst is set (default=1)")
    ap.add_argument("--use_gates", default=None,
                    help="Comma-separated gate prefixes to ALLOW (others will be -dont_use). "
                        "Example: INV,BUF,AND,NAND,NOR,OR,XOR,XNOR")

    ap.add_argument("--ban_prefixes", default="AOI,OAI,MUX",
                    help="Comma-separated cell prefixes to ban (ignored if --use_gates is set).")
    
    ap.add_argument("--gnnre", action="store_true",
                help="GNNRE mode: extract a subcircuit whose prefix category matches the root (adder/multiplier/subtractor/comparator/U); on splice-back, uniformly rename to <prefix>_OPT#")
    ap.add_argument("--trojan", action="store_true",
                    help="If set: for a non-Trojan root, extract cone only from normal gate families and exclude Trojan instances; for a Trojan root, cone can only contain instances whose names include 'Trojan'.")

    args = ap.parse_args()

    # Prepare the top-level working directory
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
        Run one iteration: use root_inst_name as the root; artifacts go to work_dir/iter_{iter_idx}/...
        Return (success, new_current_netlist)
        """
        iter_tag = f"iter_{iter_idx}"
        Wi = os.path.join(args.work_dir, iter_tag)
        os.makedirs(Wi, exist_ok=True)

        # --- Parse the current netlist and build the graph ---
        ast, _ = parse([current_netlist])
        ng = NetlistGraph(args.top)
        ng.from_ast(ast)
        g_pred, g_succ = ng.build_graph()

        # --- Verify the root exists; error if not ---
        if root_inst_name not in ng.instances:
            raise ValueError(f"[{iter_tag}] Instance {root_inst_name} not found in current netlist")

        # ============ TROJAN mode splits into two cases by root ============
        if args.trojan:  # <<< NEW
            root_is_trojan = ("trojan" in root_inst_name.lower())

            if root_is_trojan:
                # Can only traverse within the set of "instance names containing Trojan"
                allow_set = {iname for iname in ng.instances.keys()
                             if "trojan" in iname.lower()}
                if root_inst_name not in allow_set:
                    # raise ValueError(f"[{iter_tag}] Trojan root not in Trojan set?")
                    return (False, current_netlist)
                cone = k_hop_fanin_cone_filtered(root_inst_name, g_pred, args.k, allow_set)
                cone = {iname for iname in cone if not inst_has_const_pin(ng.instances[iname])}
                if root_inst_name not in cone:
                    # raise ValueError(f"[{iter_tag}] root '{root_inst_name}' has constant-connected pins; skipped this root.")
                    return (False, current_netlist)
            else:
                # Non-Trojan root: only traverse within the set of "normal gate family & instance name not containing Trojan"
                allow_set = {iname for iname, data in ng.instances.items()
                             if "trojan" not in iname.lower()
                             and is_normal_gate_family(data.get("cell", ""))}
                if root_inst_name not in allow_set:
                    # If the root itself is not a normal gate family (e.g. a complex cell), also block it
                    # raise ValueError(f"[{iter_tag}] --trojan: root '{root_inst_name}' must be a normal-gate instance and not Trojan.")
                    return (False, current_netlist)
                cone = k_hop_fanin_cone_filtered(root_inst_name, g_pred, args.k, allow_set)
                cone = {iname for iname in cone if not inst_has_const_pin(ng.instances[iname])}
                if root_inst_name not in cone:
                    # raise ValueError(f"[{iter_tag}] root '{root_inst_name}' has constant-connected pins; skipped this root.")
                    return (False, current_netlist)

        else:
        # --- Extract the fanin cone ---
        # cone = k_hop_fanin_cone(root_inst_name, g_pred, args.k)
            root_cat = inst_category(root_inst_name)
            if args.gnnre and root_cat is not None:
                # Allow set: instances of the same category as the root
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

        # --- Finishing & normalization ---
        post_v = os.path.join(Wi, f"netlist_spliced_post_{iter_idx}.v")
        postprocess_netlist(
            in_v=spliced_v,
            out_v=post_v,
            topmod=args.top,
            inst_prefix=f"U_OPT{iter_idx}_",
            net_prefix=f"W_OPT{iter_idx}_"
        )
        print(f"[{iter_tag}] post -> {post_v}")

        # === Embedded CEC: accept this iteration only if it passes; otherwise roll back and stop ===
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
        # return (True, post_v)

    # ============== Run strategy =================
    if args.root_inst:
        # Single iteration: use the index given by --iter (default 1) as the output directory index
        iter_idx = int(args.iter) if args.iter is not None else 1
        ok, new_netlist = run_one_iter(iter_idx, args.root_inst)
        # current_netlist = new_netlist
        if ok:
            current_netlist = new_netlist
        else:
            print(f"[iter_{iter_idx}] root_inst {args.root_inst} not found or failed, keep original netlist.")

    else:
        # Multiple iterations: automatically pick roots and run iterations 1..num_roots in order; the index is the iteration number
        for i in range(1, args.num_roots + 1):
            iter_tag = f"iter_{i}"

            # Parse the current netlist and pick a root (drivers preferred)
            ast, _ = parse([current_netlist])
            ng = NetlistGraph(args.top)
            ng.from_ast(ast)
            g_pred, g_succ = ng.build_graph()
            driver_insts = set(inst for net, drivers in ng.net_drivers.items() for inst, _ in drivers)
            cands = sorted(driver_insts) if driver_insts else list(ng.instances.keys())
            if not cands:
                print(f"[{iter_tag}] no candidate root; stop.")
                break
            random.seed(42 + i)
            root_inst = random.choice(cands)

            ok, new_netlist = run_one_iter(i, root_inst)
            current_netlist = new_netlist
            # if not ok:
            #     break
            if ok:
                current_netlist = new_netlist
            else:
                # If this iteration fails, continue to the next one without stopping
                print(f"[{iter_tag}] skipped  # <<< SKIP")
                continue

    print(f"[RESULT] Final netlist: {current_netlist}")


if __name__ == "__main__":
    main()
