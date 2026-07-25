#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import shutil
from pathlib import Path
from typing import List

# ===== Configuration =====
TARGETS: List[str] = [
    "add_mul_4bit","add_mul_8bit","add_mul_16bit","add_mul_32bit",
    "add_mul_combine_4bit","add_mul_combine_8bit","add_mul_combine_16bit","add_mul_combine_32bit",
    "add_mul_comp_4bit","add_mul_comp_8bit","add_mul_comp_16bit","add_mul_comp_32bit",
    "add_mul_comp_sub_4bit","add_mul_comp_sub_8bit","add_mul_comp_sub_16bit","add_mul_comp_sub_32bit",
    "add_mul_mix_4bit","add_mul_mix_8bit","add_mul_mix_16bit","add_mul_mix_32bit",
    "add_mul_sub_4bit","add_mul_sub_8bit","add_mul_sub_16bit","add_mul_sub_32bit",
]

# One-to-one correspondence with TARGETS
INDICES: List[int] = [
    18, 4, 28, 13, 4,
    3, 3, 4, 8, 48,
    3, 49, 13, 4, 28,
    44, 39, 24, 4, 14,
    9, 8, 39, 3
]
# INDICES = [i + 1 for i in INDICES]  # convert to 0-base
LLMS = ["gpt5"]#, "gpt", "llama4", "qwen3", "gemini", "gpt5"]

ROOT = Path("./circuit-transformer")  # fixed root path


def norm_circuit_dir_name(c: str) -> str:
    """Convert ..._4bit -> ..._4_bit etc., to match the directory naming on disk."""
    return re.sub(r"_(\d+)bit$", r"_\1_bit", c)


def list_existing_iters(tmp_dir: Path) -> List[int]:
    iters = []
    if not tmp_dir.is_dir():
        return iters
    for d in tmp_dir.iterdir():
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if m and (d / f"netlist_spliced_post_{m.group(1)}.v").is_file():
            iters.append(int(m.group(1)))
    return sorted(iters)


def copy_one(circuit: str, llm: str, idx: int):
    c_dir = norm_circuit_dir_name(circuit)
    tmp_dir = ROOT / f"tmp_{c_dir}_{llm}"
    src = tmp_dir / f"iter_{idx}" / f"netlist_spliced_post_{idx}.v"

    if not src.is_file():
        exists = list_existing_iters(tmp_dir)
        hint = f"available iters: {exists}" if exists else "this tmp directory does not exist or has no valid files"
        print(f"[MISS] {circuit}/{llm}: iter_{idx} not found | {hint}")
        return

    dest_dir = ROOT / f"GNNRE_{llm}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{circuit}_{llm}.v"
    shutil.copy2(src, dest)
    print(f"[COPY] {circuit}/{llm}: {src} -> {dest}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only-llm", type=str, default="", help="Process only a specific LLM (e.g. deepseek)")
    p.add_argument("--only-circuit", type=str, default="", help="Process only a specific circuit name")
    args = p.parse_args()

    if len(INDICES) != len(TARGETS):
        raise ValueError(f"INDICES({len(INDICES)}) and TARGETS({len(TARGETS)}) have mismatched lengths")

    circuits = TARGETS if not args.only_circuit else [args.only_circuit]
    llms = LLMS if not args.only_llm else [args.only_llm]

    total = copied = 0
    for circuit, idx in zip(TARGETS, INDICES):
        if circuit not in circuits:
            continue
        for llm in llms:
            total += 1
            copy_one(circuit, llm, idx)
            # Success is judged from the output; no counting is done here for simplicity
    print(f"\nDone: processed {total} tasks.")


if __name__ == "__main__":
    main()
