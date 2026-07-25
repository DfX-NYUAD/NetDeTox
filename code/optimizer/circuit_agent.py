"""
circuit_agent.py
=================

This module implements a **skeleton workflow** for processing hardware
designs encoded in the AIGER format.  The goal of the workflow is to
automate the following steps:

1. **Load** a circuit written in AIGER (And‑Inverter Graph) format.
2. **Rewrite** the circuit using a language model such as GPT to
   improve optimisation while maintaining functional equivalence.
3. **Verify** logic equivalence between the original and rewritten
   circuits using the ABC logic synthesis and verification tool.
4. **Convert** the circuit into a netlist representation suitable
   for graph‑based machine learning (e.g., GNN attacks) and run a
   placeholder attack to generate feedback for the next rewrite.
5. **Iterate** the rewrite/attack loop for a configurable number of
   iterations.

The implementation here relies on the
`py‑aiger` package for parsing and manipulating AIGER circuits.  The
package can be installed via pip (`pip install py‑aiger`)【198612609008009†L304-L317】.
ABC must be installed separately and is invoked as an external
command‑line tool.  A minimal Python interface to ABC is provided
via the `run_abc_cec` method.

This code is intended as a **template** rather than a complete
solution; functions such as `expression_to_aig`, `aig_to_netlist` and
`run_gnn_attack` contain simplified or placeholder implementations.
Replace those stubs with project‑specific logic when integrating with
your own parsing, netlist extraction or GNN attack pipelines.  Note
that the Python interface to ABC described in the AI4LogicSynthesis
repository【955057341307062†L84-L90】is not used here; instead, ABC is
invoked directly as a subprocess to maximise portability.
"""

import os
# ------------------------------------------------------------------

import json
import subprocess
from typing import Any, Dict, Optional

import aiger  # type: ignore
from aiger import BoolExpr  # type: ignore

import openai  # type: ignore
from openai import OpenAI


class GPTRewriter:
    """Encapsulates calls to a language model for circuit rewriting.

    The rewriter interacts with the OpenAI Chat Completion API if
    available.  When the `openai` module is not installed or no
    `OPENAI_API_KEY` is present in the environment, the
    :meth:`rewrite` method simply returns the input circuit
    representation verbatim.  Modify this class if you would like to
    integrate a different provider or a locally hosted model.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model


    def rewrite(self, circuit_repr: str, ori_repr: str=None, feedback: Optional[str] = None) -> str:
        """Rewrite an ASCII AIG (AAG) using GPT. Returns AAG text or the original on failure."""
        # Fallback when OpenAI is not configured
        if openai is None or not os.getenv("OPENAI_API_KEY"):
            return circuit_repr

        # --- helpers (local to keep function self-contained) ---
        import re

        def strip_code_fences(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                # remove opening fence
                s = re.sub(r"^```[^\n]*\n", "", s)
                # remove closing fence
                s = re.sub(r"\n```$", "", s)
            return s.strip()

        def parse_header(lines):
            if not lines:
                raise ValueError("empty output")
            if not (lines[0].startswith("aag ") or lines[0].startswith("aig ")):
                raise ValueError("output does not start with 'aag ' or 'aig '")
            toks = lines[0].split()
            if len(toks) != 6:
                raise ValueError("header must be: aag M I L O A")
            _, M, I, L, O, A = toks
            return tuple(map(int, (M, I, L, O, A)))

        def split_sections(aag_text: str):
            """Return (lines, (M,I,L,O,A), idx_gate_end, symbols_list)."""
            lines = [ln.rstrip() for ln in aag_text.splitlines() if ln.strip() != ""]
            M, I, L, O, A = parse_header(lines)
            gate_end = 1 + I + L + O + A  # index just after and-gate section
            if len(lines) < gate_end:
                raise ValueError(f"too few lines for declared counts; need >= {gate_end}")
            # symbol lines after structural part
            symbols = [ln for ln in lines[gate_end:] if ln and (ln[0] == "i" or ln[0] == "o")]
            return lines, (M, I, L, O, A), gate_end, symbols

        # --- ensure input is a clean AAG ---
        s = strip_code_fences(circuit_repr)
        m = re.search(r"_aig=(aag[\s\S]+?)\)\s*$", s)  # handles BoolExpr(_aig=aag ... )
        in_text = m.group(1).strip() if m else s
        # print("in_text:", in_text)
        if not (in_text.startswith("aag ") or in_text.startswith("aig ")):
            print("[GPTRewriter] Input is not ASCII AIG; returning unchanged.")
            return circuit_repr

        try:
            in_lines, (M0, I0, L0, O0, A0), gate_end0, in_symbols = split_sections(in_text)
        except Exception as e:
            print(f"[GPTRewriter] Bad input AAG: {e}; returning unchanged.")
            return circuit_repr

        # --- build prompts (AAG → AAG only) ---
        # sys_prompt = (
        #     "You rewrite ASCII AIG files (AAG) into logically equivalent ones with fewer AND nodes.\n"
        #     "Rules:\n"
        #     "1) Input is a valid ASCII AIG that starts with 'aag M I L O A'.\n"
        #     "2) The number of input and output must remain the same.\n"
        #     "3) Output MUST be a complete ASCII AIG text starting with 'aag ' and include exactly I input lines,\n"
        #     "   L latch lines, O output lines, and A AND-gate lines, followed by the SAME symbol lines (i*/o*)\n"
        #     "   in the same order and content as the input. Do not add comments, markdown, or prose.\n"
        #     "4) Keep I, L, O identical to the input. You may reduce A if you can, and update M accordingly.\n"
        #     "6) Maintain full logical equivalence. If you cannot reduce A safely, return the original AAG unchanged.\n"
        # )
        sys_prompt = (
            "You are an aggressive AIG circuit optimizer. Given an ASCII AIG file (AAG), your task is to output a logically equivalent circuit\n"
            "with the minimal number of AND gates, removing all redundancies and simplifying the logic as much as possible.\n"
            "\n"
            "Strict Optimization and Output Rules:\n"
            "1) The input is a valid ASCII AIG file that begins with 'aag M I L O A'.\n"
            "2) Preserve the number and order of inputs (I), latches (L), and outputs (O).\n"
            "3) The output must be a complete, valid, self-contained AAG file — and **only the raw AAG content**, with **no comments, explanations, or formatting**.\n"
            "   - It must start with a correctly updated 'aag M I L O A' header.\n"
            "   - It must include exactly I input lines, L latch lines, O output lines, and A AND-gate lines.\n"
            "   - It must be followed by the original symbol lines (i*/o*/l*/c), in the same order and content if present.\n"
            "   - If an Original AAG is provided, you must treat it as the functional reference circuit.\n"
            "   - The optimized AAG must be logically equivalent to the Original AAG for all possible input combinations.\n"
            "   - If any output literal from the Original AAG no longer exists in the optimized structure, you must update the output literal\n"
            "     to reference a new, valid literal that preserves the same logic.\n"
            "   - You must never reference undefined literals in any output or gate.\n"
            "4) You may change the **literals** used in input and output lines as needed for optimization,\n"
            "   including assigning constants (0 or 1) or reusing internal literals — but the symbolic names and order must remain exactly the same.\n"
            "5) You must update the header value M to match the highest variable index used (i.e., max literal ÷ 2).\n"
            "6) All literals used must correspond to either defined inputs or AND gate outputs. Do not reference any undefined literals.\n"
            "7) Minimize A (AND gates) as aggressively as possible: merge equivalent nodes, propagate constants, remove unused logic,\n"
            "   and simplify common patterns such as XOR/XNOR where applicable.\n"
            "8) Do not replace any output with constant 0 or 1 unless it is provably constant under all input combinations.\n"
            "9) If any part of the logic cannot be safely simplified, leave it unchanged to preserve correctness.\n"
            "10) The output must be returned as plain AAG text only — no prose, no code blocks, no markdown, and no explanations.\n"
            "11) If no safe reduction is possible, return the input AAG unchanged — still as raw AAG text only.\n"
        )




        user_prompt = "Input AAG:\n\n" + in_text
        if feedback:
            if ori_repr is not None:
                ori_aag_prompt = "Original AAG:\n\n" + ori_repr
                user_prompt = f"Feedback to consider:\n{feedback}\n\n" + user_prompt + "\n\n" + ori_aag_prompt
            else:
                user_prompt = f"Feedback to consider:\n{feedback}\n\n" + user_prompt 

        client = OpenAI()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            print("user_prompt:", user_prompt)
            # print("original output", resp.choices[0].message.content.strip())
            out_text = strip_code_fences(resp.choices[0].message.content.strip())
            print("out_text:", out_text)
            # --- ensure output is a valid AAG ---
            out_text = "aag " + out_text.split("aag ")[-1]  # ensure it starts with 'aag '
            out_text = out_text.split("i0")[0]
            out_text = out_text.split("o0")[0]
            # --- validate the model's output ---
            out_lines, (M1, I1, L1, O1, A1), gate_end1, out_symbols = split_sections(out_text)

            # Must keep I/L/O identical
            if (I1, L1, O1) != (I0, L0, O0):
                raise ValueError(f"I/L/O changed: input=({I0},{L0},{O0}) output=({I1},{L1},{O1})")

            # Must have at least the declared structural lines
            if len(out_lines) < gate_end1:
                raise ValueError("truncated AAG body relative to header counts")

            # Symbol lines must match exactly (order and content)
            if out_symbols != in_symbols:
                raise ValueError("symbol lines differ from input")

            # Optional: ensure the count of AND lines equals A1
            and_count = A1
            if len(out_lines) < 1 + I1 + L1 + O1 + and_count:
                raise ValueError("AND section length does not match A in header")

            return out_text

        except Exception as e:
            print(f"[GPTRewriter] Error or invalid AAG from model: {e}; returning original representation.")
            return circuit_repr



class CircuitAgent:
    """High‑level orchestrator for the AIGER rewrite pipeline.

    This class coordinates parsing, rewriting, equivalence checking,
    netlist conversion and attack feedback.  It is designed to be
    extensible—override or replace methods to customise behaviour.
    """

    def __init__(self, abc_binary: str = "abc", gpt_model: str = "gpt-4o") -> None:
        """Initialise the agent with paths to external tools.

        Parameters
        ----------
        abc_binary: str
            The name or path of the ABC executable to use for
            equivalence checking.  ABC is an academic tool for logic
            synthesis and formal verification supported by the AIGER
            format【198612609008009†L304-L309】.
        gpt_model: str
            Name of the OpenAI model to use for rewriting.  This
            parameter is ignored if no API key is provided.
        """
        self.abc_binary = abc_binary
        self.rewriter = GPTRewriter(model=gpt_model)

    # ------------------------------------------------------------------
    # AIGER loading and saving
    # ------------------------------------------------------------------
    def load_aiger(self, filepath: str) -> aiger.AIG:
        """Load an AIGER file into a `aiger.AIG` object.

        Parameters
        ----------
        filepath: str
            Path to the AIGER file (.aag or .aig).  ASCII AIGER files
            can be parsed directly with `aiger.load`【198612609008009†L373-L376】.

        Returns
        -------
        aiger.AIG
            The parsed circuit.
        """
        return aiger.load(filepath)

    def save_aiger(self, aig: aiger.AIG, path: str) -> None:
        """Save an AIG object to an AIGER file on disk.

        Parameters
        ----------
        aig: aiger.AIG
            The circuit to serialise.
        path: str
            Output path.  The file extension should match either
            `.aag` for ASCII or `.aig` for binary AIGER.
        """
        aig.write(path)

    # ------------------------------------------------------------------
    # Conversion between AIGs and textual representations
    # ------------------------------------------------------------------
    def aig_to_expression(self, aig) -> str:
        """
        Extract the AIGER content from an AIG object, removing i* and o* lines.
        """
        # Get the string representation of the AIG
        aiger_str = str(aig)
        # print(aiger_str)

        # Split into lines
        lines = aiger_str.split('\n')

        # Filter out i* and o* lines
        filtered_lines = []
        for line in lines:
            if not (line.startswith('i') or line.startswith('o')):
                filtered_lines.append(line)

        # Remove empty lines and join
        result = '\n'.join(line for line in filtered_lines if line.strip())
        print("here is aigtext", result)
        return result
        

    # -----------------------------------------------------------------

    def expression_to_aig(self, expr_text: str) -> "aiger.AIG":
        """
        Write the AIGER-format string to a temporary file, then load it as an AIG object.
        """
        return expr_text
        
    # ------------------------------------------------------------------
    def fix_aag_header(self,aag_lines):
        """
        Fixes the AAG header based on actual data counts.
        Expects a list of stripped lines (no \n).
        Returns corrected lines.
        """
        if not aag_lines or not aag_lines[0].startswith("aag"):
            raise ValueError("Not a valid AAG file. First line must start with 'aag'.")

        # Parse header
        header_parts = aag_lines[0].split()
        if len(header_parts) != 6:
            raise ValueError("Invalid AAG header line.")

        _, M_str, I_str, L_str, O_str, A_str = header_parts
        I = int(I_str)
        L = int(L_str)
        O = int(O_str)
        A = int(A_str)

        # Get the expected line ranges
        input_lines = aag_lines[1:1+I]
        latch_lines = aag_lines[1+I:1+I+L]
        output_lines = aag_lines[1+I+L:1+I+L+O]
        and_lines = aag_lines[1+I+L+O:1+I+L+O+A]
        rest_lines = aag_lines[1+I+L+O+A:]

        # Recalculate M (maximum variable index)
        literals = []
        for line in input_lines + output_lines:
            literals.append(int(line))
        for line in and_lines:
            a, b, c = map(int, line.strip().split())
            literals.extend([a, b, c])
        max_var = max(l // 2 for l in literals)
        
        new_header = f"aag {max_var} {I} {L} {O} {A}"
        return [new_header] + input_lines + latch_lines + output_lines + and_lines + rest_lines



    # ------------------------------------------------------------------
    # ABC equivalence checking
    # ------------------------------------------------------------------
    def run_abc_cec(self, orig_path: str, new_path: str):
        """Check equivalence of two circuits using ABC.

        ABC is an open‑source logic synthesis and verification tool.  The
        command sequence `read; read; miter; strash; dsec` loads the
        circuits, creates their miter (XOR of outputs), strashes
        (structural hashing) the miter and performs SAT‑based
        equivalence checking.  If the circuits are equivalent the
        command reports UNSAT; otherwise the miter is satisfiable.

        Parameters
        ----------
        orig_path: str
            Path to the original AIGER file.
        new_path: str
            Path to the rewritten AIGER file.

        Returns
        -------
        bool
            True if the circuits are equivalent, False otherwise.  On
            error the method prints a message and returns False.
        """
        # using aig2aig to convert AIGER to AIG
        # This is a workaround for ABC not supporting AIGER directly
        # in some versions. If you have a binary AIGER, you can skip this.
        with open(new_path, "r") as f:
            new_aig_text = f.read()
        print("new_aig_text:", repr(new_aig_text))
        fixed = self.fix_aag_header(new_aig_text.splitlines())
        print("fixed:", fixed)
        with open(new_path, "w", newline="\n", encoding="utf-8") as f:
            for line in fixed:
                f.write(line.rstrip() + "\n")

        orig_path_aig = orig_path.replace(".aag", ".aig")
        new_path_aig = new_path.replace(".aag", ".aig")
        aig2aig_cmd = [
            os.path.expanduser('~/local/bin/aigtoaig'),
            orig_path,
            orig_path_aig
        ]
        try:
            result = subprocess.run(aig2aig_cmd, capture_output=True, text=True) #, check=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
        except Exception as e:
            print(f"[CircuitAgent] Error converting original AIGER to AIG: {e}")
            return False, None

        aig2aig_cmd = [
            os.path.expanduser('~/local/bin/aigtoaig'),
            new_path,
            new_path_aig
        ]
        try:
            result = subprocess.run(aig2aig_cmd, capture_output=True, text=True) #, check=True)
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            if result.returncode != 0:
                # print(f"[CircuitAgent] Command failed with code {result.returncode}")
                return False, result.stderr
        except Exception as e:
            print(f"[CircuitAgent] Error converting new AIGER to AIG: {e}")
            return False, None
        

        cmd = [
            self.abc_binary,
            "-c",
            f"cec {orig_path_aig} {new_path_aig}"
        ]
        # print(cmd)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )
            output = result.stdout + result.stderr
            print("stdout:", result.stdout)
            print("stderr:", result.stderr)
            # remove the orig_path_aig and new_path_aig from the output
            os.remove(orig_path_aig)
            os.remove(new_path_aig)
            # ABC reports "UNSAT" when the miter is unsatisfiable,
            # meaning the circuits are equivalent.
            if "Networks are equivalent" in output:
                return True, "eql"
            elif "Networks are NOT EQUIVALENT" in output:
                return True, result.stdout
            else:
                print(f"[CircuitAgent] ABC equivalence check failed: {output.strip()}")
                return False, result.stderr
        except Exception as e:
            print(f"[CircuitAgent] Error running ABC: {e}")
            return False, None

    # ------------------------------------------------------------------
    # Netlist generation and GNN attack
    # ------------------------------------------------------------------
    def aig_to_netlist(self, aig: aiger.AIG) -> Dict[str, Any]:
        """Convert an AIG into a simple netlist structure.

        The netlist returned here is deliberately minimal: it only
        includes a list of node identifiers.  Real GNN pipelines will
        typically require a more detailed representation including
        gate types, fanin/fanout lists and attribute labels.  Extend
        this method to suit your application.

        Parameters
        ----------
        aig: aiger.AIG
            Circuit to convert.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing a list of nodes and (optionally)
            edges.  Only nodes are populated by default.
        """
        try:
            order = aiger.common.eval_order(aig)  # type: ignore
        except Exception:
            # Fallback: no evaluation order available
            order = []
        netlist = {"nodes": [], "edges": []}
        for node in order:
            netlist["nodes"].append({"id": str(node)})
        return netlist

    def run_gnn_attack(self, netlist: Dict[str, Any]) -> Dict[str, float]:
        """Run a placeholder GNN‑based attack on the netlist.

        In a realistic scenario this function would load a trained
        graph neural network and perform an attack or inference on
        the circuit graph.  Here we simply return a trivial metric
        (the number of nodes) to demonstrate how feedback can be
        passed back to the rewriter.

        Parameters
        ----------
        netlist: Dict[str, Any]
            Netlist representation of the circuit.

        Returns
        -------
        Dict[str, float]
            A dictionary of metrics.  At minimum a key 'score' is
            returned, representing the attack score.
        """
        num_nodes = len(netlist.get("nodes", []))
        return {"score": float(num_nodes)}

    # ------------------------------------------------------------------
    # High‑level optimisation loop
    # ------------------------------------------------------------------
    def optimise_circuit(
        self, aiger_path: str, iterations: int = 1, work_dir: str = "./tmp"
    ) -> str:
        """
        Perform iterative optimisation of a circuit with GPT feedback.
        """
        os.makedirs(work_dir, exist_ok=True)
        current_path = aiger_path
        with open(current_path, "r") as f:
            ori_aig_text = f.read()
        for i in range(iterations):
            # Load and serialise circuit
            # aig = self.load_aiger(current_path)
            with open(current_path, "r") as f:
                aig_text = f.read()
            # print("aig_text:", repr(aig_text))
            # print("aig", aig)
            expr_text = self.aig_to_expression(aig_text)

            # Generate feedback from previous iteration
            feedback: Optional[str] = None
            if i > 0:
                net = self.aig_to_netlist(aig_text)
                attack_result = self.run_gnn_attack(net)
                feedback = f"Attack score: {attack_result['score']}"
            
            if i == 0:
                # First iteration, use original AIG as the reference
                new_expr_text = self.rewriter.rewrite(expr_text, feedback=feedback)
            else:
                # Subsequent iterations, use previous output as reference
                new_expr_text = self.rewriter.rewrite(expr_text, ori_repr=ori_aig_text, feedback=feedback)
            # new_expr_text = self.rewriter.rewrite(expr_text, feedback=feedback)
            print("new_expr_text:", repr(new_expr_text))

            new_expr_text += "\n"  # Ensure there's a newline at the end
            # Save the candidate circuit
            new_path = os.path.join(work_dir, f"rewritten_iter_{i+1}.aag")
            with open(new_path, "w") as f:
                f.write(new_expr_text)
            # with open(new_path, "r") as f:
            #     new_aig_text = f.read()
            # # print("new_aig_text:", repr(new_aig_text))
            # fixed = self.fix_aag_header(new_aig_text.splitlines())
            # # print("fixed:", fixed)
            # with open(new_path, "w", newline="\n", encoding="utf-8") as f:
            #     for line in fixed:
            #         f.write(line.rstrip() + "\n")
            # new_expr_text = "\n".join(fixed)
            
            # load the new AIG from the rewritten text and then save it as aig file
            # new_aig = self.load_aiger(new_path)
            # self.save_aiger(new_aig, new_path)

            # self.save_aiger(aig, current_path)

            # Equivalence check against the original design
            iter_abc = 2
            while iter_abc > 0:
                run_result, error_info = self.run_abc_cec(aiger_path, new_path)
                if run_result:
                    current_path = new_path
                    if error_info == "eql":
                        break
                    else:
                        print("error_info:", error_info)
                        print(f"[CircuitAgent] Iteration {i+1}: circuits are not equivalent; continuing optimisation.")
                        new_expr_text = self.rewriter.rewrite(new_expr_text, ori_repr=ori_aig_text, feedback=error_info)
                        new_expr_text += "\n"  # Ensure there's a newline at the end
                        print("new_expr_text:", repr(new_expr_text))
                        with open(new_path, "w") as f:
                            f.write(new_expr_text)
                        iter_abc -= 1
                else:
                    print(f"[CircuitAgent] Iteration {i+1}: circuits are not correct; continuing optimisation.")
                    new_expr_text = self.rewriter.rewrite(new_expr_text, ori_repr=ori_aig_text, feedback=error_info)
                    print("new_expr_text:", repr(new_expr_text))
                    new_expr_text += "\n"  # Ensure there's a newline at the end
                    with open(new_path, "w") as f:
                        f.write(new_expr_text)
                    iter_abc -= 1

        return current_path



def _cli() -> None:
    """Command‑line entry point for circuit_agent.

    This function provides a simple CLI allowing the user to run the
    optimisation pipeline from a shell.  Use `python circuit_agent.py
    --help` to see available options.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimise an AIGER circuit using GPT and ABC"
    )
    parser.add_argument(
        "aiger_file", help="Path to the input AIGER file (.aag or .aig)"
    )
    parser.add_argument(
        "--abc",
        default="abc",
        help="Path to the ABC executable (default: 'abc')",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model identifier (used only if OPENAI_API_KEY is set)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of rewrite/attack iterations to perform",
    )
    parser.add_argument(
        "--workdir",
        default="./tmp",
        help="Directory for intermediate and output files",
    )
    args = parser.parse_args()

    agent = CircuitAgent(abc_binary=args.abc, gpt_model=args.model)
    final_path = agent.optimise_circuit(
        args.aiger_file, iterations=args.iterations, work_dir=args.workdir
    )
    print(f"Final equivalent circuit saved to: {final_path}")


if __name__ == "__main__":
    # Set your OpenAI API key via the OPENAI_API_KEY environment variable before running.
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    _cli()