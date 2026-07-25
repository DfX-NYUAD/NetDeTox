#!/bin/bash
# Example SLURM launcher: run NetdeTox against OMLA on the four locked benchmarks.
# Run from the code/ directory.  Adjust the resource lines and tool paths for your cluster.

#SBATCH --job-name=NetdeTox-OMLA
#SBATCH -p compute
#SBATCH -n 25
#SBATCH --mem=100GB
#SBATCH -t 168:00:00
#SBATCH --output=NetdeTox_OMLA_%j.out

# --- external EDA tools must be on PATH ---
export PATH=$PATH:/path/to/yosys
export PATH=$PATH:/path/to/iverilog/bin
module load abc

# --- LLM backend (set your key; OPENAI_MODEL selects the model) ---
export OPENAI_MODEL=gpt-5
# export OPENAI_API_KEY=...

DRIVER=drivers/netdetox_omla.py
BENCH=../benchmarks

for c in c1355 c1908 c2670 c3540; do
    python $DRIVER \
        --netlist $BENCH/locked_${c}.v \
        --work_dir tmp_${c} \
        --circuit_name ${c} \
        --max_iters 50 --eval_backend omla > log_netdetox_omla_${c}.txt
done
