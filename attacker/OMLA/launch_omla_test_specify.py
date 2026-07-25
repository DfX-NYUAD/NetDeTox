# Located at OMLA/launch_omla_test_new.py
import os, csv, shutil, sys, subprocess, argparse
from pathlib import Path
from typing import Optional, List, Tuple

def _clear_dir(p: Path) -> None:
    if not p.exists():
        return
    for item in p.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass

def _copy_dir_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for item in src.iterdir():
        s = item
        d = dst / item.name
        if item.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

def _run_omla_in_subproc(dataset_name: str, batch_size: int, cwd: Path) -> float:
    """
    Invoke OMLA in a separate subprocess, limiting BLAS/OMP/Torch threads and
    reclaiming resources when it finishes.
    Returns accuracy (float); on failure raises an exception carrying stdout.
    """
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    # env.setdefault("CUDA_VISIBLE_DEVICES", "")  # enable if GPU is not needed

    code = f"""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'
os.environ['TORCH_NUM_THREADS']='1'
from Main_omla_test import get_omla_key_acc_ori
acc = get_omla_key_acc_ori({dataset_name!r}, 'link.txt', 1, True, {batch_size})
print(acc)
"""
    p = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        raise RuntimeError(f"OMLA subprocess failed (rc={p.returncode}):\n{out}")
    last = out.splitlines()[-1].strip()
    try:
        return float(last)
    except Exception:
        raise RuntimeError(f"Could not parse accuracy from OMLA output:\n{out}")

def run_omla_for_iter_tmp_dataset(
    work_dir: str = "../tmp",
    circuit_name: str = "c1355",
    iter_i: Optional[int] = 1,
    batch_size: int = 64,
    verbose: bool = True,
    netlist: Optional[str] = None,   # <- new parameter
) -> Tuple[str, float]:
    """
    Run OMLA once:
      - If iter_i is provided (not None), use work_dir/iter_{iter_i}/netlist_spliced_post_{iter_i}.v
      - Otherwise, if netlist is provided, use that netlist directly
      - Otherwise, raise an error
    Returns (identifier string, accuracy)
    """
    omladir = Path.cwd()  # run from the OMLA directory
    work_dir_p = Path(work_dir)

    # circuit_datasets/{circuit_name}_tmp
    ds_dir = (omladir / "circuit_datasets" / f"{circuit_name}_tmp")
    ds_dir.mkdir(parents=True, exist_ok=True)

    # data/{circuit_name}_test -> data/{circuit_name}_tmp
    data_root = omladir / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    data_src = data_root / f"{circuit_name}_test"
    data_dst = data_root / f"{circuit_name}_tmp"
    data_dst.mkdir(parents=True, exist_ok=True)

    # ------- Choose the input netlist: prefer iter_i, then netlist ------- #
    src_netlist: Optional[Path] = None
    tag: str

    if iter_i is not None:
        # iter mode
        src_netlist = work_dir_p / f"iter_{iter_i}" / f"netlist_spliced_post_{iter_i}.v"
        tag = f"iter_{iter_i:03d}"
        if not src_netlist.exists():
            msg = f"[{tag}] missing: {src_netlist}"
            if verbose:
                print(msg)
            raise FileNotFoundError(msg)
    elif netlist is not None:
        # specified-netlist mode
        src_netlist = Path(netlist)
        tag = "netlist"
        if not src_netlist.exists():
            msg = f"[netlist] missing: {src_netlist}"
            if verbose:
                print(msg)
            raise FileNotFoundError(msg)
    else:
        raise ValueError("Either iter_i (not None) or netlist must be provided.")

    # 1) Clear circuit_datasets/{circuit_name}_tmp
    _clear_dir(ds_dir)

    # 2) Copy and add the "Test_" prefix
    dst_netlist = ds_dir / f"Test_{src_netlist.name}"
    shutil.copyfile(src_netlist, dst_netlist)

    # 3) Prepare data/{circuit_name}_tmp: clear it and copy data/{circuit_name}_test/*
    _clear_dir(data_dst)
    if data_src.exists():
        _copy_dir_contents(data_src, data_dst)
    else:
        if verbose:
            print(f"[{tag}] WARNING: data source not found: {data_src} (continuing)")

    # 4) Run OMLA in a subprocess
    acc = _run_omla_in_subproc(dataset_name=f"{circuit_name}_tmp",
                               batch_size=batch_size,
                               cwd=omladir)
    if verbose:
        print(f"[{tag}] acc = {acc:.4f}")
    return (tag, acc)


def run_omla_over_iters_tmp_dataset(
    work_dir: str = "../tmp",
    circuit_name: str = "c1355",
    max_iters: int = 100,
    batch_size: int = 64,
    csv_out: Optional[str] = None,
    verbose: bool = True,
) -> List[Tuple[int, float]]:
    omladir = Path.cwd()  # run from the OMLA directory
    work_dir_p = Path(work_dir)

    ds_dir = (omladir / "circuit_datasets" / f"{circuit_name}_tmp")
    ds_dir.mkdir(parents=True, exist_ok=True)

    data_root = omladir / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    data_src = data_root / f"{circuit_name}_test"
    data_dst = data_root / f"{circuit_name}_tmp"
    data_dst.mkdir(parents=True, exist_ok=True)

    results: List[Tuple[int, float]] = []
    missing: List[int] = []
    errors:  List[Tuple[int, str]] = []

    for i in range(1, max_iters + 1):
        src_netlist = work_dir_p / f"iter_{i}" / f"netlist_spliced_post_{i}.v"
        if not src_netlist.exists():
            missing.append(i)
            if verbose:
                print(f"[iter_{i:03d}] missing: {src_netlist}")
            continue

        _clear_dir(ds_dir)
        dst_netlist = ds_dir / f"Test_{src_netlist.name}"
        shutil.copyfile(src_netlist, dst_netlist)

        _clear_dir(data_dst)
        if data_src.exists():
            _copy_dir_contents(data_src, data_dst)
        else:
            if verbose:
                print(f"[iter_{i:03d}] WARNING: data source not found: {data_src} (continuing)")

        try:
            acc = _run_omla_in_subproc(dataset_name=f"{circuit_name}_tmp",
                                       batch_size=batch_size,
                                       cwd=omladir)
            results.append((i, acc))
            if verbose:
                print(f"[iter_{i:03d}] acc = {acc:.4f}")
        except Exception as e:
            errors.append((i, str(e)))
            if verbose:
                print(f"[iter_{i:03d}] OMLA ERROR: {e}")

    if not results:
        raise RuntimeError("No iterations were successfully tested. Check paths and OMLA pipeline.")

    results.sort(key=lambda x: x[0])
    if csv_out is None:
        csv_out = str(omladir / f"omla_acc_{circuit_name}_tmp.csv")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "accuracy"])
        w.writerows(results)
    if verbose:
        print(f"[CSV] saved → {csv_out}")
        if missing:
            print("[INFO] missing iters:", missing)
        if errors:
            print("[INFO] error iters:", errors)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OMLA security test on tmp datasets")
    parser.add_argument("--work_dir", default="../tmp", type=str)
    parser.add_argument("--circuit_name", default="c1355", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_iters", default=100, type=int,
                        help="Only used when --iter is not provided")
    parser.add_argument("--iter", type=int, default=None,
                        help="If set, run ONLY this specific iter_i")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--csv_out", type=str, default=None,
                        help="Only used in multi-iter mode")
    parser.add_argument("--netlist", type=str, default=None,
                    help="Run single test on this specified netlist when --iter is not provided")

    args = parser.parse_args()

    if args.iter is not None:
        tag, acc = run_omla_for_iter_tmp_dataset(
            work_dir=args.work_dir,
            circuit_name=args.circuit_name,
            iter_i=args.iter,
            batch_size=args.batch_size,
            verbose=args.verbose,
            netlist=None,
        )
        print(f"[RESULT] {tag} accuracy={acc:.6f}")
    elif args.netlist is not None:
        tag, acc = run_omla_for_iter_tmp_dataset(
            work_dir=args.work_dir,
            circuit_name=args.circuit_name,
            iter_i=None,                     # key: explicitly None to take the netlist branch
            batch_size=args.batch_size,
            verbose=args.verbose,
            netlist=args.netlist,
        )
        print(f"[RESULT] {tag} accuracy={acc:.6f}")
    else:
        run_omla_over_iters_tmp_dataset(
            work_dir=args.work_dir,
            circuit_name=args.circuit_name,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            csv_out=args.csv_out,
            verbose=args.verbose,
        )
