# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# =========================
# Global styling (ignore if you don't need plotting)
# =========================
plt.style.use("default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Liberation Sans"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.2,
})

# ========= Parse a single log, extracting the "final" security and area per iteration =========
def parse_log_security_area(log_path: Path) -> Tuple[List[int], List[float], List[float]]:
    """
    Returns: iters, security_list, area_list
    For each [iter k], keep only the "last" security/area of that iteration.
    """
    pat_iter = re.compile(r"\[iter\s*(\d+)\]")
    pat_val  = re.compile(r"->\s*security=([-\d\.]+)\s*,\s*area=([-\d\.]+)")

    iters, secs, areas = [], [], []
    current_iter = None
    last_sec = None
    last_area = None

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m_iter = pat_iter.search(line)
                if m_iter:
                    # flush the last entry of the previous iteration
                    if current_iter is not None and last_sec is not None and last_area is not None:
                        iters.append(current_iter)
                        secs.append(float(last_sec))
                        areas.append(float(last_area))
                    # enter a new iteration
                    current_iter = int(m_iter.group(1))
                    last_sec = None
                    last_area = None

                m_val = pat_val.search(line)
                if m_val:
                    # update this iteration's cache (keep only the latest)
                    last_sec = m_val.group(1)
                    last_area = m_val.group(2)

        # flush the last iteration of the file
        if current_iter is not None and last_sec is not None and last_area is not None:
            iters.append(current_iter)
            secs.append(float(last_sec))
            areas.append(float(last_area))

    except FileNotFoundError:
        print(f"File not found: {log_path}")
        return [], [], []

    # ensure uniqueness (prevent duplicates)
    seen = {}
    for i, s, a in zip(iters, secs, areas):
        seen[i] = (s, a)  # keep only the last entry for the same iter
    iters = sorted(seen.keys())
    secs  = [seen[i][0] for i in iters]
    areas = [seen[i][1] for i in iters]

    return iters, secs, areas

# ========= Point-wise mean/std within a group (computes both security and area) =========
def group_mean_std_both(log_paths: List[str]) -> Tuple[
    List[int], List[float], List[float], List[float], List[float],
    Dict[str, Dict[int, Tuple[float, float]]]
]:
    """
    Returns:
      union_iters,
      sec_means, sec_stds,
      area_means, area_stds,
      per_log_series: { log_path: {iter: (sec, area), ...}, ... }
    """
    per_log_series: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for p in log_paths:
        iters, secs, areas = parse_log_security_area(Path(p))
        if iters:
            per_log_series[p] = {i: (s, a) for i, s, a in zip(iters, secs, areas)}

    if not per_log_series:
        return [], [], [], [], [], {}

    union_iters = sorted(set().union(*[set(d.keys()) for d in per_log_series.values()]))

    sec_means, sec_stds = [], []
    area_means, area_stds = [], []
    for i in union_iters:
        sec_vals = [v[0] for d in per_log_series.values() if i in d for v in [d[i]]]
        area_vals = [v[1] for d in per_log_series.values() if i in d for v in [d[i]]]

        sec_means.append(float(np.mean(sec_vals)))
        sec_stds.append(float(np.std(sec_vals, ddof=0 if len(sec_vals) > 1 else 0)))
        area_means.append(float(np.mean(area_vals)))
        area_stds.append(float(np.std(area_vals, ddof=0 if len(area_vals) > 1 else 0)))

    return union_iters, sec_means, sec_stds, area_means, area_stds, per_log_series

# ========= Optional moving average =========
def smooth(y: List[float], window: int) -> List[float]:
    if window <= 1 or window > len(y):
        return y
    kernel = np.ones(window) / window
    ypad = np.r_[y[0], y, y[-1]]  # slight endpoint padding
    ys = np.convolve(ypad, kernel, mode="same")[1:-1]
    return ys.tolist()

# ========= Export CSV (summary / per-log) =========
def export_series_csv(
    groups: Dict[str, List[str]],
    out_csv: str,
    smooth_window: int = 1,
    nudge_flat_series: bool = False
) -> pd.DataFrame:
    """
    Generate a CSV of "summary data for plotting" (contains both security and area):
      columns:
        group, iter,
        sec_mean, sec_std, sec_mean_smooth,
        area_mean, area_std, area_mean_smooth
    """
    rows = []
    for label, logs in groups.items():
        iters, sec_mean, sec_std, area_mean, area_std, _ = group_mean_std_both(logs)
        if not iters:
            print(f"Group {label} has no data, skipping export.")
            continue

        # "visual perturbation" for plotting only; not written into mean_raw; optionally used only when computing mean_smooth
        sec_for_plot = sec_mean
        area_for_plot = area_mean
        if nudge_flat_series and np.allclose(sec_mean, np.mean(sec_mean), rtol=0, atol=1e-12):
            eps = 1e-4
            t = np.linspace(0, 2*np.pi, num=len(sec_mean))
            sec_for_plot = (np.array(sec_mean) + eps * np.sin(3*t)).tolist()
        if nudge_flat_series and np.allclose(area_mean, np.mean(area_mean), rtol=0, atol=1e-12):
            eps = 1e-4
            t = np.linspace(0, 2*np.pi, num=len(area_mean))
            area_for_plot = (np.array(area_mean) + eps * np.sin(3*t)).tolist()

        sec_mean_smooth  = smooth(sec_for_plot,  smooth_window)
        area_mean_smooth = smooth(area_for_plot, smooth_window)

        for i, sm, ss, sms, am, asd, ams in zip(
            iters, sec_mean, sec_std, sec_mean_smooth, area_mean, area_std, area_mean_smooth
        ):
            rows.append({
                "group": label,
                "iter": i,
                "sec_mean": sm, "sec_std": ss, "sec_mean_smooth": sms,
                "area_mean": am, "area_std": asd, "area_mean_smooth": ams
            })

    if not rows:
        print("No summary data available to export.")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values(["group", "iter"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Summary CSV exported: {out_csv}")
    return df_out

def export_perlog_csv(groups: Dict[str, List[str]], out_csv: str) -> pd.DataFrame:
    """
    Generate a CSV of "per-log raw details" (contains security and area):
      columns: group, log_file, iter, security, area
    """
    rows = []
    for label, logs in groups.items():
        for p in logs:
            iters, secs, areas = parse_log_security_area(Path(p))
            for i, s, a in zip(iters, secs, areas):
                rows.append({
                    "group": label,
                    "log_file": p,
                    "iter": i,
                    "security": s,
                    "area": a
                })

    if not rows:
        print("No per-log data available to export.")
        return pd.DataFrame()

    df_out = pd.DataFrame(rows).sort_values(["group", "log_file", "iter"]).reset_index(drop=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Per-log CSV exported: {out_csv}")
    return df_out

# ========= (Optional) plotting: the original security average plot over three groups =========
def plot_three_groups_avg_one(
    groups: Dict[str, List[str]],
    out_png: str = "three_groups_avg.png",
    xlim: Optional[Tuple[int, int]] = None,
    sec_ylim: Optional[Tuple[float, float]] = (0.0, 1.0),
    show_band: bool = True,
    smooth_window: int = 1,
    annotate_last: bool = True,
    nudge_flat_series: bool = True,
    marker_every: int = 5,
    save_svg: bool = True
):
    fig, ax = plt.subplots(figsize=(10.8, 6.4))

    palette = {
        "LLM only": "#E37E92",
        "RL only":  "#E0823F",
        "LLM+RL":   "#1B9E77",
    }
    linestyles = {"LLM only": "-.", "RL only": "--", "LLM+RL": "-"}
    markers    = {"LLM only": "o", "RL only": "s", "LLM+RL": "D"}

    processed = {}
    for label, logs in groups.items():
        iters, sec_mean, sec_std, _, _, _ = group_mean_std_both(logs)
        if not iters:
            print(f"Group {label} has no data, skipping plot.")
            continue

        sec_for_plot = sec_mean
        if nudge_flat_series and np.allclose(sec_mean, np.mean(sec_mean), rtol=0, atol=1e-12):
            eps = 1e-4
            t = np.linspace(0, 2*np.pi, num=len(sec_mean))
            sec_for_plot = (np.array(sec_mean) + eps * np.sin(3*t)).tolist()

        sec_smooth = smooth(sec_for_plot, window=smooth_window)
        processed[label] = (iters, sec_smooth, sec_mean, sec_std)

    if not processed:
        print("None of the groups have usable data.")
        return

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        xmin = min(min(v[0]) for v in processed.values())
        xmax = max(max(v[0]) for v in processed.values())
        ax.set_xlim(xmin, xmax)

    if sec_ylim is not None:
        ax.set_ylim(*sec_ylim)

    for label, (iters, mean_plot, mean_raw, std_raw) in processed.items():
        color, ls, mk = palette.get(label), linestyles.get(label, "-"), markers.get(label, "o")
        ax.plot(iters, mean_plot, label=label, color=color, linestyle=ls)

        if len(iters) >= 2 and marker_every >= 1:
            idxs = np.arange(0, len(iters), marker_every)
            ax.plot(np.array(iters)[idxs], np.array(mean_plot)[idxs],
                    linestyle="none", marker=mk, markersize=5,
                    markerfacecolor="white", markeredgewidth=1.2, color=color)

        if show_band and len(mean_raw) == len(std_raw):
            upper = np.array(mean_raw) + np.array(std_raw)
            lower = np.array(mean_raw) - np.array(std_raw)
            alpha = 0.10 if np.allclose(upper, lower) else 0.18
            ax.fill_between(iters, lower, upper, color=color, alpha=alpha)

        if annotate_last and len(iters) > 0:
            x_last = iters[-1]; y_last = mean_plot[-1]
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
            xr = max(xmax - xmin, 1e-12); yr = max(ymax - ymin, 1e-12)
            near_right = (x_last > xmax - 0.06 * xr)
            near_top   = (y_last > ymax - 0.06 * yr)
            dx = -8 if near_right else 6; dy = -8 if near_top else 6
            ha = 'right' if near_right else 'left'; va = 'top' if near_top else 'bottom'
            ax.annotate(f"{y_last:.1%}", xy=(x_last, y_last), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va=va, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.85))

    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Security (%)")
    ax.margins(x=0.02, y=0.03)
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(True)
    leg = ax.legend(frameon=True, fancybox=True, framealpha=0.9, borderpad=0.6)
    leg.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    print(f"Saved: {out_png}")
    svg_path = Path(out_png).with_suffix(".svg")
    plt.savefig(svg_path, dpi=220)
    print(f"Vector image also saved: {svg_path}")
    plt.close()

# ========================== Usage example ==========================
if __name__ == "__main__":
    group_noRL = [
        "log_exo_RL_c3540_fresh_updated_noRL1.txt",
        "log_exo_RL_c3540_fresh_updated_noRL2.txt",
        "log_exo_RL_c3540_fresh_updated_noRL3.txt",
        "log_exo_RL_c3540_fresh_updated_noRL4.txt",
        "log_exo_RL_c3540_fresh_updated_noRL5.txt",
    ]
    group_noLLM = [
        "log_exo_RL_c3540_fresh_updated_noLLM1.txt",
        "log_exo_RL_c3540_fresh_updated_noLLM2.txt",
        "log_exo_RL_c3540_fresh_updated_noLLM3.txt",
        "log_exo_RL_c3540_fresh_updated_noLLM4.txt",
        "log_exo_RL_c3540_fresh_updated_noLLM5.txt",
    ]
    group_LLM_RL = [
        "log_exo_RL_c3540_fresh_updated_gpt.txt",
        "log_exo_RL_c3540_fresh_updated_gpt2.txt",
        "log_exo_RL_c3540_fresh_updated_gpt3.txt",
        "log_exo_RL_c3540_fresh_updated_gpt4.txt",
        "log_exo_RL_c3540_fresh_updated_gpt55.txt",
    ]

    groups = {
        "LLM only": group_noRL,
        "RL only": group_noLLM,
        "LLM+RL": group_LLM_RL,
    }

    # === 1) Export CSV (core) ===
    export_series_csv(
        groups=groups,
        out_csv="out_c3540/security_area_series_summary.csv",  # group, iter, *sec*, *area*
        smooth_window=1,
        nudge_flat_series=False
    )

    export_perlog_csv(
        groups=groups,
        out_csv="out_c3540/security_area_series_perlog.csv"    # raw details: group, log_file, iter, security, area
    )

    # === 2) (Optional) plot: still plotting security -- three-group mean +/- std ===
    plot_three_groups_avg_one(
        groups=groups,
        out_png="out_c3540/c3540_three_groups_avg.png",
        xlim=(0, 50),
        sec_ylim=(0.48, 0.8),
        show_band=True,
        smooth_window=1,
        annotate_last=True,
        nudge_flat_series=True,
        marker_every=5,
        save_svg=True,
    )
