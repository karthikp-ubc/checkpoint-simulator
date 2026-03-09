# Simulator for Checkpointing and Recovery: CPEN 533 (University of British Columbia)
"""
Checkpointing simulator — entry point.

Usage
-----
    python simulator.py                      # uses config.yaml in current dir
    python simulator.py --config my.yaml     # explicit config file
"""
import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import simpy
import yaml

from distributions import make_sampler
from metrics import Metrics
from system import ParallelSystem, SystemConfig


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

# Metrics available in Metrics objects, mapped to their property names.
AVAILABLE_METRICS = {
    'efficiency':              'efficiency',
    'useful_work':             'useful_work',
    'wasted_work':             'wasted_work',
    'recovery_time':           'total_recovery_time',
    'coordination_overhead':   'total_coordination_overhead',
    'checkpoint_overhead':     'total_checkpoint_overhead',
    'n_failures':              'n_failures',
    'n_checkpoints':           'n_checkpoints',
    'total_time':              'total_time',
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Validate requested metrics
    requested = cfg.get('output', {}).get('metrics', ['efficiency'])
    unknown = [m for m in requested if m not in AVAILABLE_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metric(s) in config: {unknown}\n"
            f"Available: {list(AVAILABLE_METRICS)}"
        )
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Single run
# ──────────────────────────────────────────────────────────────────────────────

def run_once(
    n_nodes:              int,
    checkpoint_interval:  float,
    checkpoint_cost:      float,
    mtbf:                 float,
    mttr:                 float,
    mean_coordination:    float,
    sim_duration:         float,
    failure_dist:         str,
    recovery_dist:        str,
    coordination_dist:    str,
    seed:                 int = None,
    fail_dist_kwargs:     dict = None,
    coord_dist_kwargs:    dict = None,
) -> Metrics:
    """Run one simulation replicate and return its Metrics object."""
    fail_dist_kwargs  = fail_dist_kwargs  or {}
    coord_dist_kwargs = coord_dist_kwargs or {}
    rng = np.random.default_rng(seed)
    f_seed = int(rng.integers(2**31))
    r_seed = int(rng.integers(2**31))
    c_seed = int(rng.integers(2**31))

    config = SystemConfig(
        n_nodes=n_nodes,
        checkpoint_interval=checkpoint_interval,
        checkpoint_cost=checkpoint_cost,
        failure_sampler=make_sampler(
            failure_dist, mean=mtbf, seed=f_seed, **fail_dist_kwargs
        ),
        recovery_sampler=make_sampler(
            recovery_dist, mean=mttr, seed=r_seed
        ),
        coordination_sampler=make_sampler(
            coordination_dist, mean=mean_coordination, seed=c_seed,
            **coord_dist_kwargs
        ),
        sim_duration=sim_duration,
    )
    env = simpy.Environment()
    system = ParallelSystem(env, config)
    env.run(until=sim_duration)
    return system.metrics


# ──────────────────────────────────────────────────────────────────────────────
# Parameter sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep(
    n_nodes:            int,
    intervals:          List[float],
    checkpoint_cost:    float,
    mtbf:               float,
    mttr:               float,
    mean_coordination:  float,
    sim_duration:       float,
    n_reps:             int,
    failure_dist:       str,
    recovery_dist:      str,
    coordination_dist:  str,
    base_seed:          int = None,
    fail_dist_kwargs:   dict = None,
    coord_dist_kwargs:  dict = None,
    metric_names:       List[str] = None,
) -> List[dict]:
    """
    For each T in intervals, run n_reps replicates and return
    mean ± std for every requested metric.
    """
    metric_names = metric_names or ['efficiency']
    results = []

    for T in intervals:
        per_rep = {m: [] for m in metric_names}
        for rep in range(n_reps):
            seed = None if base_seed is None else base_seed + rep
            m = run_once(
                n_nodes=n_nodes, checkpoint_interval=T,
                checkpoint_cost=checkpoint_cost, mtbf=mtbf, mttr=mttr,
                mean_coordination=mean_coordination,
                sim_duration=sim_duration, failure_dist=failure_dist,
                recovery_dist=recovery_dist, coordination_dist=coordination_dist,
                seed=seed, fail_dist_kwargs=fail_dist_kwargs,
                coord_dist_kwargs=coord_dist_kwargs,
            )
            for name in metric_names:
                attr = AVAILABLE_METRICS[name]
                per_rep[name].append(getattr(m, attr))

        row = {'checkpoint_interval': T}
        for name in metric_names:
            vals = np.array(per_rep[name])
            row[f'{name}_mean'] = float(np.mean(vals))
            row[f'{name}_std']  = float(np.std(vals))
        results.append(row)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Theoretical reference (exponential failures)
# ──────────────────────────────────────────────────────────────────────────────

def daly_optimal_interval(mtbf_node: float, n_nodes: int, C: float) -> float:
    """
    Daly (2006) higher-order approximation:  T* = sqrt(2·C·M_sys + C²) − C
    where M_sys = MTBF_node / n_nodes.
    """
    M_sys = mtbf_node / n_nodes
    return np.sqrt(2 * C * M_sys + C ** 2) - C


def efficiency_theory(T: float, n_nodes: int, mtbf_node: float,
                      C: float, R: float, mean_coordination: float = 0.0) -> float:
    """Closed-form efficiency for exponential failures (renewal-reward theorem).

    Parameters
    ----------
    T                 : checkpoint interval
    n_nodes           : number of parallel nodes
    mtbf_node         : per-node MTBF
    C                 : checkpoint write cost (deterministic)
    R                 : mean recovery time (MTTR)
    mean_coordination : mean coordination delay before each checkpoint write
    """
    lam = n_nodes / mtbf_node
    p_ok   = np.exp(-lam * T)
    p_fail = 1.0 - p_ok
    if p_fail < 1e-12:
        e_x_given_fail = T / 2.0
    else:
        e_x_given_fail = (1.0 - (1.0 + lam * T) * p_ok) / (lam * p_fail)
    e_useful   = p_ok * T
    e_duration = p_ok * (T + mean_coordination + C) + p_fail * (e_x_given_fail + R)
    return e_useful / e_duration if e_duration > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def make_plots(all_results: dict, cfg: dict) -> None:
    """
    all_results : { n_nodes: list_of_row_dicts }
    cfg         : parsed YAML config
    """
    import matplotlib.pyplot as plt

    metric_names   = cfg['output']['metrics']
    node_counts    = cfg['nodes']['counts']
    C              = cfg['checkpoint']['cost']
    mtbf           = cfg['failure']['mtbf']
    mttr           = cfg['recovery']['mttr']
    mean_coord     = cfg['coordination']['mean']
    colors = plt.cm.tab10(np.linspace(0, 0.8, len(node_counts)))

    n_metrics = len(metric_names)
    # Always include the T* vs n panel as the last panel
    n_panels = n_metrics + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    # ── One panel per requested metric ────────────────────────────────────
    for ax, metric in zip(axes[:n_metrics], metric_names):
        for color, n in zip(colors, node_counts):
            rows  = all_results[n]
            T_vals = [r['checkpoint_interval'] for r in rows]
            means  = [r[f'{metric}_mean']      for r in rows]
            stds   = [r[f'{metric}_std']        for r in rows]
            T_opt  = daly_optimal_interval(mtbf, n, C)

            ax.errorbar(T_vals, means, yerr=stds, fmt='o-',
                        color=color, label=f'n={n}', markersize=3, capsize=2)

            # Overlay theoretical efficiency curve only for the efficiency panel
            if metric == 'efficiency' and cfg['failure']['distribution'] == 'exponential':
                th = [efficiency_theory(T, n, mtbf, C, mttr, mean_coord) for T in T_vals]
                ax.plot(T_vals, th, '--', color=color, alpha=0.55)

            ax.axvline(T_opt, color=color, linestyle=':', alpha=0.35)

        ax.set_xlabel('Checkpoint interval  T  (hours)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric}\n(dashed = theory, dotted = T* per n)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Final panel: T* and peak efficiency vs n ──────────────────────────
    ax_last  = axes[-1]
    n_range  = np.logspace(0, np.log10(max(node_counts) * 4), 300)
    T_opt_curve  = [daly_optimal_interval(mtbf, n, C) for n in n_range]
    eff_opt_curve = [efficiency_theory(T_opt_curve[i], n_range[i], mtbf, C, mttr, mean_coord)
                     for i in range(len(n_range))]

    ax_r = ax_last.twinx()
    l1, = ax_last.semilogx(n_range, T_opt_curve,    'k-',  label='T*  (Daly)')
    l2, = ax_r.semilogx   (n_range, eff_opt_curve,  'b--', label='Peak efficiency')
    ax_last.set_xlabel('Number of nodes  n')
    ax_last.set_ylabel('Optimal interval  T*  (hours)', color='k')
    ax_r.set_ylabel('Peak efficiency', color='b')
    ax_last.set_title('Optimal T* and Peak Efficiency vs Scale')
    ax_last.legend(handles=[l1, l2], fontsize=8)
    ax_last.grid(True, alpha=0.3)

    plt.tight_layout()
    out = cfg['output']['plot_file']
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    cfg = load_config(config_path)

    sim_cfg      = cfg['simulation']
    chkpt_cfg    = cfg['checkpoint']
    fail_cfg     = cfg['failure']
    rec_cfg      = cfg['recovery']
    coord_cfg    = cfg['coordination']
    node_counts  = cfg['nodes']['counts']
    metric_names = cfg['output']['metrics']

    # Extra distribution kwargs (e.g. Weibull shape, lognormal sigma)
    fail_dist_kwargs  = {k: v for k, v in fail_cfg.items()
                         if k not in ('distribution', 'mtbf')}
    coord_dist_kwargs = {k: v for k, v in coord_cfg.items()
                         if k not in ('distribution', 'mean')}

    base_seed = sim_cfg.get('seed')

    all_results = {}
    for n in node_counts:
        print(f"Sweeping n={n} …", flush=True)

        T_opt = daly_optimal_interval(fail_cfg['mtbf'], n, chkpt_cfg['cost'])
        iv = chkpt_cfg['interval']
        if iv['mode'] == 'manual':
            intervals = list(iv['values'])
        else:
            intervals = list(np.linspace(
                chkpt_cfg['cost'] * 1.05,
                iv['max_factor'] * T_opt,
                int(iv['n_points']),
            ))

        all_results[n] = sweep(
            n_nodes=n,
            intervals=intervals,
            checkpoint_cost=chkpt_cfg['cost'],
            mtbf=fail_cfg['mtbf'],
            mttr=rec_cfg['mttr'],
            mean_coordination=coord_cfg['mean'],
            sim_duration=sim_cfg['duration'],
            n_reps=sim_cfg['n_reps'],
            failure_dist=fail_cfg['distribution'],
            recovery_dist=rec_cfg['distribution'],
            coordination_dist=coord_cfg['distribution'],
            base_seed=base_seed,
            fail_dist_kwargs=fail_dist_kwargs,
            coord_dist_kwargs=coord_dist_kwargs,
            metric_names=metric_names,
        )

    make_plots(all_results, cfg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Checkpointing simulator')
    parser.add_argument(
        '--config', default='config.yaml',
        help='Path to YAML configuration file (default: config.yaml)',
    )
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    main(args.config)
