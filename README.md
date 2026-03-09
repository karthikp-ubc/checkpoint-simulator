# Simulator for Checkpointing and Recovery
### CPEN 533 — University of British Columbia

A discrete-event simulator (SimPy) for studying the performance trade-offs of
coordinated checkpointing in parallel systems.  Designed as a class assignment
to let students empirically discover the optimal checkpoint interval and see
how it scales with system size.

---

## Model

The system consists of **n** identical compute nodes running a long parallel
job.  Time is divided into **epochs**, each of length at most T (the checkpoint
interval).

```
Success epoch:  |<──────── T work ────────>|<─ coord ─>|<─── C ───>|
Failure epoch:  |<── t_f work ──>✕  <─────── R recovery ──────────|
                 ^last checkpoint                        ^restart from last ckpt
```

| Symbol | Meaning | Config key |
|--------|---------|------------|
| **T**  | Checkpoint interval (the tunable parameter) | `checkpoint.interval` |
| **C**  | Time to write the checkpoint (deterministic) | `checkpoint.cost` |
| **D**  | Coordination delay before each checkpoint write (random) | `coordination.mean` |
| **M**  | Per-node MTBF | `failure.mtbf` |
| **R**  | Recovery time after a failure (MTTR) | `recovery.mttr` |

**Key insight:** with *n* nodes the system failure rate is λ = n/M.  Any
single failure causes **all** nodes to roll back.  Adding nodes therefore
requires shorter checkpoint intervals — the trade-off this simulator quantifies.

### Daly's optimal interval

For exponential failures, the optimal T that maximises efficiency is:

```
T* = √(2·C·M/n + C²) − C
```

T* shrinks as 1/√n, so doubling the node count requires checkpoints ≈ 30 %
more frequently.

---

## File structure

```
simulator/
├── __init__.py          ← package exports (run_once, sweep, daly_optimal_interval, …)
├── config.yaml          ← all parameters (edit this to run experiments)
├── distributions.py     ← pluggable failure/recovery/coordination distributions
├── metrics.py           ← EpochRecord + Metrics (per-epoch and aggregated)
├── system.py            ← ParallelSystem + SystemConfig (SimPy coordinator)
├── simulator.py         ← parameter sweep, Daly formula, plotting, entry point
└── test_simulator.py    ← 48 analytical validation tests (pytest)
```

---

## Installation

Python 3.9 or later is required.  Install the dependencies with pip:

**Linux / macOS**
```bash
pip install simpy numpy scipy matplotlib pyyaml pytest
```

**Windows**
```bat
pip install simpy numpy scipy matplotlib pyyaml pytest
```

| Package | Purpose |
|---------|---------|
| `simpy` | Discrete-event simulation engine |
| `numpy` | Numerical arrays and random-number generation |
| `scipy` | Gamma function for Weibull distribution scaling |
| `matplotlib` | Result plots |
| `pyyaml` | Loading `config.yaml` |
| `pytest` | Running the test suite |

---

## Running the simulator

```bash
# Use the default config.yaml
python3 simulator.py

# Use a custom config file
python3 simulator.py --config my_experiment.yaml
```

The simulator prints progress as it sweeps each value of n, then saves a
plot to the file specified by `output.plot_file` (default:
`checkpointing_analysis.png`).

### Output plots

The figure contains one panel per requested metric plus a final summary panel:

- **Metric panels** — metric vs checkpoint interval T, one line per n value.
  For `efficiency` with exponential failures, a dashed theory curve is
  overlaid and dotted vertical lines mark each Daly-optimal T*.
- **Final panel** — T* and peak efficiency vs n (log scale), from the
  closed-form Daly formula.

---

## Configuration reference (`config.yaml`)

```yaml
simulation:
  duration: 20000.0   # simulated hours per replicate
  n_reps:   8         # replicates per (n, T) point — controls error bars
  seed:     null      # integer for reproducibility; null = random

nodes:
  counts: [1, 4, 16, 64, 256]   # values of n to sweep

checkpoint:
  cost: 0.5                     # C (hours)
  interval:
    mode: auto                  # 'auto' or 'manual'
    max_factor: 6.0             # auto: sweep up to max_factor × T*
    n_points:   35              # auto: number of T values per n
    # values: [0.6, 1.0, 2.0]  # manual: explicit list

failure:
  distribution: exponential     # exponential | weibull | lognormal
  mtbf: 100.0                   # M (hours per node)
  # shape: 0.7                  # Weibull only  (k < 1: decreasing hazard)
  # sigma: 0.5                  # lognormal only

recovery:
  distribution: exponential
  mttr: 1.0                     # mean recovery time (hours)

coordination:
  distribution: exponential     # models barrier / two-phase-commit latency
  mean: 0.05                    # mean coordination delay (hours)
  # shape: 1.5                  # Weibull k > 1: straggler-dominated barrier

output:
  plot_file: checkpointing_analysis.png
  metrics:                      # one plot panel per entry
    - efficiency                # primary figure of merit (useful work / total time)
    - wasted_work
    - recovery_time
    - coordination_overhead
    - checkpoint_overhead
    - n_failures
```

### Available metrics

| Name | Description |
|------|-------------|
| `efficiency` | useful work / total simulated time |
| `useful_work` | computation that survived to a checkpoint (hours) |
| `wasted_work` | computation lost to rollbacks (hours) |
| `recovery_time` | time spent recovering from failures (hours) |
| `coordination_overhead` | time spent in pre-checkpoint synchronisation (hours) |
| `checkpoint_overhead` | time spent writing checkpoints (hours) |
| `n_failures` | number of failure epochs |
| `n_checkpoints` | number of successful checkpoint epochs |
| `total_time` | total simulated time (hours) |

### Supported distributions

All three distributions are parameterised by their **mean** (MTBF or MTTR),
so units stay consistent when changing distribution shape.

| `distribution` | Extra kwargs | Typical use |
|----------------|--------------|-------------|
| `exponential`  | — | Memoryless failures; analytically tractable |
| `weibull`      | `shape` (k) | k < 1: decreasing hazard (ageing); k > 1: wear-out |
| `lognormal`    | `sigma`     | Heavy-tailed; common in empirical HPC failure data |

---

## Running the tests

```bash
python3 -m pytest test_simulator.py -v
```

Expected output: **48 passed** in approximately 5 minutes.

### What the tests verify

The test suite is organised in four groups, all using exponential distributions
for which exact closed-form results are available.

#### Group 0 — Analytical self-consistency
Verifies that the five time fractions derived from the renewal-reward theorem
sum to exactly 1, independently of the simulation.  This validates the
reference formulas before trusting them for simulation comparison.

#### Group 1 — Conservation and sanity
| Test | What it checks |
|------|----------------|
| `test_time_budget_conservation` | useful + wasted + recovery + coord + checkpoint = total_time (exact, floating-point only) |
| `test_efficiency_in_unit_interval` | 0 < η < 1 always |
| `test_more_nodes_more_failures` | failure count strictly increases with n |
| `test_no_failures_means_no_wasted_work` | with MTBF → ∞, zero failures → zero wasted work |

#### Group 2 — Analytical agreement (renewal-reward theorem)
Each time fraction is compared to its closed-form prediction within a 3 %
relative tolerance, across multiple (n, T) operating points spanning
10 %–50 % failure probability per epoch.

The verified quantities are:
- efficiency η = p·T / E[epoch]
- wasted fraction = (1−p)·E[X|X<T] / E[epoch]
- recovery fraction = (1−p)·MTTR / E[epoch]
- coordination fraction = p·μ_D / E[epoch]
- checkpoint fraction = p·C / E[epoch]
- failure rate = (1−p) / E[epoch]

where `p = exp(−nT/M)` and `E[X|X<T]` is the expected failure time
conditional on failing before the checkpoint.

#### Group 3 — Limiting cases
| Test | What it checks |
|------|----------------|
| `test_no_failure_limit` | η = T/(T+C+μ_D) exactly when MTBF → ∞ |
| `test_negligible_coordination_overhead` | coord fraction < 10⁻⁴ when μ_D = 10⁻⁶ |
| `test_coordination_overhead_scales_with_mean` | doubling μ_D doubles its time fraction (rare-failure regime) |
| `test_coordination_reduces_efficiency` | η decreases monotonically with μ_D |
| `test_checkpoint_cost_reduces_efficiency` | η decreases monotonically with C |
| `test_efficiency_rises_toward_optimal_from_below` | η increases as T approaches T* from below |

#### Group 4 — Optimal checkpoint interval
| Test | What it checks |
|------|----------------|
| `test_daly_formula_scaling` | T* scales as 1/√n (verified for n → 2n transitions) |
| `test_daly_formula_sqrt_2CM_limit` | exact identity T*²/(2CM_sys) = 1 − T*/M_sys |
| `test_daly_optimal_near_simulated_peak` | T* achieves within 5 % of simulated peak η, for n ∈ {1, 4, 16, 64} |


