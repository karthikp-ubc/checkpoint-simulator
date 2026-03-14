# Simulator Tutorial: Getting Started
### CPEN 533 — University of British Columbia

This tutorial walks you through everything you need to run the simulator and
extract results for the assignment.  It should take about 20–30 minutes to
work through from scratch.

---

## 1. Installation

Make sure Python 3.9 or later is installed, then install the dependencies:

```bash
pip install simpy numpy scipy matplotlib pyyaml pytest
```

Verify the installation by running the test suite from the `simulator/`
directory:

```bash
python3 -m pytest test_simulator.py -v
```

You should see **48 passed**.  If any tests fail, check that all packages
installed correctly.

---

## 2. Your first run

From the `simulator/` directory, run the simulator with the default config:

```bash
python3 simulator.py
```

This sweeps n ∈ {1, 16, 64, 256} and saves a plot to
`checkpointing_analysis.png`.  Open the plot — you should see one panel per
metric and a summary panel showing T\* vs n.

The terminal prints progress as it sweeps each value of n:

```
n=1:   sweeping 35 intervals ...  done
n=16:  sweeping 35 intervals ...  done
...
Plot saved to checkpointing_analysis.png
```

---

## 3. Reading the efficiency plot

The efficiency panel is the most important one.  For each value of n:

- The **x-axis** is the checkpoint interval T (hours).
- The **y-axis** is efficiency η (0–1).
- Each curve has a **peak** — the value of T at the peak is your simulated T\*.
- **Error bars** show the spread across replicates (controlled by `n_reps`).
- For exponential failures, a **dashed theory curve** is overlaid and
  **dotted vertical lines** mark the Daly-optimal T\* for each n.

To read off T\* for a given n, find the x-coordinate of the curve's peak.
The theory lines give you a reference — your simulated peak should be close.

---

## 4. Creating a config file for an experiment

For each assignment question, copy `config.yaml` to a new file and edit it.
For example, for Q1:

```bash
cp config.yaml q1.yaml
```

Then open `q1.yaml` in a text editor and adjust the parameters.  To run with
your new config:

```bash
python3 simulator.py --config q1.yaml
```

The plot will be saved to whatever `output.plot_file` is set to in your config.

---

## 5. Key parameters to know

### Changing the node count

```yaml
nodes:
  counts: [1, 16, 64, 256]   # edit this list
```

### Changing the checkpoint interval sweep

**Auto mode** (default) — sweeps automatically around the Daly-optimal T\*:

```yaml
checkpoint:
  interval:
    mode:       auto
    max_factor: 6.0    # sweep up to 6× T*_daly
    n_points:   30     # number of T values
```

**Manual mode** — use an explicit list of T values (useful for Q2):

```yaml
checkpoint:
  interval:
    mode:   manual
    values: [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
```

For Q2, compute T\* from the Daly formula first, then space your values
around it.  For T\* ≈ 9.5 h, a grid spanning [1.9, 28.5] (i.e. [0.2×T\*,
3×T\*]) with 25 values works well.

### Changing failure parameters

```yaml
failure:
  distribution: exponential   # exponential | weibull | lognormal
  mtbf: 100.0                 # mean time between failures per node (hours)
```

For Weibull (Q4), add the shape parameter:

```yaml
failure:
  distribution: weibull
  mtbf: 100.0
  shape: 0.5    # k < 1: decreasing hazard (early-life failures)
```

### Changing coordination delay (Q3)

```yaml
coordination:
  distribution: exponential
  mean: 0.01    # mean delay before each checkpoint write (hours)
```

### Choosing output metrics

```yaml
output:
  plot_file: my_experiment.png
  metrics:
    - efficiency
    - wasted_work
    - total_time
```

Available metrics: `efficiency`, `useful_work`, `wasted_work`,
`recovery_time`, `coordination_overhead`, `checkpoint_overhead`,
`n_failures`, `n_checkpoints`, `total_time`.

---

## 6. Extracting numbers for tables

The simulator saves a plot, but for tables you need numerical values.  The
easiest way is to write a short Python script using the simulator's API.

**Example: get peak efficiency and T\* for a single run**

```python
import sys
sys.path.insert(0, '.')          # ensure the simulator package is on the path

from simulator import sweep, daly_optimal_interval

C     = 0.5
MTBF  = 100.0
n     = 16
T_opt = daly_optimal_interval(MTBF, n, C)

# Build a grid of T values around T*
import numpy as np
T_values = np.linspace(0.2 * T_opt, 4 * T_opt, 30).tolist()

results = sweep(
    n_nodes           = n,
    intervals         = T_values,
    checkpoint_cost   = C,
    mtbf              = MTBF,
    mttr              = 1.0,
    mean_coordination = 0.05,
    sim_duration      = 20000.0,
    n_reps            = 8,
    failure_dist      = 'exponential',
    recovery_dist     = 'exponential',
    coordination_dist = 'exponential',
    metric_names      = ['efficiency', 'total_time', 'recovery_time'],
)

# Find the T with highest mean efficiency
best = max(results, key=lambda r: r['efficiency_mean'])
print(f"Simulated T* = {best['checkpoint_interval']:.2f} h")
print(f"Peak η       = {best['efficiency_mean']:.4f}")

# Compute recovery time as % of total time at T*
rec_pct = best['recovery_time_mean'] / best['total_time_mean'] * 100
print(f"Recovery %   = {rec_pct:.1f}%")
```

Each entry in `results` is a dict with keys of the form
`<metric>_mean` and `<metric>_std`, plus `T` and `n`.

---

## 7. Common mistakes

| Mistake | Fix |
|---------|-----|
| Plot is blank or shows no peak | T grid doesn't bracket T\* — widen `max_factor` or adjust manual values |
| `mode: manual` but no `values` key | Add `values: [...]` under `interval` |
| Weibull run gives same result as exponential | Check that `shape:` is at the `failure:` level, not nested elsewhere |
| Overhead % seems wrong | Make sure `total_time` is in your metrics list; divide metric mean by `total_time_mean` |
| n=256 shows large Daly error | Expected — see the Q1 note in the assignment about T\* < C clipping |

---

## 8. Quick-reference: one config per question

| Question | Key parameters to change from defaults |
|----------|----------------------------------------|
| Q1 | `nodes.counts: [1, 16, 64, 256]`; metrics: efficiency, wasted_work, checkpoint_overhead |
| Q2 | `nodes.counts: [1]`; `interval.mode: manual`; `n_reps: 20` (recommended); metrics: efficiency |
| Q3 | `nodes.counts: [64]`; `coordination.mean: 0.01` (or 0.5); metrics: efficiency, coordination_overhead, checkpoint_overhead |
| Q4 | `nodes.counts: [16]`; `failure.distribution: weibull`; `failure.shape: 0.5` (or 2.0); metrics: efficiency, wasted_work, n_failures |
| Q5 | `nodes.counts: [16]`; `recovery.mttr: 0.5` (or 1.0 or 5.0); `n_reps: 10`; metrics: efficiency, recovery_time, wasted_work, total_time |
| Bonus | Start with Q1 results; run `nodes.counts: [32]` at `mtbf: 100.0` and `nodes.counts: [8]`, `[16]` at `mtbf: 50.0`; metrics: efficiency |
