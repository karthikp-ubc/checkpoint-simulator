# CPEN 533 Assignment 3: Checkpointing Trade-offs in Parallel Systems

**Total: 10 points + 1 bonus point**

## Prerequisites

This assignment assumes familiarity with:
- **Python 3** — running scripts, reading printed output, and writing short
  snippets to extract numbers (e.g. dividing two values read from sweep output).
- **YAML** — editing configuration files; syntax is introduced in `README.md`.
- **Basic probability** — exponential distribution, mean, and the concept of a
  hazard rate.
- **SimPy / discrete-event simulation** — not required in depth; the simulator
  is a black box you configure and run.  Reading the Model section of `README.md`
  is sufficient background.

---

## Instructions

Use the provided simulator and `config.yaml` to complete the questions below.
See the [Marking Rubric](#marking-rubric) at the end of this document for
grading criteria.

For each question, create a dedicated YAML config file (e.g. `q1.yaml`), run
the simulator, and include the resulting plot(s) in your report.  All written
answers should be supported by data from your simulation runs.

---

## Q1 — Scaling study (2 points)

Using the default parameters (MTBF = 100 h, C = 0.5 h, MTTR = 1 h,
exponential distributions), sweep n ∈ {1, 16, 64, 256} and observe how
the system behaves as it scales.

**Instructions:**
1. Run the simulator with `output.metrics: [efficiency, wasted_work, checkpoint_overhead]`.
2. From the efficiency panel, read off the approximate optimal interval T\* for each value of n.
3. Compare the T\* values you read from the plot against the Daly formula:
   T\* = √(2 · C · M/n + C²) − C,  where M = MTBF_node.

**Deliverables:**
- The plot produced by the simulator.
- A table with columns: n, simulated T\*, Daly T\*, peak efficiency η, relative error (%).
- A short paragraph (4–6 sentences) explaining why T\* decreases as n grows,
  using the concept of system MTBF = MTBF_node / n.

> **Note:** For n = 256, Daly's formula yields T\* < C (the checkpoint write
> cost).  A checkpoint interval shorter than the write cost is physically
> infeasible, so the simulator clips T to a minimum of C × 1.05.  The large
> relative error you observe for this case reflects this constraint, not a
> modelling deficiency — comment on it briefly in your written answer.

---

## Q2 — Sharpness of the optimum (2 points)

For a single node (n = 1) with the default parameters, investigate how
sensitive efficiency is to deviations from T\*.

**Instructions:**
1. Compute T\* from the Daly formula.
2. Set `checkpoint.interval.mode: manual` and provide a fine grid of 20–30
   values spanning [0.2 · T\*, 3 · T\*].
3. Run with `output.metrics: [efficiency]` and `n_reps: 20` to get tight
   error bars.

**Deliverables:**
- The efficiency vs T plot with error bars.
- A written answer to: *How much does efficiency drop if T is set to 2× or
  0.5× the optimum?  Is the optimum flat or sharp?  What does this imply for
  practitioners who cannot checkpoint at exactly T\*?*

---

## Q3 — Impact of coordination overhead (2 points)

Coordination delay D models the time all nodes spend synchronising before the
checkpoint write begins (e.g. a barrier or two-phase commit).  This cost grows
with n and is paid even when no failure occurs.

**Instructions:**
1. Fix n = 64, MTBF = 100 h, C = 0.5 h, MTTR = 1 h.
2. Run two separate experiments with `coordination.mean` set to 0.01 and 0.5
   hours.  Keep all other parameters identical.
3. Use `output.metrics: [efficiency, coordination_overhead, checkpoint_overhead]`.

**Deliverables:**
- Two plots (or a single figure with two panels, one per μ_D value).
- A table showing, for each μ_D: peak efficiency and optimal T\*.
- A written answer to: *Does coordination overhead exceed checkpoint write
  overhead in either experiment?  How does increasing μ_D affect the optimal T\*?*

---

## Q4 — Effect of failure distribution shape (2 points)

The exponential distribution is memoryless, but real HPC systems often show a
Weibull distribution with shape k < 1 (decreasing hazard — early-life failures
dominate) or k > 1 (increasing hazard — wear-out).

**Instructions:**
1. Fix n = 16, MTBF = 100 h, C = 0.5 h, MTTR = 1 h, μ_D = 0.05 h.
2. Run two experiments varying only `failure.distribution` and `failure.shape`:
   - Weibull with `shape: 0.5` (strongly decreasing hazard)
   - Weibull with `shape: 2.0` (increasing hazard / wear-out)
3. Use `output.metrics: [efficiency, wasted_work, n_failures]`.
4. Use the n = 16 curve from your Q1 exponential run as the baseline for comparison.

**Deliverables:**
- Plots for both Weibull cases alongside the Q1 exponential baseline (n = 16);
  these may be presented as separate figures or combined into one multi-panel figure.
- A table of: distribution, shape k, peak efficiency, optimal T\*, mean
  failures per replicate at T\* (from the `n_failures` metric at the optimal T\*).
- A written answer to: *How does the shape parameter affect optimal T\* and
  peak efficiency compared to the exponential baseline?  Give an intuitive
  explanation for the difference between k = 0.5 and k = 2.0.*

---

## Q5 — Recovery time asymmetry (2 points)

Recovery time (MTTR) is dead time that neither computes nor checkpoints.
Unlike the checkpoint cost C, it is only paid after a failure, so its impact
on efficiency interacts non-linearly with the failure rate.

**Instructions:**
1. Fix n = 16, MTBF = 100 h, C = 0.5 h, μ_D = 0.05 h, exponential
   distributions.
2. Run three experiments with `recovery.mttr` set to 0.5, 1.0, and 5.0 hours.
3. Use `output.metrics: [efficiency, recovery_time, wasted_work, total_time]`
   and set `n_reps: 10`.  To obtain recovery time and wasted work as a
   percentage of total time, divide each metric's mean value at T\* by
   `total_time_mean` at T\*.

**Deliverables:**
- Efficiency vs T plots for all three MTTR values (overlay or side-by-side).
- A table of: MTTR, peak efficiency, optimal T\*, recovery time as a % of
  total simulated time, wasted work as a % of total simulated time.
- A written answer to: *Does increasing MTTR shift T\* to the left or right?
  At MTTR = 5 h, which term in the time budget is largest — wasted work or
  recovery time — and why?*

---

## Bonus — Deployment budget (1 point)

A research team is planning a parallel job on a cluster with MTBF = 100 h per
node and checkpoint write cost C = 0.5 h.  They require **at least 60%
efficiency** and want to maximise the number of nodes deployed.

**Instructions:**
1. Using your Q1 results as a starting point, identify the boundary between
   passing and failing the 60% target.  Run additional simulations at
   intermediate n values (e.g. n = 32) as needed to confirm the maximum n.
2. Repeat the search with MTBF = 50 h per node (all other parameters unchanged).
3. Use `output.metrics: [efficiency]` for any new runs.

**Deliverables:**
- A table of: n, MTBF, peak efficiency η, meets target (yes/no), T\*.
- A short paragraph (4–6 sentences) stating your recommendation for each MTBF
  and explaining why halving MTBF changes the maximum deployable n.

**Rubric:** 0.5 pts for the table with correct boundary identification for
both MTBF values; 0.5 pts for a written answer that correctly explains the
role of system MTBF = MTBF_node / n.

---

## Marking Rubric

Each question is worth 2 points, broken down as follows.

### Per-question breakdown (applies to Q1–Q5)

| Component | Full marks | Partial marks | No marks |
|-----------|-----------|---------------|----------|
| **Simulation output** (0.5 pts) | Correct config used; plot is present and legible | Plot present but config has minor errors | Plot missing or parameters wrong |
| **Table / numerical results** (0.75 pts) | All values correct and clearly labelled | Minor errors or missing column | Table absent or values unreasonable |
| **Written answer** (0.75 pts) | Explanation is accurate, concise, and grounded in simulation data | Partially correct or lacks supporting evidence | Missing, or contradicts the data |

### Question-specific expectations

| Q | Simulation output | Table | Written answer |
|---|-------------------|-------|----------------|
| **Q1** | Efficiency panel shows clear peak for each n; T* readable per curve | n (×4), simulated T*, Daly T*, peak η, relative error all present | Correctly links T* decrease to system MTBF = M/n |
| **Q2** | Fine-grid efficiency plot with visible error bars | T*, peak η, and efficiency drops at 0.5× and 2× T* reported (in written answer or table) | Correctly characterises flatness of optimum and practical implication |
| **Q3** | Two plots with coordination and checkpoint overhead panels | μ_D, peak η, T* | Correctly identifies which overhead dominates and effect on T* |
| **Q4** | Two Weibull plots compared against exponential baseline | Distribution, k, peak η, T*, mean failures/replicate at T* | Correctly contrasts k < 1 vs k > 1 behaviour with intuitive explanation |
| **Q5** | Sweep plot centred near Daly T*; all three MTTR curves shown | MTTR, peak η, T*, recovery %, wasted work % | Direction of T* shift stated correctly; budget term identified correctly |

### General deductions

- **−0.25 pts** per question if the YAML config file is not submitted.
- **−0.25 pts** per question if written answers exceed one paragraph without adding substance.
- Plots that are present but unreadable (no axis labels, too small) are treated as partially correct.

---

## Submission

Submit the following **two attachments** as a post to Piazza:

1. **Written report (PDF)** — one document containing all plots, tables, and
   written answers for Q1–Q5 (and the bonus, if attempted), in order.
2. **Code archive (.zip)** — a zip file containing all YAML config files and
   any Python scripts used to run experiments or extract results.  The zip
   should unpack into a single directory and include one YAML file per
   experiment run.
