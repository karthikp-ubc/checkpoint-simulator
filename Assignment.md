# Assignment: Checkpointing Trade-offs in Parallel Systems

**Total: 10 points**

Use the provided simulator and `config.yaml` to complete the questions below.
For each question, create a dedicated YAML config file (e.g. `q1.yaml`), run
the simulator, and include the resulting plot(s) in your report.  All written
answers should be supported by data from your simulation runs.

---

## Q1 — Scaling study (2 points)

Using the default parameters (MTBF = 100 h, C = 0.5 h, MTTR = 1 h,
exponential distributions), sweep n ∈ {1, 4, 16, 64, 256} and observe how
the system behaves as it scales.

**Instructions:**
1. Run the simulator with `output.metrics: [efficiency, wasted_work, checkpoint_overhead]`.
2. From the efficiency panel, read off the approximate optimal interval T\* for each value of n.
3. Compare the T\* values you read from the plot against the Daly formula:
   T\* = √(2 · C · M/n + C²) − C.

**Deliverables:**
- The plot produced by the simulator.
- A table with columns: n, simulated T\*, Daly T\*, relative error (%).
- A short paragraph (4–6 sentences) explaining why T\* decreases as n grows,
  using the concept of system MTBF = MTBF_node / n.

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
- The two plots.
- A table showing, for each μ_D: peak efficiency, optimal T\*, coordination
  overhead as a % of total time, checkpoint overhead as a % of total time.
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
- The two plots alongside the Q1 exponential baseline (n = 16).
- A table of: distribution, shape k, peak efficiency, optimal T\*, total
  failures observed.
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
2. Run four experiments with `recovery.mttr` set to 0.5, 1.0, 2.0, and 5.0
   hours.
3. Use `output.metrics: [efficiency, recovery_time, wasted_work]` and set
   `n_reps: 10`.

**Deliverables:**
- Efficiency vs T plots for all four MTTR values (overlay or side-by-side).
- A table of: MTTR, peak efficiency, optimal T\*, recovery time as a % of
  total simulated time.
- A written answer to: *Does increasing MTTR shift T\* to the left or right?
  At MTTR = 5 h, which term in the time budget is largest — wasted work or
  recovery time — and why?*
