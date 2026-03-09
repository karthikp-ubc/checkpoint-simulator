"""
Analytical validation tests for the checkpointing simulator.

All tests use exponential failure/recovery/coordination distributions,
for which exact closed-form expressions exist via the renewal-reward theorem.

─── Model recap ──────────────────────────────────────────────────────────────
n nodes, each with MTBF = M  →  system failure rate  λ = n / M

Each epoch is one of two outcomes:

  SUCCESS  (prob p = e^{-λT}):
      work T  →  coordination D ~ Exp(μ_D)  →  checkpoint write C
      epoch duration = T + D + C

  FAILURE  (prob 1−p):
      work until X ~ Exp(λ) | X < T  →  recovery R ~ Exp(MTTR)
      epoch duration = X + R
      all work since last checkpoint is lost

By the renewal-reward theorem, any time-averaged quantity converges to
E[quantity per epoch] / E[epoch duration].

─── Key expectations ─────────────────────────────────────────────────────────
  p          = e^{-λT}
  E[X|X<T]   = [1 − (1 + λT)·e^{−λT}] / [λ·(1−e^{−λT})]
  E[epoch]   = p·(T + μ_D + C) + (1−p)·(E[X|X<T] + MTTR)

  efficiency             = p·T              / E[epoch]
  wasted fraction        = (1−p)·E[X|X<T]  / E[epoch]
  recovery fraction      = (1−p)·MTTR      / E[epoch]
  coordination fraction  = p·μ_D           / E[epoch]
  checkpoint fraction    = p·C             / E[epoch]
  failure rate           = (1−p)           / E[epoch]

  All five time fractions sum to 1 (verified in test_analytical_fractions_sum_to_one).
─────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pytest

from simulator import run_once, daly_optimal_interval, sweep


# ─────────────────────────────────────────────────────────────────────────────
# Analytical reference functions
# ─────────────────────────────────────────────────────────────────────────────

def _lam(n, M):
    return n / M

def _p_ok(T, n, M):
    return np.exp(-_lam(n, M) * T)

def _e_x_given_fail(T, n, M):
    """E[X | X < T] for X ~ Exp(n/M)."""
    lam = _lam(n, M)
    p_fail = 1.0 - np.exp(-lam * T)
    if p_fail < 1e-12:           # limit as λT → 0
        return T / 2.0
    return (1.0 - (1.0 + lam * T) * np.exp(-lam * T)) / (lam * p_fail)

def _e_epoch(T, n, M, C, mu_D, MTTR):
    p = _p_ok(T, n, M)
    return p * (T + mu_D + C) + (1.0 - p) * (_e_x_given_fail(T, n, M) + MTTR)

def theory_efficiency(T, n, M, C, mu_D, MTTR):
    return _p_ok(T, n, M) * T / _e_epoch(T, n, M, C, mu_D, MTTR)

def theory_wasted_fraction(T, n, M, C, mu_D, MTTR):
    p_fail = 1.0 - _p_ok(T, n, M)
    return p_fail * _e_x_given_fail(T, n, M) / _e_epoch(T, n, M, C, mu_D, MTTR)

def theory_recovery_fraction(T, n, M, C, mu_D, MTTR):
    p_fail = 1.0 - _p_ok(T, n, M)
    return p_fail * MTTR / _e_epoch(T, n, M, C, mu_D, MTTR)

def theory_coord_fraction(T, n, M, C, mu_D, MTTR):
    return _p_ok(T, n, M) * mu_D / _e_epoch(T, n, M, C, mu_D, MTTR)

def theory_checkpoint_fraction(T, n, M, C, mu_D, MTTR):
    return _p_ok(T, n, M) * C / _e_epoch(T, n, M, C, mu_D, MTTR)

def theory_failure_rate(T, n, M, C, mu_D, MTTR):
    """Expected failures per unit simulated time."""
    p_fail = 1.0 - _p_ok(T, n, M)
    return p_fail / _e_epoch(T, n, M, C, mu_D, MTTR)


# ─────────────────────────────────────────────────────────────────────────────
# Simulation helper  (always exponential distributions for analytical tests)
# ─────────────────────────────────────────────────────────────────────────────

# Default parameters used unless overridden per test
DEFAULTS = dict(C=0.5, M=100.0, MTTR=1.0, mu_D=0.05)

SIM_DURATION = 100_000.0   # 100k hours gives ~10k+ epochs for most cases
SEED         = 42

# Statistical tolerance.  With N ≈ 10k–100k epochs the relative SE of a
# time-average is typically 0.3–1 %.  Using 3 % gives a comfortable ~3σ margin.
RTOL = 0.03


def simulate(n, T, C=DEFAULTS['C'], M=DEFAULTS['M'],
             MTTR=DEFAULTS['MTTR'], mu_D=DEFAULTS['mu_D'],
             sim_duration=SIM_DURATION, seed=SEED):
    return run_once(
        n_nodes=n, checkpoint_interval=T, checkpoint_cost=C,
        mtbf=M, mttr=MTTR, mean_coordination=mu_D,
        sim_duration=sim_duration,
        failure_dist='exponential', recovery_dist='exponential',
        coordination_dist='exponential',
        seed=seed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group 0 — Verify the analytical formulas themselves
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("T,n", [(10.0, 1), (5.0, 4), (2.0, 16)])
def test_analytical_fractions_sum_to_one(T, n):
    """
    The five time fractions derived from the renewal-reward theorem must
    partition the unit interval exactly.  This validates our reference
    formulas before any simulation comparison.
    """
    kw = dict(M=DEFAULTS['M'], C=DEFAULTS['C'],
              mu_D=DEFAULTS['mu_D'], MTTR=DEFAULTS['MTTR'])
    total = (theory_efficiency(T, n, **kw)
             + theory_wasted_fraction(T, n, **kw)
             + theory_recovery_fraction(T, n, **kw)
             + theory_coord_fraction(T, n, **kw)
             + theory_checkpoint_fraction(T, n, **kw))
    assert abs(total - 1.0) < 1e-12, f"Fractions sum to {total}, not 1"


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Conservation / sanity
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0), (16, 2.0), (64, 1.0)])
def test_time_budget_conservation(n, T):
    """
    Every simulated second must fall into exactly one category.
    This is an exact identity; we allow only floating-point rounding error.
    """
    m = simulate(n, T)
    budget = (m.useful_work
              + m.wasted_work
              + m.total_recovery_time
              + m.total_coordination_overhead
              + m.total_checkpoint_overhead)
    assert abs(budget - m.total_time) < 1e-6 * m.total_time, (
        f"n={n}, T={T}: budget={budget:.6f} != total_time={m.total_time:.6f}"
    )


@pytest.mark.parametrize("n,T", [(1, 10.0), (16, 3.0), (64, 1.0)])
def test_efficiency_in_unit_interval(n, T):
    """Efficiency is a fraction of time: must lie strictly in (0, 1)."""
    m = simulate(n, T)
    assert 0.0 < m.efficiency < 1.0


@pytest.mark.parametrize("T", [5.0, 10.0, 20.0])
def test_more_nodes_more_failures(T):
    """
    Adding nodes raises the system failure rate λ = n/M, so the raw failure
    count must increase strictly with n (for identical T and M).
    """
    counts = [simulate(n, T).n_failures for n in [1, 4, 16, 64]]
    for a, b in zip(counts, counts[1:]):
        assert b > a, (
            f"T={T}: failure counts did not increase with n: {counts}"
        )


@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0)])
def test_no_failures_means_no_wasted_work(n, T):
    """
    In any run without failures every epoch was successful, so wasted work,
    recovery time, and coordination-on-failure must all be exactly zero.
    """
    m = simulate(n, T, M=1e9)   # effectively infinite MTBF
    assert m.n_failures == 0
    assert m.wasted_work == 0.0
    assert m.total_recovery_time == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Analytical agreement (renewal-reward theorem)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n,T", [
    (1,  10.0),   # λT ≈ 0.10  →  ~10 % failure probability per epoch
    (1,  50.0),   # λT ≈ 0.50  →  ~39 % failure probability
    (4,   5.0),   # λT ≈ 0.20
    (16,  2.0),   # λT ≈ 0.32
    (64,  1.0),   # λT ≈ 0.64  →  ~47 % failure probability
])
def test_efficiency_matches_theory(n, T):
    """
    Simulated efficiency must agree with the renewal-reward prediction
    to within RTOL.  Covers a range of failure pressures.
    """
    C, M, MTTR, mu_D = DEFAULTS['C'], DEFAULTS['M'], DEFAULTS['MTTR'], DEFAULTS['mu_D']
    m = simulate(n, T)
    expected = theory_efficiency(T, n, M, C, mu_D, MTTR)
    assert abs(m.efficiency - expected) / expected < RTOL, (
        f"n={n}, T={T}: simulated η={m.efficiency:.4f}, theory η={expected:.4f}"
    )


@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0), (16, 2.0)])
def test_wasted_fraction_matches_theory(n, T):
    """wasted_work / total_time ≈ (1−p)·E[X|X<T] / E[epoch].

    Uses a longer run because wasted work is a small fraction (~5-10 % of
    time), giving it higher relative variance than efficiency.
    """
    C, M, MTTR, mu_D = DEFAULTS['C'], DEFAULTS['M'], DEFAULTS['MTTR'], DEFAULTS['mu_D']
    m = simulate(n, T, sim_duration=300_000.0)
    sim_frac = m.wasted_work / m.total_time
    expected = theory_wasted_fraction(T, n, M, C, mu_D, MTTR)
    assert abs(sim_frac - expected) / (expected + 1e-9) < RTOL, (
        f"n={n}, T={T}: wasted fraction {sim_frac:.4f} vs theory {expected:.4f}"
    )


@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0), (16, 2.0)])
def test_recovery_fraction_matches_theory(n, T):
    """recovery_time / total_time ≈ (1−p)·MTTR / E[epoch].

    Uses a longer run for the same reason as test_wasted_fraction_matches_theory.
    """
    C, M, MTTR, mu_D = DEFAULTS['C'], DEFAULTS['M'], DEFAULTS['MTTR'], DEFAULTS['mu_D']
    m = simulate(n, T, sim_duration=300_000.0)
    sim_frac = m.total_recovery_time / m.total_time
    expected = theory_recovery_fraction(T, n, M, C, mu_D, MTTR)
    assert abs(sim_frac - expected) / (expected + 1e-9) < RTOL, (
        f"n={n}, T={T}: recovery fraction {sim_frac:.4f} vs theory {expected:.4f}"
    )


@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0), (16, 2.0)])
def test_coordination_fraction_matches_theory(n, T):
    """coord_overhead / total_time ≈ p·μ_D / E[epoch]."""
    C, M, MTTR, mu_D = DEFAULTS['C'], DEFAULTS['M'], DEFAULTS['MTTR'], DEFAULTS['mu_D']
    m = simulate(n, T)
    sim_frac = m.total_coordination_overhead / m.total_time
    expected = theory_coord_fraction(T, n, M, C, mu_D, MTTR)
    assert abs(sim_frac - expected) / expected < RTOL, (
        f"n={n}, T={T}: coord fraction {sim_frac:.4f} vs theory {expected:.4f}"
    )


@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0), (16, 2.0)])
def test_checkpoint_fraction_matches_theory(n, T):
    """checkpoint_overhead / total_time ≈ p·C / E[epoch]."""
    C, M, MTTR, mu_D = DEFAULTS['C'], DEFAULTS['M'], DEFAULTS['MTTR'], DEFAULTS['mu_D']
    m = simulate(n, T)
    sim_frac = m.total_checkpoint_overhead / m.total_time
    expected = theory_checkpoint_fraction(T, n, M, C, mu_D, MTTR)
    assert abs(sim_frac - expected) / expected < RTOL, (
        f"n={n}, T={T}: checkpoint fraction {sim_frac:.4f} vs theory {expected:.4f}"
    )


@pytest.mark.parametrize("n,T", [(1, 10.0), (4, 5.0), (16, 2.0), (64, 1.0)])
def test_failure_rate_matches_theory(n, T):
    """
    failures / total_time ≈ (1−p) / E[epoch].

    Uses a larger sim_duration for better count statistics (Poisson noise
    scales as 1/sqrt(N_failures)).
    """
    C, M, MTTR, mu_D = DEFAULTS['C'], DEFAULTS['M'], DEFAULTS['MTTR'], DEFAULTS['mu_D']
    m = simulate(n, T, sim_duration=500_000.0)
    sim_rate = m.n_failures / m.total_time
    expected = theory_failure_rate(T, n, M, C, mu_D, MTTR)
    assert abs(sim_rate - expected) / expected < RTOL, (
        f"n={n}, T={T}: failure rate {sim_rate:.6f}/hr vs theory {expected:.6f}/hr"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Limiting / degenerate cases
# ─────────────────────────────────────────────────────────────────────────────

def test_no_failure_limit():
    """
    When MTBF → ∞, failures never occur and
    efficiency = T / (T + μ_D + C)  exactly.
    """
    T, C, mu_D = 10.0, 0.5, 0.05
    m = simulate(n=1, T=T, M=1e9, C=C, mu_D=mu_D, MTTR=1.0)
    assert m.n_failures == 0
    expected = T / (T + C + mu_D)
    assert abs(m.efficiency - expected) / expected < 1e-3, (
        f"No-failure efficiency {m.efficiency:.6f} != {expected:.6f}"
    )


def test_negligible_coordination_overhead():
    """
    With μ_D → 0, coordination overhead should be negligible and the
    efficiency should match the zero-coordination formula T·p / E[epoch].
    """
    T, n, C, M, MTTR = 10.0, 4, 0.5, 100.0, 1.0
    mu_D = 1e-6
    m = simulate(n=n, T=T, C=C, M=M, MTTR=MTTR, mu_D=mu_D)
    coord_fraction = m.total_coordination_overhead / m.total_time
    assert coord_fraction < 1e-4, (
        f"coord fraction {coord_fraction:.2e} should be near zero"
    )
    # Efficiency without coordination ≈ theory with mu_D = 0
    expected = theory_efficiency(T, n, M, C, mu_D=0.0, MTTR=MTTR)
    assert abs(m.efficiency - expected) / expected < RTOL


def test_coordination_overhead_scales_with_mean():
    """
    Doubling μ_D should approximately double the coordination time fraction
    (when failures are rare so almost all epochs are successes).
    """
    T, n, C, M, MTTR = 10.0, 1, 0.5, 1000.0, 1.0   # rare failures
    m1 = simulate(n=n, T=T, C=C, M=M, MTTR=MTTR, mu_D=0.1)
    m2 = simulate(n=n, T=T, C=C, M=M, MTTR=MTTR, mu_D=0.2)
    frac1 = m1.total_coordination_overhead / m1.total_time
    frac2 = m2.total_coordination_overhead / m2.total_time
    ratio = frac2 / frac1
    assert abs(ratio - 2.0) / 2.0 < 0.05, (
        f"Expected ratio ≈ 2.0 when doubling μ_D, got {ratio:.3f}"
    )


def test_coordination_reduces_efficiency():
    """
    Increasing μ_D adds pure overhead, so efficiency must decrease
    monotonically (with failure rate held constant).
    """
    T, n = 10.0, 4
    effs = [simulate(n=n, T=T, mu_D=mu_D).efficiency
            for mu_D in [0.01, 0.1, 0.5, 2.0]]
    for a, b in zip(effs, effs[1:]):
        assert b < a, f"Efficiency should fall with μ_D: {effs}"


def test_checkpoint_cost_reduces_efficiency():
    """Larger C → strictly lower efficiency (C is pure overhead)."""
    T, n = 10.0, 4
    effs = [simulate(n=n, T=T, C=C).efficiency for C in [0.1, 0.5, 1.0, 2.0]]
    for a, b in zip(effs, effs[1:]):
        assert b < a, f"Efficiency should fall with C: {effs}"


def test_efficiency_rises_toward_optimal_from_below():
    """
    For T well below T* (checkpoint overhead dominates), increasing T
    toward T* should raise efficiency.

    n=1, M=100, C=0.5  →  T* ≈ 9.5 h.  Test with T ∈ {0.6, 0.8, 1.0, 1.5}.
    """
    n, M, C, MTTR, mu_D = 1, 100.0, 0.5, 1.0, 0.05
    T_star = daly_optimal_interval(M, n, C)
    T_vals = [0.6, 0.8, 1.0, 1.5]
    assert all(T < T_star for T in T_vals), "All T should be below T*"
    effs = [simulate(n=n, T=T, C=C, M=M, MTTR=MTTR, mu_D=mu_D).efficiency
            for T in T_vals]
    for a, b in zip(effs, effs[1:]):
        assert b > a, f"Efficiency should rise toward T*: {list(zip(T_vals, effs))}"


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — Optimal checkpoint interval (Daly's formula)
# ─────────────────────────────────────────────────────────────────────────────

def test_daly_formula_scaling():
    """
    T* = sqrt(2·C·M/n + C²) − C  scales as 1/sqrt(n).
    Doubling n should reduce T* by a factor of ~1/sqrt(2) ≈ 0.707.
    Checked at large M/C so the C² correction is negligible.
    """
    M, C = 10_000.0, 0.5   # M/C = 20,000  →  C² term negligible
    for n_base in [1, 4, 16]:
        T1 = daly_optimal_interval(M, n_base,   C)
        T2 = daly_optimal_interval(M, n_base*2, C)
        ratio = T2 / T1
        assert abs(ratio - 1.0/np.sqrt(2)) < 0.01, (
            f"n={n_base}→{n_base*2}: T* ratio {ratio:.4f}, expected {1/np.sqrt(2):.4f}"
        )


def test_daly_formula_sqrt_2CM_limit():
    """
    When C << M_sys, the C² term in T* = sqrt(2·C·M_sys + C²) − C is
    negligible and T* ≈ sqrt(2·C·M_sys).  Equivalently T*²/(2·C·M_sys) → 1,
    with a first-order correction of −T*/M_sys.

    We verify this analytically: T*² / (2·C·M_sys) = 1 − T*/M_sys,
    and check the formula is self-consistent rather than testing convergence
    to 1 (which only holds in the limit M_sys → ∞).
    """
    M, C = 10_000.0, 0.5
    for n in [1, 4, 16]:
        T_star = daly_optimal_interval(M, n, C)
        M_sys = M / n
        ratio   = T_star ** 2 / (2 * C * M_sys)
        expected = 1.0 - T_star / M_sys   # exact first-order correction
        assert abs(ratio - expected) < 1e-9, (
            f"n={n}: T*²/(2CM) = {ratio:.6f}, expected 1−T*/M = {expected:.6f}"
        )


@pytest.mark.parametrize("n", [1, 4, 16, 64])
def test_daly_optimal_near_simulated_peak(n):
    """
    The checkpoint interval predicted by Daly's formula should achieve
    within 5 % of the peak simulated efficiency across a grid around T*.

    (Daly's formula is an approximation that omits MTTR; 5 % tolerance
    accounts for both this approximation and simulation noise.)
    """
    M, C, MTTR, mu_D = 100.0, 0.5, 1.0, 0.05
    T_star = daly_optimal_interval(M, n, C)

    # 20-point grid spanning [0.3·T*, 3·T*], excluding T ≤ C
    T_grid = np.linspace(T_star * 0.3, T_star * 3.0, 20)
    T_grid = T_grid[T_grid > C * 1.05]

    results = sweep(
        n_nodes=n, intervals=list(T_grid),
        checkpoint_cost=C, mtbf=M, mttr=MTTR,
        mean_coordination=mu_D, sim_duration=100_000.0,
        n_reps=5,
        failure_dist='exponential', recovery_dist='exponential',
        coordination_dist='exponential',
        base_seed=SEED,
        metric_names=['efficiency'],
    )

    peak_eta    = max(r['efficiency_mean'] for r in results)
    eta_at_Tstar = simulate(n=n, T=T_star, M=M, C=C, MTTR=MTTR,
                            mu_D=mu_D, sim_duration=200_000.0).efficiency

    assert eta_at_Tstar >= peak_eta * 0.95, (
        f"n={n}: η(T*={T_star:.2f}h)={eta_at_Tstar:.4f} "
        f"is more than 5 % below grid peak η={peak_eta:.4f}"
    )
