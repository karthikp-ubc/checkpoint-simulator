"""
Pluggable failure/recovery time distributions.

All distributions are parameterized by their *mean* (MTBF or MTTR), so
students can swap distributions without changing units elsewhere.
"""
import numpy as np
from scipy.special import gamma as _gamma
from typing import Callable


def make_sampler(
    distribution: str = 'exponential',
    mean: float = 1.0,
    seed: int = None,
    **kwargs,
) -> Callable[[], float]:
    """
    Return a callable () -> float that draws samples from the given distribution.

    Parameters
    ----------
    distribution : 'exponential' | 'weibull' | 'lognormal'
    mean         : desired mean of the distribution (e.g. MTBF in hours)
    seed         : optional RNG seed for reproducibility
    **kwargs
        shape    : Weibull shape parameter k (default 0.7, typical for HPC nodes;
                   k < 1 → decreasing hazard, k = 1 → exponential, k > 1 → increasing)
        sigma    : lognormal log-space std dev (default 0.5)
    """
    rng = np.random.default_rng(seed)

    if distribution == 'exponential':
        return lambda: rng.exponential(mean)

    elif distribution == 'weibull':
        k = kwargs.get('shape', 0.7)
        # Scale λ so that E[X] = λ * Γ(1 + 1/k) = mean
        scale = mean / _gamma(1.0 + 1.0 / k)
        return lambda: scale * rng.weibull(k)

    elif distribution == 'lognormal':
        sigma = kwargs.get('sigma', 0.5)
        mu = np.log(mean) - sigma ** 2 / 2.0
        return lambda: rng.lognormal(mu, sigma)

    else:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            "Choose from: 'exponential', 'weibull', 'lognormal'."
        )
