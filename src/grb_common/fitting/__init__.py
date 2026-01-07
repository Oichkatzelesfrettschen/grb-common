"""
MCMC and nested sampling infrastructure for GRB model fitting.

This module provides a unified interface to multiple sampling backends
for Bayesian parameter estimation in GRB afterglow modeling.

Priors:
    UniformPrior: Uniform distribution
    LogUniformPrior: Log-uniform (Jeffreys) prior
    GaussianPrior: Gaussian distribution
    TruncatedGaussianPrior: Bounded Gaussian
    DeltaPrior: Fixed value
    CompositePrior: Collection of priors

Likelihoods:
    chi_squared: Standard chi-squared
    gaussian_likelihood: Gaussian log-likelihood
    gaussian_likelihood_upper_limits: With upper limit handling
    poisson_likelihood: For count data
    cstat: Cash statistic for X-ray

Samplers:
    EmceeSampler: Affine-invariant MCMC (emcee)
    DynestySampler: Dynamic nested sampling (dynesty)
    MultiNestSampler: MultiNest via pymultinest

Results:
    SamplerResult: Unified container for sampler output

Usage:
    from grb_common.fitting import (
        LogUniformPrior, UniformPrior, CompositePrior,
        gaussian_likelihood, SamplerResult
    )
    from grb_common.fitting.backends import get_sampler

    # Define priors
    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e54),
        'n': LogUniformPrior(1e-5, 1.0),
        'epsilon_e': UniformPrior(0.01, 0.5),
        'p': UniformPrior(2.0, 3.0),
    })

    # Define likelihood
    def log_prob(theta):
        lp = priors.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + gaussian_likelihood(observed, model(theta), error)

    # Run sampler
    sampler = get_sampler('emcee', log_prob=log_prob, n_params=4)
    result = sampler.run(n_steps=5000)
    result.corner_plot()
"""

from .likelihood import (
    chi_squared,
    chi_squared_upper_limits,
    chi_squared_with_systematics,
    cstat,
    gaussian_likelihood,
    gaussian_likelihood_asymmetric,
    gaussian_likelihood_upper_limits,
    poisson_likelihood,
    reduced_chi_squared,
)
from .priors import (
    CompositePrior,
    DeltaPrior,
    GaussianPrior,
    LogUniformPrior,
    Prior,
    TruncatedGaussianPrior,
    UniformPrior,
)
from .result import (
    SamplerResult,
    weighted_percentile,
)

__all__ = [
    # Priors
    "Prior",
    "UniformPrior",
    "LogUniformPrior",
    "GaussianPrior",
    "TruncatedGaussianPrior",
    "DeltaPrior",
    "CompositePrior",
    # Likelihoods
    "chi_squared",
    "reduced_chi_squared",
    "gaussian_likelihood",
    "gaussian_likelihood_asymmetric",
    "chi_squared_upper_limits",
    "gaussian_likelihood_upper_limits",
    "chi_squared_with_systematics",
    "poisson_likelihood",
    "cstat",
    # Results
    "SamplerResult",
    "weighted_percentile",
]
