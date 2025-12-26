"""
MCMC and nested sampling infrastructure for GRB model fitting.

This module provides a unified interface to multiple sampling backends
for Bayesian parameter estimation in GRB afterglow modeling.

Supported samplers:
    - emcee: Affine-invariant ensemble MCMC
    - dynesty: Dynamic nested sampling
    - pymultinest: MultiNest via Python bindings

Key classes:
    GRBFitter: Main fitting interface
    Prior: Base class for parameter priors
    UniformPrior: Uniform distribution prior
    LogUniformPrior: Log-uniform (Jeffreys) prior
    GaussianPrior: Gaussian distribution prior
    FitResult: Container for sampling results

Key functions:
    gaussian_likelihood: Standard chi-square likelihood
    compute_evidence: Model evidence from nested sampling
"""

__all__ = [
    "GRBFitter",
    "Prior",
    "UniformPrior",
    "LogUniformPrior",
    "GaussianPrior",
    "FitResult",
    "gaussian_likelihood",
    "compute_evidence",
]

# Implementations will be added in Phase 2.6
