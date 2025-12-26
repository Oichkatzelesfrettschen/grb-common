#!/usr/bin/env python3
"""
Benchmarks for fitting infrastructure.

Run with:
    python benchmarks/bench_fitting.py
"""

import time
import numpy as np


def bench_prior_evaluation():
    """Benchmark prior probability evaluation."""
    from grb_common.fitting import (
        UniformPrior, LogUniformPrior, CompositePrior
    )

    # Single prior
    prior = LogUniformPrior(1e50, 1e55)
    n_iterations = 100000
    values = np.random.uniform(1e50, 1e55, n_iterations)

    start = time.perf_counter()
    for v in values:
        lp = prior.log_prob(v)
    elapsed = time.perf_counter() - start

    print(f"LogUniformPrior.log_prob (single):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")

    # Composite prior
    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e55),
        'n': LogUniformPrior(1e-5, 1.0),
        'epsilon_e': LogUniformPrior(1e-3, 0.5),
        'epsilon_B': LogUniformPrior(1e-6, 0.1),
        'p': UniformPrior(2.01, 3.0),
    })

    n_iterations = 10000
    theta = np.array([1e52, 1e-2, 0.1, 1e-3, 2.3])

    start = time.perf_counter()
    for _ in range(n_iterations):
        lp = priors.log_prob(theta)
    elapsed = time.perf_counter() - start

    print(f"\nCompositePrior.log_prob (5 params):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")


def bench_prior_transform():
    """Benchmark prior transform (for nested sampling)."""
    from grb_common.fitting import (
        UniformPrior, LogUniformPrior, CompositePrior
    )

    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e55),
        'n': LogUniformPrior(1e-5, 1.0),
        'epsilon_e': LogUniformPrior(1e-3, 0.5),
        'epsilon_B': LogUniformPrior(1e-6, 0.1),
        'p': UniformPrior(2.01, 3.0),
    })

    n_iterations = 10000
    u = np.random.uniform(0, 1, 5)

    start = time.perf_counter()
    for _ in range(n_iterations):
        theta = priors.prior_transform(u)
    elapsed = time.perf_counter() - start

    print(f"\nCompositePrior.prior_transform (5 params):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")


def bench_likelihood():
    """Benchmark likelihood functions."""
    from grb_common.fitting import gaussian_likelihood, chi_squared

    n_points = 100
    observed = np.random.randn(n_points) + 10
    model = observed + np.random.randn(n_points) * 0.1
    errors = np.ones(n_points) * 0.5

    n_iterations = 10000

    # Gaussian likelihood
    start = time.perf_counter()
    for _ in range(n_iterations):
        ll = gaussian_likelihood(observed, model, errors)
    elapsed = time.perf_counter() - start

    print(f"\ngaussian_likelihood ({n_points} points):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")

    # Chi-squared
    start = time.perf_counter()
    for _ in range(n_iterations):
        chi2 = chi_squared(observed, model, errors)
    elapsed = time.perf_counter() - start

    print(f"\nchi_squared ({n_points} points):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")


def bench_prior_sampling():
    """Benchmark prior sampling."""
    from grb_common.fitting import LogUniformPrior, CompositePrior, UniformPrior

    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e55),
        'n': LogUniformPrior(1e-5, 1.0),
        'epsilon_e': LogUniformPrior(1e-3, 0.5),
        'epsilon_B': LogUniformPrior(1e-6, 0.1),
        'p': UniformPrior(2.01, 3.0),
    })

    n_samples = 10000

    start = time.perf_counter()
    samples = priors.sample(n_samples)
    elapsed = time.perf_counter() - start

    print(f"\nCompositePrior.sample ({n_samples} samples, 5 params):")
    print(f"  {elapsed*1e3:.2f} ms total")
    print(f"  {n_samples/elapsed:.0f} samples/s")


def main():
    print("=" * 60)
    print("Fitting Benchmarks")
    print("=" * 60)
    print()

    bench_prior_evaluation()
    bench_prior_transform()
    bench_likelihood()
    bench_prior_sampling()


if __name__ == '__main__':
    main()
