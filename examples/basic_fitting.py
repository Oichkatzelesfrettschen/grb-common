#!/usr/bin/env python3
"""
Basic MCMC fitting example using grb-common.

This example demonstrates:
1. Setting up priors for GRB afterglow parameters
2. Defining a likelihood function
3. Running MCMC sampling
4. Analyzing and visualizing results
"""

import numpy as np

from grb_common.fitting import (
    LogUniformPrior,
    UniformPrior,
    CompositePrior,
    gaussian_likelihood,
    SamplerResult,
)
from grb_common.fitting.backends import get_sampler, available_backends
from grb_common.cosmology import luminosity_distance
from grb_common.constants import C_LIGHT, M_ELECTRON, SIGMA_T


def main():
    # Check available backends
    print(f"Available sampling backends: {available_backends()}")

    # --- Define priors ---
    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e55),    # Isotropic energy [erg]
        'n': LogUniformPrior(1e-5, 1.0),          # Circumburst density [cm^-3]
        'epsilon_e': LogUniformPrior(1e-3, 0.5), # Electron energy fraction
        'epsilon_B': LogUniformPrior(1e-6, 0.1), # Magnetic energy fraction
        'p': UniformPrior(2.01, 3.0),             # Electron power-law index
    })

    print(f"Number of parameters: {priors.n_params}")
    print(f"Parameter names: {priors.param_names}")

    # --- Synthetic data ---
    # In a real application, you would load observational data
    np.random.seed(42)
    n_points = 20
    time_obs = np.logspace(3, 6, n_points)  # 10^3 to 10^6 seconds

    # True parameters (for testing)
    true_params = {
        'E_iso': 1e52,
        'n': 1e-2,
        'epsilon_e': 0.1,
        'epsilon_B': 1e-3,
        'p': 2.3,
    }

    # Simple power-law model for demonstration
    def model_flux(theta, time):
        """Simplified afterglow flux model."""
        E_iso, n, eps_e, eps_B, p = theta
        # Simplified scaling (real models are much more complex)
        F_peak = 1e-26 * (E_iso / 1e52)**0.5 * (n / 1e-2)**0.5
        t_peak = 1e4  # seconds
        alpha = -(3*p - 2) / 4  # Late-time decay index
        flux = F_peak * (time / t_peak)**alpha
        return flux

    # Generate synthetic observations
    true_theta = [true_params[p] for p in priors.param_names]
    flux_model = model_flux(true_theta, time_obs)
    flux_err = 0.2 * flux_model  # 20% errors
    flux_obs = flux_model + np.random.randn(n_points) * flux_err

    # --- Define likelihood ---
    def log_likelihood(theta):
        """Gaussian log-likelihood."""
        model = model_flux(theta, time_obs)
        if np.any(model <= 0):
            return -np.inf
        return gaussian_likelihood(flux_obs, model, flux_err)

    def log_posterior(theta):
        """Log posterior = log prior + log likelihood."""
        lp = priors.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta)

    # --- Run sampler ---
    print("\nRunning MCMC sampling...")

    # Check if emcee is available
    if 'emcee' not in available_backends():
        print("emcee not installed. Install with: pip install emcee")
        return

    sampler = get_sampler(
        'emcee',
        log_prob=log_posterior,
        n_params=priors.n_params,
        param_names=priors.param_names,
    )

    # Run with moderate settings for demonstration
    result = sampler.run(
        n_walkers=32,
        n_steps=1000,
        progress=True,
    )

    # --- Analyze results ---
    print("\n--- Results ---")
    for param in priors.param_names:
        p16, p50, p84 = result.percentile(param, [16, 50, 84])
        true_val = true_params[param]
        print(f"{param}: {p50:.2e} (+{p84-p50:.2e} / -{p50-p16:.2e}) [true: {true_val:.2e}]")

    # --- Save results ---
    try:
        result.save('example_chains.h5')
        print("\nResults saved to example_chains.h5")
    except ImportError:
        print("\nh5py not installed, skipping save")

    # --- Plot results ---
    try:
        from grb_common.plotting import corner_plot, set_style
        import matplotlib.pyplot as plt

        set_style('default')

        fig = corner_plot(
            result,
            truths=true_params,
            quantiles=[0.16, 0.5, 0.84],
        )
        fig.savefig('example_corner.pdf', bbox_inches='tight')
        print("Corner plot saved to example_corner.pdf")
        plt.close()

    except ImportError as e:
        print(f"\nPlotting libraries not available: {e}")


if __name__ == '__main__':
    main()
