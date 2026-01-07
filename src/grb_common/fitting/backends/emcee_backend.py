"""
emcee sampler backend.

Provides an affine-invariant ensemble MCMC sampler using the emcee package.

Usage:
    from grb_common.fitting.backends import EmceeSampler

    def log_prob(theta):
        return log_prior(theta) + log_likelihood(theta)

    sampler = EmceeSampler(
        log_prob=log_prob,
        n_params=5,
        n_walkers=32,
    )
    result = sampler.run(n_steps=5000, n_burn=1000)
"""

import time
from typing import Any, Callable, List, Optional, cast

import numpy as np

from ..result import SamplerResult


class EmceeSampler:
    """
    Affine-invariant ensemble MCMC sampler.

    Wrapper around emcee.EnsembleSampler with progress tracking
    and result packaging.

    Parameters
    ----------
    log_prob : callable
        Log posterior function. Signature: log_prob(theta) -> float.
    n_params : int
        Number of parameters.
    n_walkers : int, optional
        Number of walkers (default: 2 * n_params).
    param_names : list of str, optional
        Names of parameters.
    moves : optional
        emcee moves to use.
    pool : optional
        Multiprocessing pool for parallelization.
    """

    def __init__(
        self,
        log_prob: Callable,
        n_params: int,
        n_walkers: Optional[int] = None,
        param_names: Optional[List[str]] = None,
        moves: Optional[Any] = None,
        pool: Optional[Any] = None,
    ):
        import emcee

        self.log_prob = log_prob
        self.n_params = n_params
        self.n_walkers = n_walkers or 2 * n_params

        if self.n_walkers < 2 * n_params:
            raise ValueError(
                f"n_walkers ({self.n_walkers}) must be >= 2 * n_params ({2 * n_params})"
            )

        self.param_names = param_names or [f"p{i}" for i in range(n_params)]
        self.pool = pool

        self.sampler = emcee.EnsembleSampler(
            self.n_walkers,
            self.n_params,
            self.log_prob,
            moves=moves,
            pool=pool,
        )

    def run(
        self,
        n_steps: int,
        initial_state: Optional[np.ndarray] = None,
        n_burn: int = 0,
        thin: int = 1,
        progress: bool = True,
        **kwargs,
    ) -> SamplerResult:
        """
        Run the sampler.

        Parameters
        ----------
        n_steps : int
            Number of steps per walker.
        initial_state : ndarray, optional
            Initial walker positions, shape (n_walkers, n_params).
            If None, samples from a small ball around origin.
        n_burn : int
            Number of burn-in steps to discard.
        thin : int
            Thinning factor.
        progress : bool
            Show progress bar.
        **kwargs
            Additional arguments passed to sampler.run_mcmc().

        Returns
        -------
        SamplerResult
            Sampler output container.
        """
        # Initialize walkers if not provided
        if initial_state is None:
            initial_state = np.random.randn(self.n_walkers, self.n_params) * 0.1

        start_time = time.time()

        # Run sampler
        self.sampler.run_mcmc(
            initial_state,
            n_steps,
            progress=progress,
            **kwargs,
        )

        runtime = time.time() - start_time

        # Extract chains
        samples = cast(np.ndarray, self.sampler.get_chain(discard=n_burn, thin=thin, flat=True))
        log_prob = cast(np.ndarray, self.sampler.get_log_prob(discard=n_burn, thin=thin, flat=True))

        # Package result
        return SamplerResult(
            samples=samples,
            log_likelihood=log_prob,  # Note: this is log_posterior, not just likelihood
            param_names=self.param_names,
            metadata={
                "sampler": "emcee",
                "n_walkers": self.n_walkers,
                "n_steps": n_steps,
                "n_burn": n_burn,
                "thin": thin,
                "runtime_seconds": runtime,
                "acceptance_fraction": np.mean(self.sampler.acceptance_fraction),
            },
        )

    def reset(self) -> None:
        """Reset the sampler state."""
        self.sampler.reset()

    @property
    def chain(self) -> np.ndarray:
        """Get the chain (n_steps, n_walkers, n_params)."""
        return cast(np.ndarray, self.sampler.get_chain())

    @property
    def acceptance_fraction(self) -> np.ndarray:
        """Get acceptance fraction for each walker."""
        return cast(np.ndarray, self.sampler.acceptance_fraction)

    def get_autocorr_time(self, **kwargs) -> np.ndarray:
        """
        Estimate integrated autocorrelation time.

        Returns
        -------
        ndarray
            Autocorrelation time for each parameter.
        """
        try:
            return cast(np.ndarray, self.sampler.get_autocorr_time(**kwargs))
        except Exception as e:
            # emcee raises errors if chain too short
            import warnings
            warnings.warn(f"Could not compute autocorr time: {e}")
            return cast(np.ndarray, np.full(self.n_params, np.nan))


__all__ = ["EmceeSampler"]
