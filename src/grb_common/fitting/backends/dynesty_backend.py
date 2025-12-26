"""
dynesty sampler backend.

Provides dynamic nested sampling using the dynesty package.

Nested sampling is particularly useful for:
- Multi-modal posteriors
- Evidence (marginal likelihood) calculation
- Efficient exploration of complex parameter spaces

Usage:
    from grb_common.fitting.backends import DynestySampler

    def log_likelihood(theta):
        return compute_log_likelihood(theta)

    def prior_transform(u):
        # Transform uniform [0,1] to prior
        return transform(u)

    sampler = DynestySampler(
        log_likelihood=log_likelihood,
        prior_transform=prior_transform,
        n_params=5,
    )
    result = sampler.run()
"""

from typing import Callable, Optional, List, Literal
import time
import numpy as np

from ..result import SamplerResult


class DynestySampler:
    """
    Dynamic nested sampler.

    Wrapper around dynesty.DynamicNestedSampler (or NestedSampler)
    with result packaging.

    Parameters
    ----------
    log_likelihood : callable
        Log likelihood function. Signature: log_likelihood(theta) -> float.
    prior_transform : callable
        Prior transform function. Signature: prior_transform(u) -> theta.
        Maps uniform [0,1]^n to parameter space.
    n_params : int
        Number of parameters.
    param_names : list of str, optional
        Names of parameters.
    dynamic : bool
        Use dynamic nested sampling (default: True).
    bound : str
        Bounding method ('multi', 'single', 'balls', 'cubes').
    sample : str
        Sampling method ('auto', 'unif', 'rwalk', 'slice', 'rslice').
    nlive : int
        Number of live points.
    pool : optional
        Multiprocessing pool.
    """

    def __init__(
        self,
        log_likelihood: Callable,
        prior_transform: Callable,
        n_params: int,
        param_names: Optional[List[str]] = None,
        dynamic: bool = True,
        bound: str = "multi",
        sample: str = "auto",
        nlive: int = 500,
        pool: Optional = None,
        **kwargs,
    ):
        import dynesty

        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.n_params = n_params
        self.param_names = param_names or [f"p{i}" for i in range(n_params)]
        self.dynamic = dynamic

        # Common kwargs
        sampler_kwargs = {
            "loglikelihood": log_likelihood,
            "prior_transform": prior_transform,
            "ndim": n_params,
            "bound": bound,
            "sample": sample,
            "pool": pool,
        }
        sampler_kwargs.update(kwargs)

        if dynamic:
            self.sampler = dynesty.DynamicNestedSampler(
                nlive_init=nlive,
                **sampler_kwargs,
            )
        else:
            self.sampler = dynesty.NestedSampler(
                nlive=nlive,
                **sampler_kwargs,
            )

        self.nlive = nlive

    def run(
        self,
        dlogz: float = 0.01,
        maxiter: Optional[int] = None,
        maxcall: Optional[int] = None,
        print_progress: bool = True,
        **kwargs,
    ) -> SamplerResult:
        """
        Run the sampler.

        Parameters
        ----------
        dlogz : float
            Stopping criterion: delta log(evidence).
        maxiter : int, optional
            Maximum number of iterations.
        maxcall : int, optional
            Maximum number of likelihood calls.
        print_progress : bool
            Print progress updates.
        **kwargs
            Additional arguments passed to run_nested().

        Returns
        -------
        SamplerResult
            Sampler output container.
        """
        start_time = time.time()

        run_kwargs = {
            "dlogz_init": dlogz if self.dynamic else None,
            "print_progress": print_progress,
        }
        if not self.dynamic:
            run_kwargs["dlogz"] = dlogz
        if maxiter is not None:
            run_kwargs["maxiter"] = maxiter
        if maxcall is not None:
            run_kwargs["maxcall"] = maxcall

        run_kwargs.update(kwargs)

        self.sampler.run_nested(**run_kwargs)

        runtime = time.time() - start_time

        # Extract results
        results = self.sampler.results

        # Get weighted samples
        samples = results.samples
        weights = np.exp(results.logwt - results.logwt.max())
        weights /= weights.sum()

        log_likelihood = results.logl

        # Evidence and uncertainty
        log_evidence = results.logz[-1]
        log_evidence_err = results.logzerr[-1]

        return SamplerResult(
            samples=samples,
            log_likelihood=log_likelihood,
            param_names=self.param_names,
            weights=weights,
            metadata={
                "sampler": "dynesty",
                "dynamic": self.dynamic,
                "nlive": self.nlive,
                "log_evidence": log_evidence,
                "log_evidence_err": log_evidence_err,
                "n_likelihood_calls": results.ncall.sum(),
                "n_iterations": len(results.logz),
                "runtime_seconds": runtime,
            },
        )

    def add_batch(self, **kwargs) -> None:
        """
        Add additional samples for dynamic nested sampling.

        Only valid for dynamic samplers.
        """
        if not self.dynamic:
            raise ValueError("add_batch only available for dynamic samplers")
        self.sampler.add_batch(**kwargs)

    @property
    def results(self):
        """Get the raw dynesty results object."""
        return self.sampler.results


__all__ = ["DynestySampler"]
