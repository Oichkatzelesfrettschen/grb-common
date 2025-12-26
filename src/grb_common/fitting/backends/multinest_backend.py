"""
PyMultiNest sampler backend.

Provides MultiNest nested sampling via pymultinest bindings.

Note: Requires MultiNest library to be installed separately.

Usage:
    from grb_common.fitting.backends import MultiNestSampler

    sampler = MultiNestSampler(
        log_likelihood=log_likelihood,
        prior_transform=prior_transform,
        n_params=5,
        output_dir='chains/',
    )
    result = sampler.run()
"""

from typing import Callable, Optional, List
from pathlib import Path
import time
import numpy as np

from ..result import SamplerResult


class MultiNestSampler:
    """
    MultiNest nested sampler.

    Wrapper around pymultinest.run with result packaging.

    Parameters
    ----------
    log_likelihood : callable
        Log likelihood function.
    prior_transform : callable
        Prior transform function.
    n_params : int
        Number of parameters.
    param_names : list of str, optional
        Names of parameters.
    output_dir : str or Path
        Directory for MultiNest output files.
    basename : str
        Prefix for output files.
    nlive : int
        Number of live points.
    **kwargs
        Additional arguments passed to pymultinest.run().
    """

    def __init__(
        self,
        log_likelihood: Callable,
        prior_transform: Callable,
        n_params: int,
        param_names: Optional[List[str]] = None,
        output_dir: str = "chains",
        basename: str = "grb_",
        nlive: int = 400,
        **kwargs,
    ):
        self.log_likelihood = log_likelihood
        self.prior_transform = prior_transform
        self.n_params = n_params
        self.param_names = param_names or [f"p{i}" for i in range(n_params)]
        self.output_dir = Path(output_dir)
        self.basename = basename
        self.nlive = nlive
        self.extra_kwargs = kwargs

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _pymn_prior(self, cube, ndim, nparams):
        """PyMultiNest prior wrapper."""
        theta = self.prior_transform(np.array([cube[i] for i in range(ndim)]))
        for i in range(ndim):
            cube[i] = theta[i]

    def _pymn_loglike(self, cube, ndim, nparams):
        """PyMultiNest likelihood wrapper."""
        theta = np.array([cube[i] for i in range(ndim)])
        return self.log_likelihood(theta)

    def run(
        self,
        evidence_tolerance: float = 0.5,
        sampling_efficiency: float = 0.8,
        importance_nested_sampling: bool = True,
        const_efficiency_mode: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> SamplerResult:
        """
        Run the sampler.

        Parameters
        ----------
        evidence_tolerance : float
            Evidence tolerance for convergence.
        sampling_efficiency : float
            Sampling efficiency parameter.
        importance_nested_sampling : bool
            Use importance nested sampling.
        const_efficiency_mode : bool
            Use constant efficiency mode.
        verbose : bool
            Print progress.
        **kwargs
            Additional arguments passed to pymultinest.run().

        Returns
        -------
        SamplerResult
            Sampler output container.
        """
        import pymultinest

        start_time = time.time()

        # Merge kwargs
        run_kwargs = {
            "n_live_points": self.nlive,
            "evidence_tolerance": evidence_tolerance,
            "sampling_efficiency": sampling_efficiency,
            "importance_nested_sampling": importance_nested_sampling,
            "const_efficiency_mode": const_efficiency_mode,
            "verbose": verbose,
            "resume": False,
            "outputfiles_basename": str(self.output_dir / self.basename),
        }
        run_kwargs.update(self.extra_kwargs)
        run_kwargs.update(kwargs)

        pymultinest.run(
            LogLikelihood=self._pymn_loglike,
            Prior=self._pymn_prior,
            n_dims=self.n_params,
            **run_kwargs,
        )

        runtime = time.time() - start_time

        # Load results
        analyzer = pymultinest.Analyzer(
            n_params=self.n_params,
            outputfiles_basename=str(self.output_dir / self.basename),
        )

        data = analyzer.get_data()
        samples = data[:, 2:]  # Skip weight and likelihood columns
        weights = data[:, 0]
        log_likelihood = data[:, 1]

        # Normalize weights
        weights = weights / weights.sum()

        # Get evidence
        stats = analyzer.get_stats()
        log_evidence = stats["nested sampling global log-evidence"]
        log_evidence_err = stats["nested sampling global log-evidence error"]

        return SamplerResult(
            samples=samples,
            log_likelihood=log_likelihood,
            param_names=self.param_names,
            weights=weights,
            metadata={
                "sampler": "pymultinest",
                "nlive": self.nlive,
                "log_evidence": log_evidence,
                "log_evidence_err": log_evidence_err,
                "output_dir": str(self.output_dir),
                "runtime_seconds": runtime,
            },
        )


__all__ = ["MultiNestSampler"]
