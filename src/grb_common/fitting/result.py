"""
Sampler result container for Bayesian parameter estimation.

This module provides a unified container for storing and analyzing
results from MCMC and nested sampling runs.

Usage:
    result = sampler.run(n_samples=10000)

    # Statistics
    print(result.median('E_iso'))
    print(result.percentile('n', [16, 50, 84]))

    # Visualization
    result.corner_plot()
    result.trace_plot()

    # Persistence
    result.save('chains.h5')
    result = SamplerResult.load('chains.h5')
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, overload

import numpy as np


@dataclass
class SamplerResult:
    """
    Container for sampler output.

    Attributes
    ----------
    samples : ndarray
        Parameter samples, shape (n_samples, n_params).
    log_likelihood : ndarray
        Log-likelihood values for each sample.
    log_prior : ndarray, optional
        Log-prior values for each sample.
    param_names : list of str
        Names of parameters.
    metadata : dict
        Additional information (sampler name, runtime, etc.).
    weights : ndarray, optional
        Sample weights (for nested sampling).
    """
    samples: np.ndarray
    log_likelihood: np.ndarray
    param_names: List[str]
    log_prior: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate shapes."""
        n_samples, n_params = self.samples.shape
        if len(self.log_likelihood) != n_samples:
            raise ValueError(
                f"log_likelihood length {len(self.log_likelihood)} != "
                f"samples length {n_samples}"
            )
        if len(self.param_names) != n_params:
            raise ValueError(
                f"param_names length {len(self.param_names)} != "
                f"n_params {n_params}"
            )

    @property
    def n_samples(self) -> int:
        """Number of samples."""
        return int(self.samples.shape[0])

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return int(self.samples.shape[1])

    def _get_param_index(self, param: Union[str, int]) -> int:
        """Get parameter index from name or index."""
        if isinstance(param, int):
            return param
        return self.param_names.index(param)

    def _get_param_samples(self, param: Union[str, int]) -> np.ndarray:
        """Get samples for a specific parameter."""
        idx = self._get_param_index(param)
        return self.samples[:, idx]

    @overload
    def percentile(self, param: Union[str, int], q: float) -> float: ...

    @overload
    def percentile(self, param: Union[str, int], q: Sequence[float]) -> np.ndarray: ...

    def percentile(
        self,
        param: Union[str, int],
        q: Union[float, Sequence[float]],
    ) -> Union[float, np.ndarray]:
        """
        Compute percentile(s) for a parameter.

        Parameters
        ----------
        param : str or int
            Parameter name or index.
        q : float or list of float
            Percentile(s) to compute (0-100).

        Returns
        -------
        float or ndarray
            Percentile value(s).
        """
        samples = self._get_param_samples(param)
        if isinstance(q, (int, float)):
            q_scalar = float(q)
            if self.weights is not None:
                return float(weighted_percentile(samples, q_scalar, self.weights))
            return float(np.percentile(samples, q_scalar))

        q_list = list(q)
        if self.weights is not None:
            return cast(np.ndarray, weighted_percentile(samples, q_list, self.weights))
        return cast(np.ndarray, np.percentile(samples, q_list))

    def median(self, param: Union[str, int]) -> float:
        """Compute median for a parameter."""
        return float(self.percentile(param, 50))

    def mean(self, param: Union[str, int]) -> float:
        """Compute mean for a parameter."""
        samples = self._get_param_samples(param)
        if self.weights is not None:
            return float(np.average(samples, weights=self.weights))
        return float(np.mean(samples))

    def std(self, param: Union[str, int]) -> float:
        """Compute standard deviation for a parameter."""
        samples = self._get_param_samples(param)
        if self.weights is not None:
            mean = np.average(samples, weights=self.weights)
            variance = np.average((samples - mean)**2, weights=self.weights)
            return float(np.sqrt(variance))
        return float(np.std(samples))

    def credible_interval(
        self,
        param: Union[str, int],
        level: float = 0.68,
    ) -> Tuple[float, float]:
        """
        Compute credible interval for a parameter.

        Parameters
        ----------
        param : str or int
            Parameter name or index.
        level : float
            Credible level (default 0.68 for 1-sigma).

        Returns
        -------
        tuple of float
            Lower and upper bounds.
        """
        alpha = (1 - level) / 2
        q_lo = alpha * 100
        q_hi = (1 - alpha) * 100
        lo, hi = cast(np.ndarray, self.percentile(param, [q_lo, q_hi]))
        return (float(lo), float(hi))

    def summary(self) -> str:
        """Generate summary string for all parameters."""
        lines = [f"Sampler Result ({self.n_samples} samples, {self.n_params} params)"]
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<20} {'Median':>12} {'Mean':>12} {'Std':>12}")
        lines.append("-" * 60)

        for name in self.param_names:
            median = self.median(name)
            mean = self.mean(name)
            std = self.std(name)
            lines.append(f"{name:<20} {median:>12.4e} {mean:>12.4e} {std:>12.4e}")

        return "\n".join(lines)

    def corner_plot(
        self,
        params: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Generate corner plot.

        Parameters
        ----------
        params : list of str, optional
            Parameters to include (default: all).
        **kwargs
            Additional arguments passed to corner.corner().

        Returns
        -------
        matplotlib.figure.Figure
            Corner plot figure.
        """
        try:
            import corner
        except ImportError:
            raise ImportError(
                "corner is required for corner plots. "
                "Install with: pip install grb-common[plotting]"
            )

        if params is None:
            params = self.param_names
            data = self.samples
        else:
            indices = [self._get_param_index(p) for p in params]
            data = self.samples[:, indices]

        default_kwargs = {
            "labels": params,
            "quantiles": [0.16, 0.5, 0.84],
            "show_titles": True,
            "title_fmt": ".3e",
        }
        default_kwargs.update(kwargs)

        if self.weights is not None:
            default_kwargs["weights"] = self.weights

        return corner.corner(data, **default_kwargs)

    def trace_plot(
        self,
        params: Optional[List[str]] = None,
    ):
        """
        Generate trace plot for MCMC diagnostics.

        Parameters
        ----------
        params : list of str, optional
            Parameters to include (default: all).

        Returns
        -------
        matplotlib.figure.Figure
            Trace plot figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for trace plots. "
                "Install with: pip install grb-common[plotting]"
            )

        if params is None:
            params = self.param_names

        n_params = len(params)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2 * n_params), sharex=True)

        if n_params == 1:
            axes = [axes]

        for ax, param in zip(axes, params):
            samples = self._get_param_samples(param)
            ax.plot(samples, alpha=0.5, lw=0.5)
            ax.set_ylabel(param)

        axes[-1].set_xlabel("Sample")
        fig.tight_layout()

        return fig

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save result to HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Output file path.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for saving results")

        filepath = Path(filepath)

        with h5py.File(filepath, "w") as f:
            f.create_dataset("samples", data=self.samples, compression="gzip")
            f.create_dataset("log_likelihood", data=self.log_likelihood, compression="gzip")

            if self.log_prior is not None:
                f.create_dataset("log_prior", data=self.log_prior, compression="gzip")
            if self.weights is not None:
                f.create_dataset("weights", data=self.weights, compression="gzip")

            # Store param names as string array
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("param_names", data=self.param_names, dtype=dt)

            # Store metadata as attributes
            for key, val in self.metadata.items():
                if isinstance(val, (str, int, float, bool)):
                    f.attrs[key] = val

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SamplerResult":
        """
        Load result from HDF5 file.

        Parameters
        ----------
        filepath : str or Path
            Input file path.

        Returns
        -------
        SamplerResult
            Loaded result object.
        """
        try:
            import h5py
        except ImportError:
            raise ImportError("h5py required for loading results")

        filepath = Path(filepath)

        with h5py.File(filepath, "r") as f:
            samples = f["samples"][:]
            log_likelihood = f["log_likelihood"][:]
            param_names = [s.decode() if isinstance(s, bytes) else s
                          for s in f["param_names"][:]]

            log_prior = f["log_prior"][:] if "log_prior" in f else None
            weights = f["weights"][:] if "weights" in f else None

            metadata = dict(f.attrs)

        return cls(
            samples=samples,
            log_likelihood=log_likelihood,
            param_names=param_names,
            log_prior=log_prior,
            weights=weights,
            metadata=metadata,
        )


def weighted_percentile(
    data: np.ndarray,
    percentiles: Union[float, Sequence[float]],
    weights: np.ndarray,
) -> Union[float, np.ndarray]:
    """
    Compute weighted percentiles.

    Parameters
    ----------
    data : ndarray
        Data values.
    percentiles : float or list of float
        Percentile(s) to compute (0-100).
    weights : ndarray
        Sample weights.

    Returns
    -------
    float or ndarray
        Percentile value(s).
    """
    percentiles_arr = cast(np.ndarray, np.atleast_1d(percentiles)) / 100.0

    # Sort data and weights
    sort_idx = np.argsort(data)
    sorted_data = data[sort_idx]
    sorted_weights = weights[sort_idx]

    # Compute cumulative weights
    cumsum = np.cumsum(sorted_weights)
    cumsum /= cumsum[-1]

    # Interpolate to find percentiles
    result = cast(np.ndarray, np.interp(percentiles_arr, cumsum, sorted_data))

    return float(result[0]) if int(result.size) == 1 else result


__all__ = [
    "SamplerResult",
    "weighted_percentile",
]
