"""
Prior distributions for Bayesian parameter estimation.

This module provides prior distribution classes for use with MCMC and
nested sampling algorithms in GRB afterglow fitting.

Each prior class provides:
    - sample(): Draw random samples from the prior
    - log_prob(): Compute log probability density
    - ppf(): Percent point function (inverse CDF) for nested sampling

Usage:
    from grb_common.fitting import UniformPrior, LogUniformPrior

    priors = {
        'E_iso': LogUniformPrior(1e50, 1e54),
        'n': LogUniformPrior(1e-5, 1.0),
        'epsilon_e': UniformPrior(0.01, 0.5),
        'p': UniformPrior(2.0, 3.0),
    }
"""

from abc import ABC, abstractmethod
from typing import Union, Optional
import numpy as np

ArrayLike = Union[float, np.ndarray]


class Prior(ABC):
    """
    Abstract base class for prior distributions.

    Subclasses must implement:
        - sample(n): Draw n random samples
        - log_prob(x): Log probability density at x
        - ppf(q): Percent point function (inverse CDF)
    """

    @abstractmethod
    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Draw random samples from the prior.

        Parameters
        ----------
        n : int
            Number of samples.
        rng : numpy.random.Generator, optional
            Random number generator.

        Returns
        -------
        ndarray
            Array of samples.
        """
        pass

    @abstractmethod
    def log_prob(self, x: ArrayLike) -> ArrayLike:
        """
        Compute log probability density at x.

        Parameters
        ----------
        x : float or array-like
            Value(s) at which to evaluate.

        Returns
        -------
        float or ndarray
            Log probability density.
        """
        pass

    @abstractmethod
    def ppf(self, q: ArrayLike) -> ArrayLike:
        """
        Percent point function (inverse CDF).

        Used by nested sampling to transform uniform [0,1] samples
        to the prior distribution.

        Parameters
        ----------
        q : float or array-like
            Quantile(s) in [0, 1].

        Returns
        -------
        float or ndarray
            Values corresponding to quantiles.
        """
        pass

    def __call__(self, q: ArrayLike) -> ArrayLike:
        """Alias for ppf() for use as prior transform."""
        return self.ppf(q)


class UniformPrior(Prior):
    """
    Uniform prior distribution.

    Parameters
    ----------
    low : float
        Lower bound.
    high : float
        Upper bound.

    Examples
    --------
    >>> prior = UniformPrior(0, 1)
    >>> prior.sample(3)
    array([0.42, 0.67, 0.15])
    >>> prior.log_prob(0.5)
    0.0  # log(1) for uniform on [0,1]
    """

    def __init__(self, low: float, high: float):
        if high <= low:
            raise ValueError(f"high ({high}) must be greater than low ({low})")
        self.low = float(low)
        self.high = float(high)
        self._range = high - low
        self._log_prob_value = -np.log(self._range)

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high, size=n)

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x)
        result = np.where(
            (x >= self.low) & (x <= self.high),
            self._log_prob_value,
            -np.inf,
        )
        return float(result) if result.ndim == 0 else result

    def ppf(self, q: ArrayLike) -> ArrayLike:
        return self.low + q * self._range

    def __repr__(self) -> str:
        return f"UniformPrior({self.low}, {self.high})"


class LogUniformPrior(Prior):
    """
    Log-uniform (Jeffreys) prior distribution.

    Uniform in log space, giving equal probability per decade.
    Common for scale parameters like energy, density.

    Parameters
    ----------
    low : float
        Lower bound (must be positive).
    high : float
        Upper bound (must be > low).

    Examples
    --------
    >>> prior = LogUniformPrior(1e50, 1e54)
    >>> prior.sample(3)
    array([2.5e52, 1.1e51, 8.3e53])
    """

    def __init__(self, low: float, high: float):
        if low <= 0:
            raise ValueError(f"low ({low}) must be positive for log-uniform prior")
        if high <= low:
            raise ValueError(f"high ({high}) must be greater than low ({low})")
        self.low = float(low)
        self.high = float(high)
        self._log_low = np.log(low)
        self._log_high = np.log(high)
        self._log_range = self._log_high - self._log_low

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        log_samples = rng.uniform(self._log_low, self._log_high, size=n)
        return np.exp(log_samples)

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x)
        # p(x) = 1 / (x * log(high/low))
        # log p(x) = -log(x) - log(log(high/low))
        result = np.where(
            (x >= self.low) & (x <= self.high),
            -np.log(x) - np.log(self._log_range),
            -np.inf,
        )
        return float(result) if result.ndim == 0 else result

    def ppf(self, q: ArrayLike) -> ArrayLike:
        log_val = self._log_low + q * self._log_range
        return np.exp(log_val)

    def __repr__(self) -> str:
        return f"LogUniformPrior({self.low:.2e}, {self.high:.2e})"


class GaussianPrior(Prior):
    """
    Gaussian (normal) prior distribution.

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation.

    Examples
    --------
    >>> prior = GaussianPrior(2.5, 0.1)  # p ~ 2.5 +/- 0.1
    >>> prior.sample(3)
    array([2.48, 2.53, 2.41])
    """

    def __init__(self, mu: float, sigma: float):
        if sigma <= 0:
            raise ValueError(f"sigma ({sigma}) must be positive")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self._log_norm = -0.5 * np.log(2 * np.pi) - np.log(sigma)

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(self.mu, self.sigma, size=n)

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x)
        return self._log_norm - 0.5 * ((x - self.mu) / self.sigma) ** 2

    def ppf(self, q: ArrayLike) -> ArrayLike:
        from scipy.special import erfinv
        return self.mu + self.sigma * np.sqrt(2) * erfinv(2 * q - 1)

    def __repr__(self) -> str:
        return f"GaussianPrior({self.mu}, {self.sigma})"


class TruncatedGaussianPrior(Prior):
    """
    Truncated Gaussian prior distribution.

    Gaussian distribution bounded to [low, high].

    Parameters
    ----------
    mu : float
        Mean of underlying Gaussian.
    sigma : float
        Standard deviation of underlying Gaussian.
    low : float
        Lower bound.
    high : float
        Upper bound.
    """

    def __init__(self, mu: float, sigma: float, low: float, high: float):
        if sigma <= 0:
            raise ValueError(f"sigma ({sigma}) must be positive")
        if high <= low:
            raise ValueError(f"high ({high}) must be greater than low ({low})")
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.low = float(low)
        self.high = float(high)

        # Pre-compute normalization
        from scipy.special import erf
        self._alpha = (low - mu) / sigma
        self._beta = (high - mu) / sigma
        self._phi_alpha = 0.5 * (1 + erf(self._alpha / np.sqrt(2)))
        self._phi_beta = 0.5 * (1 + erf(self._beta / np.sqrt(2)))
        self._Z = self._phi_beta - self._phi_alpha

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        # Use inverse CDF sampling
        u = rng.uniform(0, 1, size=n)
        return self.ppf(u)

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x)
        # Truncated normal PDF
        in_bounds = (x >= self.low) & (x <= self.high)
        gaussian_log_prob = -0.5 * ((x - self.mu) / self.sigma) ** 2
        log_norm = -0.5 * np.log(2 * np.pi) - np.log(self.sigma) - np.log(self._Z)
        result = np.where(in_bounds, gaussian_log_prob + log_norm, -np.inf)
        return float(result) if result.ndim == 0 else result

    def ppf(self, q: ArrayLike) -> ArrayLike:
        from scipy.special import erfinv
        # Transform uniform to truncated normal
        q_scaled = self._phi_alpha + q * self._Z
        return self.mu + self.sigma * np.sqrt(2) * erfinv(2 * q_scaled - 1)

    def __repr__(self) -> str:
        return f"TruncatedGaussianPrior({self.mu}, {self.sigma}, [{self.low}, {self.high}])"


class DeltaPrior(Prior):
    """
    Delta (fixed value) prior.

    Used for parameters that should be held constant during fitting.

    Parameters
    ----------
    value : float
        Fixed parameter value.
    """

    def __init__(self, value: float):
        self.value = float(value)

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        return np.full(n, self.value)

    def log_prob(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x)
        result = np.where(np.isclose(x, self.value), 0.0, -np.inf)
        return float(result) if result.ndim == 0 else result

    def ppf(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q)
        return np.full_like(q, self.value)

    def __repr__(self) -> str:
        return f"DeltaPrior({self.value})"


class CompositePrior:
    """
    Collection of priors for multiple parameters.

    Provides convenience methods for working with parameter vectors.

    Parameters
    ----------
    priors : dict
        Dictionary mapping parameter names to Prior objects.

    Examples
    --------
    >>> priors = CompositePrior({
    ...     'E_iso': LogUniformPrior(1e50, 1e54),
    ...     'n': LogUniformPrior(1e-5, 1.0),
    ...     'p': UniformPrior(2.0, 3.0),
    ... })
    >>> theta = priors.sample()
    >>> log_p = priors.log_prob(theta)
    """

    def __init__(self, priors: dict):
        self.priors = priors
        self.param_names = list(priors.keys())
        self.n_params = len(priors)

    def sample(self, n: int = 1, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample from all priors.

        Returns
        -------
        ndarray
            Shape (n, n_params) array of samples.
        """
        if rng is None:
            rng = np.random.default_rng()
        samples = np.column_stack([
            self.priors[name].sample(n, rng)
            for name in self.param_names
        ])
        return samples

    def log_prob(self, theta: np.ndarray) -> float:
        """
        Compute total log prior probability.

        Parameters
        ----------
        theta : array-like
            Parameter vector.

        Returns
        -------
        float
            Sum of log probabilities from all priors.
        """
        theta = np.atleast_1d(theta)
        if len(theta) != self.n_params:
            raise ValueError(
                f"theta has {len(theta)} values, expected {self.n_params}"
            )
        total = 0.0
        for i, name in enumerate(self.param_names):
            lp = self.priors[name].log_prob(theta[i])
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return total

    def prior_transform(self, u: np.ndarray) -> np.ndarray:
        """
        Transform uniform [0,1]^n to prior distribution.

        Used by nested sampling algorithms.

        Parameters
        ----------
        u : array-like
            Uniform samples in [0, 1].

        Returns
        -------
        ndarray
            Samples from prior distribution.
        """
        u = np.atleast_1d(u)
        return np.array([
            self.priors[name].ppf(u[i])
            for i, name in enumerate(self.param_names)
        ])

    def __repr__(self) -> str:
        prior_strs = [f"  {name}: {prior}" for name, prior in self.priors.items()]
        return "CompositePrior({\n" + "\n".join(prior_strs) + "\n})"


__all__ = [
    "Prior",
    "UniformPrior",
    "LogUniformPrior",
    "GaussianPrior",
    "TruncatedGaussianPrior",
    "DeltaPrior",
    "CompositePrior",
]
