"""
Likelihood functions for GRB model fitting.

This module provides likelihood functions for comparing model predictions
to observed data, including proper handling of:
- Symmetric and asymmetric errors
- Upper limits (non-detections)
- Systematic uncertainties
- Poisson-distributed counts

Usage:
    from grb_common.fitting import chi_squared, gaussian_likelihood

    log_like = gaussian_likelihood(observed, model, error)
"""

from typing import Union, Optional, Tuple
import numpy as np

ArrayLike = Union[float, np.ndarray]


def chi_squared(
    observed: ArrayLike,
    model: ArrayLike,
    error: ArrayLike,
) -> float:
    """
    Compute chi-squared statistic.

    Parameters
    ----------
    observed : array-like
        Observed data values.
    model : array-like
        Model predictions.
    error : array-like
        Measurement uncertainties (1-sigma).

    Returns
    -------
    float
        Chi-squared value.
    """
    observed = np.asarray(observed)
    model = np.asarray(model)
    error = np.asarray(error)

    return np.sum(((observed - model) / error) ** 2)


def reduced_chi_squared(
    observed: ArrayLike,
    model: ArrayLike,
    error: ArrayLike,
    n_params: int = 0,
) -> float:
    """
    Compute reduced chi-squared statistic.

    Parameters
    ----------
    observed : array-like
        Observed data values.
    model : array-like
        Model predictions.
    error : array-like
        Measurement uncertainties.
    n_params : int
        Number of free parameters in model.

    Returns
    -------
    float
        Reduced chi-squared (chi^2 / dof).
    """
    chi2 = chi_squared(observed, model, error)
    dof = len(np.asarray(observed)) - n_params
    if dof <= 0:
        raise ValueError(f"Degrees of freedom ({dof}) must be positive")
    return chi2 / dof


def gaussian_likelihood(
    observed: ArrayLike,
    model: ArrayLike,
    error: ArrayLike,
) -> float:
    """
    Compute Gaussian log-likelihood.

    Assumes independent Gaussian errors on each data point.

    Parameters
    ----------
    observed : array-like
        Observed data values.
    model : array-like
        Model predictions.
    error : array-like
        Measurement uncertainties (1-sigma).

    Returns
    -------
    float
        Log-likelihood value.
    """
    observed = np.asarray(observed)
    model = np.asarray(model)
    error = np.asarray(error)

    n = len(observed)

    # log L = -0.5 * sum[(y - m)^2 / sigma^2] - sum[log(sigma)] - n/2 * log(2*pi)
    chi2 = chi_squared(observed, model, error)
    log_norm = -np.sum(np.log(error)) - 0.5 * n * np.log(2 * np.pi)

    return -0.5 * chi2 + log_norm


def gaussian_likelihood_asymmetric(
    observed: ArrayLike,
    model: ArrayLike,
    error_lo: ArrayLike,
    error_hi: ArrayLike,
) -> float:
    """
    Compute Gaussian log-likelihood with asymmetric errors.

    Uses error_lo when model < observed, error_hi otherwise.

    Parameters
    ----------
    observed : array-like
        Observed data values.
    model : array-like
        Model predictions.
    error_lo : array-like
        Lower uncertainties (positive values).
    error_hi : array-like
        Upper uncertainties (positive values).

    Returns
    -------
    float
        Log-likelihood value.
    """
    observed = np.asarray(observed)
    model = np.asarray(model)
    error_lo = np.asarray(error_lo)
    error_hi = np.asarray(error_hi)

    # Select appropriate error based on residual sign
    residual = observed - model
    error = np.where(residual > 0, error_lo, error_hi)

    return gaussian_likelihood(observed, model, error)


def chi_squared_upper_limits(
    observed: ArrayLike,
    model: ArrayLike,
    error: ArrayLike,
    upper_limits: ArrayLike,
    sigma_ul: float = 3.0,
) -> float:
    """
    Compute chi-squared including upper limits.

    Upper limits contribute when model exceeds the limit.
    Non-detections are penalized via a one-sided Gaussian.

    Parameters
    ----------
    observed : array-like
        Observed data (flux for detections, limit for non-detections).
    model : array-like
        Model predictions.
    error : array-like
        Uncertainties for detections.
    upper_limits : array-like
        Boolean mask (True = upper limit).
    sigma_ul : float
        Sigma level of upper limits (default 3-sigma).

    Returns
    -------
    float
        Modified chi-squared value.
    """
    observed = np.asarray(observed)
    model = np.asarray(model)
    error = np.asarray(error)
    upper_limits = np.asarray(upper_limits, dtype=bool)

    # Detections: standard chi-squared
    det_mask = ~upper_limits
    chi2_det = chi_squared(observed[det_mask], model[det_mask], error[det_mask])

    # Upper limits: penalize if model > limit
    ul_mask = upper_limits
    if np.any(ul_mask):
        limits = observed[ul_mask]
        model_ul = model[ul_mask]
        # Upper limit "sigma" estimated from the limit value
        sigma_ul_values = limits / sigma_ul

        # One-sided: only penalize when model exceeds limit
        excess = model_ul - limits
        chi2_ul = np.sum(np.where(excess > 0, (excess / sigma_ul_values) ** 2, 0))
    else:
        chi2_ul = 0.0

    return chi2_det + chi2_ul


def gaussian_likelihood_upper_limits(
    observed: ArrayLike,
    model: ArrayLike,
    error: ArrayLike,
    upper_limits: ArrayLike,
    sigma_ul: float = 3.0,
) -> float:
    """
    Compute Gaussian log-likelihood including upper limits.

    Parameters
    ----------
    observed : array-like
        Observed data (flux for detections, limit for non-detections).
    model : array-like
        Model predictions.
    error : array-like
        Uncertainties for detections.
    upper_limits : array-like
        Boolean mask (True = upper limit).
    sigma_ul : float
        Sigma level of upper limits.

    Returns
    -------
    float
        Log-likelihood value.
    """
    observed = np.asarray(observed)
    model = np.asarray(model)
    error = np.asarray(error)
    upper_limits = np.asarray(upper_limits, dtype=bool)

    det_mask = ~upper_limits

    # Detections contribute normal Gaussian likelihood
    log_like_det = gaussian_likelihood(
        observed[det_mask], model[det_mask], error[det_mask]
    ) if np.any(det_mask) else 0.0

    # Upper limits: use CDF-based likelihood
    # P(data | model) = integral from 0 to limit of Gaussian
    # log P = log(CDF(limit | model, sigma))
    ul_mask = upper_limits
    if np.any(ul_mask):
        from scipy.special import erf

        limits = observed[ul_mask]
        model_ul = model[ul_mask]
        sigma_ul_values = limits / sigma_ul

        # CDF of upper limit: probability of observing <= limit given model
        # For model < limit, this is high (good)
        # For model > limit, this is low (bad)
        z = (limits - model_ul) / sigma_ul_values
        cdf_values = 0.5 * (1 + erf(z / np.sqrt(2)))

        # Avoid log(0)
        cdf_values = np.clip(cdf_values, 1e-300, 1.0)
        log_like_ul = np.sum(np.log(cdf_values))
    else:
        log_like_ul = 0.0

    return log_like_det + log_like_ul


def chi_squared_with_systematics(
    observed: ArrayLike,
    model: ArrayLike,
    stat_error: ArrayLike,
    sys_error: ArrayLike,
) -> float:
    """
    Compute chi-squared with systematic uncertainties.

    Combines statistical and systematic errors in quadrature.

    Parameters
    ----------
    observed : array-like
        Observed data values.
    model : array-like
        Model predictions.
    stat_error : array-like
        Statistical uncertainties.
    sys_error : array-like
        Systematic uncertainties.

    Returns
    -------
    float
        Chi-squared value.
    """
    stat_error = np.asarray(stat_error)
    sys_error = np.asarray(sys_error)
    total_error = np.sqrt(stat_error**2 + sys_error**2)
    return chi_squared(observed, model, total_error)


def poisson_likelihood(
    counts: ArrayLike,
    model: ArrayLike,
) -> float:
    """
    Compute Poisson log-likelihood.

    For count data (e.g., gamma-ray photon counts).

    Parameters
    ----------
    counts : array-like
        Observed counts (integers).
    model : array-like
        Model-predicted counts (can be non-integer).

    Returns
    -------
    float
        Log-likelihood value.
    """
    counts = np.asarray(counts)
    model = np.asarray(model)

    # Ensure model is positive
    model = np.maximum(model, 1e-300)

    # log L = sum[k * log(lambda) - lambda - log(k!)]
    # Ignoring constant factorial term
    from scipy.special import gammaln

    log_like = np.sum(counts * np.log(model) - model - gammaln(counts + 1))

    return log_like


def cstat(
    counts: ArrayLike,
    model: ArrayLike,
) -> float:
    """
    Compute Cash statistic (C-stat).

    Alternative to chi-squared for Poisson data, commonly used
    in X-ray astronomy.

    Parameters
    ----------
    counts : array-like
        Observed counts.
    model : array-like
        Model-predicted counts.

    Returns
    -------
    float
        C-statistic value (lower is better, like chi-squared).
    """
    counts = np.asarray(counts)
    model = np.asarray(model)

    model = np.maximum(model, 1e-300)

    # C = 2 * sum[m - n + n*log(n/m)]
    # For n=0: contribution is just 2*m
    result = 2 * np.sum(
        model - counts + np.where(counts > 0, counts * np.log(counts / model), 0)
    )

    return result


__all__ = [
    "chi_squared",
    "reduced_chi_squared",
    "gaussian_likelihood",
    "gaussian_likelihood_asymmetric",
    "chi_squared_upper_limits",
    "gaussian_likelihood_upper_limits",
    "chi_squared_with_systematics",
    "poisson_likelihood",
    "cstat",
]
