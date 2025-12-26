"""
Sampler backends for Bayesian parameter estimation.

This module provides a unified interface to multiple sampling backends:
- emcee: Affine-invariant ensemble MCMC
- dynesty: Dynamic nested sampling
- pymultinest: MultiNest via Python bindings (optional)

Usage:
    from grb_common.fitting.backends import get_sampler, EmceeSampler

    # Auto-detect available backend
    sampler = get_sampler('emcee', log_prob, n_walkers=32, ...)

    # Or import directly
    sampler = EmceeSampler(log_prob, n_walkers=32, ...)
"""

from typing import Callable, Dict, Any, Optional, List
import warnings

# Track available backends
_AVAILABLE_BACKENDS: Dict[str, bool] = {}


def _check_emcee() -> bool:
    """Check if emcee is available."""
    try:
        import emcee  # noqa: F401
        return True
    except ImportError:
        return False


def _check_dynesty() -> bool:
    """Check if dynesty is available."""
    try:
        import dynesty  # noqa: F401
        return True
    except ImportError:
        return False


def _check_pymultinest() -> bool:
    """Check if pymultinest is available."""
    try:
        import pymultinest  # noqa: F401
        return True
    except ImportError:
        return False


def available_backends() -> List[str]:
    """
    Get list of available sampler backends.

    Returns
    -------
    list of str
        Names of available backends.
    """
    backends = []
    if _check_emcee():
        backends.append("emcee")
    if _check_dynesty():
        backends.append("dynesty")
    if _check_pymultinest():
        backends.append("pymultinest")
    return backends


def get_sampler(
    name: str,
    log_prob: Optional[Callable] = None,
    log_likelihood: Optional[Callable] = None,
    prior_transform: Optional[Callable] = None,
    **kwargs,
):
    """
    Get sampler instance by name.

    Parameters
    ----------
    name : str
        Backend name: 'emcee', 'dynesty', or 'pymultinest'.
    log_prob : callable, optional
        Log posterior function (for MCMC).
    log_likelihood : callable, optional
        Log likelihood function (for nested sampling).
    prior_transform : callable, optional
        Prior transform function (for nested sampling).
    **kwargs
        Backend-specific arguments.

    Returns
    -------
    BaseSampler
        Sampler instance.

    Raises
    ------
    ImportError
        If requested backend is not installed.
    ValueError
        If backend name is unknown.
    """
    name = name.lower()

    if name == "emcee":
        if not _check_emcee():
            raise ImportError(
                "emcee is not installed. "
                "Install with: pip install grb-common[fitting]"
            )
        from .emcee_backend import EmceeSampler
        if log_prob is None:
            raise ValueError("log_prob required for emcee")
        return EmceeSampler(log_prob=log_prob, **kwargs)

    elif name == "dynesty":
        if not _check_dynesty():
            raise ImportError(
                "dynesty is not installed. "
                "Install with: pip install grb-common[nested]"
            )
        from .dynesty_backend import DynestySampler
        if log_likelihood is None:
            raise ValueError("log_likelihood required for dynesty")
        if prior_transform is None:
            raise ValueError("prior_transform required for dynesty")
        return DynestySampler(
            log_likelihood=log_likelihood,
            prior_transform=prior_transform,
            **kwargs,
        )

    elif name == "pymultinest":
        if not _check_pymultinest():
            raise ImportError(
                "pymultinest is not installed. "
                "Install with: pip install grb-common[multinest]"
            )
        from .multinest_backend import MultiNestSampler
        if log_likelihood is None:
            raise ValueError("log_likelihood required for pymultinest")
        if prior_transform is None:
            raise ValueError("prior_transform required for pymultinest")
        return MultiNestSampler(
            log_likelihood=log_likelihood,
            prior_transform=prior_transform,
            **kwargs,
        )

    else:
        available = available_backends()
        raise ValueError(
            f"Unknown sampler: {name}. "
            f"Available: {available if available else 'none (install a backend)'}"
        )


__all__ = [
    "get_sampler",
    "available_backends",
]

# Lazy imports for direct access
def __getattr__(name):
    if name == "EmceeSampler":
        if not _check_emcee():
            raise ImportError("emcee required")
        from .emcee_backend import EmceeSampler
        return EmceeSampler
    elif name == "DynestySampler":
        if not _check_dynesty():
            raise ImportError("dynesty required")
        from .dynesty_backend import DynestySampler
        return DynestySampler
    elif name == "MultiNestSampler":
        if not _check_pymultinest():
            raise ImportError("pymultinest required")
        from .multinest_backend import MultiNestSampler
        return MultiNestSampler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
