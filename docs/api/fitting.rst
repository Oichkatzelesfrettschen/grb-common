Fitting Module
==============

The fitting module provides a unified interface for Bayesian parameter
estimation using various sampling backends.

.. automodule:: grb_common.fitting
   :members:
   :undoc-members:
   :show-inheritance:

Priors
------

Available prior distributions:

.. autoclass:: grb_common.fitting.UniformPrior
   :members:
   :special-members: __init__

.. autoclass:: grb_common.fitting.LogUniformPrior
   :members:
   :special-members: __init__

.. autoclass:: grb_common.fitting.GaussianPrior
   :members:
   :special-members: __init__

.. autoclass:: grb_common.fitting.TruncatedGaussianPrior
   :members:
   :special-members: __init__

.. autoclass:: grb_common.fitting.DeltaPrior
   :members:
   :special-members: __init__

.. autoclass:: grb_common.fitting.CompositePrior
   :members:
   :special-members: __init__

Likelihoods
-----------

.. autofunction:: grb_common.fitting.chi_squared
.. autofunction:: grb_common.fitting.gaussian_likelihood
.. autofunction:: grb_common.fitting.gaussian_likelihood_upper_limits
.. autofunction:: grb_common.fitting.poisson_likelihood
.. autofunction:: grb_common.fitting.cstat

Results
-------

.. autoclass:: grb_common.fitting.SamplerResult
   :members:
   :special-members: __init__

Sampler Backends
----------------

.. automodule:: grb_common.fitting.backends
   :members:

Usage Examples
--------------

Basic MCMC with emcee
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from grb_common.fitting import LogUniformPrior, CompositePrior
    from grb_common.fitting.backends import get_sampler

    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e55),
        'n': LogUniformPrior(1e-5, 1.0),
    })

    def log_posterior(theta):
        return priors.log_prob(theta) + log_likelihood(theta)

    sampler = get_sampler('emcee', log_prob=log_posterior, n_params=2)
    result = sampler.run(n_walkers=32, n_steps=5000)

Nested Sampling with dynesty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from grb_common.fitting.backends import get_sampler

    sampler = get_sampler(
        'dynesty',
        log_likelihood=log_likelihood,
        prior_transform=priors.prior_transform,
        n_params=2,
    )
    result = sampler.run(nlive=500)
    print(f"Log evidence: {result.metadata['log_evidence']}")
