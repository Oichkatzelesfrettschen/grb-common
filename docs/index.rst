grb-common Documentation
========================

Shared infrastructure for gamma-ray burst (GRB) astrophysics analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   MIGRATION
   api/index
   examples

Overview
--------

grb-common consolidates common utilities used across multiple GRB afterglow
modeling codes, providing:

* **Physical Constants**: CGS units derived from astropy.constants
* **Cosmology**: Distance calculations with standard cosmologies
* **Extinction**: Dust corrections (Fitzpatrick99, Cardelli89, SMC, LMC)
* **Data I/O**: Multi-format loaders with standardized HDF5 schema
* **Fitting**: Unified sampler interface (emcee, dynesty, pymultinest)
* **Plotting**: Publication-quality figures for light curves, SEDs, corners

Installation
------------

.. code-block:: bash

    # Core package
    pip install grb-common

    # With all dependencies
    pip install grb-common[all]

Quick Example
-------------

.. code-block:: python

    from grb_common.constants import C_LIGHT, SIGMA_T
    from grb_common.cosmology import luminosity_distance
    from grb_common.io import load_lightcurve
    from grb_common.plotting import plot_lightcurve, set_style

    # Load data
    lc = load_lightcurve("grb_data.txt")

    # Calculate distance
    d_L = luminosity_distance(lc.redshift)

    # Plot
    set_style('apj')
    ax = plot_lightcurve(lc)
    ax.figure.savefig("lightcurve.pdf")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
