Quick Start Guide
=================

This guide covers the essential features of grb-common.

Installation
------------

.. code-block:: bash

    # Core package (constants, cosmology)
    pip install grb-common

    # With fitting backends
    pip install grb-common[fitting]

    # With plotting support
    pip install grb-common[plotting]

    # Everything
    pip install grb-common[all]

Physical Constants
------------------

All constants are in CGS units:

.. code-block:: python

    from grb_common.constants import (
        C_LIGHT,      # Speed of light [cm/s]
        M_ELECTRON,   # Electron mass [g]
        M_PROTON,     # Proton mass [g]
        SIGMA_T,      # Thomson cross-section [cm^2]
        H_PLANCK,     # Planck constant [erg s]
        K_BOLTZMANN,  # Boltzmann constant [erg/K]
        M_SUN,        # Solar mass [g]
        PC, MPC,      # Parsec, Megaparsec [cm]
    )

    # Example: Compton Y-parameter calculation
    y_compton = (4/3) * (K_BOLTZMANN * T / (M_ELECTRON * C_LIGHT**2))

Cosmology
---------

Calculate cosmological distances:

.. code-block:: python

    from grb_common.cosmology import (
        luminosity_distance,
        angular_diameter_distance,
        comoving_distance,
        lookback_time,
    )

    z = 0.5

    # Default: Planck18 cosmology
    d_L = luminosity_distance(z)        # Returns cm
    d_A = angular_diameter_distance(z)  # Returns cm
    d_C = comoving_distance(z)          # Returns cm
    t_lb = lookback_time(z)             # Returns seconds

    # Use different cosmology
    d_L_wmap = luminosity_distance(z, cosmology='WMAP9')

Extinction Corrections
----------------------

Apply dust extinction corrections:

.. code-block:: python

    from grb_common.extinction import (
        fitzpatrick99,
        cardelli89,
        deredden_flux,
        redden_flux,
    )

    # Get A_lambda at 5500 Angstrom
    wavelength = 5500.0  # Angstrom
    Av = 0.5
    A_lambda = fitzpatrick99(wavelength, Av=Av, Rv=3.1)

    # Deredden flux
    flux_corrected = deredden_flux(flux, wavelength, E_BV=0.1, Rv=3.1)

Data Loading
------------

Load light curves in various formats:

.. code-block:: python

    from grb_common.io import (
        load_lightcurve,
        load_grb,
        save_grb,
        LightCurve,
    )

    # Auto-detect format and load
    lc = load_lightcurve("data.txt")

    # Access fields
    print(lc.time)       # Time array [s]
    print(lc.flux)       # Flux array [erg/cm^2/s/Hz]
    print(lc.flux_err)   # Errors
    print(lc.band)       # Band name

    # Load full observation
    obs = load_grb("grb170817.h5")
    xray = obs.get_band("X-ray")
    optical = obs.get_band("optical_R")

    # Save to HDF5
    save_grb(obs, "output.h5")

MCMC Fitting
------------

Set up and run parameter estimation:

.. code-block:: python

    from grb_common.fitting import (
        UniformPrior,
        LogUniformPrior,
        CompositePrior,
        gaussian_likelihood,
    )
    from grb_common.fitting.backends import get_sampler

    # Define priors
    priors = CompositePrior({
        'E_iso': LogUniformPrior(1e50, 1e55),
        'n': LogUniformPrior(1e-5, 1.0),
        'p': UniformPrior(2.01, 3.0),
    })

    # Define likelihood
    def log_posterior(theta):
        lp = priors.log_prob(theta)
        if not np.isfinite(lp):
            return -np.inf
        model = my_model(theta, time, freq)
        return lp + gaussian_likelihood(data, model, errors)

    # Run sampler
    sampler = get_sampler('emcee', log_prob=log_posterior, n_params=3)
    result = sampler.run(n_walkers=32, n_steps=5000)

    # Analyze results
    print(result.mean('E_iso'))
    print(result.percentile('p', [16, 50, 84]))

Plotting
--------

Create publication-quality figures:

.. code-block:: python

    from grb_common.plotting import (
        set_style,
        plot_lightcurve,
        plot_multiband,
        corner_plot,
    )

    # Set journal style
    set_style('apj')      # Astrophysical Journal
    set_style('mnras')    # Monthly Notices
    set_style('nature')   # Nature/Science

    # Plot light curve
    ax = plot_lightcurve(lc, flux_unit='mJy')
    ax.figure.savefig("lc.pdf")

    # Multi-band plot
    ax = plot_multiband(obs, offset_bands=True)
    ax.figure.savefig("multiband.pdf")

    # Corner plot from MCMC result
    fig = corner_plot(result, truths={'E_iso': 1e52, 'p': 2.3})
    fig.savefig("corner.pdf")
