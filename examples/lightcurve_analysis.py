#!/usr/bin/env python3
"""
Light curve analysis example using grb-common.

This example demonstrates:
1. Loading light curve data
2. Applying extinction corrections
3. Creating publication-quality plots
"""

import numpy as np

from grb_common.io import LightCurve, GRBMetadata, GRBObservation
from grb_common.extinction import deredden_flux
from grb_common.cosmology import luminosity_distance
from grb_common.constants import MPC


def main():
    # --- Create synthetic multi-band data ---
    # In practice, you would use load_lightcurve() or load_grb()

    # Common observation times
    time = np.logspace(3, 6, 30)  # 10^3 to 10^6 seconds

    # X-ray light curve (Swift-XRT like)
    xray_flux = 1e-12 * (time / 1e4)**(-1.2)
    xray_err = 0.15 * xray_flux
    xray_lc = LightCurve(
        time=time,
        flux=xray_flux,
        flux_err=xray_err,
        band='X-ray',
    )

    # Optical R-band (with some upper limits at late times)
    optical_flux = 1e-27 * (time / 1e4)**(-1.0)
    optical_err = 0.2 * optical_flux
    optical_upper_limits = time > 5e5  # Late times are upper limits
    optical_lc = LightCurve(
        time=time,
        flux=optical_flux,
        flux_err=optical_err,
        band='optical_R',
        upper_limits=optical_upper_limits,
    )

    # Radio 5 GHz (delayed onset)
    radio_time = time[time > 1e4]
    radio_flux = 1e-28 * (radio_time / 1e5)**0.5 * np.exp(-radio_time / 1e6)
    radio_err = 0.25 * radio_flux
    radio_lc = LightCurve(
        time=radio_time,
        flux=radio_flux,
        flux_err=radio_err,
        band='radio_5GHz',
    )

    # Create observation container
    metadata = GRBMetadata(
        name='GRB Example',
        redshift=0.5,
        ra=180.0,
        dec=30.0,
    )

    obs = GRBObservation(
        metadata=metadata,
        lightcurves={'X-ray': xray_lc, 'optical_R': optical_lc, 'radio_5GHz': radio_lc},
    )

    print(f"GRB: {obs.metadata.name}")
    print(f"Redshift: {obs.metadata.redshift}")
    print(f"Available bands: {obs.bands}")

    # --- Cosmological calculations ---
    z = obs.metadata.redshift
    d_L = luminosity_distance(z)
    print(f"\nLuminosity distance: {d_L:.2e} cm = {d_L/MPC:.1f} Mpc")

    # --- Extinction correction example ---
    # Galactic extinction toward this direction
    E_BV = 0.05  # mag
    R_V = 3.1

    # Correct optical R-band (effective wavelength ~6400 Angstrom)
    wavelength_R = 6400.0  # Angstrom
    optical_flux_corrected = deredden_flux(
        optical_lc.flux,
        wavelength_R,
        E_BV=E_BV,
        R_V=R_V,
    )

    print(f"\nOptical extinction correction:")
    print(f"  E(B-V) = {E_BV} mag")
    print(f"  Flux increase factor: {optical_flux_corrected[0] / optical_lc.flux[0]:.2f}")

    # --- Plotting ---
    try:
        from grb_common.plotting import (
            set_style, plot_multiband, plot_lightcurve
        )
        import matplotlib.pyplot as plt

        # Set publication style
        set_style('apj')

        # Multi-band plot
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_multiband(obs, ax=ax, flux_unit='mJy', show_legend=True)
        ax.set_title(f'{obs.metadata.name} (z={z})')
        fig.tight_layout()
        fig.savefig('example_multiband.pdf', dpi=300)
        print("\nMulti-band plot saved to example_multiband.pdf")
        plt.close()

        # Single band with model overlay
        fig, ax = plt.subplots(figsize=(5, 4))
        plot_lightcurve(xray_lc, ax=ax, flux_unit='cgs', color='blue', label='Data')

        # Simple power-law model
        model_time = np.logspace(3, 6, 100)
        model_flux = 1.1e-12 * (model_time / 1e4)**(-1.2)
        ax.plot(model_time, model_flux, 'k-', linewidth=1.5, label='Model')
        ax.legend()
        ax.set_title('X-ray Light Curve')
        fig.tight_layout()
        fig.savefig('example_xray.pdf', dpi=300)
        print("X-ray plot saved to example_xray.pdf")
        plt.close()

        # Presentation style
        set_style('presentation')
        fig, ax = plt.subplots()
        plot_multiband(obs, ax=ax, offset_bands=True, offset_factor=100)
        ax.set_title('Multi-band Light Curves (offset for clarity)')
        fig.savefig('example_presentation.png', dpi=150)
        print("Presentation plot saved to example_presentation.png")
        plt.close()

    except ImportError as e:
        print(f"\nPlotting libraries not available: {e}")
        print("Install with: pip install grb-common[plotting]")

    # --- Save to HDF5 ---
    try:
        from grb_common.io import save_grb, load_grb

        save_grb(obs, 'example_grb.h5')
        print("\nObservation saved to example_grb.h5")

        # Load it back
        obs_loaded = load_grb('example_grb.h5')
        print(f"Loaded: {obs_loaded.metadata.name} with bands {obs_loaded.bands}")

    except ImportError:
        print("\nh5py not installed, skipping HDF5 save")


if __name__ == '__main__':
    main()
