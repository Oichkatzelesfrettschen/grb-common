#!/usr/bin/env python3
"""
Benchmarks for extinction calculations.

Run with:
    python benchmarks/bench_extinction.py
"""

import time
import numpy as np


def bench_fitzpatrick99():
    """Benchmark Fitzpatrick99 extinction law."""
    from grb_common.extinction import fitzpatrick99

    # Single wavelength
    n_iterations = 10000
    wavelength = 5500.0  # V-band
    Av = 1.0

    start = time.perf_counter()
    for _ in range(n_iterations):
        A_lambda = fitzpatrick99(wavelength, Av=Av, Rv=3.1)
    elapsed = time.perf_counter() - start

    print(f"fitzpatrick99 (single wavelength):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")

    # Array of wavelengths
    wavelengths = np.linspace(1000, 30000, 1000)  # UV to NIR
    n_iterations = 1000

    start = time.perf_counter()
    for _ in range(n_iterations):
        A_lambda = fitzpatrick99(wavelengths, Av=Av, Rv=3.1)
    elapsed = time.perf_counter() - start

    print(f"\nfitzpatrick99 (1000 wavelengths):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e3:.2f} ms per call")


def bench_deredden():
    """Benchmark flux dereddening."""
    from grb_common.extinction import deredden_flux

    flux = np.random.uniform(1e-26, 1e-24, 100)
    wavelengths = np.linspace(3000, 10000, 100)
    n_iterations = 1000

    start = time.perf_counter()
    for _ in range(n_iterations):
        flux_corrected = deredden_flux(flux, wavelengths, E_BV=0.1, Rv=3.1)
    elapsed = time.perf_counter() - start

    print(f"\nderedden_flux (100 points):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e3:.2f} ms per call")


def bench_law_comparison():
    """Compare different extinction laws."""
    from grb_common.extinction import fitzpatrick99, cardelli89

    wavelengths = np.linspace(1000, 30000, 100)
    Av = 1.0
    n_iterations = 1000

    for law_func, name in [(fitzpatrick99, 'Fitzpatrick99'), (cardelli89, 'Cardelli89')]:
        start = time.perf_counter()
        for _ in range(n_iterations):
            A_lambda = law_func(wavelengths, Av=Av, Rv=3.1)
        elapsed = time.perf_counter() - start

        print(f"{name}: {elapsed/n_iterations*1e3:.2f} ms per 100 wavelengths")


def main():
    print("=" * 60)
    print("Extinction Benchmarks")
    print("=" * 60)
    print()

    bench_fitzpatrick99()
    bench_deredden()
    print()
    bench_law_comparison()


if __name__ == '__main__':
    main()
