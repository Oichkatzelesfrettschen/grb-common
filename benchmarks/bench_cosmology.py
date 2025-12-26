#!/usr/bin/env python3
"""
Benchmarks for cosmology calculations.

Run with:
    python benchmarks/bench_cosmology.py
"""

import time
import numpy as np


def bench_luminosity_distance():
    """Benchmark luminosity distance calculations."""
    from grb_common.cosmology import luminosity_distance

    # Single value
    n_iterations = 1000
    z = 0.5

    start = time.perf_counter()
    for _ in range(n_iterations):
        d_L = luminosity_distance(z)
    elapsed = time.perf_counter() - start

    print(f"luminosity_distance (single z):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e6:.2f} us per call")
    print(f"  {n_iterations/elapsed:.0f} calls/s")

    # Array of values
    z_array = np.linspace(0.01, 2.0, 1000)
    n_iterations = 100

    start = time.perf_counter()
    for _ in range(n_iterations):
        d_L = luminosity_distance(z_array)
    elapsed = time.perf_counter() - start

    print(f"\nluminosity_distance (1000 z values):")
    print(f"  {n_iterations} iterations in {elapsed:.3f} s")
    print(f"  {elapsed/n_iterations*1e3:.2f} ms per call")
    print(f"  {n_iterations*1000/elapsed:.0f} z-values/s")


def bench_cosmology_comparison():
    """Compare different cosmologies."""
    from grb_common.cosmology import luminosity_distance

    z_array = np.linspace(0.01, 2.0, 100)
    cosmologies = ['Planck18', 'Planck15', 'WMAP9']

    for cosmo in cosmologies:
        n_iterations = 100
        start = time.perf_counter()
        for _ in range(n_iterations):
            d_L = luminosity_distance(z_array, cosmology=cosmo)
        elapsed = time.perf_counter() - start

        print(f"{cosmo}: {elapsed/n_iterations*1e3:.2f} ms per 100 values")


def main():
    print("=" * 60)
    print("Cosmology Benchmarks")
    print("=" * 60)
    print()

    bench_luminosity_distance()
    print()
    bench_cosmology_comparison()


if __name__ == '__main__':
    main()
