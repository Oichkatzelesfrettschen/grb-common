#!/usr/bin/env python3
"""
Run all benchmarks for grb-common.

Usage:
    python benchmarks/run_all.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    benchmark_dir = Path(__file__).parent
    benchmarks = [
        'bench_cosmology.py',
        'bench_extinction.py',
        'bench_fitting.py',
    ]

    print("=" * 70)
    print("grb-common Benchmark Suite")
    print("=" * 70)

    for bench in benchmarks:
        bench_path = benchmark_dir / bench
        if bench_path.exists():
            print(f"\n{'=' * 70}")
            print(f"Running: {bench}")
            print("=" * 70)
            subprocess.run([sys.executable, str(bench_path)])
        else:
            print(f"Warning: {bench} not found")

    print("\n" + "=" * 70)
    print("Benchmark suite complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
