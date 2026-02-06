#!/usr/bin/env python
############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

"""
Benchmark script for xtensor-python with pybind11 and nanobind backends.

Both backends are REQUIRED. The script will exit with a non-zero status
if either backend fails to build or import.

Usage:
    python run_benchmarks.py [--iterations N] [--size N] [--warmup N]
"""

import os
import sys
import subprocess
import argparse
from timeit import timeit

import numpy as np

here = os.path.abspath(os.path.dirname(__file__))

# Maximum characters to show from build output on failure
MAX_STDERR_CHARS = 5000
MAX_STDOUT_CHARS = 5000


def build_extension(name, setup_script):
    """Build an extension and return True if successful."""
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(here, setup_script), 'build_ext', '--inplace'],
            cwd=here,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return True
        # If build fails, show the error for debugging
        print(f"Failed to build {name}:")
        if result.stderr:
            stderr = result.stderr
            if len(stderr) > MAX_STDERR_CHARS:
                stderr = stderr[:MAX_STDERR_CHARS] + "\n... (truncated)"
            print(f"  stderr: {stderr}")
        if result.stdout:
            stdout = result.stdout
            if len(stdout) > MAX_STDOUT_CHARS:
                stdout = stdout[:MAX_STDOUT_CHARS] + "\n... (truncated)"
            print(f"  stdout: {stdout}")
        return False
    except Exception as e:
        print(f"Failed to build {name}: {e}")
        return False


def import_extension(name):
    """Import an extension module by name."""
    try:
        return __import__(name)
    except ImportError as e:
        print(f"Failed to import {name}: {e}")
        return None


# Build and import extensions
print("Building extensions...")

# pybind11 version
HAS_PYBIND11 = build_extension("pybind11", "setup.py")
xt_pybind11 = import_extension("benchmark_xtensor_python") if HAS_PYBIND11 else None
HAS_PYBIND11 = xt_pybind11 is not None

# nanobind version
HAS_NANOBIND = build_extension("nanobind", "setup_nanobind.py")
xt_nanobind = import_extension("benchmark_xtensor_nanobind") if HAS_NANOBIND else None
HAS_NANOBIND = xt_nanobind is not None

print("\nAvailable backends:")
print(f"  pybind11: {'[OK]' if HAS_PYBIND11 else '[FAIL]'}")
print(f"  nanobind: {'[OK]' if HAS_NANOBIND else '[FAIL]'}")

# Both backends are required - fail if either is missing
if not HAS_PYBIND11 or not HAS_NANOBIND:
    missing = []
    if not HAS_PYBIND11:
        missing.append("pybind11")
    if not HAS_NANOBIND:
        missing.append("nanobind")
    print(f"\nError: Required backend(s) not available: {', '.join(missing)}")
    print("Both pybind11 and nanobind backends are required for benchmarks.")
    sys.exit(1)


def run_benchmark(func, setup_vars, iterations):
    """Run a benchmark and return the time in seconds."""
    setup = "; ".join(f"from __main__ import {v}" for v in setup_vars.split(","))
    return timeit(func, setup=setup, number=iterations)


def main():
    parser = argparse.ArgumentParser(description="xtensor-python benchmarks")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--size", type=int, default=1000000, help="Array size")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    args = parser.parse_args()

    iterations = args.iterations
    size = args.size
    warmup = args.warmup

    u = np.ones(size, dtype=float)

    # Warmup
    for _ in range(warmup):
        if HAS_PYBIND11:
            xt_pybind11.sum_array(u)
            xt_pybind11.pybind_sum_array(u)
        if HAS_NANOBIND:
            xt_nanobind.nanobind_sum_array(u)
        np.sum(u)

    print(f"\n{'='*70}")
    print(f"BENCHMARK: pybind11 vs nanobind vs numpy")
    print(f"Array size: {size:,}, Iterations: {iterations:,}")
    print(f"{'='*70}")

    # pybind11 benchmarks
    print(f"\n--- pybind11 backend ---")
    t_pyarray = timeit('xt_pybind11.sum_array(u)', globals={'xt_pybind11': xt_pybind11, 'u': u}, number=iterations)
    print(f"  xt::pyarray sum:    {t_pyarray:.4f}s")

    t_pytensor = timeit('xt_pybind11.sum_tensor(u)', globals={'xt_pybind11': xt_pybind11, 'u': u}, number=iterations)
    print(f"  xt::pytensor sum:   {t_pytensor:.4f}s")

    t_pybind = timeit('xt_pybind11.pybind_sum_array(u)', globals={'xt_pybind11': xt_pybind11, 'u': u}, number=iterations)
    print(f"  pybind11 array sum: {t_pybind:.4f}s")

    # nanobind benchmarks
    print(f"\n--- nanobind backend ---")
    t_nanobind = timeit('xt_nanobind.nanobind_sum_array(u)', globals={'xt_nanobind': xt_nanobind, 'u': u}, number=iterations)
    print(f"  nanobind array sum: {t_nanobind:.4f}s")

    # numpy reference
    print(f"\n--- numpy reference ---")
    t_numpy = timeit('np.sum(u)', globals={'np': np, 'u': u}, number=iterations)
    print(f"  numpy sum:          {t_numpy:.4f}s")

    # Comparison
    print(f"\n--- Summary ---")
    fastest = min(t_pyarray, t_pytensor, t_pybind, t_nanobind, t_numpy)
    for name, t in [("xt::pyarray (pybind11)", t_pyarray),
                    ("xt::pytensor (pybind11)", t_pytensor),
                    ("pybind11 native", t_pybind),
                    ("nanobind native", t_nanobind),
                    ("numpy", t_numpy)]:
        ratio = t / fastest if fastest > 0 else float('inf')
        print(f"  {name:<25} {t:.4f}s  ({ratio:.2f}x)")

    print(f"\n{'='*70}")
    print("Benchmarks completed successfully.")


if __name__ == "__main__":
    main()
