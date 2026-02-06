#!/usr/bin/env python
############################################################################
# Copyright (c) 2016, Johan Mabille and Sylvain Corlay                     #
#                                                                          #
# Distributed under the terms of the BSD 3-Clause License.                 #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

"""
Comprehensive benchmark script for xtensor-python.

Output is organized in two clear sections:
1. PYBIND11 vs NANOBIND: Framework comparison (binding overhead)
2. XTENSOR vs NUMPY: Library comparison (uses fastest available framework)

Tests cover:
- Basic sum operations (pyarray, pytensor, native array)
- Auto conversion (xtensor/xarray copy from numpy)
- Return value conversion (xtensor/xarray copy to numpy)
- Inplace operations (by reference)
- View operations (strided views, math on views)
- Vectorize operations
- Full round-trip operations
- Broadcasting and slicing

Usage:
    python run_benchmarks.py [--iterations N] [--size N] [--warmup N]
"""

import os
import sys
import subprocess
import argparse
from timeit import timeit
from typing import Optional, List, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm", "-q"])
    from tqdm import tqdm

here = os.path.abspath(os.path.dirname(__file__))

# Maximum characters to show from build output on failure
MAX_STDERR_CHARS = 5000
MAX_STDOUT_CHARS = 5000


def build_extension(name: str, setup_script: str) -> bool:
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


def import_extension(name: str):
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

if not HAS_PYBIND11 and not HAS_NANOBIND:
    print("\nWarning: No backend available! Skipping benchmarks.")
    sys.exit(0)


class BenchmarkResult:
    """Store benchmark results for comparison."""
    def __init__(self, name: str, pybind11_time: Optional[float] = None,
                 nanobind_time: Optional[float] = None, numpy_time: Optional[float] = None,
                 category: str = "general"):
        self.name = name
        self.pybind11_time = pybind11_time
        self.nanobind_time = nanobind_time
        self.numpy_time = numpy_time
        self.category = category
    
    def get_best_xtensor_time(self) -> Optional[float]:
        """Return the best xtensor time (faster of pybind11/nanobind)."""
        times = [t for t in [self.pybind11_time, self.nanobind_time] if t is not None]
        return min(times) if times else None
    
    def get_winner(self) -> str:
        """Return 'pybind11', 'nanobind', or 'lean pb'/'lean nb' for small differences."""
        if not (self.pybind11_time and self.nanobind_time):
            return "N/A"
        ratio = self.pybind11_time / self.nanobind_time
        if ratio > 1.07:  # nanobind is >7% faster
            return "nanobind"
        elif ratio < 0.93:  # pybind11 is >7% faster
            return "pybind11"
        elif ratio > 1.0:
            return "lean nb"
        else:
            return "lean pb"


def run_benchmark(name: str, setup_code: str,
                  pybind11_code: Optional[str], nanobind_code: Optional[str],
                  iterations: int, warmup: int = 3,
                  numpy_code: Optional[str] = None,
                  category: str = "general",
                  pbar: Optional[tqdm] = None) -> BenchmarkResult:
    """Run a benchmark for both backends and optionally NumPy."""
    result = BenchmarkResult(name, category=category)
    
    if pbar:
        pbar.set_description(f"{name[:40]:<40}")
        pbar.update(1)

    # Warmup and benchmark pybind11
    if HAS_PYBIND11 and pybind11_code:
        full_setup = f"import benchmark_xtensor_python as xt\n{setup_code}"
        # Warmup
        for _ in range(warmup):
            timeit(pybind11_code, setup=full_setup, number=1, globals=globals())
        # Actual benchmark
        result.pybind11_time = timeit(
            pybind11_code, setup=full_setup,
            number=iterations, globals=globals()) / iterations

    # Warmup and benchmark nanobind
    if HAS_NANOBIND and nanobind_code:
        full_setup = f"import benchmark_xtensor_nanobind as xt\n{setup_code}"
        # Warmup
        for _ in range(warmup):
            timeit(nanobind_code, setup=full_setup, number=1, globals=globals())
        # Actual benchmark
        result.nanobind_time = timeit(
            nanobind_code, setup=full_setup,
            number=iterations, globals=globals()) / iterations

    # Warmup and benchmark NumPy
    if numpy_code:
        full_setup = f"import numpy as np\n{setup_code}"
        # Warmup
        for _ in range(warmup):
            timeit(numpy_code, setup=full_setup, number=1, globals=globals())
        # Actual benchmark
        result.numpy_time = timeit(
            numpy_code, setup=full_setup,
            number=iterations, globals=globals()) / iterations

    return result


def print_header(title: str, char: str = "="):
    """Print a section header."""
    print(f"\n{char*80}")
    print(f" {title}")
    print(f"{char*80}")


def format_time(t: Optional[float]) -> str:
    """Format a time value in milliseconds."""
    if t is None:
        return "N/A"
    return f"{t*1000:.3f}ms"


def format_speedup(ratio: float) -> str:
    """Format a speedup ratio."""
    if ratio > 1.05:
        return f"{ratio:.2f}x faster"
    elif ratio < 0.95:
        return f"{1/ratio:.2f}x slower"
    return "~equal"


def count_benchmarks(size: int) -> int:
    """Count total number of benchmarks to run."""
    # Base benchmarks that always run
    count = 29  # approximate number of benchmarks
    return count


def run_all_benchmarks(iterations: int, size: int, warmup: int) -> List[BenchmarkResult]:
    """Run all benchmarks and return results."""
    results = []
    
    # Estimate total benchmarks for progress bar
    total_benchmarks = 50  # Updated for new benchmarks
    pbar = tqdm(total=total_benchmarks, desc="Running benchmarks", unit="test",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # ========================================================================
    # Basic Sum Operations
    # ========================================================================
    setup = f"import numpy as np\nu = np.ones({size}, dtype=float)"
    
    results.append(run_benchmark(
        "sum_tensor (pytensor, 1D)", setup,
        "xt.sum_tensor(u)", "xt.sum_tensor(u)",
        iterations, warmup, numpy_code="np.sum(u)", category="sum",
        pbar=pbar
    ))
    
    if HAS_PYBIND11:
        results.append(run_benchmark(
            "sum_array (pyarray)", setup,
            "xt.sum_array(u)", None,
            iterations, warmup, numpy_code="np.sum(u)", category="sum",
            pbar=pbar
        ))
        results.append(run_benchmark(
            "pybind_sum_array (native)", setup,
            "xt.pybind_sum_array(u)", None,
            iterations, warmup, category="sum",
            pbar=pbar
        ))
    
    if HAS_NANOBIND:
        results.append(run_benchmark(
            "nanobind_sum_array (native)", setup,
            None, "xt.nanobind_sum_array(u)",
            iterations, warmup, category="sum",
            pbar=pbar
        ))
    
    # ========================================================================
    # Auto Conversion (Copy from NumPy to xtensor)
    # ========================================================================
    results.append(run_benchmark(
        "auto_convert_xtensor_input", setup,
        "xt.auto_convert_xtensor_input(u)", "xt.auto_convert_xtensor_input(u)",
        iterations, warmup, category="conversion",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "auto_convert_xarray_input", setup,
        "xt.auto_convert_xarray_input(u)", "xt.auto_convert_xarray_input(u)",
        iterations, warmup, category="conversion",
        pbar=pbar
    ))
    
    # ========================================================================
    # Return Value Conversion (xtensor to NumPy)
    # ========================================================================
    setup_size = f"size = {size}"
    
    results.append(run_benchmark(
        "return_xtensor", setup_size,
        "xt.return_xtensor(size)", "xt.return_xtensor(size)",
        iterations, warmup, category="conversion",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "return_xarray", setup_size,
        "xt.return_xarray(size)", "xt.return_xarray(size)",
        iterations, warmup, category="conversion",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "return_pytensor (zero-copy)", setup_size,
        "xt.return_pytensor(size)", "xt.return_pytensor(size)",
        iterations, warmup, category="conversion",
        pbar=pbar
    ))
    if HAS_PYBIND11:
        results.append(run_benchmark(
            "return_pyarray (zero-copy)", setup_size,
            "xt.return_pyarray(size)", None,
            iterations, warmup, category="conversion",
            pbar=pbar
        ))
    
    # ========================================================================
    # Inplace Operations (By Reference)
    # ========================================================================
    setup_inplace = f"""import numpy as np
def get_array():
    return np.ones({size}, dtype=float)
"""
    
    results.append(run_benchmark(
        "inplace_multiply_pytensor",
        setup_inplace + "u = get_array()",
        "xt.inplace_multiply_pytensor(u)", "xt.inplace_multiply_pytensor(u)",
        iterations, warmup, numpy_code="u *= 2.0", category="inplace",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "inplace_add_pytensor",
        setup_inplace + "u = get_array()",
        "xt.inplace_add_pytensor(u, 2.0)", "xt.inplace_add_pytensor(u, 2.0)",
        iterations, warmup, numpy_code="u += 2.0", category="inplace",
        pbar=pbar
    ))
    if HAS_PYBIND11:
        results.append(run_benchmark(
            "inplace_multiply_pyarray", 
            setup_inplace + "u = get_array()",
            "xt.inplace_multiply_pyarray(u)", None,
            iterations, warmup, category="inplace",
            pbar=pbar
        ))
    
    # ========================================================================
    # View Operations
    # ========================================================================
    setup_view = f"import numpy as np\nu = np.ones({size}, dtype=float)"
    
    results.append(run_benchmark(
        "sum_strided_view (every 2nd)", setup_view,
        "xt.sum_strided_view(u)", "xt.sum_strided_view(u)",
        iterations, warmup, numpy_code="np.sum(u[::2])", category="view",
        pbar=pbar
    ))
    
    # ========================================================================
    # Math Operations
    # ========================================================================
    setup_math = f"import numpy as np\nu = np.random.rand({size})"
    
    results.append(run_benchmark(
        "math_on_pytensor (sin+cos)", setup_math,
        "xt.math_on_pytensor(u)", "xt.math_on_pytensor(u)",
        iterations // 10, warmup, numpy_code="np.sum(np.sin(u) + np.cos(u))", category="math",
        pbar=pbar
    ))
    if HAS_PYBIND11:
        results.append(run_benchmark(
            "math_on_pyarray (sin+cos)", setup_math,
            "xt.math_on_pyarray(u)", None,
            iterations // 10, warmup, numpy_code="np.sum(np.sin(u) + np.cos(u))", category="math",
            pbar=pbar
        ))
    results.append(run_benchmark(
        "math_on_xtensor (with copy)", setup_math,
        "xt.math_on_xtensor(u)", "xt.math_on_xtensor(u)",
        iterations // 10, warmup, numpy_code="np.sum(np.sin(u) + np.cos(u))", category="math",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "math_on_xarray (with copy)", setup_math,
        "xt.math_on_xarray(u)", "xt.math_on_xarray(u)",
        iterations // 10, warmup, category="math",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "math_on_view_pytensor (strided)", setup_math,
        "xt.math_on_view_pytensor(u)", "xt.math_on_view_pytensor(u)",
        iterations // 10, warmup, numpy_code="np.sum(np.sin(u[::2]) + np.cos(u[::2]))", category="math",
        pbar=pbar
    ))
    
    # ========================================================================
    # Full Round-trip
    # ========================================================================
    setup_round = f"import numpy as np\nu = np.ones({size}, dtype=float)"
    
    results.append(run_benchmark(
        "roundtrip_pytensor (zero-copy)", setup_round,
        "xt.roundtrip_pytensor(u)", "xt.roundtrip_pytensor(u)",
        iterations, warmup, numpy_code="u * 2.0 + 1.0", category="roundtrip",
        pbar=pbar
    ))
    if HAS_PYBIND11:
        results.append(run_benchmark(
            "roundtrip_pyarray (zero-copy)", setup_round,
            "xt.roundtrip_pyarray(u)", None,
            iterations, warmup, category="roundtrip",
            pbar=pbar
        ))
    results.append(run_benchmark(
        "roundtrip_xtensor (with copy)", setup_round,
        "xt.roundtrip_xtensor(u)", "xt.roundtrip_xtensor(u)",
        iterations, warmup, category="roundtrip",
        pbar=pbar
    ))
    
    # ========================================================================
    # Large 2D Array Processing
    # ========================================================================
    large_size = int(np.sqrt(size))
    setup_large = f"import numpy as np\nu = np.ones(({large_size}, {large_size}), dtype=float)"
    
    results.append(run_benchmark(
        f"process_large_pytensor ({large_size}x{large_size})", setup_large,
        "xt.process_large_pytensor(u)", "xt.process_large_pytensor(u)",
        iterations, warmup, numpy_code="np.sum(u)", category="large",
        pbar=pbar
    ))
    if HAS_PYBIND11:
        results.append(run_benchmark(
            f"process_large_pyarray ({large_size}x{large_size})", setup_large,
            "xt.process_large_pyarray(u)", None,
            iterations, warmup, category="large",
            pbar=pbar
        ))
    results.append(run_benchmark(
        f"process_large_xtensor ({large_size}x{large_size}, copy)", setup_large,
        "xt.process_large_xtensor(u)", "xt.process_large_xtensor(u)",
        iterations // 10, warmup, category="large",
        pbar=pbar
    ))
    
    # ========================================================================
    # Vectorize Operations
    # ========================================================================
    setup_vec = f"import numpy as np\nw = np.ones({size // 10}, dtype=complex)"
    
    results.append(run_benchmark(
        "rect_to_polar (pyvectorize)", setup_vec,
        "xt.rect_to_polar(w)", "xt.rect_to_polar(w)",
        iterations, warmup, numpy_code="np.abs(w)", category="vectorize",
        pbar=pbar
    ))
    
    setup_vec_strided = f"import numpy as np\nw = np.ones({size // 5}, dtype=complex)"
    results.append(run_benchmark(
        "rect_to_polar (strided)", setup_vec_strided,
        "xt.rect_to_polar(w[::2])", "xt.rect_to_polar(w[::2])",
        iterations, warmup, numpy_code="np.abs(w[::2])", category="vectorize",
        pbar=pbar
    ))
    if HAS_PYBIND11:
        results.append(run_benchmark(
            "pybind_rect_to_polar (native)", setup_vec_strided,
            "xt.pybind_rect_to_polar(w[::2])", None,
            iterations, warmup, category="vectorize",
            pbar=pbar
        ))
    
    # ========================================================================
    # Type Conversion
    # ========================================================================
    setup_int32 = f"import numpy as np\nu = np.ones({size}, dtype=np.int32)"
    results.append(run_benchmark(
        "type_convert_int32_to_double", setup_int32,
        "xt.type_convert_int32_to_double(u)", "xt.type_convert_int32_to_double(u)",
        iterations, warmup, category="type",
        pbar=pbar
    ))
    
    setup_float32 = f"import numpy as np\nu = np.ones({size}, dtype=np.float32)"
    results.append(run_benchmark(
        "type_convert_float32_to_double", setup_float32,
        "xt.type_convert_float32_to_double(u)", "xt.type_convert_float32_to_double(u)",
        iterations, warmup, category="type",
        pbar=pbar
    ))
    
    setup_double = f"import numpy as np\nu = np.ones({size}, dtype=np.float64)"
    results.append(run_benchmark(
        "type_no_convert_double", setup_double,
        "xt.type_no_convert_double(u)", "xt.type_no_convert_double(u)",
        iterations, warmup, category="type",
        pbar=pbar
    ))
    
    # ========================================================================
    # Broadcasting Operations
    # ========================================================================
    setup_broadcast = f"import numpy as np\nu = np.ones({size}, dtype=float)"
    results.append(run_benchmark(
        "broadcast_scalar_add", setup_broadcast,
        "xt.broadcast_scalar_add(u, 2.5)", "xt.broadcast_scalar_add(u, 2.5)",
        iterations, warmup, numpy_code="u + 2.5", category="broadcast",
        pbar=pbar
    ))
    
    broadcast_size = 1000
    setup_broadcast_2d = f"""import numpy as np
u = np.ones(({broadcast_size}, {broadcast_size}), dtype=float)
row = np.arange({broadcast_size}, dtype=float)
"""
    results.append(run_benchmark(
        f"broadcast_1d_to_2d ({broadcast_size}x{broadcast_size})", setup_broadcast_2d,
        "xt.broadcast_1d_to_2d(u, row)", "xt.broadcast_1d_to_2d(u, row)",
        iterations // 10, warmup, numpy_code="u + row", category="broadcast",
        pbar=pbar
    ))
    results.append(run_benchmark(
        f"broadcast_and_reduce ({broadcast_size}x{broadcast_size})", setup_broadcast_2d,
        "xt.broadcast_and_reduce(u, row)", "xt.broadcast_and_reduce(u, row)",
        iterations // 10, warmup, numpy_code="np.sum(u * row)", category="broadcast",
        pbar=pbar
    ))
    
    # ========================================================================
    # Slicing Operations
    # ========================================================================
    setup_slice = f"import numpy as np\nu = np.arange({size}, dtype=float)"
    slice_start, slice_end = size // 4, 3 * size // 4
    
    results.append(run_benchmark(
        "slice_contiguous (middle 50%)", setup_slice,
        f"xt.slice_contiguous(u, {slice_start}, {slice_end})",
        f"xt.slice_contiguous(u, {slice_start}, {slice_end})",
        iterations, warmup, numpy_code=f"np.sum(u[{slice_start}:{slice_end}])", category="slice",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "slice_strided (every 4th)", setup_slice,
        "xt.slice_strided(u, 4)", "xt.slice_strided(u, 4)",
        iterations, warmup, numpy_code="np.sum(u[::4])", category="slice",
        pbar=pbar
    ))
    
    slice_2d_size = 1000
    setup_slice_2d = f"import numpy as np\nu = np.arange({slice_2d_size * slice_2d_size}, dtype=float).reshape({slice_2d_size}, {slice_2d_size})"
    results.append(run_benchmark(
        f"slice_2d_submatrix (500x500 of {slice_2d_size}x{slice_2d_size})", setup_slice_2d,
        "xt.slice_2d_submatrix(u, 250, 750, 250, 750)",
        "xt.slice_2d_submatrix(u, 250, 750, 250, 750)",
        iterations, warmup, numpy_code="np.sum(u[250:750, 250:750])", category="slice",
        pbar=pbar
    ))
    
    setup_slice_modify = f"""import numpy as np
def get_array():
    return np.ones({size}, dtype=float)
u = get_array()
"""
    results.append(run_benchmark(
        "slice_and_modify (inplace)", setup_slice_modify,
        f"xt.slice_and_modify(u, {slice_start}, {slice_end}, 1.0)",
        f"xt.slice_and_modify(u, {slice_start}, {slice_end}, 1.0)",
        iterations, warmup, numpy_code=f"u[{slice_start}:{slice_end}].__iadd__(1.0)", category="slice",
        pbar=pbar
    ))
    
    # ========================================================================
    # Pure C++ Benchmarks (pytensor vs native xtensor)
    # ========================================================================
    cpp_size = 100000
    setup_cpp = f"""import numpy as np
u = np.ones({cpp_size}, dtype=float)
u2d = np.ones((1000, 1000), dtype=float)
row = np.arange(1000, dtype=float)
"""
    
    results.append(run_benchmark(
        f"cpp_native_xtensor_math ({cpp_size})",
        f"size = {cpp_size}",
        "xt.cpp_native_xtensor_math(size)", "xt.cpp_native_xtensor_math(size)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        f"cpp_pytensor_math ({cpp_size})", setup_cpp,
        "xt.cpp_pytensor_math(u)", "xt.cpp_pytensor_math(u)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "cpp_native_xtensor_2d_sum (1000x1000)",
        "rows, cols = 1000, 1000",
        "xt.cpp_native_xtensor_2d_sum(rows, cols)", "xt.cpp_native_xtensor_2d_sum(rows, cols)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "cpp_pytensor_2d_sum (1000x1000)", setup_cpp,
        "xt.cpp_pytensor_2d_sum(u2d)", "xt.cpp_pytensor_2d_sum(u2d)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        f"cpp_native_strided_view_sum ({cpp_size})",
        f"size = {cpp_size}",
        "xt.cpp_native_strided_view_sum(size)", "xt.cpp_native_strided_view_sum(size)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        f"cpp_pytensor_strided_view_sum ({cpp_size})", setup_cpp,
        "xt.cpp_pytensor_strided_view_sum(u)", "xt.cpp_pytensor_strided_view_sum(u)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "cpp_native_broadcast_reduce (1000x1000)",
        "rows, cols = 1000, 1000",
        "xt.cpp_native_broadcast_reduce(rows, cols)", "xt.cpp_native_broadcast_reduce(rows, cols)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "cpp_pytensor_broadcast_reduce (1000x1000)", setup_cpp,
        "xt.cpp_pytensor_broadcast_reduce(u2d, row)", "xt.cpp_pytensor_broadcast_reduce(u2d, row)",
        iterations // 10, warmup, category="cpp_pure",
        pbar=pbar
    ))
    
    # ========================================================================
    # No-convert Benchmarks (ensure no dtype conversion overhead)
    # ========================================================================
    setup_noconv = f"import numpy as np\nu = np.ones({size}, dtype=np.float64)"
    setup_noconv_2d = f"import numpy as np\nu = np.ones((1000, 1000), dtype=np.float64)"
    
    results.append(run_benchmark(
        "noconvert_sum_tensor (1D)", setup_noconv,
        "xt.noconvert_sum_tensor(u)", "xt.noconvert_sum_tensor(u)",
        iterations, warmup, numpy_code="np.sum(u)", category="noconvert",
        pbar=pbar
    ))
    results.append(run_benchmark(
        "noconvert_sum_tensor_2d", setup_noconv_2d,
        "xt.noconvert_sum_tensor_2d(u)", "xt.noconvert_sum_tensor_2d(u)",
        iterations, warmup, numpy_code="np.sum(u)", category="noconvert",
        pbar=pbar
    ))
    setup_noconv_inplace = f"""import numpy as np
def get_array():
    return np.ones({size}, dtype=np.float64)
u = get_array()
"""
    results.append(run_benchmark(
        "noconvert_inplace_multiply", setup_noconv_inplace,
        "xt.noconvert_inplace_multiply(u)", "xt.noconvert_inplace_multiply(u)",
        iterations, warmup, numpy_code="u.__imul__(2.0)", category="noconvert",
        pbar=pbar
    ))
    setup_noconv_math = f"import numpy as np\nu = np.random.rand({size})"
    results.append(run_benchmark(
        "noconvert_math (sin+cos)", setup_noconv_math,
        "xt.noconvert_math(u)", "xt.noconvert_math(u)",
        iterations // 10, warmup, numpy_code="np.sum(np.sin(u) + np.cos(u))", category="noconvert",
        pbar=pbar
    ))
    
    pbar.close()
    return results


def verify_reference_semantics():
    """Test that pytensor correctly references (not copies) numpy arrays."""
    print("\n" + "="*80)
    print(" REFERENCE SEMANTICS VERIFICATION")
    print("="*80)
    
    import numpy as np
    
    tests_passed = 0
    tests_total = 0
    
    # Test pybind11
    if HAS_PYBIND11:
        tests_total += 1
        arr = np.zeros((3, 4), dtype=np.float64)
        xt_pybind11.reference_test_modify(arr)
        
        # Check expected pattern: arr[i,j] = i * 1000 + j
        expected = np.array([[i * 1000 + j for j in range(4)] for i in range(3)], dtype=np.float64)
        if np.allclose(arr, expected):
            print("  [OK] pybind11: pytensor correctly references numpy array (not copied)")
            tests_passed += 1
        else:
            print("  [FAIL] pybind11: FAILED - array was copied, not referenced!")
            print(f"    Expected:\n{expected}")
            print(f"    Got:\n{arr}")
    
    # Test nanobind
    if HAS_NANOBIND:
        tests_total += 1
        arr = np.zeros((3, 4), dtype=np.float64)
        xt_nanobind.reference_test_modify(arr)
        
        # Check expected pattern: arr[i,j] = i * 1000 + j
        expected = np.array([[i * 1000 + j for j in range(4)] for i in range(3)], dtype=np.float64)
        if np.allclose(arr, expected):
            print("  [OK] nanobind: pytensor correctly references numpy array (not copied)")
            tests_passed += 1
        else:
            print("  [FAIL] nanobind: FAILED - array was copied, not referenced!")
            print(f"    Expected:\n{expected}")
            print(f"    Got:\n{arr}")
    
    print(f"\nReference tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def print_pybind_vs_nanobind(results: List[BenchmarkResult]):
    """Print pybind11 vs nanobind comparison section."""
    print_header("SECTION 1: PYBIND11 vs NANOBIND (binding framework comparison)")
    
    # Filter results that have both measurements
    comparable = [r for r in results if r.pybind11_time and r.nanobind_time]
    
    if not comparable:
        print("\nNo comparable benchmarks (need both pybind11 and nanobind)")
        return
    
    print("\nNote: xarray auto-conversion is slower in nanobind (~1.8x) due to type caster differences.")
    print("      This affects copy operations only; pytensor zero-copy operations are unaffected.")
    
    # Group by category
    categories = {}
    for r in comparable:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    # Print table
    print(f"\n{'Benchmark':<45} {'pybind11':>12} {'nanobind':>12} {'Winner':>12}")
    print("-" * 83)
    
    stats = {"pybind11": 0, "nanobind": 0, "lean pb": 0, "lean nb": 0}
    
    for category, cat_results in categories.items():
        print(f"\n[{category}]")
        for r in cat_results:
            pb = format_time(r.pybind11_time)
            nb = format_time(r.nanobind_time)
            winner = r.get_winner()
            
            # Colorize winner
            if winner == "nanobind":
                ratio = r.pybind11_time / r.nanobind_time
                winner_str = f"nb {ratio:.2f}x *"
            elif winner == "pybind11":
                ratio = r.nanobind_time / r.pybind11_time
                winner_str = f"pb {ratio:.2f}x *"
            elif winner == "lean nb":
                winner_str = "lean nb"
            else:
                winner_str = "lean pb"
            
            stats[winner] += 1
            print(f"  {r.name:<43} {pb:>12} {nb:>12} {winner_str:>12}")
    
    # Print summary
    print("\n" + "-" * 83)
    total = sum(stats.values())
    print(f"\nSummary ({total} benchmarks):")
    print(f"  nanobind faster (>7%): {stats['nanobind']:>3} ({100*stats['nanobind']/total:.0f}%)")
    print(f"  pybind11 faster (>7%): {stats['pybind11']:>3} ({100*stats['pybind11']/total:.0f}%)")
    lean_total = stats['lean pb'] + stats['lean nb']
    print(f"  within 7% margin:      {lean_total:>3} ({100*lean_total/total:.0f}%) [lean pb: {stats['lean pb']}, lean nb: {stats['lean nb']}]")


def print_xtensor_vs_numpy(results: List[BenchmarkResult]):
    """Print xtensor vs NumPy comparison section (using faster binding framework)."""
    print_header("SECTION 2: XTENSOR vs NUMPY (using best available framework)")
    
    # Filter results that have numpy comparison
    with_numpy = [r for r in results if r.numpy_time is not None]
    
    if not with_numpy:
        print("\nNo benchmarks with NumPy comparison")
        return
    
    # Determine which framework is generally faster
    if HAS_PYBIND11 and HAS_NANOBIND:
        comparable = [r for r in results if r.pybind11_time and r.nanobind_time]
        pb_wins = sum(1 for r in comparable if r.get_winner() == "pybind11")
        nb_wins = sum(1 for r in comparable if r.get_winner() == "nanobind")
        preferred_framework = "nanobind" if nb_wins >= pb_wins else "pybind11"
        print(f"\nUsing: {preferred_framework} (faster in {max(nb_wins, pb_wins)}/{len(comparable)} tests)")
    elif HAS_NANOBIND:
        preferred_framework = "nanobind"
        print("\nUsing: nanobind (only available)")
    else:
        preferred_framework = "pybind11"
        print("\nUsing: pybind11 (only available)")
    
    # Note about comparison fairness
    print("\nNote: These benchmarks test element-wise operations (sum, sin, cos), not BLAS.")
    print("      Both NumPy and xtensor use SIMD (AVX/AVX-512) for these operations.")
    print("      Performance differences are due to implementation details and overhead.")
    
    # Group by category
    categories = {}
    for r in with_numpy:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)
    
    # Print table
    print(f"\n{'Benchmark':<45} {'xtensor':>12} {'NumPy':>12} {'Comparison':>15}")
    print("-" * 86)
    
    stats = {"xtensor": 0, "numpy": 0, "lean xt": 0, "lean np": 0}
    
    for category, cat_results in categories.items():
        print(f"\n[{category}]")
        for r in cat_results:
            # Get the best xtensor time (or the one from preferred framework)
            if preferred_framework == "nanobind" and r.nanobind_time:
                xt_time = r.nanobind_time
            elif preferred_framework == "pybind11" and r.pybind11_time:
                xt_time = r.pybind11_time
            else:
                xt_time = r.get_best_xtensor_time()
            
            xt = format_time(xt_time)
            npy = format_time(r.numpy_time)
            
            # Calculate comparison
            if xt_time and r.numpy_time:
                ratio = r.numpy_time / xt_time
                if ratio > 1.07:
                    cmp_str = f"xt {ratio:.1f}x faster *"
                    stats["xtensor"] += 1
                elif ratio < 0.93:
                    cmp_str = f"np {1/ratio:.1f}x faster"
                    stats["numpy"] += 1
                elif ratio >= 1.0:
                    cmp_str = "lean xt"
                    stats["lean xt"] += 1
                else:
                    cmp_str = "lean np"
                    stats["lean np"] += 1
            else:
                cmp_str = "N/A"
            
            print(f"  {r.name:<43} {xt:>12} {npy:>12} {cmp_str:>15}")
    
    # Print summary
    print("\n" + "-" * 86)
    total = sum(stats.values())
    if total > 0:
        print(f"\nSummary ({total} benchmarks):")
        print(f"  xtensor faster (>7%): {stats['xtensor']:>3} ({100*stats['xtensor']/total:.0f}%)")
        print(f"  numpy faster (>7%):   {stats['numpy']:>3} ({100*stats['numpy']/total:.0f}%)")
        lean_total = stats['lean xt'] + stats['lean np']
        print(f"  within 7% margin:     {lean_total:>3} ({100*lean_total/total:.0f}%) [lean xt: {stats['lean xt']}, lean np: {stats['lean np']}]")


def main():
    parser = argparse.ArgumentParser(description="Benchmark xtensor-python bindings")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                        help="Number of iterations per benchmark")
    parser.add_argument("--size", "-s", type=int, default=1000000,
                        help="Array size for benchmarks")
    parser.add_argument("--warmup", "-w", type=int, default=3,
                        help="Number of warmup iterations")
    args = parser.parse_args()
    
    iterations = args.iterations
    size = args.size
    warmup = args.warmup
    
    print("\n" + "="*80)
    print(" XTENSOR-PYTHON BENCHMARK SUITE")
    print("="*80)
    print(f"\nSettings:")
    print(f"  Array size: {size:,}")
    print(f"  Iterations: {iterations:,}")
    print(f"  Warmup: {warmup}")
    
    # Run reference semantics verification first
    ref_ok = verify_reference_semantics()
    if not ref_ok:
        print("\n[!] WARNING: Reference semantics test FAILED!")
        print("  pytensor may be copying arrays instead of referencing them.")
    
    # Run all benchmarks
    print("\nRunning benchmarks...")
    results = run_all_benchmarks(iterations, size, warmup)
    
    # Print Section 1: pybind11 vs nanobind
    if HAS_PYBIND11 and HAS_NANOBIND:
        print_pybind_vs_nanobind(results)
    
    # Print Section 2: xtensor vs NumPy
    print_xtensor_vs_numpy(results)
    
    print("\n" + "="*80)
    print(" BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
