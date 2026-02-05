#!/usr/bin/env python3
"""
Test that noconvert is working correctly for pybind11 and nanobind.

When noconvert is enabled, passing an array with the wrong dtype should raise
a TypeError (or similar exception), not silently convert the data.
"""

import numpy as np
import pytest
import sys


def test_pybind11_noconvert():
    """Test that pybind11 noconvert correctly rejects wrong dtypes."""
    try:
        import benchmark_xtensor_python as pb
    except ImportError:
        pytest.skip("benchmark_xtensor_python not available")
    
    # Create arrays with correct and wrong dtypes
    correct_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    wrong_dtype_int = np.array([1, 2, 3], dtype=np.int32)
    wrong_dtype_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # These should work (correct dtype)
    result = pb.noconvert_sum_tensor(correct_array)
    assert abs(result - 6.0) < 1e-10, f"Expected 6.0, got {result}"
    
    # These should raise TypeError (wrong dtype with noconvert)
    with pytest.raises(TypeError):
        pb.noconvert_sum_tensor(wrong_dtype_int)
    
    with pytest.raises(TypeError):
        pb.noconvert_sum_tensor(wrong_dtype_float32)
    
    # Test 2D as well
    correct_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    wrong_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
    
    result = pb.noconvert_sum_tensor_2d(correct_2d)
    assert abs(result - 10.0) < 1e-10
    
    with pytest.raises(TypeError):
        pb.noconvert_sum_tensor_2d(wrong_2d)
    
    print("✓ pybind11 noconvert tests passed")


def test_nanobind_noconvert():
    """Test that nanobind noconvert correctly rejects wrong dtypes."""
    try:
        import benchmark_xtensor_nanobind as nb
    except ImportError:
        pytest.skip("benchmark_xtensor_nanobind not available")
    
    # Create arrays with correct and wrong dtypes
    correct_array = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    wrong_dtype_int = np.array([1, 2, 3], dtype=np.int32)
    wrong_dtype_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    
    # These should work (correct dtype)
    result = nb.noconvert_sum_tensor(correct_array)
    assert abs(result - 6.0) < 1e-10, f"Expected 6.0, got {result}"
    
    # These should raise TypeError (wrong dtype with noconvert)
    with pytest.raises(TypeError):
        nb.noconvert_sum_tensor(wrong_dtype_int)
    
    with pytest.raises(TypeError):
        nb.noconvert_sum_tensor(wrong_dtype_float32)
    
    # Test 2D as well
    correct_2d = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    wrong_2d = np.array([[1, 2], [3, 4]], dtype=np.int32)
    
    result = nb.noconvert_sum_tensor_2d(correct_2d)
    assert abs(result - 10.0) < 1e-10
    
    with pytest.raises(TypeError):
        nb.noconvert_sum_tensor_2d(wrong_2d)
    
    print("✓ nanobind noconvert tests passed")


def test_pybind11_noconvert_inplace():
    """Test that inplace operations with noconvert reject wrong dtypes."""
    try:
        import benchmark_xtensor_python as pb
    except ImportError:
        pytest.skip("benchmark_xtensor_python not available")
    
    correct = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    wrong = np.array([1, 2, 3], dtype=np.int32)
    
    # Should work
    pb.noconvert_inplace_multiply(correct)
    np.testing.assert_array_almost_equal(correct, [2.0, 4.0, 6.0])
    
    # Should fail
    with pytest.raises(TypeError):
        pb.noconvert_inplace_multiply(wrong)
    
    print("✓ pybind11 noconvert inplace tests passed")


def test_nanobind_noconvert_inplace():
    """Test that inplace operations with noconvert reject wrong dtypes."""
    try:
        import benchmark_xtensor_nanobind as nb
    except ImportError:
        pytest.skip("benchmark_xtensor_nanobind not available")
    
    correct = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    wrong = np.array([1, 2, 3], dtype=np.int32)
    
    # Should work
    nb.noconvert_inplace_multiply(correct)
    np.testing.assert_array_almost_equal(correct, [2.0, 4.0, 6.0])
    
    # Should fail
    with pytest.raises(TypeError):
        nb.noconvert_inplace_multiply(wrong)
    
    print("✓ nanobind noconvert inplace tests passed")


def test_pybind11_noconvert_math():
    """Test math operations with noconvert."""
    try:
        import benchmark_xtensor_python as pb
    except ImportError:
        pytest.skip("benchmark_xtensor_python not available")
    
    correct = np.array([0.0, np.pi/2, np.pi], dtype=np.float64)
    wrong = np.array([0, 1, 2], dtype=np.int32)
    
    # Should work
    result = pb.noconvert_math(correct)
    expected = np.sum(np.sin(correct) + np.cos(correct))
    assert abs(result - expected) < 1e-10
    
    # Should fail
    with pytest.raises(TypeError):
        pb.noconvert_math(wrong)
    
    print("✓ pybind11 noconvert math tests passed")


def test_nanobind_noconvert_math():
    """Test math operations with noconvert."""
    try:
        import benchmark_xtensor_nanobind as nb
    except ImportError:
        pytest.skip("benchmark_xtensor_nanobind not available")
    
    correct = np.array([0.0, np.pi/2, np.pi], dtype=np.float64)
    wrong = np.array([0, 1, 2], dtype=np.int32)
    
    # Should work
    result = nb.noconvert_math(correct)
    expected = np.sum(np.sin(correct) + np.cos(correct))
    assert abs(result - expected) < 1e-10
    
    # Should fail
    with pytest.raises(TypeError):
        nb.noconvert_math(wrong)
    
    print("✓ nanobind noconvert math tests passed")


def run_all_tests():
    """Run all noconvert tests and report results."""
    tests = [
        test_pybind11_noconvert,
        test_nanobind_noconvert,
        test_pybind11_noconvert_inplace,
        test_nanobind_noconvert_inplace,
        test_pybind11_noconvert_math,
        test_nanobind_noconvert_math,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⊘ {test.__name__}: SKIPPED - {e}")
            skipped += 1
        except Exception as e:
            print(f"✗ {test.__name__}: FAILED - {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    # Allow running with pytest or standalone
    if "pytest" in sys.modules:
        # Running under pytest - tests will be discovered automatically
        pass
    else:
        # Running standalone
        success = run_all_tests()
        sys.exit(0 if success else 1)
