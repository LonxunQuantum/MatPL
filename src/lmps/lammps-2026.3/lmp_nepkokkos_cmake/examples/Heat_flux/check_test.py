# -*- coding: utf-8 -*-
"""
Validate MIX-precision Heat_flux test results against FP64 reference.

Compares the accumulated heat flux curve (from compute_HeatFlux.out) between
the test run and the FP64 reference. The compute_Energy_Temp.out is expected
to have some deviation under MIX precision, so only HeatFlux is checked.
"""
import sys
import os
import numpy as np

def load_heatflux(path):
    """Load compute_HeatFlux.out and return accumulated heat flux curve."""
    start = 500
    dt = 0.001  # ps
    Ns = 1000   # Sample interval
    BLOCK_LENGTH = 426

    jp = np.loadtxt(os.path.join(path, "compute_HeatFlux.out"))
    jp = jp[start:]
    t = dt * np.arange(1, len(jp) + 1) * Ns / 1000  # ns

    jpy = jp[:, 2] - jp[:, 5]
    jpy = jpy / BLOCK_LENGTH / 10 * 1000  # eV/ns
    accum_jpy = np.cumsum(jpy) * 0.001 / 1000  # keV
    return t, -accum_jpy

def check_test(test_path, ref_path, rtol=0.05, r2_min=0.99):
    """
    Compare MIX-precision heat flux against FP64 reference.

    Parameters:
        test_path: directory containing test compute_HeatFlux.out
        ref_path:  directory containing FP64 reference compute_HeatFlux.out
        rtol:      relative tolerance for max error (default 5%)
        r2_min:    minimum R-squared (default 0.99)

    Returns:
        True if test passes, False otherwise.
    """
    t_test, hf_test = load_heatflux(test_path)
    t_ref, hf_ref = load_heatflux(ref_path)

    # Align lengths
    n = min(len(hf_test), len(hf_ref))
    hf_test = hf_test[:n]
    hf_ref = hf_ref[:n]
    t = t_ref[:n]

    # Compare in [0, 1.5] ns range
    mask = t <= 1.5
    test_range = hf_test[mask]
    ref_range = hf_ref[mask]

    ref_max = np.max(np.abs(ref_range))
    if ref_max < 1e-10:
        print("FAIL: Reference data is essentially zero.")
        return False

    # Metrics
    diff = np.abs(test_range - ref_range)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_max_err = max_diff / ref_max

    ss_res = np.sum((test_range - ref_range) ** 2)
    ss_tot = np.sum((ref_range - np.mean(ref_range)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"=== Heat Flux Validation (MIX vs FP64) ===")
    print(f"Data points compared: {np.sum(mask)}")
    print(f"Max absolute difference: {max_diff:.6f} keV")
    print(f"Mean absolute difference: {mean_diff:.6f} keV")
    print(f"Relative max error: {rel_max_err:.4f} ({rel_max_err*100:.2f}%)")
    print(f"R-squared: {r_squared:.6f}")

    passed = True
    reasons = []

    if rel_max_err > rtol:
        passed = False
        reasons.append(f"Relative max error {rel_max_err*100:.2f}% exceeds {rtol*100:.1f}%")

    if r_squared < r2_min:
        passed = False
        reasons.append(f"R-squared {r_squared:.6f} < {r2_min}")

    if passed:
        print(f"\nRESULT: PASS")
    else:
        print(f"\nRESULT: FAIL")
        for r in reasons:
            print(f"  - {r}")

    return passed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_test.py <test_dir> [ref_dir]")
        print("  test_dir: directory with MIX-precision results")
        print("  ref_dir:  directory with FP64 reference (default: current dir)")
        sys.exit(1)

    test_dir = sys.argv[1]
    ref_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    success = check_test(test_dir, ref_dir)
    sys.exit(0 if success else 1)
