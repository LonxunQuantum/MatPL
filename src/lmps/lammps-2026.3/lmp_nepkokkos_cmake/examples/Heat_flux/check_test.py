# -*- coding: utf-8 -*-
import sys
import numpy as np

def lammps_heatflux(path):
    start = 500
    dt = 0.001  # ps
    Ns = 1000   # Sample interval
    thermo = np.loadtxt(path + "/compute_Energy_Temp.out")
    jp = np.loadtxt(path + "/compute_HeatFlux.out")
    BLOCK_LENGTH = 426
    Ein = thermo[start:, 1]
    Eout = thermo[start:, 2]
    Etol = (Eout - Ein) / 2 / 1000  # in units of KeV
    Etol = Etol - Etol[0]
    t = dt * np.arange(1, len(Etol) + 1) * Ns / 1000  # unit in ns
    jpy = jp[start:, 2] - jp[start:, 5]
    jpy = jpy / BLOCK_LENGTH / 10 * 1000  # in units of eV/ns
    accum_jpy = np.cumsum(jpy) * 0.001 / 1000  # in units of KeV
    return t, accum_jpy, Etol

def check_test(path=".", rtol=0.05, atol=0.02):
    """
    Check if heat flux from atoms and from thermostats are consistent.

    Parameters:
        path: directory containing the .out files
        rtol: relative tolerance (default 5%)
        atol: absolute tolerance in keV (default 0.02 keV)

    Returns:
        True if test passes, False otherwise.
    """
    t, jp, etol = lammps_heatflux(path)

    jp_neg = -1 * jp

    # Compare in the plotted range [0, 1.5] ns
    mask = t <= 1.5
    jp_range = jp_neg[mask]
    etol_range = etol[mask]

    # Use the thermostat curve as reference
    ref_max = np.max(np.abs(etol_range))
    if ref_max < 1e-10:
        print("FAIL: Reference data is essentially zero.")
        return False

    # Compute differences
    diff = np.abs(jp_range - etol_range)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # R-squared (coefficient of determination)
    ss_res = np.sum((jp_range - etol_range) ** 2)
    ss_tot = np.sum((etol_range - np.mean(etol_range)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Relative max error
    rel_max_err = max_diff / ref_max

    print(f"Max absolute difference: {max_diff:.6f} keV")
    print(f"Mean absolute difference: {mean_diff:.6f} keV")
    print(f"Relative max error: {rel_max_err:.4f} ({rel_max_err*100:.2f}%)")
    print(f"R-squared: {r_squared:.6f}")

    passed = True
    reasons = []

    if rel_max_err > rtol:
        passed = False
        reasons.append(f"Relative max error {rel_max_err:.4f} exceeds tolerance {rtol}")

    if max_diff > atol * ref_max + atol:
        passed = False
        reasons.append(f"Max absolute difference {max_diff:.6f} exceeds tolerance")

    if r_squared < 0.95:
        passed = False
        reasons.append(f"R-squared {r_squared:.6f} < 0.95, curves do not match well")

    if passed:
        print("\nRESULT: PASS")
    else:
        print("\nRESULT: FAIL")
        for r in reasons:
            print(f"  - {r}")

    return passed

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "."
    success = check_test(path)
    sys.exit(0 if success else 1)
