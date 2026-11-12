#!/usr/bin/env python3
"""Compare GPUMD ``nep`` prediction outputs with MatPL ``test`` outputs.

Usage: python check_gpumd_matpl.py GPUMD_TEST_DIR TEST_RESULT_DIR
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


COMPONENTS = ("x", "y", "z")
VIRIAL_COMPONENTS = ("xx", "xy", "xz", "yy", "yz", "zz")
# GPUMD writes xx, yy, zz, xy, yz, zx; MatPL writes xx, xy, xz, yy, yz, zz.
GPUMD_VIRIAL_TO_MATPL = (0, 3, 5, 1, 4, 2)


def read_matrix(path: Path, columns: int) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2)
    if data.shape[0] == 0 or data.shape[1] != columns:
        raise ValueError(f"{path}: expected a nonempty table with {columns} columns, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: contains NaN or infinity")
    return data


def read_atom_counts(path: Path) -> np.ndarray:
    counts = []
    with path.open(encoding="utf-8") as xyz:
        while (header := xyz.readline()):
            if not header.strip():
                continue
            try:
                count = int(header.strip())
            except ValueError as exc:
                raise ValueError(f"{path}: invalid XYZ atom count in image {len(counts) + 1}") from exc
            if count <= 0 or not xyz.readline():
                raise ValueError(f"{path}: incomplete XYZ image {len(counts) + 1}")
            for _ in range(count):
                if not xyz.readline():
                    raise ValueError(f"{path}: incomplete XYZ image {len(counts) + 1}")
            counts.append(count)
    if not counts:
        raise ValueError(f"{path}: no XYZ images found")
    return np.asarray(counts)


def require_rows(name: str, actual: int, expected: int) -> None:
    if actual != expected:
        raise ValueError(f"{name}: {actual} rows, expected {expected}; cannot compare row by row")


def report_difference(
    name: str, matpl: np.ndarray, gpumd: np.ndarray, unit: str,
    row_name: str = "image", components: Optional[Tuple[str, ...]] = None,
) -> None:
    error = matpl - gpumd
    absolute = np.abs(error)
    location = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    component = f", {components[location[1]]}" if components else ""
    print(f"{name}: max abs = {absolute[location]:.9g} {unit} "
          f"({row_name} {location[0] + 1}{component})")
    print(f"  MatPL = {matpl[location]:.9g}, GPUMD = {gpumd[location]:.9g}")
    print(f"  MAE = {absolute.mean():.9g}, RMSE = {np.sqrt(np.mean(error**2)):.9g} {unit}")


def compare(gpumd_dir: Path, matpl_dir: Path) -> None:
    counts = read_atom_counts(gpumd_dir / "train.xyz")
    images = len(counts)
    atoms = int(counts.sum())

    gpumd_force = read_matrix(gpumd_dir / "force_train.out", 6)
    matpl_force = read_matrix(matpl_dir / "inference_force.txt", 3)
    dft_force = read_matrix(matpl_dir / "dft_force.txt", 3)
    for name, data in (("GPUMD force", gpumd_force), ("MatPL force", matpl_force), ("DFT force", dft_force)):
        require_rows(name, len(data), atoms)

    gpumd_energy = read_matrix(gpumd_dir / "energy_train.out", 2)
    matpl_energy = read_matrix(matpl_dir / "inference_total_energy.txt", 1)
    dft_energy = read_matrix(matpl_dir / "dft_total_energy.txt", 1)
    gpumd_virial = read_matrix(gpumd_dir / "virial_train.out", 12)
    matpl_virial = read_matrix(matpl_dir / "inference_virial.txt", 6)
    dft_virial = read_matrix(matpl_dir / "dft_virial.txt", 6)
    for name, data in (
        ("GPUMD energy", gpumd_energy), ("MatPL energy", matpl_energy), ("DFT energy", dft_energy),
        ("GPUMD virial", gpumd_virial), ("MatPL virial", matpl_virial), ("DFT virial", dft_virial),
    ):
        require_rows(name, len(data), images)

    print(f"Images: {images}; atoms: {atoms}")
    force_error = matpl_force - gpumd_force[:, :3]
    absolute = np.abs(force_error)
    atom, component = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
    vector_errors = np.linalg.norm(force_error, axis=1)
    vector_atom = int(np.argmax(vector_errors))
    print(f"Force: max component abs = {absolute[atom, component]:.9g} eV/Angstrom "
          f"(atom {atom + 1}, {COMPONENTS[component]})")
    print(f"  MatPL = {matpl_force[atom, component]:.9g}, GPUMD = {gpumd_force[atom, component]:.9g}")
    print(f"  Max atom vector norm = {vector_errors[vector_atom]:.9g} eV/Angstrom "
          f"(atom {vector_atom + 1})")
    print(f"  MAE = {absolute.mean():.9g}, RMSE = {np.sqrt(np.mean(force_error**2)):.9g} eV/Angstrom")
    print(f"  Atoms with max component error > 0.01 eV/Angstrom: "
          f"{np.count_nonzero(absolute.max(axis=1) > 0.01)}")
    report_difference("DFT force alignment", dft_force, gpumd_force[:, 3:],
                      "eV/Angstrom", "atom", COMPONENTS)

    per_atom = counts.reshape(-1, 1)
    report_difference("Energy/atom", matpl_energy / per_atom, gpumd_energy[:, :1], "eV/atom")
    report_difference("DFT energy/atom alignment", dft_energy / per_atom, gpumd_energy[:, 1:], "eV/atom")
    gpumd_virial_matpl_order = gpumd_virial[:, GPUMD_VIRIAL_TO_MATPL]
    gpumd_dft_virial = gpumd_virial[:, [index + 6 for index in GPUMD_VIRIAL_TO_MATPL]]
    report_difference("Virial/atom", matpl_virial / per_atom, gpumd_virial_matpl_order,
                      "eV/atom", components=VIRIAL_COMPONENTS)
    report_difference("DFT virial/atom alignment", dft_virial / per_atom, gpumd_dft_virial,
                      "eV/atom", components=VIRIAL_COMPONENTS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpumd_test", type=Path, help="directory containing GPUMD *_train.out and train.xyz")
    parser.add_argument("test_result", type=Path, help="directory containing MatPL inference_*.txt and dft_*.txt")
    args = parser.parse_args()
    try:
        compare(args.gpumd_test, args.test_result)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
