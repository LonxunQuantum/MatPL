#!/usr/bin/env python3
"""Compare quick-train metrics with the stored batch-size baselines."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence


CASES = (
    "16_metal",
    "AuAg",
    "LiSiC",
    "MPtraj",
    "Si-SiO2-La2O3-HfO2-TiN",
)
BATCH_SIZES = (1, 32)
METRICS = (
    "loss",
    "RMSE_Etot(eV/atom)",
    "RMSE_F(eV/Å)",
    "RMSE_virial(eV/atom)",
)
TOLERANCE = 1.0e-10
TARGET_EPOCH = 1


class EpochDataError(ValueError):
    """Raised when an epoch_train.dat file is missing or malformed."""


class Mismatch:
    def __init__(
        self,
        epoch: int,
        metric: str,
        actual: float,
        reference: float,
        error: float,
    ) -> None:
        self.epoch = epoch
        self.metric = metric
        self.actual = actual
        self.reference = reference
        self.error = error


class ComparisonResult:
    def __init__(self, max_errors: dict[str, float], mismatches: list[Mismatch]) -> None:
        self.max_errors = max_errors
        self.mismatches = mismatches

    @property
    def passed(self) -> bool:
        return not self.mismatches


def read_epoch_metrics(path: Path) -> dict[int, dict[str, float]]:
    """Read epoch identifiers and the four metrics used by this regression test."""
    if not path.is_file():
        raise EpochDataError(f"file not found: {path}")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EpochDataError(f"cannot read {path}: {exc}") from exc

    required_columns = ("epoch", *METRICS)
    column_indexes: dict[str, int] | None = None
    rows: dict[int, dict[str, float]] = {}

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header = stripped[1:].split()
            if all(column in header for column in required_columns):
                column_indexes = {column: header.index(column) for column in required_columns}
            continue
        if column_indexes is None:
            raise EpochDataError(f"missing required header in {path}")

        fields = stripped.split()
        largest_index = max(column_indexes.values())
        if len(fields) <= largest_index:
            raise EpochDataError(f"too few columns in {path}:{line_number}")

        try:
            epoch_value = float(fields[column_indexes["epoch"]])
        except ValueError as exc:
            raise EpochDataError(f"invalid epoch in {path}:{line_number}") from exc
        if not math.isfinite(epoch_value) or not epoch_value.is_integer():
            raise EpochDataError(f"invalid epoch in {path}:{line_number}")
        epoch = int(epoch_value)
        if epoch in rows:
            raise EpochDataError(f"duplicate epoch {epoch} in {path}")

        metrics: dict[str, float] = {}
        for metric in METRICS:
            try:
                value = float(fields[column_indexes[metric]])
            except ValueError as exc:
                raise EpochDataError(
                    f"invalid {metric} value in {path}:{line_number}"
                ) from exc
            if not math.isfinite(value):
                raise EpochDataError(
                    f"non-finite {metric} value in {path}:{line_number}"
                )
            metrics[metric] = value
        rows[epoch] = metrics

    if column_indexes is None:
        raise EpochDataError(f"missing required header in {path}")
    if not rows:
        raise EpochDataError(f"no epoch data in {path}")
    return rows


def compare_epoch_files(
    actual_path: Path,
    reference_path: Path,
    tolerance: float = TOLERANCE,
) -> ComparisonResult:
    """Compare the four selected metrics for epoch 1 only."""
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")

    actual_rows = read_epoch_metrics(Path(actual_path))
    reference_rows = read_epoch_metrics(Path(reference_path))
    missing_sources = []
    if TARGET_EPOCH not in actual_rows:
        missing_sources.append("actual")
    if TARGET_EPOCH not in reference_rows:
        missing_sources.append("reference")
    if missing_sources:
        raise EpochDataError(
            f"missing required epoch {TARGET_EPOCH} in "
            f"{', '.join(missing_sources)} file"
        )

    max_errors = {metric: 0.0 for metric in METRICS}
    mismatches: list[Mismatch] = []
    for metric in METRICS:
        actual = actual_rows[TARGET_EPOCH][metric]
        reference = reference_rows[TARGET_EPOCH][metric]
        error = abs(actual - reference)
        max_errors[metric] = error
        if error > tolerance:
            mismatches.append(
                Mismatch(TARGET_EPOCH, metric, actual, reference, error)
            )

    return ComparisonResult(max_errors, mismatches)


def _format_max_errors(max_errors: dict[str, float]) -> str:
    return ", ".join(f"{metric}={max_errors[metric]:.3e}" for metric in METRICS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare quick-train loss and RMSE values with stored baselines."
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=Path.cwd() / "quick_train",
        help="quick_train directory (default: ./quick_train)",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()

    total = len(CASES) * len(BATCH_SIZES)
    passed = 0
    print(f"Output directory: {output_dir}")
    print(f"Compared epoch: {TARGET_EPOCH}")
    print(f"Absolute tolerance: {TOLERANCE:.1e}")

    for case in CASES:
        for batch_size in BATCH_SIZES:
            label = f"{case}/batch{batch_size}"
            job_dir = output_dir / case / f"batch{batch_size}"
            actual = job_dir / "model_record" / "epoch_train.dat"
            reference = job_dir / f"batch{batch_size}_epoch_train.dat"
            try:
                result = compare_epoch_files(actual, reference)
            except EpochDataError as exc:
                print(f"[FAIL] {label}: {exc}")
                continue

            if result.passed:
                passed += 1
                print(f"[PASS] {label}: {_format_max_errors(result.max_errors)}")
                continue

            print(f"[FAIL] {label}: {_format_max_errors(result.max_errors)}")
            for mismatch in result.mismatches:
                print(
                    "       "
                    f"epoch={mismatch.epoch}, column={mismatch.metric}, "
                    f"actual={mismatch.actual:.16e}, "
                    f"reference={mismatch.reference:.16e}, "
                    f"abs_error={mismatch.error:.3e}"
                )

    failed = total - passed
    print(f"Summary: {passed}/{total} passed, {failed}/{total} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
