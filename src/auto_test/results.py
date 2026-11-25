"""Parse MatPL artifacts and compare results in physical units."""

import json
import math
import os


TRAIN_COLUMNS = ("loss", "RMSE_Etot(eV/atom)", "RMSE_F(eV/Å)", "RMSE_virial(eV/atom)")
ATOMIC_PROPERTIES = {"atomic_energy", "force", "charge", "bec"}
PREDICTION_FILES = {
    "energy": ("inference_total_energy.txt", 1),
    "atomic_energy": ("inference_atomic_energy.txt", 1),
    "force": ("inference_force.txt", 3),
    "virial": ("inference_virial.txt", 6),
    "charge": ("inference_charge.txt", 1),
    "bec": ("inference_bec.txt", 9),
}


def write_json(path, value):
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def read_epochs(path):
    header = None
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "#":
            columns = fields[1:]
            if columns and columns[0] == "epoch":
                if header is not None and header != columns:
                    raise ValueError(f"epoch header changed in {path}")
                header = columns
            continue
        if header is None or len(header) != len(fields):
            raise ValueError(f"malformed epoch row in {path}: {line}")
        epoch = int(fields[0])
        if epoch < 1 or epoch in rows:
            raise ValueError(f"invalid or duplicate epoch {epoch} in {path}")
        values = {key: float(value) for key, value in zip(header[1:], fields[1:])
                  if key != "time(s)"}
        if not values or not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"nonfinite or missing metrics at epoch {epoch} in {path}")
        rows[epoch] = values
    if not rows:
        raise ValueError(f"no epoch metrics in {path}")
    return rows


def compare_metrics(actual, reference, atol, rtol):
    differences = []
    if actual.keys() != reference.keys():
        differences.append(f"epochs differ: current={list(actual)} base={list(reference)}")
    for epoch in sorted(actual.keys() & reference.keys()):
        current, expected = actual[epoch], reference[epoch]
        if current.keys() != expected.keys():
            differences.append(f"epoch={epoch} metric columns differ")
        for name in expected.keys() & current.keys():
            delta = abs(current[name] - expected[name])
            limit = atol + rtol * abs(expected[name])
            if not math.isfinite(delta) or delta > limit:
                differences.append(
                    f"epoch={epoch} {name} current={current[name]:.12g} "
                    f"base={expected[name]:.12g} delta={delta:.3g} limit={limit:.3g}"
                )
    return differences


def compare_predictions(actual, reference, atol, rtol):
    if any(actual[key] != reference[key] for key in ("frame_ids", "natoms")):
        return ["frame IDs or atom counts differ"]
    differences = []
    if actual.keys() != reference.keys():
        differences.append("prediction fields differ")
    for name in sorted((actual.keys() & reference.keys()) - {"frame_ids", "natoms"}):
        current, expected = actual[name], reference[name]
        if len(current) != len(expected):
            differences.append(f"{name}: row counts differ")
            continue
        frame = atom = 0
        for row, (current_row, reference_row) in enumerate(zip(current, expected)):
            if name not in ATOMIC_PROPERTIES:
                frame = row
            elif atom == actual["natoms"][frame]:
                frame += 1
                atom = 0
            label = f"{name} frame={actual['frame_ids'][frame]}"
            if name in ATOMIC_PROPERTIES:
                label += f" atom={atom}"
            if len(current_row) != len(reference_row):
                differences.append(f"{label}: component counts differ")
                break
            for component, (value, target) in enumerate(zip(current_row, reference_row)):
                delta = abs(value - target)
                if not math.isfinite(delta) or delta > atol + rtol * abs(target):
                    differences.append(f"{label} component={component} "
                                       f"current={value:.10g} base={target:.10g} delta={delta:.3g}")
                    break
            else:
                atom += 1
                continue
            break
    return differences


def read_per_frame_scalars(path, counts):
    frames = [[float(value) for value in line.split()]
              for line in path.read_text(encoding="utf-8").splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if [len(frame) for frame in frames] != counts:
        raise ValueError(f"{path.name}: per-frame atom counts differ")
    return [[value] for frame in frames for value in frame]


def read_predictions(directory, frames):
    import numpy as np

    counts = [int(frame[1].splitlines()[0]) for frame in frames]
    nframes, natoms = len(frames), sum(counts)
    reported = np.loadtxt(directory / "image_atom_nums.txt", ndmin=1)
    if reported.tolist() != counts:
        raise ValueError(f"{directory}: per-frame atom counts differ")
    result = {"frame_ids": [frame[0] for frame in frames], "natoms": counts}
    for name, (filename, columns) in PREDICTION_FILES.items():
        path = directory / filename
        if name in {"atomic_energy", "charge"}:
            array = np.asarray(read_per_frame_scalars(path, counts))
        else:
            array = np.loadtxt(path, ndmin=2)
        size = natoms if name in ATOMIC_PROPERTIES else nframes
        if array.shape != (size, columns) or not np.isfinite(array).all():
            raise ValueError(f"{filename}: expected {size}×{columns} finite values, got {array.shape}")
        result[name] = array.tolist()
    offset = 0
    for index, count in enumerate(counts):
        sl = slice(offset, offset + count)
        if abs(sum(row[0] for row in result["charge"][sl])) > 1e-4:
            raise ValueError(f"frame {result['frame_ids'][index]}: predicted charge does not sum to zero")
        total = sum(row[0] for row in result["atomic_energy"][sl])
        if not math.isclose(total, result["energy"][index][0], rel_tol=1e-7, abs_tol=1e-5):
            raise ValueError(f"frame {result['frame_ids'][index]}: predicted atomic energies do not sum")
        offset += count
    expected_labels = {
        "dft_virial.txt": (nframes, 6, [row[2] == "stress" for row in frames]),
        "dft_bec.txt": (natoms, 9, [row[2] == "bec" for row, count in zip(frames, counts)
                                     for _ in range(count)]),
    }
    for name, (size, columns, mask) in expected_labels.items():
        labels = np.loadtxt(directory / name, ndmin=2)
        if labels.shape != (size, columns) or not np.isfinite(labels).all():
            raise ValueError(f"{name}: missing or invalid labels")
        if (labels[:, 0] > -1e6).tolist() != mask:
            raise ValueError(f"{name}: labeled frame or atom mask differs")
    return result
