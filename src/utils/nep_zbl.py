"""GPUMD flexible ZBL parameters, ordered by upper-triangular type pairs."""

from pathlib import Path
import numpy as np


def validate_zbl_parameters(values, types, angular_cutoff):
    if not 1 <= types <= 10:
        raise ValueError("flexible ZBL supports 1..10 element types")
    expected = 10 * types * (types + 1) // 2
    array = np.asarray(values, dtype=np.float64)
    if array.size != expected:
        raise ValueError(f"zbl.in requires {expected} numbers for {types} types; got {array.size}")
    array = array.reshape(-1, 10)
    if not np.isfinite(array).all():
        raise ValueError("zbl.in parameters must be finite")
    if np.any(array[:, 0] < 0) or np.any(array[:, 1] <= array[:, 0]):
        raise ValueError("zbl.in requires 0 <= inner cutoff < outer cutoff for every pair")
    if np.any(array[:, 1] > angular_cutoff):
        raise ValueError("zbl.in outer cutoff exceeds the angular neighbor cutoff")
    return array.copy()


def read_zbl_in(path, types, angular_cutoff):
    try:
        values = [float(token) for token in Path(path).read_text().split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read zbl_in {path}: {exc}") from exc
    return validate_zbl_parameters(values, types, angular_cutoff)


def pair_index(i, j, types):
    lo, hi = sorted((i, j))
    return lo * types - lo * (lo - 1) // 2 + hi - lo


def reorder_zbl_parameters(values, indices, source_types):
    return np.asarray([
        values[pair_index(source_i, source_j, source_types)]
        for n, source_i in enumerate(indices)
        for source_j in indices[n:]
    ], dtype=np.float64)
