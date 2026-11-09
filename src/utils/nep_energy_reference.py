"""Versioned per-element DFT energy references for LMDB NEP training."""
import math

REFERENCE_VERSION = 1
REFERENCE_TABLES = {
    "mptrj": (
        -3.4631, -0.27309, -2.8321, -3.6718, -7.6282, -8.3344,
        -7.5666, -7.472, -5.0677, -0.030058, -2.0124, -1.6125,
        -4.4889, -6.3631, -6.4289, -5.2847, -3.0972, -0.063163,
        -1.7411, -3.7065, -6.8949, -9.5306, -8.7439, -9.0234,
        -8.4526, -7.4091, -6.1867, -5.4593, -3.3224, -0.82911,
        -3.2546, -4.9264, -4.6854, -4.5014, -2.702, 0.77055,
        -1.6062, -3.4319, -7.722, -9.6074, -10.678, -9.2717,
        -6.8685, -8.2849, -7.0391, -5.0754, -1.7952, -0.35316,
        -2.5871, -3.9603, -3.9316, -3.6352, -2.1676, 2.7399,
        -2.1183, -3.9493, -7.1591, -7.4358, -6.5724, -6.7311,
        -5.1406, -6.7412, -11.751, -16.219, -6.4131, -6.4793,
        -6.3707, -6.3809, -6.3572, -2.8174, -6.4382, -10.711,
        -12.433, -10.198, -10.641, -9.197, -8.222, -6.0345,
        -2.7692, 0.6655, -1.5987, -3.2421, -3.4098, 1.0903e-16,
        6.1043e-16, 1.2149e-15, 3.4812e-16, -3.3371e-18, -4.2623, -9.0547,
        -10.285, -12.643, -12.739, -14.317,
    ),
    "omat24": (
        -3.0332, 0.027781, -2.0769, -2.8506, -5.5714, -6.9814,
        -7.5916, -6.8493, -4.5307, 0.37653, -1.2547, -1.2771,
        -3.5325, -5.1621, -5.2792, -5.1759, -2.9546, 0.094408,
        -0.52747, -1.9239, -6.3525, -7.7363, -8.106, -8.0094,
        -7.96, -6.9398, -5.9301, -4.8372, -2.9134, -0.9236,
        -2.844, -4.3353, -4.5904, -4.3398, -2.4037, 0.23849,
        -0.90706, -1.9711, -6.5034, -8.4291, -9.0198, -9.169,
        -8.955, -8.0029, -6.88, -5.212, -2.0796, -0.30074,
        -2.2362, -3.7322, -3.9127, -3.5653, -1.9106, 3.2219,
        -0.96049, -2.0196, -5.0013, -6.514, -4.8854, -4.8524,
        -4.9177, -4.79, -10.408, -13.873, -4.7775, -4.5076,
        -4.4782, -4.4732, -4.4918, -4.9103, -4.4665, -9.2834,
        -10.675, -10.502, -10.141, -9.1863, -7.9159, -6.0584,
        -3.0528, 0.17787, -1.9486, -3.1409, -3.5926, 1.2319e-13,
        -3.4435e-13, 2.9553e-12, 8.9567e-16, -2.9915e-17, -4.4779, -7.9138,
        -10.146, -12.389, -13.565, -14.339,
    ),
    "mae": (
        -3.1737, -0.26391, -2.2369, -3.4445, -6.8277, -8.1473,
        -8.8195, -7.5632, -5.0891, 0.0011973, -1.568, -1.6372,
        -4.0944, -5.8544, -6.0222, -5.5525, -3.0572, -0.035725,
        -0.89308, -2.414, -6.4643, -8.3088, -8.9037, -8.8312,
        -8.5999, -7.8178, -6.703, -5.7014, -3.7271, -1.2587,
        -3.3469, -5.0977, -5.1029, -4.7616, -2.5345, 0.70565,
        -1.3512, -2.0171, -6.4545, -8.696, -9.9447, -9.8603,
        -9.643, -8.9905, -7.6219, -5.5586, -2.4417, -0.78289,
        -2.6984, -4.1773, -4.2378, -3.8948, -2.3448, 2.6678,
        -1.4716, -2.3332, -5.1416, -6.557, -4.8068, -4.7959,
        -4.8117, -4.7422, -10.256, -13.897, -4.5928, -4.5515,
        -4.5212, -4.5335, -4.4311, -2.4671, -4.7002, -10.072,
        -11.422, -11.575, -11.294, -10.299, -8.8652, -6.5876,
        -3.3634, -0.12897, -2.0918, -3.4916, -3.9858, 1.0642e-14,
        1.1423e-16, -2.4249e-15, 6.9084e-17, -5.6083e-18, -4.4531, -7.612,
        -9.9345, -11.656, -12.598, -14.325,
    ),
}


def normalize_lmdb_type(value):
    if not isinstance(value, str):
        raise ValueError("lmdb_type must be one of mptrj, omat24, mae")
    result = value.strip().lower()
    if result == "omat":
        result = "omat24"
    if result not in REFERENCE_TABLES:
        raise ValueError("lmdb_type must be one of mptrj, omat24, mae")
    return result


def validate_reference_elements(lmdb_type, atom_types):
    table = REFERENCE_TABLES[normalize_lmdb_type(lmdb_type)]
    for number in atom_types:
        if isinstance(number, bool) or int(number) != number or not 1 <= number <= len(table):
            raise ValueError(f"missing energy reference for atomic number {number}")
    return True


def reference_for_numbers(lmdb_type, numbers):
    table = REFERENCE_TABLES[normalize_lmdb_type(lmdb_type)]
    total = 0.0
    for number in numbers:
        if isinstance(number, bool) or int(number) != number or not 1 <= number <= len(table):
            raise ValueError(f"missing energy reference for atomic number {number}")
        total += table[int(number)-1]
    return total


def make_energy_reference_metadata(lmdb_type):
    active_type = normalize_lmdb_type(lmdb_type)
    return {
        "version": REFERENCE_VERSION,
        "representation": "residual",
        "active_type": active_type,
        "atomic_numbers": list(range(1, 95)),
        "tables": {name: list(values) for name, values in REFERENCE_TABLES.items()},
    }


def validate_energy_reference_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("energy_reference metadata must be an object")
    if metadata.get("version") != REFERENCE_VERSION or metadata.get("representation") != "residual":
        raise ValueError("unsupported energy_reference version or representation")
    active_type = normalize_lmdb_type(metadata.get("active_type"))
    if metadata.get("atomic_numbers") != list(range(1, 95)):
        raise ValueError("energy_reference atomic_numbers must be 1..94")
    saved = metadata.get("tables")
    if not isinstance(saved, dict) or set(saved) != set(REFERENCE_TABLES):
        raise ValueError("energy_reference tables must contain mptrj, omat24, mae")
    for name, expected in REFERENCE_TABLES.items():
        values = saved[name]
        if (not isinstance(values, (list, tuple)) or len(values) != 94
                or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values)
                or tuple(values) != expected):
            raise ValueError(f"energy_reference table {name} differs from this code")
    return active_type


def checkpoint_reference_for_config(json_file):
    lmdb_type = json_file.get("lmdb_type")
    return make_energy_reference_metadata(lmdb_type) if lmdb_type is not None else None


def prepare_energy_reference_checkpoint(
        checkpoint, target_type, atom_types, migrate_legacy=False):
    """Validate checkpoint convention; convert legacy bias only by explicit request."""
    metadata = checkpoint.get("energy_reference")
    if metadata is not None:
        source_type = validate_energy_reference_metadata(metadata)
        if target_type is None:
            raise ValueError("residual checkpoint requires lmdb_type")
        target_type = normalize_lmdb_type(target_type)
        validate_reference_elements(target_type, atom_types)
        return source_type != target_type, source_type
    if target_type is None:
        return False, "legacy"
    target_type = normalize_lmdb_type(target_type)
    validate_reference_elements(target_type, atom_types)
    if not migrate_legacy:
        raise ValueError(
            "legacy total-energy checkpoint requires migrate_legacy_energy_reference=true"
        )
    state = dict(checkpoint["state_dict"])
    prefix = "module." if any(key.startswith("module.") for key in state) else ""
    for index, atomic_number in enumerate(atom_types):
        shift = reference_for_numbers(target_type, [atomic_number])
        keys = (
            f"{prefix}fitting_net.{index}.layers.1.bias",
            f"{prefix}fitting_net.{index}.energy_head.bias",
        )
        key = next((key for key in keys if key in state), None)
        if key is None:
            raise ValueError(
                f"cannot migrate energy bias for atomic number {atomic_number}"
            )
        state[key] = state[key].clone() - shift
    checkpoint["state_dict"] = state
    return True, "legacy"
