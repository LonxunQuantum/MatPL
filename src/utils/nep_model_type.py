"""Canonical NEP model types used by configuration, training and export."""


NEP_MODEL_TYPES = frozenset({"NEP", "NEP5", "NEP4", "NEP_CHARGE2", "NEP4_CHARGE2"})


def is_nep_model_type(model_type):
    return isinstance(model_type, str) and model_type.upper() in NEP_MODEL_TYPES


def nep_model_settings(model_type):
    """Return (NEP version, internal charge mode) for a configured type."""
    if not is_nep_model_type(model_type):
        raise ValueError(f"Unsupported NEP model_type: {model_type!r}")
    name = model_type.upper()
    return (4 if name.startswith("NEP4") else 5,
            2 if name.endswith("_CHARGE2") else 0)


def nep_model_type_from_txt_header(header):
    """Map an existing GPUMD text header to the corresponding model type."""
    name = header.split()[0].lower()
    for version in (4, 5):
        for charge_suffix in ("", "_charge2"):
            for zbl_suffix in ("", "_zbl"):
                if name == f"nep{version}{zbl_suffix}{charge_suffix}":
                    return f"NEP{version if version == 4 else ''}{charge_suffix}".upper()
    raise ValueError(f"Unsupported NEP text header: {name!r}")
