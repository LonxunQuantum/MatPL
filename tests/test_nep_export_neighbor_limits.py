import pytest
import torch

from src.utils.nep_to_gpumd import extract_model


@pytest.mark.parametrize(
    "prefix, legacy, expected",
    [
        ("", None, "cutoff 6.0 5.0 167 98"),
        ("module.", None, "cutoff 6.0 5.0 167 98"),
        (None, [88, 44], "cutoff 6.0 5.0 88 44"),
        (None, None, "cutoff 6.0 5.0 500 300"),
    ],
)
def test_extract_model_uses_saved_neighbor_limits(tmp_path, prefix, legacy, expected):
    key_prefix = prefix or ""
    state = {
        f"{key_prefix}q_scaler": torch.ones(1),
        f"{key_prefix}fitting_net.0.layers.0.weight": torch.ones(1, 1),
        f"{key_prefix}fitting_net.0.layers.0.bias": torch.zeros(1),
        f"{key_prefix}fitting_net.0.layers.1.weight": torch.ones(1, 1),
        f"{key_prefix}fitting_net.0.layers.1.bias": torch.zeros(1),
        f"{key_prefix}c_param_2": torch.ones(1, 1, 1, 1),
    }
    if prefix is not None:
        state[f"{prefix}max_NN_radial"] = torch.tensor(167)
        state[f"{prefix}max_NN_angular"] = torch.tensor(98)
    checkpoint = {
        "json_file": {
            "model_type": "NEP",
            "atom_type": [1],
            "model": {
                "descriptor": {
                    "cutoff": [6.0, 5.0],
                    "n_max": [0, 0],
                    "basis_size": [0, 0],
                    "l_max": [0, 0, 0],
                },
                "fitting_net": {"network_size": [1, 1]},
            },
        },
        "state_dict": state,
    }
    if legacy is not None:
        checkpoint["max_neighbor"] = legacy
    path = tmp_path / "nep_model.ckpt"
    torch.save(checkpoint, path)
    content, _, _ = extract_model(str(path))
    cutoff_line = next(line for line in content.splitlines() if line.startswith("cutoff "))
    assert cutoff_line == expected
