"""Model-type semantics and GPUMD NEP4 text parameter order."""

import numpy as np
import pytest
import torch

from src.model.nep_net import NEP
from src.user.input_param import InputParam
from src.user.nep_param import NepParam
from src.user.convert_model import get_model_type
from src.utils.file_operation import check_model_type
from src.utils.nep_to_gpumd import extract_model, nep_ckpt_to_gpumd


def _config(model_type):
    return {
        "model_type": model_type,
        "atom_type": [1, 2],
        "model": {
            "descriptor": {
                "cutoff": [6.0, 5.0],
                "n_max": [4, 4],
                "basis_size": [4, 4],
                "l_max": [4, 0, 0],
            },
            "fitting_net": {"network_size": [30, 1]},
        },
        "optimizer": {"optimizer": "ADAM"},
    }


@pytest.mark.parametrize("model_type,charge_mode", [
    ("NEP", 0), ("NEP4", 0), ("NEP_charge2", 2), ("NEP4_charge2", 2),
])
def test_model_type_is_the_only_serialized_version_and_charge_setting(
        tmp_path, monkeypatch, model_type, charge_mode):
    monkeypatch.chdir(tmp_path)
    params = InputParam(_config(model_type), "TRAIN")
    saved = params.to_dict()
    assert params.model_type == model_type.upper()
    assert params.nep_param.charge_mode == charge_mode
    assert saved["model_type"] == model_type.upper()
    assert "charge_mode" not in saved["model"]["descriptor"]
    assert "gpumd_nep4" not in saved["model"]["descriptor"]


@pytest.mark.parametrize("model_type,is_nep4,has_charge", [
    ("NEP", False, False),
    ("NEP4", True, False),
    ("NEP_charge2", False, True),
    ("NEP4_charge2", True, True),
])
def test_energy_bias_structure_and_gradient(
        tmp_path, monkeypatch, model_type, is_nep4, has_charge):
    monkeypatch.chdir(tmp_path)
    params = InputParam(_config(model_type), "TRAIN")
    net = NEP(params, energy_shift=np.array([0.5, 1.5]),
              q_scaler=torch.ones(params.nep_param.feature_nums),
              max_NN_radial=10, max_NN_angular=10,
              dtype=torch.float64, device=torch.device("cpu")).to(torch.float64)
    assert isinstance(net.common_bias, torch.nn.Parameter) == is_nep4
    for fitting in net.fitting_net:
        output_bias = (fitting.energy_head.bias if has_charge
                       else fitting.layers[1].bias)
        assert (output_bias is None) == is_nep4

    features = torch.zeros(2, params.nep_param.feature_nums, dtype=torch.float64)
    types = torch.tensor([0, 1])
    energies, charges, grad_e, grad_q = net.calculate_Ei_with_grad(
        types, features, torch.device("cpu"))
    direct = []
    for index, fitting in enumerate(net.fitting_net):
        output = fitting(features[index:index + 1])
        direct.append((output[0] if has_charge else output).reshape(()))
    expected = torch.stack(direct)
    if is_nep4:
        expected = expected + net.common_bias
    torch.testing.assert_close(energies, expected)
    assert (charges is not None) == has_charge
    assert (grad_q is not None) == has_charge
    assert grad_e.shape == features.shape
    energies.sum().backward()
    if is_nep4:
        torch.testing.assert_close(net.common_bias.grad, torch.tensor(2.0, dtype=torch.float64))


@pytest.mark.parametrize("model_type,charge,is_nep4", [
    ("NEP", False, False),
    ("NEP4", False, True),
    ("NEP_charge2", True, False),
    ("NEP4_charge2", True, True),
])
def test_nep_text_matches_parser_and_parameter_order(
        tmp_path, monkeypatch, model_type, charge, is_nep4):
    config = _config(model_type)
    features, hidden, types = 25, 30, 2
    state = {
        "q_scaler": torch.arange(1, features + 1, dtype=torch.float64),
        "c_param_2": torch.arange(100, dtype=torch.float64).reshape(2, 2, 5, 5),
        "c_param_3": torch.arange(100, dtype=torch.float64).reshape(2, 2, 5, 5) + 100,
        "max_NN_radial": torch.tensor(20),
        "max_NN_angular": torch.tensor(15),
    }
    if is_nep4:
        state["common_bias"] = torch.tensor(0.75, dtype=torch.float64)
    if charge:
        state["sqrt_epsilon_inf"] = torch.tensor(2.5, dtype=torch.float64)
    expected_values = []
    for t in range(types):
        w0 = torch.arange(features * hidden, dtype=torch.float64).reshape(features, hidden) / 100 + t
        b0 = torch.arange(hidden, dtype=torch.float64) / 10 + t
        we = torch.arange(hidden, dtype=torch.float64) / 20 + t
        state[f"fitting_net.{t}.layers.0.weight"] = w0
        state[f"fitting_net.{t}.layers.0.bias"] = b0
        expected_values += w0.T.flatten().tolist() + (-b0).tolist() + we.tolist()
        if charge:
            wq = torch.arange(hidden, dtype=torch.float64) / 30 + t
            state[f"fitting_net.{t}.energy_head.weight"] = we.reshape(hidden, 1)
            state[f"fitting_net.{t}.charge_head.weight"] = wq.reshape(hidden, 1)
            expected_values += wq.tolist()
            if not is_nep4:
                output_bias = torch.tensor([[0.1 * (t + 1)]], dtype=torch.float64)
                state[f"fitting_net.{t}.energy_head.bias"] = output_bias
                expected_values.append(-output_bias.item())
        else:
            state[f"fitting_net.{t}.layers.1.weight"] = we.reshape(hidden, 1)
            if not is_nep4:
                output_bias = torch.tensor([[0.1 * (t + 1)]], dtype=torch.float64)
                state[f"fitting_net.{t}.layers.1.bias"] = output_bias
                expected_values.append(-output_bias.item())
    if charge:
        expected_values.append(2.5)
    expected_values.append(-0.75 if is_nep4 else 0.0)

    checkpoint = tmp_path / "model.ckpt"
    torch.save({"json_file": config, "state_dict": state}, checkpoint)
    assert get_model_type(str(checkpoint)) == model_type.upper()
    assert check_model_type(str(checkpoint)) == model_type.upper()
    content, _, _ = extract_model(str(checkpoint))
    version = 4 if is_nep4 else 5
    assert content.splitlines()[0].startswith(
        f"nep{version}{'_charge2' if charge else ''} ")
    ann_index = next(i for i, line in enumerate(content.splitlines())
                     if line.startswith("ANN "))
    values = [float(line) for line in content.splitlines()[ann_index + 1:]]
    np.testing.assert_allclose(values[:len(expected_values)], expected_values)

    force_field = tmp_path / f"nep{version}.txt"
    force_field.write_text(content)
    assert get_model_type(str(force_field)) == model_type.upper()
    assert check_model_type(str(force_field)) == model_type.upper()
    parsed = NepParam()
    parsed.set_nep_nn_c_param_from_nep_txt(str(force_field), [1, 2])
    assert parsed.charge_mode == (2 if charge else 0)
    expected_bias = [0.75, 0.75] if is_nep4 else [0.1, 0.2]
    np.testing.assert_allclose(parsed.bias_lastlayer[:, 0] if charge
                               else parsed.bias_lastlayer, expected_bias)
    for t in range(types):
        np.testing.assert_allclose(parsed.model_wb[t * 3],
                                   state[f"fitting_net.{t}.layers.0.weight"].numpy())
    np.testing.assert_allclose(parsed.c2_param, state["c_param_2"].numpy())
    np.testing.assert_allclose(parsed.c3_param, state["c_param_3"].numpy())
    np.testing.assert_allclose(parsed.q_scaler, state["q_scaler"].numpy())

    monkeypatch.chdir(tmp_path)
    nep_ckpt_to_gpumd([str(checkpoint)])
    assert (tmp_path / f"nep{version}.txt").read_text() == content
    initialized_config = _config(model_type)
    initialized_config["nep_txt_file"] = str(force_field)
    params = InputParam(initialized_config, "TRAIN")
    net = NEP(params, energy_shift=np.array([0.0, 0.0]),
              q_scaler=torch.as_tensor(parsed.q_scaler),
              dtype=torch.float64, device=torch.device("cpu")).to(torch.float64)
    if is_nep4:
        torch.testing.assert_close(net.common_bias, torch.tensor(0.75, dtype=torch.float64))
    else:
        assert net.common_bias is None
    assert all(((f.energy_head.bias if charge else f.layers[1].bias) is None)
               == is_nep4 for f in net.fitting_net)
