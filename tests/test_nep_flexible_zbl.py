import numpy as np
import pytest
import torch

from src.user.input_param import InputParam
from src.model.nep_net import NEP
from src.user.nep_param import NepParam
from src.utils.nep_to_gpumd import extract_model
from src.utils.nep_zbl import read_zbl_in, reorder_zbl_parameters


def _config():
    return {
        "model_type": "NEP4", "atom_type": ["H", "He"],
        "model": {
            "descriptor": {"cutoff": [6, 4], "n_max": [4, 4],
                           "basis_size": [4, 4], "l_max": [4, 0, 0],
                           "zbl": 2, "zbl_in": "zbl.in"},
            "fitting_net": {"network_size": [30, 1]},
        },
        "optimizer": {"optimizer": "ADAM"},
    }


def test_zbl_file_order_and_checkpoint_snapshot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    values = np.array([
        [.2, 2.2, 1, 2, 3, 4, 5, 6, 7, 8],
        [.3, 2.3, 2, 3, 4, 5, 6, 7, 8, 9],
        [.4, 2.4, 3, 4, 5, 6, 7, 8, 9, 10],
    ])
    (tmp_path / "zbl.in").write_text("\n".join(map(str, values.flatten())))
    parsed = read_zbl_in(tmp_path / "zbl.in", 2, 4)
    np.testing.assert_array_equal(parsed, values)
    np.testing.assert_array_equal(reorder_zbl_parameters(parsed, [1, 0], 2),
                                  values[[2, 1, 0]])
    params = InputParam(_config(), "TRAIN")
    saved = params.to_dict()
    np.testing.assert_array_equal(saved["model"]["descriptor"]["zbl_parameters"], values)
    (tmp_path / "zbl.in").unlink()
    restored = InputParam(saved, "TRAIN")
    np.testing.assert_array_equal(restored.nep_param.zbl_parameters, values)
    net = NEP(params, energy_shift=np.zeros(2),
              q_scaler=torch.ones(params.nep_param.feature_nums),
              max_NN_radial=10, max_NN_angular=10,
              dtype=torch.float64, device=torch.device("cpu"))
    ckpt = tmp_path / "model.ckpt"
    torch.save({"json_file": saved, "state_dict": net.state_dict()}, ckpt)
    content, _, _ = extract_model(str(ckpt))
    assert content.splitlines()[0].startswith("nep4_zbl  ")
    assert content.splitlines()[1] == "zbl 0 0"
    np.testing.assert_array_equal([float(v) for v in content.splitlines()[-30:]],
                                  values.flatten())
    txt = tmp_path / "nep.txt"
    txt.write_text(content)
    from_txt = NepParam()
    from_txt.set_nep_nn_c_param_from_nep_txt(str(txt), [1, 2])
    np.testing.assert_array_equal(from_txt.zbl_parameters, values)


def test_zbl_file_validation(tmp_path):
    path = tmp_path / "zbl.in"
    path.write_text("1 2 3")
    with pytest.raises(ValueError, match="requires 30"):
        read_zbl_in(path, 2, 4)
    path.write_text("0 5 " + "1 " * 8)
    with pytest.raises(ValueError, match="angular neighbor cutoff"):
        read_zbl_in(path, 1, 4)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_flexible_zbl_pair_energy_and_gradient(tmp_path, monkeypatch, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    device = torch.device(device_name)
    monkeypatch.chdir(tmp_path)
    # Only the mixed pair has a nonzero screening coefficient.
    values = np.zeros((3, 10))
    values[:, 0] = .2
    values[:, 1] = 2.5
    values[1, 2] = 1.0
    values[:, 3] = 1.0
    (tmp_path / "zbl.in").write_text("\n".join(map(str, values.flatten())))
    config = InputParam(_config(), "TRAIN")
    net = NEP(config, energy_shift=np.zeros(2),
              q_scaler=torch.ones(config.nep_param.feature_nums),
              max_NN_radial=10, max_NN_angular=10,
              dtype=torch.float64, device=device)
    r = torch.tensor([[1.0], [1.0]], dtype=torch.float64, device=device, requires_grad=True)
    ri = torch.zeros((2, 1, 4), dtype=torch.float64, device=device)
    ri[:, :, 0] = r
    energy = net.cal_zbl(ri, torch.tensor([[1], [0]], device=device),
                         torch.tensor([0, 1], device=device),
                         torch.full((2, 1), 2.5, dtype=torch.float64, device=device))
    x = (1.0**.23 + 2.0**.23) * 2.134563
    fc = .5 + .5 * np.cos(np.pi * (1.0 - .2) / (2.5 - .2))
    expected = .5 * 14.399645 * 2 * np.exp(-x) * fc
    torch.testing.assert_close(energy, torch.full_like(energy, expected))
    assert torch.isfinite(torch.autograd.grad(energy.sum(), r)[0]).all()


def test_flexible_zbl_follows_module_to(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    values = np.tile([.2, 2.5, 1, 2, 3, 4, 5, 6, 7, 8], (3, 1))
    (tmp_path / "zbl.in").write_text("\n".join(map(str, values.flatten())))
    params = InputParam(_config(), "TRAIN")
    net = NEP(params, energy_shift=np.zeros(2),
              q_scaler=torch.ones(params.nep_param.feature_nums),
              max_NN_radial=10, max_NN_angular=10,
              dtype=torch.float64, device=torch.device("cpu"))
    assert "flexible_zbl" not in net.state_dict()
    net = net.to(dtype=torch.float32)
    assert net.flexible_zbl.dtype == torch.float32
    if torch.cuda.is_available():
        net = net.to(device="cuda")
        assert net.flexible_zbl.device.type == "cuda"
