import torch

from src.model.nep_fitting import FittingNet, QNEPFittingNet
from src.model.nep_net import NEP


def _make_nep(charge_mode, gpumd_nep4=False):
    model = NEP.__new__(NEP)
    torch.nn.Module.__init__(model)
    model.dtype = torch.float64
    model.charge_mode = charge_mode
    model.gpumd_nep4 = gpumd_nep4
    model.q_scaler = torch.tensor([0.5, 1.5, -2.0], dtype=torch.float64)
    model.common_bias = torch.nn.Parameter(torch.tensor(0.375, dtype=torch.float64))
    model.fitting_net = torch.nn.ModuleList()
    for atom_type in range(3):
        torch.manual_seed(20260727 + atom_type + 10 * charge_mode)
        if charge_mode:
            fit_net = QNEPFittingNet(
                network_size=[4, 4],
                bias=True,
                resnet_dt=True,
                activation="tanh",
                input_dim=3,
                ener_shift=0.1 * (atom_type + 1),
                charge_mode=2,
                last_bias=not gpumd_nep4,
            )
        else:
            fit_net = FittingNet(
                network_size=[4, 4, 1],
                bias=True,
                resnet_dt=True,
                activation="tanh",
                input_dim=3,
                ener_shift=0.1 * (atom_type + 1),
            )
        model.fitting_net.append(fit_net.double())
    return model


def _compare_with_autograd(charge_mode, gpumd_nep4=False):
    model = _make_nep(charge_mode=charge_mode, gpumd_nep4=gpumd_nep4)
    device = torch.device("cpu")
    imagetype_map = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    feats_scaled = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)

    result = model.calculate_Ei_with_grad(imagetype_map, feats_scaled, device)
    ei, charge, grad_e_scaled, grad_q_scaled = result

    ei_ref, charge_ref = model.calculate_Ei(imagetype_map, feats_scaled, device)
    grad_e_ref = torch.autograd.grad(ei_ref.sum(), feats_scaled, retain_graph=True)[0]

    torch.testing.assert_close(ei, ei_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(grad_e_scaled, grad_e_ref, rtol=1e-10, atol=1e-10)

    if charge_mode:
        grad_q_ref = torch.autograd.grad(charge_ref.sum(), feats_scaled)[0]
        torch.testing.assert_close(charge, charge_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(grad_q_scaled, grad_q_ref, rtol=1e-10, atol=1e-10)
    else:
        assert charge is None
        assert grad_q_scaled is None

    loss = ei.sum() + grad_e_scaled.square().sum()
    if charge is not None:
        loss = loss + charge.sum() + grad_q_scaled.square().sum()
    loss.backward()
    for name, param in model.named_parameters():
        if name == "common_bias" and not gpumd_nep4:
            continue
        if name.startswith("fitting_net.2."):
            continue
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_calculate_ei_with_grad_matches_autograd_for_energy_only_model():
    _compare_with_autograd(charge_mode=0)


def test_calculate_ei_with_grad_matches_autograd_for_charge_model():
    _compare_with_autograd(charge_mode=2)


def test_calculate_ei_with_grad_keeps_gpumd_common_bias_behavior():
    _compare_with_autograd(charge_mode=2, gpumd_nep4=True)


def test_calculate_ei_with_grad_uses_scaled_feature_convention():
    model = _make_nep(charge_mode=2)
    device = torch.device("cpu")
    imagetype_map = torch.tensor([0, 1, 0, 1, 1], dtype=torch.long)
    feats_raw = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    feats_scaled = feats_raw * model.q_scaler

    ei, charge, grad_e_scaled, grad_q_scaled = model.calculate_Ei_with_grad(
        imagetype_map,
        feats_scaled,
        device,
    )
    grad_e_raw_ref = torch.autograd.grad(ei.sum(), feats_raw, retain_graph=True)[0]
    grad_q_raw_ref = torch.autograd.grad(charge.sum(), feats_raw)[0]

    torch.testing.assert_close(
        grad_e_scaled * model.q_scaler,
        grad_e_raw_ref,
        rtol=1e-10,
        atol=1e-10,
    )
    torch.testing.assert_close(
        grad_q_scaled * model.q_scaler,
        grad_q_raw_ref,
        rtol=1e-10,
        atol=1e-10,
    )


if __name__ == "__main__":
    test_calculate_ei_with_grad_matches_autograd_for_energy_only_model()
    test_calculate_ei_with_grad_matches_autograd_for_charge_model()
    test_calculate_ei_with_grad_keeps_gpumd_common_bias_behavior()
    test_calculate_ei_with_grad_uses_scaled_feature_convention()
