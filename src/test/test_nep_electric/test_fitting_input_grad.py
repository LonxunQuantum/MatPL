import torch

from src.model.nep_fitting import FittingNet, QNEPFittingNet


def _finite_parameter_grads(module):
    missing = []
    not_finite = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
        elif not torch.isfinite(param.grad).all():
            not_finite.append(name)
    return missing, not_finite


def test_fitting_forward_with_input_grad_matches_autograd():
    cases = [
        ([4, 1], False),
        ([4, 4, 1], False),
        ([4, 4, 1], True),
        ([5, 1], False),
    ]
    for network_size, resnet_dt in cases:
        torch.manual_seed(20260727)
        input_dim = 4
        net = FittingNet(
            network_size=network_size,
            bias=True,
            resnet_dt=resnet_dt,
            activation="tanh",
            input_dim=input_dim,
            ener_shift=0.125,
        ).double()
        x = torch.randn(5, input_dim, dtype=torch.float64, requires_grad=True)

        energy, de_dx = net.forward_with_input_grad(x)

        energy_ref = net(x)
        de_ref = torch.autograd.grad(energy_ref.sum(), x, create_graph=True)[0]
        torch.testing.assert_close(energy, energy_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(de_dx, de_ref, rtol=1e-10, atol=1e-10)

        loss = energy.sum() + (de_dx.square()).sum()
        loss.backward()
        missing, not_finite = _finite_parameter_grads(net)
        assert missing == []
        assert not_finite == []


def test_qnep_forward_with_input_grad_matches_autograd():
    cases = [
        ([4], False),
        ([4, 4], False),
        ([4, 4], True),
        ([5], False),
    ]
    for network_size, resnet_dt in cases:
        torch.manual_seed(20260727)
        input_dim = 4
        net = QNEPFittingNet(
            network_size=network_size,
            bias=True,
            resnet_dt=resnet_dt,
            activation="tanh",
            input_dim=input_dim,
            ener_shift=0.25,
            charge_mode=2,
        ).double()
        x = torch.randn(6, input_dim, dtype=torch.float64, requires_grad=True)

        energy, charge, de_dx, dq_dx = net.forward_with_input_grad(x)

        energy_ref, charge_ref = net(x)
        de_ref = torch.autograd.grad(
            energy_ref.sum(),
            x,
            retain_graph=True,
            create_graph=True,
        )[0]
        dq_ref = torch.autograd.grad(charge_ref.sum(), x, create_graph=True)[0]

        torch.testing.assert_close(energy, energy_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(charge, charge_ref, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(de_dx, de_ref, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(dq_dx, dq_ref, rtol=1e-10, atol=1e-10)

        loss = energy.sum() + charge.sum() + (de_dx.square()).sum() + (dq_dx.square()).sum()
        loss.backward()
        missing, not_finite = _finite_parameter_grads(net)
        assert missing == []
        assert not_finite == []


def test_qnep_gpumd_common_bias_style_energy_biasless_output():
    torch.manual_seed(20260727)
    net = QNEPFittingNet(
        network_size=[3, 3],
        bias=True,
        resnet_dt=True,
        activation="tanh",
        input_dim=3,
        ener_shift=-0.5,
        charge_mode=2,
        last_bias=False,
    ).double()
    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)

    energy, charge, de_dx, dq_dx = net.forward_with_input_grad(x)
    energy_ref, charge_ref = net(x)

    torch.testing.assert_close(energy, energy_ref, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(charge, charge_ref, rtol=1e-12, atol=1e-12)
    assert de_dx.shape == x.shape
    assert dq_dx.shape == x.shape


if __name__ == "__main__":
    test_fitting_forward_with_input_grad_matches_autograd()
    test_qnep_forward_with_input_grad_matches_autograd()
    test_qnep_gpumd_common_bias_style_energy_biasless_output()
