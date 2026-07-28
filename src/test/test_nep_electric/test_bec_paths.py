import torch

from src.model.nep_fitting import QNEPFittingNet
from src.model.nep_net import CalcOps, NEP


def _make_charge_nep():
    model = NEP.__new__(NEP)
    torch.nn.Module.__init__(model)
    model.train_2b = True
    model.l_max_3b = 0
    model.dtype = torch.float64
    model.charge_mode = 2
    model.gpumd_nep4 = False
    model.two_feat_num = 2
    model.multi_feat_num = 0
    model.sqrt_epsilon_inf = torch.nn.Parameter(torch.tensor(1.7, dtype=torch.float64))
    return model


def test_radial_analytical_bec_matches_autograd_on_cuda():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260728)
    device = torch.device("cuda")
    dtype = torch.float64
    model = _make_charge_nep().to(device=device, dtype=dtype)
    model.q_scaler = torch.tensor([0.8, -1.1], dtype=dtype, device=device)
    model.input_param = type("InputParamStub", (), {})()
    model.input_param.nep_param = type("NepParamStub", (), {"fix_cij": False})()
    model.fitting_net = torch.nn.ModuleList([
        QNEPFittingNet(
            network_size=[4],
            bias=True,
            resnet_dt=False,
            activation="tanh",
            input_dim=2,
            ener_shift=0.0,
            charge_mode=2,
        ).to(device=device, dtype=dtype)
    ])
    model.c_param_2 = torch.nn.Parameter(
        torch.randn(1, 1, 2, 2, dtype=dtype, device=device))

    atom_type_map = torch.zeros(2, dtype=torch.int64, device=device)
    nl = torch.tensor([[1], [0]], dtype=torch.int64, device=device)
    ri = torch.tensor(
        [
            [[1.0, 0.35, 0.10, 0.05]],
            [[1.1, -0.25, 0.15, -0.08]],
        ],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    ri_d = torch.zeros(2, 1, 4, 3, dtype=dtype, device=device)
    ri_d[:, :, 1, 0] = 1.0
    ri_d[:, :, 2, 1] = 1.0
    ri_d[:, :, 3, 2] = 1.0
    feats0 = torch.zeros(2, 2, dtype=dtype, device=device, requires_grad=True)

    feat, dfeat_c2, dfeat_2b, dfeat_2b_noc = CalcOps.calculateNepFeatWithGradContext(
        model.c_param_2.contiguous(),
        ri,
        nl,
        atom_type_map,
        feats0,
        5.0,
        0,
        0,
    )
    feats_scaled = feat * model.q_scaler
    _, charge, _, grad_feat_Q_scaled = model.calculate_Ei_with_grad(atom_type_map, feats_scaled, device)
    charge_shifted = charge + torch.tensor([0.2, -0.3], dtype=dtype, device=device)

    bec_ref = model.calculate_bec(
        charge,
        charge_shifted,
        ri,
        ri_d,
        nl,
        None,
        None,
        None,
        device,
        dtype,
    )
    grad_q_raw = grad_feat_Q_scaled * model.q_scaler
    dQ_radial = CalcOps.calculateNepFeatInputGrad(
        grad_q_raw,
        model.c_param_2.contiguous(),
        ri,
        nl,
        dfeat_c2,
        dfeat_2b,
        dfeat_2b_noc,
        atom_type_map,
        0,
        0,
    )
    bec = model.calculate_bec_from_descriptor_grad(
        charge_shifted,
        dQ_radial,
        ri,
        ri_d,
        nl,
        None,
        None,
        None,
        None,
        device,
        dtype,
    )

    torch.testing.assert_close(bec, bec_ref, rtol=1e-7, atol=1e-7)
    loss = bec.square().sum()
    checked_params = [
        param
        for name, param in model.fitting_net.named_parameters()
        if not name.startswith("0.energy_head.")
    ]
    grads = torch.autograd.grad(
        loss,
        checked_params + [model.c_param_2, model.sqrt_epsilon_inf],
        allow_unused=False,
    )
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_angular_analytical_bec_parameter_grad_matches_autograd_on_cuda():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260728)
    device = torch.device("cuda")
    dtype = torch.float64
    model = NEP.__new__(NEP)
    torch.nn.Module.__init__(model)
    model.train_2b = False
    model.l_max_3b = 4
    model.l_max_4b = 2
    model.l_max_5b = 1
    model.dtype = dtype
    model.charge_mode = 2
    model.gpumd_nep4 = False
    model.two_feat_num = 0
    model.multi_feat_num = 30
    model.sqrt_epsilon_inf = torch.nn.Parameter(torch.tensor(1.3, dtype=dtype, device=device))
    model.q_scaler = torch.linspace(0.7, 1.3, model.multi_feat_num, dtype=dtype, device=device)
    model.input_param = type("InputParamStub", (), {})()
    model.input_param.nep_param = type("NepParamStub", (), {"fix_cij": False})()
    model.fitting_net = torch.nn.ModuleList([
        QNEPFittingNet(
            network_size=[8],
            bias=True,
            resnet_dt=False,
            activation="tanh",
            input_dim=model.multi_feat_num,
            ener_shift=0.0,
            charge_mode=2,
        ).to(device=device, dtype=dtype)
    ])
    model.c_param_3 = torch.nn.Parameter(torch.randn(1, 1, 5, 9, dtype=dtype, device=device))

    atom_type_map = torch.zeros(3, dtype=torch.int64, device=device)
    nl = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.int64, device=device)
    ri = torch.randn(3, 2, 4, dtype=dtype, device=device, requires_grad=True)
    ri = ri.clone()
    ri[:, :, 0] = ri[:, :, 0].abs() + 0.8
    ri_d = torch.randn(3, 2, 4, 3, dtype=dtype, device=device)
    feats0 = torch.zeros(3, model.multi_feat_num, dtype=dtype, device=device, requires_grad=True)

    feat, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz = CalcOps.calculateNepMbFeatWithGradContext(
        model.c_param_3.contiguous(),
        ri,
        nl,
        atom_type_map,
        feats0,
        0,
        model.l_max_3b,
        model.l_max_4b,
        model.l_max_5b,
        5.0,
        0,
    )
    feats_scaled = feat * model.q_scaler
    _, charge, _, grad_feat_Q_scaled = model.calculate_Ei_with_grad(atom_type_map, feats_scaled, device)
    charge_shifted = charge + torch.tensor([0.2, -0.3, 0.1], dtype=dtype, device=device)
    bec_ref = model.calculate_bec(
        charge,
        charge_shifted,
        None,
        None,
        None,
        ri,
        ri_d,
        nl,
        device,
        dtype,
    )
    grad_q_raw = grad_feat_Q_scaled * model.q_scaler
    dQ_angular = CalcOps.calculateNepMbFeatInputGrad(
        grad_q_raw,
        model.c_param_3.contiguous(),
        ri,
        nl,
        dfeat_c3,
        dfeat_3b,
        dfeat_3b_noc,
        sum_fxyz,
        atom_type_map,
        0,
        model.l_max_3b,
        model.l_max_4b,
        model.l_max_5b,
        5.0,
        0,
    )
    bec = model.calculate_bec_from_descriptor_grad(
        charge_shifted,
        None,
        None,
        None,
        None,
        dQ_angular,
        ri,
        ri_d,
        nl,
        device,
        dtype,
    )
    torch.testing.assert_close(bec, bec_ref, rtol=1e-7, atol=1e-7)

    params = [p for n, p in model.fitting_net.named_parameters() if not n.startswith("0.energy_head.")]
    params = params + [model.c_param_3, model.sqrt_epsilon_inf]
    ref_grads = torch.autograd.grad(bec_ref.square().sum(), params, retain_graph=True)
    new_grads = torch.autograd.grad(bec.square().sum(), params)
    for new_grad, ref_grad in zip(new_grads, ref_grads):
        torch.testing.assert_close(new_grad, ref_grad, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    test_radial_analytical_bec_matches_autograd_on_cuda()
    test_angular_analytical_bec_parameter_grad_matches_autograd_on_cuda()
