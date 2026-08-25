import torch

from src.model.nep_fitting import FittingNet
from src.model.nep_net import CalcOps
from src.model.nep_net import NEP


def _make_nep(train_2b=True, l_max_3b=0):
    model = NEP.__new__(NEP)
    torch.nn.Module.__init__(model)
    model.train_2b = train_2b
    model.l_max_3b = l_max_3b
    return model


def test_force_virial_helper_matches_radial_cpu_formula():
    model = _make_nep(train_2b=True, l_max_3b=0)
    device = torch.device("cpu")
    dtype = torch.float64
    num_atom = torch.tensor([2], dtype=torch.int64, device=device)
    Ri = torch.tensor(
        [
            [[1.0, 0.5, 0.0, 0.0]],
            [[1.0, -0.5, 0.0, 0.0]],
        ],
        dtype=dtype,
        device=device,
    )
    Ri_d = torch.zeros(2, 1, 4, 3, dtype=dtype, device=device)
    Ri_d[:, :, 1, 0] = 1.0
    dE = torch.tensor(
        [
            [[0.0, 2.0, 0.0, 0.0]],
            [[0.0, -3.0, 0.0, 0.0]],
        ],
        dtype=dtype,
        device=device,
    )
    list_neigh = torch.tensor([[1], [0]], dtype=torch.int64, device=device)

    force, virial = model.calculate_force_virial_from_descriptor_grad(
        dE,
        None,
        None,
        Ri,
        Ri_d,
        None,
        None,
        None,
        None,
        list_neigh,
        None,
        None,
        num_atom,
        device,
        dtype,
    )

    expected_force = torch.tensor(
        [
            [5.0, 0.0, 0.0],
            [-5.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    expected_virial = torch.tensor(
        [[-2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=dtype,
        device=device,
    )
    torch.testing.assert_close(force, expected_force)
    torch.testing.assert_close(virial, expected_virial)


def test_radial_analytical_force_matches_autograd_on_cuda():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20260728)
    device = torch.device("cuda")
    dtype = torch.float64
    model = _make_nep(train_2b=True, l_max_3b=0)
    model.dtype = dtype
    model.charge_mode = 0
    model.gpumd_nep4 = False
    model.two_feat_num = 2
    model.multi_feat_num = 0
    model.q_scaler = torch.tensor([0.7, -1.3], dtype=dtype, device=device)
    model.input_param = type("InputParamStub", (), {})()
    model.input_param.nep_param = type("NepParamStub", (), {"fix_cij": False})()
    fit_net = FittingNet(
        network_size=[4, 1],
        bias=True,
        resnet_dt=False,
        activation="tanh",
        input_dim=2,
        ener_shift=0.0,
    ).to(device=device, dtype=dtype)
    model.fitting_net = torch.nn.ModuleList([fit_net])
    model.c_param_2 = torch.nn.Parameter(
        torch.randn(1, 1, 2, 2, dtype=dtype, device=device))

    num_atom = torch.tensor([2], dtype=torch.int64, device=device)
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
    ei, _, grad_feat_E_scaled, _ = model.calculate_Ei_with_grad(atom_type_map, feats_scaled, device)
    etot = ei.sum().reshape(1, 1)

    force_ref, virial_ref = model.calculate_force_virial(
        ri,
        ri_d,
        None,
        None,
        None,
        None,
        etot,
        2,
        nl,
        None,
        None,
        num_atom,
        device,
        dtype,
    )
    grad_feat_raw = grad_feat_E_scaled * model.q_scaler
    dE_radial = CalcOps.calculateNepFeatInputGrad(
        grad_feat_raw,
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
    force, virial = model.calculate_force_virial_from_descriptor_grad(
        dE_radial,
        None,
        None,
        ri,
        ri_d,
        None,
        None,
        None,
        None,
        nl,
        None,
        None,
        num_atom,
        device,
        dtype,
    )
    torch.testing.assert_close(force, force_ref, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(virial, virial_ref, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    test_force_virial_helper_matches_radial_cpu_formula()
    test_radial_analytical_force_matches_autograd_on_cuda()
