import os

import torch


_LIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "op",
    "build",
    "lib",
    "libCalcOps_bind.so",
)
try:
    torch.ops.load_library(_LIB_PATH)
    CalcOps = torch.ops.CalcOps_cuda
except OSError as exc:
    if "libcuda.so" not in str(exc):
        raise
    CalcOps = None


def _require_cuda():
    if CalcOps is None or not torch.cuda.is_available():
        return False
    return True


def test_descriptor_vjp_ops_are_registered():
    if CalcOps is None:
        return
    names = dir(CalcOps)
    assert "calculateNepFeatWithGradContext" in names
    assert "calculateNepFeatInputGrad" in names
    assert "calculateNepMbFeatWithGradContext" in names
    assert "calculateNepMbFeatInputGrad" in names


def test_radial_descriptor_vjp_matches_autograd():
    if not _require_cuda():
        return
    torch.manual_seed(20260727)
    device = torch.device("cuda")
    dtype = torch.float64
    natoms = 2
    max_neigh = 1
    ntypes = 1
    n_max = 2
    n_base = 2
    coeff = torch.randn(ntypes, ntypes, n_max, n_base, dtype=dtype, device=device, requires_grad=True)
    d12 = torch.tensor(
        [[[1.0, 0.25, 0.10, 0.05]], [[1.1, -0.20, 0.15, -0.07]]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    nl = torch.tensor([[1], [0]], dtype=torch.int64, device=device)
    atom_map = torch.zeros(natoms, dtype=torch.int64, device=device)
    feats = torch.zeros(natoms, n_max, dtype=dtype, device=device, requires_grad=True)

    feat, dfeat_c2, dfeat_2b, dfeat_2b_noc = CalcOps.calculateNepFeatWithGradContext(
        coeff,
        d12,
        nl,
        atom_map,
        feats,
        5.0,
        0,
        0,
    )
    seed = torch.randn_like(feat, requires_grad=True)
    vjp = CalcOps.calculateNepFeatInputGrad(
        seed,
        coeff,
        d12,
        nl,
        dfeat_c2,
        dfeat_2b,
        dfeat_2b_noc,
        atom_map,
        0,
        0,
    )
    ref = torch.autograd.grad(feat, d12, grad_outputs=seed, create_graph=True)[0]
    torch.testing.assert_close(vjp, ref, rtol=1e-7, atol=1e-7)

    loss = vjp.square().sum()
    grad_seed, grad_coeff = torch.autograd.grad(loss, (seed, coeff), allow_unused=False)
    assert torch.isfinite(grad_seed).all()
    assert torch.isfinite(grad_coeff).all()


def test_angular_descriptor_vjp_matches_autograd():
    if not _require_cuda():
        return
    torch.manual_seed(20260727)
    device = torch.device("cuda")
    dtype = torch.float64
    natoms = 2
    max_neigh = 1
    ntypes = 1
    n_max = 2
    n_base = 2
    feat_2b_num = 0
    lmax_3 = 1
    lmax_4 = 0
    lmax_5 = 0
    coeff = torch.randn(ntypes, ntypes, n_max, n_base, dtype=dtype, device=device, requires_grad=True)
    d12 = torch.tensor(
        [[[1.0, 0.30, 0.20, 0.10]], [[1.2, -0.10, 0.25, -0.05]]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    nl = torch.tensor([[1], [0]], dtype=torch.int64, device=device)
    atom_map = torch.zeros(natoms, dtype=torch.int64, device=device)
    feats = torch.zeros(natoms, n_max * lmax_3, dtype=dtype, device=device, requires_grad=True)

    feat, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz = CalcOps.calculateNepMbFeatWithGradContext(
        coeff,
        d12,
        nl,
        atom_map,
        feats,
        feat_2b_num,
        lmax_3,
        lmax_4,
        lmax_5,
        5.0,
        0,
    )
    seed = torch.randn_like(feat, requires_grad=True)
    vjp = CalcOps.calculateNepMbFeatInputGrad(
        seed,
        coeff,
        d12,
        nl,
        dfeat_c3,
        dfeat_3b,
        dfeat_3b_noc,
        sum_fxyz,
        atom_map,
        feat_2b_num,
        lmax_3,
        lmax_4,
        lmax_5,
        5.0,
        0,
    )
    ref = torch.autograd.grad(feat, d12, grad_outputs=seed, create_graph=True)[0]
    torch.testing.assert_close(vjp, ref, rtol=1e-7, atol=1e-7)

    loss = vjp.square().sum()
    grad_seed, grad_coeff = torch.autograd.grad(loss, (seed, coeff), allow_unused=False)
    assert torch.isfinite(grad_seed).all()
    assert torch.isfinite(grad_coeff).all()


def test_angular_descriptor_vjp_matches_autograd_for_real_lmax_shape():
    if not _require_cuda():
        return
    torch.manual_seed(20260728)
    device = torch.device("cuda")
    dtype = torch.float64
    natoms = 3
    max_neigh = 2
    ntypes = 2
    n_max = 4
    n_base = 8
    feat_2b_num = 6
    lmax_3 = 4
    lmax_4 = 2
    lmax_5 = 1
    multi_feat_num = n_max * lmax_3 + n_max * lmax_4 + n_max * lmax_5
    coeff = torch.randn(ntypes, ntypes, n_max, n_base, dtype=dtype, device=device, requires_grad=True)
    d12 = torch.randn(natoms, max_neigh, 4, dtype=dtype, device=device, requires_grad=True)
    d12 = d12.clone()
    d12[:, :, 0] = d12[:, :, 0].abs() + 0.8
    nl = torch.tensor([[1, 2], [0, 2], [0, 1]], dtype=torch.int64, device=device)
    atom_map = torch.tensor([0, 1, 0], dtype=torch.int64, device=device)
    feats = torch.zeros(natoms, feat_2b_num + multi_feat_num, dtype=dtype, device=device, requires_grad=True)

    feat, dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz = CalcOps.calculateNepMbFeatWithGradContext(
        coeff,
        d12,
        nl,
        atom_map,
        feats,
        feat_2b_num,
        lmax_3,
        lmax_4,
        lmax_5,
        5.0,
        0,
    )
    seed = torch.randn_like(feat, requires_grad=True)
    vjp = CalcOps.calculateNepMbFeatInputGrad(
        seed,
        coeff,
        d12,
        nl,
        dfeat_c3,
        dfeat_3b,
        dfeat_3b_noc,
        sum_fxyz,
        atom_map,
        feat_2b_num,
        lmax_3,
        lmax_4,
        lmax_5,
        5.0,
        0,
    )
    ref = torch.autograd.grad(feat, d12, grad_outputs=seed, create_graph=True)[0]
    torch.testing.assert_close(vjp, ref, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    test_radial_descriptor_vjp_matches_autograd()
    test_angular_descriptor_vjp_matches_autograd()
