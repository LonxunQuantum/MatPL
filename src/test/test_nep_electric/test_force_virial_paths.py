import torch

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


if __name__ == "__main__":
    test_force_virial_helper_matches_radial_cpu_formula()
