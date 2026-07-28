import torch

from src.PWMLFF.nep_mods.nep_trainer import _get_fragment_charge_loss, _get_model_output_requests


class _Args:
    def __init__(self):
        self.optimizer_param = type("OptimizerParamStub", (), {})()


def test_fragment_charge_loss_aggregates_per_image_fragment_namespace():
    atomic_charge = torch.tensor([0.2, 0.3, 1.0, -0.4, 0.6], dtype=torch.float64)
    sample = {
        "num_atom": torch.tensor([3, 2], dtype=torch.int64),
        "fragment": torch.tensor([0, 0, 1, 0, 0], dtype=torch.int64),
        "fragment_charge": torch.tensor([0.7, 0.7, float("nan"), 0.1, 0.1], dtype=torch.float64),
    }
    criterion = torch.nn.MSELoss()

    loss = _get_fragment_charge_loss(atomic_charge, sample, criterion)

    pred = torch.tensor([[0.5], [0.2]], dtype=torch.float64)
    label = torch.tensor([[0.7], [0.1]], dtype=torch.float64)
    expected = criterion(pred, label)
    torch.testing.assert_close(loss, expected)


def test_fragment_charge_loss_ignores_negative_fragment_and_nan_labels():
    atomic_charge = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    sample = {
        "num_atom": torch.tensor([3], dtype=torch.int64),
        "fragment": torch.tensor([-1, 0, 0], dtype=torch.int64),
        "fragment_charge": torch.tensor([0.0, float("nan"), float("nan")], dtype=torch.float64),
    }
    assert _get_fragment_charge_loss(atomic_charge, sample, torch.nn.MSELoss()) is None


def test_model_output_requests_skip_coordinate_graph_for_fragment_charge_only():
    args = _Args()
    args.optimizer_param.train_force = False
    args.optimizer_param.train_energy = False
    args.optimizer_param.train_bec = False
    sample = {
        "fragment": torch.tensor([0]),
        "fragment_charge": torch.tensor([1.0]),
    }

    requests = _get_model_output_requests(sample, args, train_virial=False)

    assert requests == {
        "need_force": False,
        "need_bec": False,
        "need_charge_virial": False,
        "need_charge_energy": False,
    }


def test_model_output_requests_keep_charge_energy_for_force_or_virial():
    args = _Args()
    args.optimizer_param.train_force = True
    args.optimizer_param.train_energy = False
    args.optimizer_param.train_bec = False
    sample = {}

    requests = _get_model_output_requests(sample, args, train_virial=True)

    assert requests["need_force"] is True
    assert requests["need_charge_virial"] is True
    assert requests["need_charge_energy"] is True


if __name__ == "__main__":
    test_fragment_charge_loss_aggregates_per_image_fragment_namespace()
    test_fragment_charge_loss_ignores_negative_fragment_and_nan_labels()
    test_model_output_requests_skip_coordinate_graph_for_fragment_charge_only()
    test_model_output_requests_keep_charge_energy_for_force_or_virial()
