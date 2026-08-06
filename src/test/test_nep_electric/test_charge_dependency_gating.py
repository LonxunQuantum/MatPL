import math

import numpy as np
import torch

from src.PWMLFF.nep_mods.nep_trainer import (
    _build_predict_metric_row,
    _collect_charge_outputs_for_inference,
    _get_fragment_charge_loss,
    _get_model_output_requests,
    get_charge_loss_stats,
)
from src.model.nep_net import NEP
from src.pre_data.nep_data_loader import _build_default_ion_bec
from src.user.optimizer_param import OptimizerParam
from src.user.nep_work import _prepare_nep_test_ckpt_json
from src.utils.train_log import AverageMeter, Summary


class _Args:
    def __init__(self):
        self.optimizer_param = type("OptimizerParamStub", (), {})()


def test_shift_total_charge_converts_naked_label_to_screened_target():
    model = object.__new__(NEP)
    torch.nn.Module.__init__(model)
    model.sqrt_epsilon_inf = torch.nn.Parameter(
        torch.tensor(2.0, dtype=torch.float64))
    screened_charge = torch.tensor([0.2, 0.4], dtype=torch.float64)

    physical_charge_predict, shifted_charge = model.shift_total_charge(
        screened_charge,
        num_atom=torch.tensor([2], dtype=torch.int64),
        charge_label=torch.tensor([1.0], dtype=torch.float64),
    )

    torch.testing.assert_close(
        physical_charge_predict,
        torch.tensor([[1.2]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        shifted_charge,
        torch.tensor([0.15, 0.35], dtype=torch.float64),
    )
    torch.testing.assert_close(
        shifted_charge.sum() * model.sqrt_epsilon_inf,
        torch.tensor(1.0, dtype=torch.float64),
    )


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


def test_fragment_charge_epoch_rmse_is_weighted_by_fragment_count():
    args = _Args()
    args.optimizer_param.train_charge = True
    criterion = torch.nn.MSELoss()
    meter = AverageMeter("Charge", summary_type=Summary.ROOT)

    batches = [
        (
            torch.tensor([1.0], dtype=torch.float64),
            {
                "num_atom": torch.tensor([1], dtype=torch.int64),
                "fragment": torch.tensor([0], dtype=torch.int64),
                "fragment_charge": torch.tensor([0.0], dtype=torch.float64),
            },
        ),
        (
            torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64),
            {
                "num_atom": torch.tensor([3], dtype=torch.int64),
                "fragment": torch.tensor([0, 1, 2], dtype=torch.int64),
                "fragment_charge": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            },
        ),
    ]

    for atomic_charge, sample in batches:
        optimizer_loss, log_loss, target_count = get_charge_loss_stats(
            charge_predict=None,
            sample=sample,
            criterion=criterion,
            args=args,
            atomic_charge=atomic_charge,
        )
        torch.testing.assert_close(optimizer_loss, log_loss)
        meter.update(log_loss.item(), target_count)

    assert meter.count == 4
    assert math.isclose(meter.root, math.sqrt(7.0), rel_tol=0.0, abs_tol=1e-12)


def test_total_charge_stats_keep_per_atom_log_loss_and_count_images():
    args = _Args()
    args.optimizer_param.train_charge = True
    sample = {
        "num_atom": torch.tensor([2, 4], dtype=torch.int64),
        "charge": torch.tensor([0.0, 2.0], dtype=torch.float64),
    }
    charge_predict = torch.tensor([[2.0], [6.0]], dtype=torch.float64)

    optimizer_loss, log_loss, target_count = get_charge_loss_stats(
        charge_predict=charge_predict,
        sample=sample,
        criterion=torch.nn.MSELoss(),
        args=args,
    )

    torch.testing.assert_close(optimizer_loss, torch.tensor(10.0, dtype=torch.float64))
    torch.testing.assert_close(log_loss, torch.tensor(1.0, dtype=torch.float64))
    assert target_count == 2


def test_ion_training_flags_default_to_false_and_serialize():
    optimizer_param = OptimizerParam()
    optimizer_param.set_optimizer({})

    assert optimizer_param.train_charge_ion is False
    assert optimizer_param.train_bec_ion is False
    optimizer_dict = optimizer_param.to_dict()
    assert optimizer_dict["train_charge_ion"] is False
    assert optimizer_dict["train_bec_ion"] is False

    enabled = OptimizerParam()
    enabled.set_optimizer({
        "optimizer": {
            "train_charge_ion": True,
            "train_bec_ion": True,
        }
    })
    assert enabled.train_charge_ion is True
    assert enabled.train_bec_ion is True
    enabled_dict = enabled.to_dict()
    assert enabled_dict["train_charge_ion"] is True
    assert enabled_dict["train_bec_ion"] is True


def test_charge_loss_excludes_ionic_fragments_by_default():
    args = _Args()
    args.optimizer_param.train_charge = True
    args.optimizer_param.train_charge_ion = False
    sample = {
        "num_atom": torch.tensor([3], dtype=torch.int64),
        "fragment": torch.tensor([0, 0, 1], dtype=torch.int64),
        "fragment_charge": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        "charge": torch.tensor([1.0], dtype=torch.float64),
    }
    atomic_charge = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)

    optimizer_loss, log_loss, target_count = get_charge_loss_stats(
        charge_predict=torch.tensor([[0.5]], dtype=torch.float64),
        sample=sample,
        criterion=torch.nn.MSELoss(),
        args=args,
        atomic_charge=atomic_charge,
        charge_scale=torch.tensor(2.0, dtype=torch.float64),
    )

    expected = torch.tensor(0.01, dtype=torch.float64)
    torch.testing.assert_close(optimizer_loss, expected)
    torch.testing.assert_close(log_loss, expected)
    assert target_count == 1


def test_charge_loss_includes_ionic_fragments_when_enabled():
    args = _Args()
    args.optimizer_param.train_charge = True
    args.optimizer_param.train_charge_ion = True
    sample = {
        "num_atom": torch.tensor([3], dtype=torch.int64),
        "fragment": torch.tensor([0, 0, 1], dtype=torch.int64),
        "fragment_charge": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    }
    atomic_charge = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    charge_scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    optimizer_loss, log_loss, target_count = get_charge_loss_stats(
        charge_predict=None,
        sample=sample,
        criterion=torch.nn.MSELoss(),
        args=args,
        atomic_charge=atomic_charge,
        charge_scale=charge_scale,
    )

    expected = torch.tensor(((2.0 * 0.1) ** 2 + (2.0 * 0.4 - 1.0) ** 2) / 2, dtype=torch.float64)
    torch.testing.assert_close(optimizer_loss, expected)
    torch.testing.assert_close(log_loss, expected)
    assert target_count == 2
    optimizer_loss.backward()
    torch.testing.assert_close(
        charge_scale.grad,
        torch.tensor(-0.06, dtype=torch.float64),
    )


def test_all_ionic_fragment_batch_skips_charge_loss_without_total_fallback():
    args = _Args()
    args.optimizer_param.train_charge = True
    args.optimizer_param.train_charge_ion = False
    sample = {
        "num_atom": torch.tensor([2], dtype=torch.int64),
        "fragment": torch.tensor([0, 1], dtype=torch.int64),
        "fragment_charge": torch.tensor([1.0, -1.0], dtype=torch.float64),
        "charge": torch.tensor([0.0], dtype=torch.float64),
    }

    stats = get_charge_loss_stats(
        charge_predict=torch.tensor([[0.3]], dtype=torch.float64),
        sample=sample,
        criterion=torch.nn.MSELoss(),
        args=args,
        atomic_charge=torch.tensor([0.2, -0.2], dtype=torch.float64),
    )

    assert stats == (None, None, 0)


def test_predict_metric_row_keeps_enabled_optional_columns_without_labels():
    args = _Args()
    args.optimizer_param.train_charge = True
    args.optimizer_param.train_bec = True
    args.optimizer_param.train_egroup = True
    args.optimizer_param.train_virial = True

    row = _build_predict_metric_row(
        image_index=3,
        etot_rmse=torch.tensor(1.0),
        etot_atom_rmse=torch.tensor(2.0),
        ei_rmse=torch.tensor(3.0),
        force_rmse=torch.tensor(4.0),
        args=args,
        charge_loss=None,
        bec_loss=None,
        egroup_loss=None,
        virial_loss=None,
        virial_per_atom_loss=None,
    )

    assert row["img_idx"] == 3
    assert set(row) == {
        "img_idx", "RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F",
        "RMSE_charge", "RMSE_BEC", "RMSE_Egroup", "RMSE_virial",
        "RMSE_virial_per_atom",
    }
    for key in (
            "RMSE_charge", "RMSE_BEC", "RMSE_Egroup", "RMSE_virial",
            "RMSE_virial_per_atom"):
        assert math.isnan(row[key])


def test_inference_fragment_charge_rmse_excludes_ions_when_disabled():
    sample = {
        "num_atom": torch.tensor([3]),
        "fragment": torch.tensor([0, 0, 1]),
        "fragment_charge": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    }
    atomic_charge = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)

    labels, predictions, rmses = _collect_charge_outputs_for_inference(
        atomic_charge,
        total_charge_predict=None,
        sample=sample,
        train_charge_ion=False,
        charge_scale=torch.tensor(2.0, dtype=torch.float64),
    )

    np.testing.assert_allclose(labels[0], [0.0])
    np.testing.assert_allclose(predictions[0], [0.2, -0.1, 0.4])
    np.testing.assert_allclose(rmses, [0.1])


def test_inference_fragment_charge_rmse_scales_ions_when_enabled():
    sample = {
        "num_atom": torch.tensor([3]),
        "fragment": torch.tensor([0, 0, 1]),
        "fragment_charge": torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
    }
    atomic_charge = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)

    labels, predictions, rmses = _collect_charge_outputs_for_inference(
        atomic_charge,
        total_charge_predict=None,
        sample=sample,
        train_charge_ion=True,
        charge_scale=torch.tensor(2.0, dtype=torch.float64),
    )

    np.testing.assert_allclose(labels[0], [0.0, 1.0])
    np.testing.assert_allclose(predictions[0], [0.2, -0.1, 0.4])
    np.testing.assert_allclose(rmses, [0.2])


def test_inference_total_charge_rmse_uses_physical_model_output():
    sample = {
        "num_atom": torch.tensor([2]),
        "charge": torch.tensor([1.0], dtype=torch.float64),
    }

    labels, predictions, rmses = _collect_charge_outputs_for_inference(
        torch.tensor([0.1, 0.4], dtype=torch.float64),
        total_charge_predict=torch.tensor([[1.2]], dtype=torch.float64),
        sample=sample,
        train_charge_ion=False,
        charge_scale=torch.tensor(2.0, dtype=torch.float64),
    )

    np.testing.assert_allclose(labels[0], [1.0])
    np.testing.assert_allclose(predictions[0], [0.1, 0.4])
    np.testing.assert_allclose(rmses, [0.2])


def test_prepare_nep_test_ckpt_json_does_not_mutate_checkpoint_metadata():
    checkpoint_json = {
        "datasets_path": ["train_dataset"],
        "train_data": ["train.xyz"],
        "valid_data": ["valid.xyz"],
        "test_data": ["old_test.xyz"],
        "format": "pwmat/movement",
        "optimizer": {"train_charge": True},
    }
    input_json = {
        "test_data": ["new_test.xyz"],
        "format": "extxyz",
    }

    prepared = _prepare_nep_test_ckpt_json(checkpoint_json, input_json)

    assert prepared["datasets_path"] == []
    assert prepared["train_data"] == []
    assert prepared["valid_data"] == []
    assert prepared["test_data"] == ["new_test.xyz"]
    assert prepared["format"] == "extxyz"
    assert prepared["optimizer"] == {"train_charge": True}
    assert checkpoint_json["datasets_path"] == ["train_dataset"]
    assert checkpoint_json["test_data"] == ["old_test.xyz"]


def test_default_ion_bec_uses_valence_on_diagonal():
    bec = _build_default_ion_bec(np.array([3, 11, 19, 12, 20, 8, 26]))

    np.testing.assert_allclose(bec[0, [0, 4, 8]], 1.0)
    np.testing.assert_allclose(bec[1, [0, 4, 8]], 1.0)
    np.testing.assert_allclose(bec[2, [0, 4, 8]], 1.0)
    np.testing.assert_allclose(bec[3, [0, 4, 8]], 2.0)
    np.testing.assert_allclose(bec[4, [0, 4, 8]], 2.0)
    np.testing.assert_allclose(bec[:5, [1, 2, 3, 5, 6, 7]], 0.0)
    np.testing.assert_allclose(bec[5:], -1e6)


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
    test_shift_total_charge_converts_naked_label_to_screened_target()
    test_fragment_charge_loss_aggregates_per_image_fragment_namespace()
    test_fragment_charge_loss_ignores_negative_fragment_and_nan_labels()
    test_fragment_charge_epoch_rmse_is_weighted_by_fragment_count()
    test_total_charge_stats_keep_per_atom_log_loss_and_count_images()
    test_ion_training_flags_default_to_false_and_serialize()
    test_charge_loss_excludes_ionic_fragments_by_default()
    test_charge_loss_includes_ionic_fragments_when_enabled()
    test_all_ionic_fragment_batch_skips_charge_loss_without_total_fallback()
    test_predict_metric_row_keeps_enabled_optional_columns_without_labels()
    test_inference_fragment_charge_rmse_excludes_ions_when_disabled()
    test_inference_fragment_charge_rmse_scales_ions_when_enabled()
    test_inference_total_charge_rmse_uses_physical_model_output()
    test_prepare_nep_test_ckpt_json_does_not_mutate_checkpoint_metadata()
    test_default_ion_bec_uses_valence_on_diagonal()
    test_model_output_requests_skip_coordinate_graph_for_fragment_charge_only()
    test_model_output_requests_keep_charge_energy_for_force_or_virial()
