import math

import numpy as np
import pytest
import torch

from src.PWMLFF.nep_network import (
    build_nep_checkpoint,
    load_nep_checkpoint_with_fallback,
    restore_nep_training_state,
)
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


def _minimal_charge_model(sqrt_epsilon_inf=1.4):
    model = object.__new__(NEP)
    torch.nn.Module.__init__(model)
    if not torch.is_tensor(sqrt_epsilon_inf):
        sqrt_epsilon_inf = torch.tensor(
            sqrt_epsilon_inf, dtype=torch.float64)
    model._set_sqrt_epsilon_inf(sqrt_epsilon_inf)
    return model


def _minimal_configured_charge_model(initial=1.4, fixed=None):
    model = object.__new__(NEP)
    torch.nn.Module.__init__(model)
    model.register_buffer("_fixed_sqrt_epsilon_inf", None, persistent=False)
    model.register_parameter("raw_sqrt_epsilon_inf", None)
    model._configure_sqrt_epsilon_inf(
        initial,
        fixed_sqrt_epsilon_inf=fixed,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    return model


def _resume_optimizer_and_scheduler(parameter):
    optimizer = torch.optim.Adam([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2, T_mult=2, eta_min=1.0e-5)
    return optimizer, scheduler


def _checkpoint_with_training_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer, scheduler = _resume_optimizer_and_scheduler(parameter)
    parameter.square().sum().backward()
    optimizer.step()
    scheduler.step(1.5)
    return {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def test_resume_restores_optimizer_moments_and_scheduler_position():
    checkpoint = _checkpoint_with_training_state()
    parameter = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))
    optimizer, scheduler = _resume_optimizer_and_scheduler(parameter)

    restored = restore_nep_training_state(
        checkpoint, optimizer, scheduler, reset_epoch=False)

    assert restored == (True, True)
    state = optimizer.state[parameter]
    assert state["step"].item() == 1
    assert torch.allclose(
        state["exp_avg"], torch.tensor([0.2], dtype=torch.float64))
    assert scheduler.last_epoch == 1
    assert scheduler.T_cur == 1.5


def test_legacy_checkpoint_without_training_state_uses_fresh_objects():
    parameter = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))
    optimizer, scheduler = _resume_optimizer_and_scheduler(parameter)
    initial_scheduler_state = scheduler.state_dict()

    restored = restore_nep_training_state(
        {"state_dict": {}}, optimizer, scheduler, reset_epoch=False)

    assert restored == (False, False)
    assert optimizer.state == {}
    assert scheduler.state_dict() == initial_scheduler_state


def test_reset_epoch_ignores_available_training_state():
    checkpoint = _checkpoint_with_training_state()
    parameter = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))
    optimizer, scheduler = _resume_optimizer_and_scheduler(parameter)
    initial_scheduler_state = scheduler.state_dict()

    restored = restore_nep_training_state(
        checkpoint, optimizer, scheduler, reset_epoch=True)

    assert restored == (False, False)
    assert optimizer.state == {}
    assert scheduler.state_dict() == initial_scheduler_state


def test_checkpoint_without_scheduler_restores_optimizer_only():
    checkpoint = _checkpoint_with_training_state()
    checkpoint.pop("scheduler")
    parameter = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))
    optimizer, scheduler = _resume_optimizer_and_scheduler(parameter)

    restored = restore_nep_training_state(
        checkpoint, optimizer, scheduler, reset_epoch=False)

    assert restored == (True, False)
    assert optimizer.state[parameter]["step"].item() == 1
    assert scheduler.last_epoch == 0


def test_new_checkpoint_contains_optimizer_and_scheduler_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    optimizer, scheduler = _resume_optimizer_and_scheduler(parameter)
    scheduler.step(1.5)

    checkpoint = build_nep_checkpoint(
        {"model_type": "NEP"}, 12, model, optimizer, scheduler)

    assert checkpoint["json_file"] == {"model_type": "NEP"}
    assert checkpoint["epoch"] == 12
    assert checkpoint["state_dict"].keys() == model.state_dict().keys()
    assert abs(
        checkpoint["optimizer"]["param_groups"][0]["lr"]
        - 0.0014730016279731956
    ) < 1.0e-15
    assert checkpoint["scheduler"]["T_cur"] == 1.5


def test_new_checkpoint_records_absent_scheduler():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    checkpoint = build_nep_checkpoint(
        {"model_type": "NEP"}, 3, model, optimizer, None)

    assert checkpoint["scheduler"] is None


def test_recover_uses_latest_periodic_checkpoint_when_primary_is_nep_text(tmp_path):
    primary = tmp_path / "nep_model.ckpt"
    primary.write_text("nep5_charge2 1 Li\n", encoding="utf-8")
    torch.save(
        {"epoch": 6, "state_dict": {"weight": torch.tensor([6.0])}},
        tmp_path / "epoch_6_nep_model.ckpt",
    )
    torch.save(
        {"epoch": 12, "state_dict": {"weight": torch.tensor([12.0])}},
        tmp_path / "epoch_12_nep_model.ckpt",
    )

    checkpoint, loaded_path = load_nep_checkpoint_with_fallback(
        str(primary), map_location="cpu", allow_periodic_fallback=True)

    assert checkpoint["epoch"] == 12
    assert loaded_path == str(tmp_path / "epoch_12_nep_model.ckpt")


def test_explicit_load_does_not_silently_fallback_from_nep_text(tmp_path):
    primary = tmp_path / "nep_model.ckpt"
    primary.write_text("nep5_charge2 1 Li\n", encoding="utf-8")
    torch.save(
        {"epoch": 12, "state_dict": {"weight": torch.tensor([12.0])}},
        tmp_path / "epoch_12_nep_model.ckpt",
    )

    with pytest.raises(RuntimeError, match="not a valid PyTorch NEP checkpoint"):
        load_nep_checkpoint_with_fallback(
            str(primary), map_location="cpu", allow_periodic_fallback=False)


def test_fixed_sqrt_epsilon_inf_is_not_a_parameter_or_optimizer_variable():
    model = _minimal_configured_charge_model(fixed=1.4)
    model.other = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    assert "raw_sqrt_epsilon_inf" not in dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    (model.other * model.sqrt_epsilon_inf).backward()
    optimizer.step()

    torch.testing.assert_close(
        model.sqrt_epsilon_inf, torch.tensor(1.4, dtype=torch.float64))
    assert model.sqrt_epsilon_inf.grad is None


@pytest.mark.parametrize("invalid", [0.999, float("nan"), float("inf")])
def test_fixed_sqrt_epsilon_inf_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="fixed_sqrt_epsilon_inf"):
        _minimal_configured_charge_model(fixed=invalid)


def test_fixed_sqrt_epsilon_inf_accepts_boundary_value_one():
    model = _minimal_configured_charge_model(fixed=1.0)
    torch.testing.assert_close(
        model.sqrt_epsilon_inf, torch.tensor(1.0, dtype=torch.float64))


def test_fixed_json_value_overrides_trainable_checkpoint_value():
    checkpoint_state = _minimal_configured_charge_model(
        initial=1.8).state_dict()
    fixed_model = _minimal_configured_charge_model(fixed=1.3)

    fixed_model.load_state_dict(checkpoint_state, strict=True)

    torch.testing.assert_close(
        fixed_model.sqrt_epsilon_inf,
        torch.tensor(1.3, dtype=torch.float64))
    assert "raw_sqrt_epsilon_inf" not in dict(fixed_model.named_parameters())


def test_fixed_checkpoint_can_initialize_trainable_sqrt_epsilon_inf():
    checkpoint_state = _minimal_configured_charge_model(
        fixed=1.4).state_dict()
    assert "raw_sqrt_epsilon_inf" not in checkpoint_state
    assert "sqrt_epsilon_inf" in checkpoint_state
    trainable_model = _minimal_configured_charge_model(initial=1.8)

    trainable_model.load_state_dict(checkpoint_state, strict=True)

    torch.testing.assert_close(
        trainable_model.sqrt_epsilon_inf,
        torch.tensor(1.4, dtype=torch.float64))
    assert trainable_model.raw_sqrt_epsilon_inf.requires_grad


def test_old_trainable_optimizer_is_skipped_for_fixed_parameter_layout():
    source_model = _minimal_configured_charge_model(initial=1.4)
    source_model.other = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    checkpoint = {"optimizer": source_optimizer.state_dict()}
    fixed_model = _minimal_configured_charge_model(fixed=1.4)
    fixed_model.other = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    fixed_optimizer = torch.optim.Adam(fixed_model.parameters(), lr=0.01)

    restored = restore_nep_training_state(
        checkpoint,
        fixed_optimizer,
        scheduler=None,
        reset_epoch=False,
        allow_optimizer_param_group_mismatch=True,
    )

    assert restored == (False, False)
    assert fixed_optimizer.state == {}


def test_matching_fixed_optimizer_state_is_restored():
    source_model = _minimal_configured_charge_model(fixed=1.4)
    source_model.other = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=0.01)
    source_model.other.square().backward()
    source_optimizer.step()
    target_model = _minimal_configured_charge_model(fixed=1.4)
    target_model.other = torch.nn.Parameter(torch.tensor(3.0, dtype=torch.float64))
    target_optimizer = torch.optim.Adam(target_model.parameters(), lr=0.01)

    restored = restore_nep_training_state(
        {"optimizer": source_optimizer.state_dict()},
        target_optimizer,
        scheduler=None,
        reset_epoch=False,
        allow_optimizer_param_group_mismatch=True,
    )

    assert restored == (True, False)
    assert target_optimizer.state[target_model.other]["step"].item() == 1


def test_sqrt_epsilon_inf_stays_above_one_for_unbounded_raw_parameter():
    model = _minimal_charge_model(1.4)

    with torch.no_grad():
        model.raw_sqrt_epsilon_inf.fill_(-100.0)

    assert model.sqrt_epsilon_inf.item() >= 1.0
    physical_charge, shifted_charge = model.shift_total_charge(
        torch.tensor([0.2, 0.4], dtype=torch.float64),
        num_atom=torch.tensor([2], dtype=torch.int64),
        charge_label=torch.tensor([1.0], dtype=torch.float64),
    )
    assert torch.isfinite(physical_charge).all()
    assert torch.isfinite(shifted_charge).all()


def test_sqrt_epsilon_inf_state_dict_keeps_physical_compatibility_value():
    model = _minimal_charge_model(1.4)

    state_dict = model.state_dict()

    assert "raw_sqrt_epsilon_inf" in state_dict
    torch.testing.assert_close(
        state_dict["sqrt_epsilon_inf"],
        torch.tensor(1.4, dtype=torch.float64),
    )


def test_legacy_sqrt_epsilon_inf_state_dict_is_migrated():
    model = _minimal_charge_model(1.4)

    model.load_state_dict({
        "sqrt_epsilon_inf": torch.tensor(1.7, dtype=torch.float64),
    })

    torch.testing.assert_close(
        model.sqrt_epsilon_inf,
        torch.tensor(1.7, dtype=torch.float64),
    )


def test_legacy_nonphysical_sqrt_epsilon_inf_is_rejected():
    model = _minimal_charge_model(1.4)

    with pytest.raises(RuntimeError, match="greater than 1"):
        model.load_state_dict({
            "sqrt_epsilon_inf": torch.tensor(0.5, dtype=torch.float64),
        })


def test_shift_total_charge_converts_naked_label_to_screened_target():
    model = object.__new__(NEP)
    torch.nn.Module.__init__(model)
    model._set_sqrt_epsilon_inf(torch.tensor(2.0, dtype=torch.float64))
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
