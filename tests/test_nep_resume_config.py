"""Checkpoint state must not override the current NEP training configuration."""
import copy
import importlib
import math
from types import SimpleNamespace

import pytest
import torch

from src.loss.loss import adjust_lr, get_loss
from src.PWMLFF.nep_network import build_nep_checkpoint, nep_network
from src.user.optimizer_param import OptimizerParam
from src.utils.learning_rate import (
    calculate_loss_weight_progress, calculate_warmup_lr, optimizer_step_with_lr,
)


def settings(**changes):
    result = dict(optimizer="ADAM", reset_epoch=False, learning_rate=0.001,
                  lambda_2=0.01, t_0=2, t_mult=2, stop_lr=1e-7,
                  warm_epochs=1, stop_step=100, decay_step=10,
                  scale_lr=False, batch_size=4, train_energy=False,
                  start_pre_fac_force=100, end_pre_fac_force=1)
    result.update(changes)
    return result


@pytest.fixture
def load_network(monkeypatch, tmp_path):
    module = importlib.import_module("src.PWMLFF.nep_network")
    # Keep the real checkpoint/optimizer initialization; descriptor math is
    # independent of configuration precedence and covered by CPU training tests.
    monkeypatch.setattr(module, "NEP", lambda *args, **kwargs:
                        torch.nn.Linear(2, 1, dtype=torch.float64))
    monkeypatch.setattr(module, "prepare_fitting_jit", lambda model: False)

    def load(config, checkpoint=None, iterations=4):
        params = OptimizerParam()
        params.set_optimizer({"optimizer": config})
        path = tmp_path / "source.ckpt"
        if checkpoint is not None:
            torch.save(checkpoint, path)
        network = object.__new__(nep_network)
        network.input_param = SimpleNamespace(
            optimizer_param=params, world_size=1, rank=0, local_rank=0,
            inference=False, recover_train=True,
            nep_param=SimpleNamespace(model_wb=None),
            file_paths=SimpleNamespace(model_load_path=str(path) if checkpoint else None,
                                      model_save_path=str(tmp_path / "absent.ckpt")))
        network.device = torch.device("cpu")
        network.training_type = torch.float64
        network.is_rank_0 = True
        model, optimizer, scheduler = network.load_model_optimizer([], iterations=iterations)
        return network, model, optimizer, scheduler
    return load


def advance(network, model, optimizer, scheduler, updates):
    params = network.input_param.optimizer_param
    used_lrs = []
    for _ in range(updates):
        step = network.completed_optimizer_updates
        warm = step < network.warmup_optimizer_updates
        if warm:
            lr = calculate_warmup_lr(step, network.warmup_optimizer_updates,
                                     params.stop_lr, network.optimizer_peak_lr)
        elif scheduler is None:
            lr = adjust_lr(step, network.optimizer_peak_lr,
                           params.stop_step, params.decay_step, params.stop_lr)
        elif step == network.warmup_optimizer_updates:
            lr = network.optimizer_peak_lr
        else:
            lr = optimizer.param_groups[0]["lr"]
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        sum(p.square().sum() for p in model.parameters()).backward()
        used_lrs.append(optimizer_step_with_lr(optimizer, None if warm else scheduler))
        network.completed_optimizer_updates += 1
    return used_lrs


def make_checkpoint(source):
    network, model, optimizer, scheduler = source
    return copy.deepcopy(build_nep_checkpoint(
        {}, 2, model, optimizer, scheduler,
        network.completed_optimizer_updates, network.warmup_optimizer_updates))


@pytest.mark.parametrize("name", ["ADAM", "ADAMW"])
@pytest.mark.parametrize("warm_epochs", [0, 3])
@pytest.mark.parametrize("saved_scheduler", [True, False])
def test_resume_current_cosine_config_and_moments(
        load_network, name, warm_epochs, saved_scheduler):
    source = load_network(settings(optimizer=name))
    advance(*source, 8)
    checkpoint = make_checkpoint(source)
    if not saved_scheduler:
        checkpoint.pop("scheduler")
    config = settings(optimizer=name, learning_rate=0.002, lambda_2=0.2,
                      t_0=3, t_mult=1, stop_lr=2e-5, warm_epochs=warm_epochs,
                      scale_lr=True, scale_method="sqrt_batch", batch_size=4,
                      start_pre_fac_force=8, end_pre_fac_force=4)
    target = load_network(config, checkpoint, iterations=5)
    network, model, optimizer, scheduler = target
    assert network.completed_optimizer_updates == 8
    assert network.warmup_optimizer_updates == warm_epochs * 5
    assert network.input_param.optimizer_param.start_epoch == 3
    assert scheduler.T_0 == 15
    assert scheduler.T_mult == 1
    assert scheduler.eta_min == 2e-5
    assert scheduler.base_lrs == pytest.approx([0.004])
    assert optimizer.param_groups[0]["weight_decay"] == 0.2
    if warm_epochs:
        expected_lr = 2e-5 + 8 / 15 * (0.004 - 2e-5)
    else:
        expected_lr = 2e-5 + (0.004 - 2e-5) * (1 + math.cos(math.pi * 8 / 15)) / 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)
    for old, new in zip(source[1].parameters(), model.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        for key in ["step", "exp_avg", "exp_avg_sq"]:
            torch.testing.assert_close(source[2].state[old][key], optimizer.state[new][key])
    loss = get_loss(network.input_param, expected_lr, 1,
                    torch.tensor(1., dtype=torch.float64), torch.tensor(0.),
                    loss_weight_progress=calculate_loss_weight_progress(8, 100))
    assert loss.item() == pytest.approx(8 + (4 - 8) * 8 / 100)
    assert advance(*target, 1)[0] == pytest.approx(expected_lr)


@pytest.mark.parametrize("resume_at", [0, 2, 4, 7, 12, 28])
@pytest.mark.parametrize("cosine", [True, False])
def test_same_config_resume_matches_uninterrupted_updates(load_network, resume_at, cosine):
    config = settings(**({} if cosine else {"t_0": None, "t_mult": None}))
    source = load_network(config)
    advance(*source, resume_at)
    checkpoint = make_checkpoint(source)
    target = load_network(config, checkpoint)
    expected_lrs = advance(*source, 5)
    actual_lrs = advance(*target, 5)
    assert actual_lrs == pytest.approx(expected_lrs, rel=1e-13)
    for old, new in zip(source[1].parameters(), target[1].parameters()):
        torch.testing.assert_close(old, new, rtol=1e-13, atol=1e-14)


def test_resume_exponential_schedule_uses_current_config(load_network):
    source = load_network(settings())
    advance(*source, 8)
    target = load_network(settings(learning_rate=0.003, lambda_2=0.4,
                                  t_0=None, t_mult=None, warm_epochs=0,
                                  stop_lr=3e-6, decay_step=2, stop_step=20),
                          make_checkpoint(source), iterations=5)
    _, _, optimizer, scheduler = target
    expected = 0.003 * (3e-6 / 0.003) ** (8 / 20)
    assert scheduler is None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(expected)
    assert optimizer.param_groups[0]["weight_decay"] == 0.4
    assert advance(*target, 1)[0] == pytest.approx(expected)


def test_legacy_checkpoint_uses_epoch_fallback_and_current_warmup(load_network):
    source = load_network(settings())
    advance(*source, 8)
    checkpoint = make_checkpoint(source)
    checkpoint.pop("optimizer_updates")
    checkpoint.pop("warmup_updates")
    checkpoint.pop("scheduler")
    target = load_network(settings(warm_epochs=3), checkpoint, iterations=5)
    network, _, optimizer, scheduler = target
    assert network.completed_optimizer_updates == 10
    assert network.warmup_optimizer_updates == 15
    assert scheduler.T_cur == 0
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-7 + 10 / 15 * (0.001 - 1e-7))


def test_reset_epoch_keeps_weights_and_restarts_state_with_json_config(load_network):
    source = load_network(settings())
    advance(*source, 8)
    target = load_network(settings(reset_epoch=True, lambda_2=0.3, stop_lr=2e-5,
                                  warm_epochs=2), make_checkpoint(source), iterations=5)
    network, model, optimizer, scheduler = target
    assert network.completed_optimizer_updates == 0
    assert network.warmup_optimizer_updates == 10
    assert network.input_param.optimizer_param.start_epoch == 1
    assert not optimizer.state
    assert optimizer.param_groups[0]["weight_decay"] == 0.3
    assert optimizer.param_groups[0]["lr"] == 2e-5
    assert scheduler.T_cur == 0
    for old, new in zip(source[1].parameters(), model.parameters()):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
