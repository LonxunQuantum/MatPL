import math
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.utils.train_log import RMSEMeter


def test_rmse_meter_weights_batches_by_scalar_targets():
    meter = RMSEMeter("Force", device=torch.device("cpu"))
    meter.update(1.0, 3)   # one atom, three force components
    meter.update(4.0, 9)   # three atoms, nine force components
    assert meter.root == pytest.approx(math.sqrt(39.0 / 12.0))
    assert meter.count == 12


def _distributed_rmse_worker(rank, port):
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
    try:
        meter = RMSEMeter("Force", device=torch.device("cpu"))
        meter.update(1.0 if rank == 0 else 4.0, 3 if rank == 0 else 9)
        meter.all_reduce()
        assert meter.root == pytest.approx(math.sqrt(39.0 / 12.0))
        if rank == 0:
            meter.update(9.0, 3)
        meter.all_reduce()
        assert meter.root == pytest.approx(math.sqrt(66.0 / 15.0))
    finally:
        dist.destroy_process_group()


def test_rmse_meter_sums_squares_and_counts_across_ranks(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_distributed_rmse_worker, args=(port,), nprocs=2, join=True)


def test_virial_log_metrics_match_inference_validity_and_six_components():
    from src.PWMLFF.nep_mods.nep_trainer import get_virial_log_mse

    labels = torch.zeros((3, 9), dtype=torch.float64)
    labels[1, 5] = -1e7  # first component is present, another is missing
    predictions = labels + 2.0
    predictions[1] = 100.0
    atoms = torch.tensor([[2.0], [4.0], [8.0]], dtype=torch.float64)
    raw, per_atom, count = get_virial_log_mse(
        predictions, labels, atoms, torch.nn.MSELoss())
    assert count == 12
    assert raw.item() == pytest.approx(4.0)
    assert per_atom.item() == pytest.approx((1.0 + 0.0625) / 2)


def test_nep_energy_and_virial_objective_uses_per_atom_mse():
    from types import SimpleNamespace
    from src.loss.loss import get_loss

    atoms = torch.tensor([[2.0], [8.0]], dtype=torch.float64)
    energy = torch.tensor([[2.0], [8.0]], dtype=torch.float64, requires_grad=True)
    virial = torch.tensor([[2.0] * 6, [8.0] * 6], dtype=torch.float64,
                          requires_grad=True)
    criterion = torch.nn.MSELoss()
    energy_mse = criterion(energy / atoms, torch.zeros_like(energy))
    virial_mse = criterion(virial / atoms, torch.zeros_like(virial))
    optimizer = SimpleNamespace(
        train_force=False, train_energy=True, train_egroup=False,
        train_charge=False, train_bec=False,
        start_pre_fac_etot=2.0, end_pre_fac_etot=2.0,
        start_pre_fac_virial=3.0, end_pre_fac_virial=3.0)
    loss = get_loss(SimpleNamespace(optimizer_param=optimizer), 0.001, 1,
                    torch.zeros_like(energy_mse), energy_mse, virial_mse,
                    train_virial=True)
    assert loss.item() == pytest.approx(5.0)
    loss.backward()
    torch.testing.assert_close(energy.grad[:, 0],
                               torch.tensor([1.0, 0.25], dtype=torch.float64))
    torch.testing.assert_close(virial.grad[:, 0],
                               torch.tensor([0.25, 0.0625], dtype=torch.float64))
