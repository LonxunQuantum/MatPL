"""Rank discovery must follow the launcher, not the number of GPUs."""
from types import SimpleNamespace

import pytest
import torch


def params(device="auto", backend="auto", optimizer="ADAM"):
    return SimpleNamespace(device=device, dist_backend=backend,
                           master_addr=None, master_port=None,
                           optimizer_param=SimpleNamespace(opt_name=optimizer))


def test_slurm_cpu_uses_global_rank_and_all_tasks():
    from src.utils.nep_distributed import configure_nep_runtime
    p = params("cpu")
    spawn = configure_nep_runtime(p, {
        "SLURM_NNODES": "2", "SLURM_NTASKS": "8", "SLURM_PROCID": "5",
        "SLURM_LOCALID": "1", "MASTER_ADDR": "node-a", "MASTER_PORT": "29500",
    })
    assert not spawn
    assert (p.world_size, p.rank, p.local_rank, p.multi_nodes) == (8, 5, 1, True)
    assert (p.master_addr, p.master_port) == ("node-a", "29500")


def test_torchrun_takes_precedence_over_slurm_allocation():
    from src.utils.nep_distributed import configure_nep_runtime
    p = params("cpu")
    spawn = configure_nep_runtime(p, {
        "WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "2",
        "SLURM_NTASKS": "2", "SLURM_PROCID": "1", "SLURM_NNODES": "2",
        "MASTER_ADDR": "node-a", "MASTER_PORT": "29501",
    })
    assert not spawn
    assert (p.world_size, p.rank, p.local_rank) == (4, 2, 0)


def test_single_node_srun_cpu_is_distributed():
    from src.utils.nep_distributed import configure_nep_runtime
    p = params("cpu")
    assert not configure_nep_runtime(p, {
        "SLURM_NNODES": "1", "SLURM_NTASKS": "4", "SLURM_PROCID": "3",
        "SLURM_LOCALID": "3", "MASTER_ADDR": "localhost", "MASTER_PORT": "29502",
    })
    assert (p.world_size, p.rank, p.multi_nodes) == (4, 3, False)


def test_bare_cpu_defaults_to_one_process():
    from src.utils.nep_distributed import configure_nep_runtime
    p = params("cpu")
    assert not configure_nep_runtime(p, {})
    assert (p.world_size, p.rank, p.local_rank) == (1, 0, 0)


def test_bare_gpu_preserves_automatic_spawn(monkeypatch):
    from src.utils.nep_distributed import configure_nep_runtime
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    p = params()
    assert configure_nep_runtime(p, {})
    assert p.world_size == 4
    assert p.master_addr == "127.0.0.1" and 0 < int(p.master_port) < 65536


def test_one_torchrun_rank_does_not_spawn_visible_gpus(monkeypatch):
    from src.utils.nep_distributed import configure_nep_runtime
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    p = params()
    assert not configure_nep_runtime(p, {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"})
    assert p.world_size == 1


@pytest.mark.parametrize("optimizer", ["LKF", "GKF"])
def test_kalman_rejects_cpu_distributed(optimizer):
    from src.utils.nep_distributed import configure_nep_runtime
    with pytest.raises(ValueError, match="LKF|GKF"):
        configure_nep_runtime(params("cpu", optimizer=optimizer), {
            "WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0",
            "MASTER_ADDR": "localhost", "MASTER_PORT": "29500",
        })


def test_external_ranks_require_shared_port():
    from src.utils.nep_distributed import configure_nep_runtime
    with pytest.raises(ValueError, match="MASTER_PORT|master_port"):
        configure_nep_runtime(params("cpu"), {
            "WORLD_SIZE": "2", "RANK": "0", "LOCAL_RANK": "0", "MASTER_ADDR": "node-a",
        })


def test_cpu_rejects_explicit_nccl():
    from src.utils.nep_distributed import configure_nep_runtime
    with pytest.raises(ValueError, match="[Gg]loo|nccl"):
        configure_nep_runtime(params("cpu", "nccl"), {})


def test_cpu_ops_selected_even_when_cuda_is_visible(monkeypatch):
    from src.utils.op_loader import select_runtime_backend
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_runtime_backend(device="cpu") == "cpu"


def test_training_device_is_serialized(tmp_path, monkeypatch):
    from src.user.input_param import InputParam
    monkeypatch.chdir(tmp_path)
    p = InputParam({"model_type": "NEP", "atom_type": [1], "device": "cpu",
                    "dist_backend": "gloo", "optimizer": {"optimizer": "ADAM"}}, "TRAIN")
    saved = p.to_dict()
    assert saved["device"] == "cpu"
    assert saved["dist_backend"] == "gloo"
