"""CPU NEP DDP must synchronize energy/force/virial training gradients."""
import copy
import socket

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def config():
    return {
        "model_type": "NEP", "atom_type": [1, 8, 29],
        "device": "cpu", "dist_backend": "gloo", "seed": 123,
        "precision": "float64", "recover_train": False,
        "model": {
            "descriptor": {"cutoff": [5.0, 4.0], "n_max": [1, 1],
                           "basis_size": [2, 2], "l_max": [4, 2, 1]},
            "fitting_net": {"network_size": [30, 1]},
        },
        "optimizer": {"optimizer": "ADAM", "epochs": 2,
                      "train_energy": True, "train_force": True,
                      "train_virial": True, "scale_lr": False},
    }


def sample(indices):
    from src.pre_data.nep_lmdb_dataset import NepLmdbDataset
    from src.pre_data.nep_data_loader import variable_length_collate_fn
    dataset = NepLmdbDataset([], [1, 8, 29], 5.0, 4.0)
    frames = [
        {"numbers": [1, 1], "positions": [[1, 1, 1], [2.2, 1.3, 1]]},
        {"numbers": [8, 8], "positions": [[1, 1, 1], [2.5, 1.2, 1.4]]},
    ]
    rows = []
    for index in indices:
        frame = dict(frames[index], cell=np.eye(3) * 12, pbc=[True] * 3,
                     energy=-1., forces=np.zeros((2, 3)), stress=np.zeros(6))
        rows.append(dataset._convert_frame(frame))
    return variable_length_collate_fn(rows)


def forward(model, batch):
    from src.utils.op_loader import load_calc_ops
    ops = load_calc_ops()
    module = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    nnr, nna, nlr, nla, rr, ra = ops.calculate_neighbor(
        batch["num_atom"], batch["atom_type_map"], module.atom_type_device - 1,
        batch["box"], batch["box_original"], batch["num_cell"], batch["position"],
        module.cutoff_radial, module.cutoff_angular, 10, 10, True)
    return model(nnr, nlr, rr, nna, nla, ra, batch["num_atom"], batch["atom_type_map"])


def loss(outputs):
    etot, _, force, _, virial = outputs[:5]
    return etot.square().sum() + force.square().sum() + .01 * virial.square().sum()


def ddp_worker(rank, port, directory):
    import os
    os.chdir(directory)
    torch.set_num_threads(1)
    from src.user.input_param import InputParam
    from src.PWMLFF.nep_network import nep_network
    params = InputParam(config(), "TRAIN")
    params.world_size = 2
    params.rank = params.local_rank = rank
    params.multi_gpus = True
    params.master_addr, params.master_port = "127.0.0.1", str(port)
    try:
        network = nep_network(params)
        assert network.device.type == "cpu"
        assert dist.get_backend() == "gloo"
        wrapped, optimizer, _ = network.load_model_optimizer(
            [0., 0., 0.], avg_atom_num=2, iterations=3,
            q_scaler=np.ones(14), max_NN_radial=10, max_NN_angular=10)
        assert isinstance(wrapped, torch.nn.parallel.DistributedDataParallel)
        reference = copy.deepcopy(wrapped.module)
        ref_optimizer = torch.optim.Adam(reference.parameters(), **{
            key: value for key, value in optimizer.defaults.items()
        })
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            loss(forward(wrapped, sample([rank]))).backward()
            (loss(forward(reference, sample([0, 1]))) / 2).backward()
            for (name, actual), (_, expected) in zip(
                    wrapped.module.named_parameters(), reference.named_parameters()):
                if expected.grad is None:
                    assert actual.grad is None, name
                else:
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-8, atol=2e-9, msg=name)
            optimizer.step()
            ref_optimizer.step()
            for actual, expected in zip(wrapped.module.parameters(), reference.parameters()):
                torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
        from src.utils.train_log import AverageMeter
        meter = AverageMeter("loss", device=torch.device("cpu"), world_size=2)
        meter.update(2. if rank == 0 else 6.)
        meter.all_reduce()
        assert meter.avg == 4.
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_cpu_ddp_gradients_and_adam_match_serial(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(ddp_worker, args=(port, str(tmp_path)), nprocs=2, join=True)


def test_meter_without_process_group_is_noop():
    from src.utils.train_log import AverageMeter
    meter = AverageMeter("loss", device=torch.device("cpu"))
    meter.update(3.)
    meter.all_reduce()
    assert meter.avg == 3.
