import os
import torch
import json
from contextlib import closing
import socket

from src.user.input_param import InputParam
from src.utils.json_operation import get_parameter, get_required_parameter


def find_free_port():
    """Find a free TCP port."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def main_worker(rank, world_size, nep_param):
    try:
        if nep_param.multi_nodes is False and nep_param.multi_gpus:
            nep_param.rank = rank
            nep_param.local_rank = rank
        from src.PWMLFF.tnep_network import tnep_network
        tnep_net = tnep_network(nep_param)
        tnep_net.train()
    except Exception as e:
        print(f"Rank {nep_param.rank}, LocalRank {nep_param.local_rank}: Error occurred: {e}")
        raise


def tnep_train(input_json: json, cmd: str):
    """
    Train a tNEP (Tensorial NEP) model.

    Args:
        input_json: Parsed JSON configuration dict
        cmd: Command string ("TRAIN" or "TEST")
    """
    nep_param = InputParam(input_json, cmd)
    num_nodes = os.environ.get("SLURM_NNODES", None)
    if num_nodes is not None and int(num_nodes) > 1:
        world_size = int(os.environ["SLURM_NTASKS"])
        rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        num_nodes = int(num_nodes)
    else:
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
        else:
            world_size = 1
        rank = 0
        local_rank = 0

    if world_size > 1:
        torch.multiprocessing.spawn(main_worker, args=(world_size, nep_param), nprocs=world_size, join=True)
    else:
        main_worker(0, 1, nep_param)


def tnep_test(input_json: json, cmd: str):
    """
    Test/inference for a tNEP model.

    Args:
        input_json: Parsed JSON configuration dict
        cmd: Command string ("TEST")
    """
    nep_param = InputParam(input_json, cmd)
    from src.PWMLFF.tnep_network import tnep_network
    tnep_net = tnep_network(nep_param)
    # Test mode: load checkpoint and run validation
    tnep_net.load_checkpoint()
    tnep_net.valid()
