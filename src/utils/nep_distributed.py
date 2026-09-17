"""Device and launcher selection shared by NEP CPU and GPU training."""
import os
import socket

import torch


def training_device_type(params):
    requested = getattr(params, "device", "auto")
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("NEP device must be auto, cpu, or cuda")
    if requested == "cpu":
        return "cpu"
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise ValueError("NEP device=cuda requested, but no CUDA/HIP device is available")
    return "cuda" if available else "cpu"


def training_backend(params, device_type):
    backend = getattr(params, "dist_backend", "auto")
    if backend == "auto":
        return "gloo" if device_type == "cpu" else "nccl"
    if backend not in ("gloo", "nccl"):
        raise ValueError("NEP dist_backend must be auto, gloo, or nccl")
    if device_type == "cpu" and backend != "gloo":
        raise ValueError("CPU distributed training requires Gloo; use dist_backend=gloo or auto")
    return backend


def configure_nep_runtime(params, environ=None):
    """Set ranks from torchrun/srun, or request local GPU spawning.

    CPU process counts are controlled by the launcher. A bare Python invocation
    remains single-process on CPU and uses the visible GPUs on CUDA/HIP.
    """
    env = os.environ if environ is None else environ
    device_type = training_device_type(params)
    training_backend(params, device_type)  # Reject incompatible settings before starting ranks.
    if "RANK" in env or "WORLD_SIZE" in env:
        if not all(key in env for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
            raise ValueError("torchrun requires RANK, WORLD_SIZE and LOCAL_RANK")
        world_size, rank, local_rank = (int(env[key]) for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
        local_world_size = int(env.get("LOCAL_WORLD_SIZE", world_size))
        multi_nodes = world_size > local_world_size
        external = True
    elif "SLURM_PROCID" in env:
        if not all(key in env for key in ("SLURM_NTASKS", "SLURM_LOCALID")):
            raise ValueError("srun requires SLURM_NTASKS and SLURM_LOCALID")
        world_size, rank, local_rank = (int(env[key]) for key in ("SLURM_NTASKS", "SLURM_PROCID", "SLURM_LOCALID"))
        multi_nodes = int(env.get("SLURM_NNODES", "1")) > 1
        external = True
    else:
        if int(env.get("SLURM_NNODES", "1")) > 1:
            raise ValueError("Multi-node NEP training must be launched with srun or torchrun")
        world_size = torch.cuda.device_count() if device_type == "cuda" else 1
        rank = local_rank = 0
        multi_nodes = external = False
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise ValueError("Invalid distributed world_size, rank or local_rank")
    if world_size > 1 and params.optimizer_param.opt_name in ("LKF", "GKF"):
        raise ValueError("LKF and GKF support only single-process training")

    params.world_size, params.rank, params.local_rank = world_size, rank, local_rank
    params.multi_nodes = multi_nodes
    # Keep the existing flag for downstream code; it denotes distributed ranks.
    params.multi_gpus = params.distributed = world_size > 1
    if world_size > 1:
        address = params.master_addr or env.get("MASTER_ADDR")
        port = params.master_port or env.get("MASTER_PORT")
        if external and (not address or not port):
            raise ValueError("Set shared MASTER_ADDR and MASTER_PORT (or master_addr/master_port) for all ranks")
        if not address:
            address = "127.0.0.1"
        if not port:
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
        if not str(port).isdigit() or not 0 < int(port) < 65536:
            raise ValueError("MASTER_PORT/master_port must be an integer in 1..65535")
        params.master_addr, params.master_port = str(address), str(port)
    else:
        params.master_addr = params.master_port = None
    return not external and world_size > 1
