#!/bin/bash
#SBATCH --job-name=muon_route_probe
#SBATCH --partition=4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=00:05:00
#SBATCH --output=/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL/phase0_benchmarks/muon_route_probe.log

module load gcc/8.3.1 cuda/11.8-share openmpi/4.1.6
source /data/home/pfsuo/pfsuo/software/build/PWMLFF_test/libtorch_version/2026.3/MatPL-2026.3/matpl-2026.3/bin/activate

CODE_DIR="/data/home/pfsuo/pfsuo/software/build/PWMLFF_test/MatPL_Agent_dev/MatPL"
export PYTHONPATH="$CODE_DIR"
cd "$CODE_DIR"

python - << 'PY'
import functools
import torch
import torch.nn as nn
from src.optimizer.hybrid_muon import (
    get_adam_route, get_effective_shape, get_matrix_view_shape
)

# === Build a mock module that mirrors NEP's parameter names exactly ===
# Based on direct reading of:
#   src/model/nep_net.py: c_param_2, c_param_3 (Cij descriptor coefficients)
#   src/model/nep_fitting.py: layers.{i}.{weight,bias,resnet_dt} per element type
class FakeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, has_bias=True, has_resnet=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(1, out_dim)) if has_bias else None
        self.resnet_dt = nn.Parameter(torch.empty(1, out_dim)) if has_resnet else None

class FakeFittingNet(nn.Module):
    def __init__(self, network=[100, 50, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1, len(network) - 1):
            self.layers.append(FakeLayer(network[i-1], network[i], has_bias=True, has_resnet=(i > 1)))
        # Last layer: typically last_bias=True -> has bias
        self.layers.append(FakeLayer(network[-2], network[-1], has_bias=True, has_resnet=False))

class FakeNEP(nn.Module):
    """Mirror of NEP:
       - c_param_2: shape (ntypes, ntypes, n_max_radial+1, n_base_radial+1)
       - c_param_3: shape (ntypes, ntypes, n_max_angular+1, n_base_angular+1)
       - fitting_net.{i}.layers.{j}.{weight,bias,resnet_dt} per element type
    """
    def __init__(self, ntypes=2, n_max=4, n_base=4, neuron=[50, 1]):
        super().__init__()
        self.c_param_2 = nn.Parameter(torch.empty(ntypes, ntypes, n_max+1, n_base+1))
        self.c_param_3 = nn.Parameter(torch.empty(ntypes, ntypes, n_max+1, n_base+1))
        feat_dim = (n_max+1)*4 + (n_max+1)*3   # rough Q descriptor dim
        self.fitting_net = nn.ModuleList([FakeFittingNet([feat_dim] + neuron) for _ in range(ntypes)])

# === Probe routing ===
def probe(model, label):
    print(f"\n{'='*100}")
    print(f"=== {label} ===")
    print(f"{'='*100}")
    print(f"{'name':<55} {'shape':<25} {'route':<25} {'Muon view'}")
    print("-" * 100)

    buckets = {"muon": [], "adam_no_decay": [], "adam_decay": []}
    muon_mode = "slice"

    for name, p in model.named_parameters():
        eff = get_effective_shape(p.shape)
        name_route = get_adam_route(name)

        if name_route == "adam":
            route = "adam_no_decay"
            mv_str = "-"
        elif name_route == "adamw":
            route = "adam_decay"
            mv_str = "-"
        elif len(eff) < 2:
            route = "adam_no_decay (rank<2)"
            mv_str = "-"
        else:
            mv = get_matrix_view_shape(eff, muon_mode)
            if mv is None:
                route = "adam_decay (non-matrix)"
                mv_str = "-"
            else:
                route = "muon"
                mv_str = f"B={mv[0]}, R={mv[1]}, C={mv[2]}"

        buckets[route.split()[0]].append((name, tuple(p.shape)))
        print(f"{name:<55} {str(tuple(p.shape)):<25} {route:<25} {mv_str}")

    print("\n--- Bucket summary ---")
    for k in ["muon", "adam_no_decay", "adam_decay"]:
        lst = buckets[k]
        elems = sum(functools.reduce(lambda a,b: a*b, s, 1) for _, s in lst)
        print(f"  [{k}] {len(lst)} params, {elems:,} elements")
        for n, s in lst:
            print(f"      {n:<55} {s}")

# Single-element [50, 1] (small_Si setup)
probe(FakeNEP(ntypes=1, n_max=4, n_base=4, neuron=[50, 1]), "small_Si: ntypes=1, neuron=[50,1]")

# Two-element [100, 1] (water_5k setup)
probe(FakeNEP(ntypes=2, n_max=4, n_base=4, neuron=[100, 1]), "water_5k: ntypes=2, neuron=[100,1]")

# Multi-element [100, 100, 1] (deeper net)
probe(FakeNEP(ntypes=3, n_max=4, n_base=4, neuron=[100, 100, 1]), "ntypes=3, neuron=[100,100,1]")
PY
echo "exit code: $?"
