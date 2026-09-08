"""Python adapter for the bounded, fused NEP fitting CUDA operator.

This module deliberately does not load CalcOps at import time.  Dataset workers and
CPU-only tools can therefore construct fitting metadata without a CUDA library.
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch
from torch.autograd.function import once_differentiable


@dataclass(frozen=True)
class FittingGroups:
    """Stable atom grouping metadata, with host-side type/count information."""

    type_ids: Tuple[int, ...]
    counts: Tuple[int, ...]
    atom_ids: torch.Tensor
    offsets: torch.Tensor

    def to(self, *args, **kwargs):
        return FittingGroups(
            self.type_ids,
            self.counts,
            self.atom_ids.to(*args, **kwargs),
            self.offsets.to(*args, **kwargs),
        )

    def pin_memory(self):
        return FittingGroups(
            self.type_ids,
            self.counts,
            self.atom_ids.pin_memory(),
            self.offsets.pin_memory(),
        )


def build_fitting_groups(atom_type_map_cpu: torch.Tensor) -> FittingGroups:
    """Group row indices by atom type in first-appearance order."""
    if not isinstance(atom_type_map_cpu, torch.Tensor):
        raise TypeError("atom_type_map_cpu must be a torch.Tensor")
    if atom_type_map_cpu.device.type != "cpu":
        raise ValueError("atom_type_map_cpu must be on CPU")
    if atom_type_map_cpu.ndim != 1:
        raise ValueError("atom_type_map_cpu must be one-dimensional")
    if atom_type_map_cpu.dtype not in (
        torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8
    ):
        raise TypeError("atom_type_map_cpu must have an integer dtype")

    grouped = {}
    for atom_id, type_id in enumerate(atom_type_map_cpu.tolist()):
        type_id = int(type_id)
        if type_id < 0:
            raise ValueError("atom type ids must be non-negative")
        grouped.setdefault(type_id, []).append(atom_id)

    type_ids = tuple(grouped)
    counts = tuple(len(grouped[type_id]) for type_id in type_ids)
    flat_ids = [atom_id for type_id in type_ids for atom_id in grouped[type_id]]
    cumulative = 0
    offset_values = [0]
    for count in counts:
        cumulative += count
        offset_values.append(cumulative)
    return FittingGroups(
        type_ids,
        counts,
        torch.tensor(flat_ids, dtype=torch.int64),
        torch.tensor(offset_values, dtype=torch.int64),
    )


def _zero_bias(weight: torch.Tensor, width: int) -> torch.Tensor:
    return weight.new_zeros(width)


def pack_fitting_parameters(fitting_net: Sequence, type_ids: Sequence[int]):
    """Stack only active single-hidden-layer FittingNet/QNEPFittingNet parameters."""
    if not type_ids:
        if not fitting_net:
            raise ValueError("cannot infer empty pack shapes without a fitting network")
        net = fitting_net[0]
        is_qnep = hasattr(net, "energy_head") and hasattr(net, "charge_head")
        expected_layers = 1 if is_qnep else 2
        if not hasattr(net, "layers") or len(net.layers) != expected_layers:
            raise ValueError("fused NEP fitting requires exactly one hidden layer")
        W0 = net.layers[0].weight
        V0 = (
            torch.cat((net.energy_head.weight, net.charge_head.weight), dim=1)
            if is_qnep else net.layers[1].weight
        )
        D, H = W0.shape
        Q = V0.shape[1]
        return (
            W0.new_empty((0, D, H)),
            W0.new_empty((0, H)),
            W0.new_empty((0, H, Q)),
            W0.new_empty((0, Q)),
        )

    weights = []
    hidden_biases = []
    output_weights = []
    output_biases = []

    for raw_type_id in type_ids:
        type_id = int(raw_type_id)
        if type_id < 0 or type_id >= len(fitting_net):
            raise ValueError(f"atom type id {type_id} has no fitting network")
        net = fitting_net[type_id]

        if hasattr(net, "energy_head") and hasattr(net, "charge_head"):
            if len(net.layers) != 1:
                raise ValueError("fused QNEP fitting requires exactly one hidden layer")
            hidden = net.layers[0]
            W = hidden.weight
            b = (
                hidden.bias.reshape(-1)
                if net.bias_flag and hidden.bias is not None
                else _zero_bias(W, W.shape[1])
            )
            V = torch.cat((net.energy_head.weight, net.charge_head.weight), dim=1)
            energy_bias = (
                net.energy_head.bias.reshape(-1)
                if net.energy_head.bias is not None
                else _zero_bias(W, 1)
            )
            charge_bias = (
                net.charge_head.bias.reshape(-1)
                if net.charge_head.bias is not None
                else _zero_bias(W, 1)
            )
            c = torch.cat((energy_bias, charge_bias))
        else:
            if not hasattr(net, "layers") or len(net.layers) != 2:
                raise ValueError("fused NEP fitting requires exactly one hidden layer")
            hidden, output = net.layers
            W = hidden.weight
            b = (
                hidden.bias.reshape(-1)
                if net.bias_flag and hidden.bias is not None
                else _zero_bias(W, W.shape[1])
            )
            V = output.weight
            c = (
                output.bias.reshape(-1)
                if net.last_bias and output.bias is not None
                else _zero_bias(W, V.shape[1])
            )

        if W.ndim != 2 or V.ndim != 2 or W.shape[1] != V.shape[0]:
            raise ValueError("invalid fitting parameter shapes")
        weights.append(W)
        hidden_biases.append(b)
        output_weights.append(V)
        output_biases.append(c)

    return tuple(torch.stack(values) for values in (
        weights, hidden_biases, output_weights, output_biases
    ))


def _load_raw_ops():
    from src.utils.op_loader import load_calc_ops

    load_calc_ops()
    return torch.ops.CalcOps_cuda


class _FusedFitting(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, b, V, c, atom_ids, offsets, counts):
        ctx.set_materialize_grads(False)
        ctx.counts = tuple(counts)
        ctx.save_for_backward(X, W, b, V, c, atom_ids, offsets)
        return tuple(_load_raw_ops().nep_fitting_forward(
            X, W, b, V, c, atom_ids, offsets, list(ctx.counts)
        ))

    @staticmethod
    @once_differentiable
    def backward(ctx, gradY, gradG):
        X, W, b, V, c, atom_ids, offsets = ctx.saved_tensors
        Q, N, D = V.shape[2], X.shape[0], X.shape[1]
        has_gradY = gradY is not None
        if gradY is None:
            gradY = X.new_zeros((Q, N))
        if gradG is None:
            gradG = X.new_zeros((Q, N, D))
        gradients = list(_load_raw_ops().nep_fitting_backward(
            X, W, b, V, c, atom_ids, offsets, list(ctx.counts),
            gradY.contiguous(), gradG.contiguous(),
        ))
        if not has_gradY:
            gradients[4] = None
        return (*gradients, None, None, None)


def fused_fitting(X, W, b, V, c, groups: FittingGroups):
    """Evaluate energy/charge and descriptor gradients with the fused raw op."""
    if X.dtype != torch.float64 or any(t.dtype != torch.float64 for t in (W, b, V, c)):
        raise TypeError("fused NEP fitting requires float64 inputs")
    if X.ndim != 2 or W.ndim != 3 or b.ndim != 2 or V.ndim != 3 or c.ndim != 2:
        raise ValueError("invalid fused fitting tensor ranks")
    N, D = X.shape
    Ta, packed_D, H = W.shape
    if packed_D != D or b.shape != (Ta, H) or V.shape[:2] != (Ta, H):
        raise ValueError("inconsistent fused fitting tensor shapes")
    Q = V.shape[2]
    if c.shape != (Ta, Q):
        raise ValueError("inconsistent fused fitting output bias shape")
    if D > 96 or H > 100 or Q not in (1, 2):
        raise ValueError("fused fitting supports D<=96, H<=100, and Q=1 or 2")
    if len(groups.type_ids) != Ta or len(groups.counts) != Ta or sum(groups.counts) != N:
        raise ValueError("fitting groups do not match packed parameters and input rows")
    if groups.atom_ids.device != X.device or groups.offsets.device != X.device:
        raise ValueError("fitting group indices must be on the input device")
    return _FusedFitting.apply(X, W, b, V, c, groups.atom_ids, groups.offsets, groups.counts)


__all__ = ["FittingGroups", "build_fitting_groups", "pack_fitting_parameters", "fused_fitting"]
