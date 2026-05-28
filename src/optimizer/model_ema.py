"""Phase 2.4: opt-in exponential moving average over model parameters.

Shadow tensors live on the same device/dtype as the source parameters, the
update happens after ``optimizer.step()``, and ``apply_shadow`` / ``restore``
swap the live weights for evaluation, checkpointing, and NEP-text export.
``state_dict`` reload remains symmetric for resume.

The LKF/GKF Kalman optimizers are not wrapped — they have their own averaging
behavior and the Phase 2.4 plan keeps them on the legacy path.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not (0.0 < decay < 1.0):
            raise ValueError(
                "ModelEMA decay must be in (0, 1), got {}.".format(decay)
            )
        self.decay = float(decay)
        module = model.module if hasattr(model, "module") else model
        self._names: list[str] = []
        self._params: list[torch.Tensor] = []
        self._shadows: list[torch.Tensor] = []
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            self._names.append(name)
            self._params.append(param)
            self._shadows.append(param.detach().clone())
        self._backup: list[torch.Tensor] | None = None

    @torch.no_grad()
    def update(self) -> None:
        if not self._shadows:
            return
        torch._foreach_lerp_(self._shadows, self._params, 1.0 - self.decay)

    @torch.no_grad()
    def apply_shadow(self) -> None:
        if self._backup is not None:
            raise RuntimeError("apply_shadow called twice without restore()")
        self._backup = [p.detach().clone() for p in self._params]
        torch._foreach_copy_(self._params, self._shadows)

    @torch.no_grad()
    def restore(self) -> None:
        if self._backup is None:
            return
        torch._foreach_copy_(self._params, self._backup)
        self._backup = None

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": {name: t for name, t in zip(self._names, self._shadows)},
        }

    def load_state_dict(self, state: dict) -> None:
        if "decay" in state:
            decay = state["decay"]
            self.decay = float(decay.item() if torch.is_tensor(decay) else decay)
        shadow_state = state.get("shadow", state)
        for name, shadow in zip(self._names, self._shadows):
            if name in shadow_state:
                shadow.copy_(shadow_state[name])
