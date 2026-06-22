"""
TNEP (Tensorial Neuroevolution Potential) model class.

Extends the NEP class to support training of tensorial properties:
  - train_mode=1: dipole moment (3 components: μ_x, μ_y, μ_z) → IR spectrum
  - train_mode=2: polarizability tensor (6 components: α_xx, α_yy, α_zz, α_xy, α_yz, α_zx) → Raman spectrum

Reference:
  Xu et al., J. Chem. Theory Comput. 20, 3273 (2024)
  GPUMD implementation: tnep.cu / tnep.cuh
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List

from src.model.nep_net import NEP
from src.model.nep_fitting import FittingNet
from src.user.input_param import InputParam

# Load CalcOps following the same pattern as nep_net.py
if torch.cuda.is_available():
    _op_lib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(_op_lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    _op_lib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(_op_lib_path)
    CalcOps = torch.ops.CalcOps_cpu


class TNEP(NEP):
    """
    Tensorial NEP: extends NEP for dipole/polarizability prediction.

    The descriptor computation (radial Chebyshev + angular spherical harmonics)
    is IDENTICAL to regular NEP. The differences are:
      1. For polarizability (train_mode=2): TWO fitting nets per element type
         - scalar head (self.fitting_net_pol): isotropic diagonal contribution
         - tensorial head (self.fitting_net): anisotropic contribution via Fp
      2. For dipole (train_mode=1): single fitting net, output interpreted as
         site virial → summed to dipole moment
      3. Output is dipole (3 comp) or polarizability (6 comp), NOT site energy
      4. Loss is target-property-only (λ_e = λ_f = 0)

    Internally, dipole/polarizability are stored in the same virial tensor slots
    as regular NEP (for GPUMD compatibility), but user-facing names are physical.
    """

    def __init__(self, input_param: InputParam = None, energy_shift=None, rank=0,
                 q_scaler=None, max_NN_radial=-1, max_NN_angular=-1,
                 dtype=None, device=None):
        # Call NEP.__init__ which sets up all descriptor machinery
        super().__init__(input_param, energy_shift, rank, q_scaler,
                         max_NN_radial, max_NN_angular, dtype, device)

        # Compat: older NEP versions may not set accumulate_dtype
        if not hasattr(self, 'accumulate_dtype'):
            self.accumulate_dtype = torch.float64 if getattr(
                input_param, 'precision', 'float64') in ("float64", "mixed") else torch.float32

        self.train_mode = input_param.nep_param.train_mode

        # Validate train_mode
        if self.train_mode not in [0, 1, 2]:
            raise ValueError(f"train_mode must be 0, 1, or 2, got {self.train_mode}")

        # For polarizability (mode=2): create second set of fitting nets
        # (scalar head) in addition to the inherited self.fitting_net (tensorial head)
        self.fitting_net_pol = None
        if self.train_mode == 2:
            self.fitting_net_pol = nn.ModuleList()
            for i in range(self.ntypes):
                nep_txt_param = None
                if input_param.nep_param.c2_param is not None:
                    # Load from nep.txt if available
                    # For polarizability, wb_pol params come after the regular wb params
                    # The offset accounts for the doubled parameter set
                    pol_offset = self.ntypes * 3  # w0, b0, w1 per type for regular head
                    wb_pol = input_param.nep_param.model_wb_pol if hasattr(
                        input_param.nep_param, 'model_wb_pol') else None
                    if wb_pol is not None:
                        nep_txt_param = [
                            wb_pol[i * 3 + 0],
                            wb_pol[i * 3 + 1],
                            wb_pol[i * 3 + 2],
                            input_param.nep_param.bias_lastlayer_pol[i]
                            if hasattr(input_param.nep_param, 'bias_lastlayer_pol')
                            else input_param.nep_param.bias_lastlayer[i]
                        ]

                self.fitting_net_pol.append(
                    FittingNet(
                        network_size=self.neuron,
                        bias=True,
                        resnet_dt=False,
                        activation="tanh",
                        input_dim=self.feature_nums,
                        ener_shift=energy_shift[i] if energy_shift is not None else 0.0,
                        magic=False,
                        nep_txt_param=nep_txt_param,
                        last_bias=True,
                    )
                )

            # Optionally compile the scalar head too
            if getattr(input_param, "compile_fitting", False):
                for fit_net in self.fitting_net_pol:
                    fit_net.forward = torch.compile(
                        fit_net.forward, mode="reduce-overhead", dynamic=False,
                    )

        # Mark fitting nets appropriately for checkpoint save/load
        self._tnep_has_pol_head = (self.train_mode == 2)

    def _calculate_Ei_scalar(self,
                              Imagetype_map: torch.Tensor,
                              feats: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
        """
        Compute per-atom output from the scalar (polarizability) head.

        For polarizability mode, this scalar per-atom value contributes
        isotropically to the diagonal of the polarizability tensor: each
        atom's scalar is added equally to α_xx, α_yy, α_zz.
        """
        Ei_pol = torch.zeros_like(Imagetype_map, dtype=self.dtype)
        for idx, fit_net in enumerate(self.fitting_net_pol):
            mask = (Imagetype_map == idx)
            if not mask.any():
                continue
            indices = torch.arange(len(Imagetype_map.flatten()), device=device)[mask]
            feat = feats[indices, :]
            Ei_ntype = fit_net.forward(feat)
            Ei_pol[mask] = Ei_ntype.squeeze()
        return Ei_pol

    def forward(self,
                NN_radial: torch.Tensor,
                NL_radial: torch.Tensor,
                Ri_radial: torch.Tensor,
                NN_angular: torch.Tensor,
                NL_angular: torch.Tensor,
                Ri_angular: torch.Tensor,
                num_atom: torch.Tensor,
                atom_type_map: torch.Tensor,
                Egroup_weight: Optional[torch.Tensor] = None,
                divider: Optional[torch.Tensor] = None,
                is_calc_f: Optional[bool] = True) -> Tuple[
                    torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                    Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for tNEP.

        For train_mode=0: identical to NEP (scalar potential)
        For train_mode=1: Ei = site virial, outputs dipole (3 comp) in Virial
        For train_mode=2: dual-head forward, outputs polarizability (6 comp) in Virial

        Returns:
            Etot: total energy (train_mode=0) or zero tensor (train_mode>0)
            Ei: per-atom output
            Force: atomic forces (None for train_mode>0 with is_calc_f=False)
            Egroup: energy group decomposition (None for tNEP)
            Virial: dipole (3 comp) or polarizability (6 comp)
        """
        # If train_mode=0, delegate entirely to parent NEP.forward
        if self.train_mode == 0:
            return super().forward(
                NN_radial, NL_radial, Ri_radial,
                NN_angular, NL_angular, Ri_angular,
                num_atom, atom_type_map, Egroup_weight, divider, is_calc_f
            )

        # --- tNEP path (train_mode=1 or 2) ---
        device = Ri_radial.device
        dtype = self.dtype
        natoms_sum = NL_radial.shape[0]

        # Step 1: Compute descriptors (IDENTICAL to NEP)
        Ri, Ri_d, Ri_angular, Ri_d_angular = self.calculate_Ri(
            Ri_radial, Ri_angular, device, dtype)
        Ri = Ri_radial.to(dtype)
        Ri.requires_grad_()
        Ri_angular = Ri_angular.to(dtype)
        Ri_angular.requires_grad_()

        if device.type == "cpu":
            NL_radial_type = NL_radial.new_full(NL_radial.shape, -1)
            mask = NL_radial != -1
            NL_radial_type[mask] = atom_type_map[NL_radial[mask]]
            NL_angular_type = NL_angular.new_full(NL_angular.shape, -1)
            mask = NL_angular != -1
            NL_angular_type[mask] = atom_type_map[NL_angular[mask]]
            feats = self.calculate_qn(atom_type_map, NL_radial_type, Ri,
                                      NL_angular_type, Ri_angular, device, dtype)
        else:
            # GPU path: use existing CalcOps (same as NEP)
            if self.train_2b:
                feat_2b = torch.zeros(natoms_sum, self.two_feat_num,
                                      dtype=dtype, device=device)
                feat_2b = CalcOps.calculateNepFeat(
                    self.c_param_2, Ri, NL_radial, atom_type_map, feat_2b,
                    self.cutoff_radial, self.multi_feat_num,
                    int(self.input_param.nep_param.fix_cij))[0]
            if self.l_max_3b > 0:
                feat_3b = torch.zeros(natoms_sum, self.multi_feat_num,
                                      dtype=dtype, device=device)
                feat_3b = CalcOps.calculateNepMbFeat(
                    self.c_param_3, Ri_angular, NL_angular, atom_type_map,
                    feat_3b, self.two_feat_num, self.l_max_3b,
                    self.l_max_4b, self.l_max_5b, self.cutoff_angular,
                    int(self.input_param.nep_param.fix_cij))[0]
                feats = torch.concat([feat_2b, feat_3b], dim=-1) if self.train_2b else feat_3b
            else:
                feats = feat_2b

        # Step 2: Scale descriptors
        feats_in = self.q_scaler * feats

        # Step 3: Compute per-atom outputs
        if self.train_mode == 1:
            # Dipole: single head → per-atom scalar (site virial contribution)
            Ei = self.calculate_Ei(atom_type_map, feats_in, device)
            Ei_pol = None
        elif self.train_mode == 2:
            # Polarizability: dual head
            # Tensorial head → per-atom scalar (drives anisotropic virial via Fp)
            Ei = self.calculate_Ei(atom_type_map, feats_in, device)
            # Scalar head → per-atom scalar (isotropic diagonal contribution)
            Ei_pol = self._calculate_Ei_scalar(atom_type_map, feats_in, device)

        assert Ei is not None

        # Step 4: Compute total "Etot" (sum of per-atom outputs)
        # For tNEP, this is NOT physically energy — it's the sum of site virial contributions
        # Used only for autograd chain rule to get forces/virial
        split_sizes = num_atom.reshape(-1).tolist()
        Ei_acc = Ei.to(self.accumulate_dtype) if self.accumulate_dtype != Ei.dtype else Ei
        energy_per_image = Ei_acc.split(split_sizes)
        Etot = torch.stack([x.sum() for x in energy_per_image]).unsqueeze(-1)

        # Step 5: Compute forces and virial
        if not is_calc_f:
            Force, Virial = None, None
        else:
            Force, Virial = self._calculate_force_virial_tnep(
                Ri, Ri_d, Ri_angular, Ri_d_angular,
                None, None,  # no ZBL for tNEP
                Etot, natoms_sum,
                NL_radial, NL_angular,
                None,  # neigh_zbl: no ZBL neighbors for tNEP
                num_atom, device,
                Ei_pol  # scalar head output for polarizability mode
            )

        # Step 6: Apply scalar polarizability contribution to diagonal virial
        if self.train_mode == 2 and Ei_pol is not None and Virial is not None:
            batch_size = len(split_sizes)
            # Sum scalar head output per image
            pol_per_image = Ei_pol.to(self.accumulate_dtype).split(split_sizes)
            pol_sums = torch.stack([x.sum() for x in pol_per_image])  # [batch_size]

            # Add isotropic contribution to diagonal virial components (xx, yy, zz)
            # Virial layout: [batch, 9] where indices 0=xx, 4=yy, 8=zz
            Virial = Virial.clone()
            Virial[:, 0] = Virial[:, 0] + pol_sums  # xx
            Virial[:, 4] = Virial[:, 4] + pol_sums  # yy
            Virial[:, 8] = Virial[:, 8] + pol_sums  # zz

        # For tNEP, Etot is not physical energy — return zeros to avoid confusion
        # The actual output is in Virial (dipole or polarizability)
        Etot_zero = torch.zeros_like(Etot) if self.train_mode > 0 else Etot
        Egroup = None  # No energy group decomposition for tNEP

        return Etot_zero, Ei, Force, Egroup, Virial

    def _calculate_force_virial_tnep(self,
                                      Ri: torch.Tensor,
                                      Ri_d: torch.Tensor,
                                      Ri_angular: torch.Tensor,
                                      Ri_d_angular: torch.Tensor,
                                      ri_zbl,
                                      ri_d_zbl,
                                      Etot: torch.Tensor,
                                      natoms_sum: int,
                                      NL_radial: torch.Tensor,
                                      NL_angular: torch.Tensor,
                                      neigh_zbl,
                                      num_atom: torch.Tensor,
                                      device: torch.device,
                                      Ei_pol: Optional[torch.Tensor] = None
                                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute forces and virial for tNEP.

        For polarizability mode (train_mode=2):
          - Forces are computed from the tensorial head output (Etot)
            via autograd, same chain rule as regular NEP
          - The scalar head (Ei_pol) contributes isotropically to the
            diagonal virial — applied in forward(), not here
          - No ZBL contribution for tNEP

        For dipole mode (train_mode=1):
          - Same as regular NEP force/virial computation, but the output
            is interpreted as dipole (3 diagonal components)

        NOTE: This method reuses NEP's calculate_force_virial for the core
        autograd chain rule. In a future optimization, a custom CUDA kernel
        could fuse the dual-head virial accumulation.
        """
        # Reuse parent's calculate_force_virial for the core computation.
        # The Etot here is from the tensorial head (or single head for dipole).
        # This gives us forces and the anisotropic virial contribution.
        Force, Virial = super().calculate_force_virial(
            Ri, Ri_d, Ri_angular, Ri_d_angular,
            ri_zbl, ri_d_zbl,
            Etot, natoms_sum,
            NL_radial, NL_angular,
            neigh_zbl, num_atom,
            device, self.dtype
        )
        return Force, Virial

    def get_checkpoint_state(self) -> dict:
        """
        Return state dict for checkpoint saving, including polarizability head.
        """
        state = super().state_dict()
        if self._tnep_has_pol_head and self.fitting_net_pol is not None:
            for i, fit_net in enumerate(self.fitting_net_pol):
                for j, layer in enumerate(fit_net.layers):
                    state[f'fitting_net_pol.{i}.layers.{j}.weight'] = layer.weight
                    if layer.bias is not None:
                        state[f'fitting_net_pol.{i}.layers.{j}.bias'] = layer.bias
        return state
