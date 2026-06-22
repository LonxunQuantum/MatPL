"""
tNEP training and validation step functions.

For tNEP (train_mode=1 or 2), the loss is computed solely on the
target tensorial property (dipole or polarizability), with zero
weight on energy and force terms (λ_e = λ_f = 0).
"""

import torch
import numpy as np
from collections import defaultdict

from src.user.input_param import InputParam
from src.loss.dploss import adjust_lr


def _compute_target_loss(predicted_virial, target_virial, train_mode,
                         natoms_sum, batch_size):
    """
    Compute MSE loss on dipole or polarizability.

    Args:
        predicted_virial: shape [batch, 9] — model output (virial tensor)
        target_virial: shape [batch, 6] — reference data
        train_mode: 1 (dipole) or 2 (polarizability)
        natoms_sum: total atoms across all images in batch
        batch_size: number of images in batch

    Returns:
        loss: scalar MSE loss
        per_comp_loss: dict of per-component MSE losses
    """
    if train_mode == 1:
        # Dipole: only diagonal components (xx, yy, zz) = virial indices [0, 4, 8]
        # But target_virial is shape [batch, 6] with standard layout;
        # For dipole, the target stores μ_x, μ_y, μ_z in slots [0, 4, 8]? No —
        # Actually, looking at GPUMD's dataset, for dipole the virial_ref stores
        # 3 values (xx, yy, zz), and the raw data has 3 components.
        # In MatPL/GPUMD convention:
        #   Virial tensor: [0]=xx, [4]=yy, [8]=zz (diagonal)
        #   We extract diagonal from predicted_virial and compare to target
        pred_xx = predicted_virial[:, 0]   # xx component
        pred_yy = predicted_virial[:, 4]   # yy component
        pred_zz = predicted_virial[:, 8]   # zz component
        target_xx = target_virial[:, 0]    # μ_x
        target_yy = target_virial[:, 1]    # μ_y
        target_zz = target_virial[:, 2]    # μ_z

        diff_xx = pred_xx - target_xx
        diff_yy = pred_yy - target_yy
        diff_zz = pred_zz - target_zz

        mse = (diff_xx.pow(2).sum() + diff_yy.pow(2).sum() +
               diff_zz.pow(2).sum()) / (3.0 * batch_size)

        per_comp = {
            'mu_x': diff_xx.pow(2).mean().sqrt().item(),
            'mu_y': diff_yy.pow(2).mean().sqrt().item(),
            'mu_z': diff_zz.pow(2).mean().sqrt().item(),
        }
    elif train_mode == 2:
        # Polarizability: all 6 components
        # predicted_virial layout: [0]=xx, [4]=yy, [8]=zz, [1]=xy, [2]=xz, [5]=yz
        # But the indices don't match standard Voigt notation exactly...
        # Following GPUMD's blockToCompIdx = {0, 1, 2, 3, 5, 7} for polarizability:
        # predicted indices: 0(xx), 1(yy?), 2(zz?), 3(xy), 5(yz), 7(zx)
        # Let's use the GPUMD convention directly: xx=0, yy=1, zz=2, xy=3, yz=5, zx=7
        # Actually, for a 9-component virial tensor in row-major 3x3:
        # [0]=xx, [1]=xy, [2]=xz, [3]=yx, [4]=yy, [5]=yz, [6]=zx, [7]=zy, [8]=zz
        # GPUMD's blockToCompIdx for polarizability: {0, 1, 2, 3, 5, 7}
        # But wait — GPUMD stores virial differently. Let me look at get_rmse_virial:
        #   Virial indexing in GPUMD is Voigt: 0=xx, 1=yy, 2=zz, 3=yz, 4=xz, 5=xy
        # For the 9-comp atom virial: 0=xx, 4=yy, 8=zz, 1=xy, 2=xz, 5=yz
        # For polarizability target (6 comp, Voigt): xx, yy, zz, yz, xz, xy
        # Mapping predicted 9-comp → comparison:
        #   pred[0] vs target[0] (xx)
        #   pred[4] vs target[1] (yy)
        #   pred[8] vs target[2] (zz)
        #   pred[1] vs target[5] (xy) — wait, target[5] in Voigt = xy
        #   pred[5] vs target[3] (yz) — target[3] in Voigt = yz
        #   pred[2] vs target[4] (xz) — target[4] in Voigt = xz

        # For simplicity, we compare both in the same 6-comp Voigt order.
        # We need to extract from 9-comp predicted to 6-comp Voigt:
        #   Voigt: xx, yy, zz, yz, xz, xy
        #   From 9-comp: pred[0](xx), pred[4](yy), pred[8](zz),
        #                pred[5](yz), pred[2](xz), pred[1](xy)

        pred_voigt = torch.stack([
            predicted_virial[:, 0],   # xx
            predicted_virial[:, 4],   # yy
            predicted_virial[:, 8],   # zz
            predicted_virial[:, 5],   # yz
            predicted_virial[:, 2],   # xz
            predicted_virial[:, 1],   # xy
        ], dim=1)  # [batch, 6]

        # target_virial is assumed to be in Voigt order: [xx, yy, zz, yz, xz, xy]
        diff = pred_voigt - target_virial
        mse = diff.pow(2).sum() / (6.0 * batch_size)

        comp_names = ['a_xx', 'a_yy', 'a_zz', 'a_yz', 'a_xz', 'a_xy']
        per_comp = {
            comp_names[c]: diff[:, c].pow(2).mean().sqrt().item()
            for c in range(6)
        }
    else:
        raise ValueError(f"Invalid train_mode={train_mode} for tNEP")

    return mse, per_comp


def tnep_train_step(train_loader, model, optimizer, epoch, real_lr,
                    start_lr, device, input_param: InputParam):
    """
    Single training epoch for tNEP.

    Args:
        train_loader: DataLoader for training data
        model: TNEP model
        optimizer: optimizer instance
        epoch: current epoch number
        real_lr: current learning rate
        start_lr: initial learning rate
        device: torch device
        input_param: InputParam config

    Returns:
        avg_loss: average loss over the epoch
        avg_rmse: average RMSE over the epoch
    """
    train_mode = input_param.nep_param.train_mode
    mode_name = {1: "dipole", 2: "polarizability"}.get(train_mode, "potential")

    model.train()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0

    for batch_idx, batch_data in enumerate(train_loader):
        # Unpack batch
        if len(batch_data) == 7:
            list_neigh, Imagetype_map, atom_type, ImageDR, \
                list_neigh_a, Imagetype_map_a, ImageDR_a = batch_data
            Ei_label, Force_label, Virial_label = None, None, None
        elif len(batch_data) >= 8:
            list_neigh, Imagetype_map, atom_type, ImageDR, \
                list_neigh_a, Imagetype_map_a, ImageDR_a, \
                Ei_label, Force_label, Virial_label = batch_data[:10]
        else:
            print(f"Warning: unexpected batch_data length {len(batch_data)}")
            continue

        # Move to device
        list_neigh = [x.to(device) if isinstance(x, torch.Tensor) else x
                      for x in list_neigh]
        Imagetype_map = Imagetype_map.to(device)
        ImageDR = [x.to(device) if isinstance(x, torch.Tensor) else x
                   for x in ImageDR]

        # Build neighbor tensors
        NN_radial = list_neigh[0].to(torch.int32)
        NL_radial = list_neigh[1].to(torch.int32)
        Ri_radial = ImageDR[0]
        NN_angular = list_neigh_a[0].to(torch.int32) if list_neigh_a is not None else NN_radial
        NL_angular = list_neigh_a[1].to(torch.int32) if list_neigh_a is not None else NL_radial
        Ri_angular = ImageDR_a[0] if ImageDR_a is not None else Ri_radial

        num_atom = Imagetype_map_a if Imagetype_map_a is not None else Imagetype_map
        atom_type_map = atom_type.to(torch.int32)

        # Forward pass
        Etot, Ei, Force, Egroup, Virial = model(
            NN_radial, NL_radial, Ri_radial,
            NN_angular, NL_angular, Ri_angular,
            num_atom, atom_type_map,
            Egroup_weight=None, divider=None, is_calc_f=True
        )

        # Compute loss
        if Virial_label is not None and Virial is not None:
            Virial_label = Virial_label.to(device).to(model.dtype)
            natoms_sum = Ri_radial.shape[0]
            batch_size = len(num_atom.reshape(-1))
            loss, per_comp = _compute_target_loss(
                Virial, Virial_label, train_mode, natoms_sum, batch_size)
        else:
            # No labels available — skip this batch
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += np.sqrt(loss.item())  # RMSE ~ sqrt(MSE)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_rmse = total_rmse / max(n_batches, 1)

    if n_batches > 0:
        print(f"Epoch {epoch:4d} | Train {mode_name}_RMSE: {avg_rmse:.6f} | "
              f"LR: {real_lr:.2e}")

    return avg_loss, avg_rmse


def tnep_valid_step(val_loader, model, epoch, device, input_param: InputParam):
    """
    Single validation epoch for tNEP.

    Args:
        val_loader: DataLoader for validation data
        model: TNEP model
        epoch: current epoch number
        device: torch device
        input_param: InputParam config

    Returns:
        avg_loss: average validation loss
        avg_rmse: average validation RMSE
    """
    train_mode = input_param.nep_param.train_mode
    mode_name = {1: "dipole", 2: "polarizability"}.get(train_mode, "potential")

    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    n_batches = 0
    all_per_comp = defaultdict(list)

    with torch.no_grad():
        for batch_data in val_loader:
            if len(batch_data) >= 8:
                list_neigh, Imagetype_map, atom_type, ImageDR, \
                    list_neigh_a, Imagetype_map_a, ImageDR_a, \
                    Ei_label, Force_label, Virial_label = batch_data[:10]
            else:
                continue

            list_neigh = [x.to(device) if isinstance(x, torch.Tensor) else x
                          for x in list_neigh]
            Imagetype_map = Imagetype_map.to(device)
            ImageDR = [x.to(device) if isinstance(x, torch.Tensor) else x
                       for x in ImageDR]

            NN_radial = list_neigh[0].to(torch.int32)
            NL_radial = list_neigh[1].to(torch.int32)
            Ri_radial = ImageDR[0]
            NN_angular = list_neigh_a[0].to(torch.int32) if list_neigh_a is not None else NN_radial
            NL_angular = list_neigh_a[1].to(torch.int32) if list_neigh_a is not None else NL_radial
            Ri_angular = ImageDR_a[0] if ImageDR_a is not None else Ri_radial

            num_atom = Imagetype_map_a if Imagetype_map_a is not None else Imagetype_map
            atom_type_map = atom_type.to(torch.int32)

            _, _, _, _, Virial = model(
                NN_radial, NL_radial, Ri_radial,
                NN_angular, NL_angular, Ri_angular,
                num_atom, atom_type_map,
                Egroup_weight=None, divider=None, is_calc_f=True
            )

            if Virial_label is not None and Virial is not None:
                Virial_label = Virial_label.to(device).to(model.dtype)
                natoms_sum = Ri_radial.shape[0]
                batch_size = len(num_atom.reshape(-1))
                loss, per_comp = _compute_target_loss(
                    Virial, Virial_label, train_mode, natoms_sum, batch_size)

                total_loss += loss.item()
                total_rmse += np.sqrt(loss.item())
                n_batches += 1

                for k, v in per_comp.items():
                    all_per_comp[k].append(v)

    avg_loss = total_loss / max(n_batches, 1)
    avg_rmse = total_rmse / max(n_batches, 1)

    # Report per-component RMSE
    if all_per_comp:
        comp_avg = {k: np.mean(v) for k, v in all_per_comp.items()}
        comp_str = "  ".join(f"{k}={v:.4f}" for k, v in comp_avg.items())
        print(f"Epoch {epoch:4d} | Valid {mode_name}_RMSE: {avg_rmse:.6f} | "
              f"{comp_str}")

    return avg_loss, avg_rmse
