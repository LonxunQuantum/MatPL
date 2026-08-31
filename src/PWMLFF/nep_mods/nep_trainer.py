import os
import glob
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from src.loss.loss import adjust_lr, get_loss, print_l1_l2
from src.utils.learning_rate import (
    calculate_loss_weight_progress,
    calculate_warmup_lr,
    optimizer_step_with_lr,
    optimizer_update_step,
)

from src.optimizer.KFWrapper import KFOptimizerWrapper
# import horovod.torch as hvd
# from torch.profiler import profile, record_function, ProfilerActivity
from src.user.input_param import InputParam
from src.utils.debug_operation import check_cuda_memory
from collections import defaultdict
from src.utils.train_log import AverageMeter, Summary, ProgressMeter
from src.utils.op_loader import load_calc_ops

CalcOps = load_calc_ops()

def get_model_module(model, args:InputParam):
    return model.module if getattr(args, "world_size", 1) > 1 else model


def _get_fragment_charge_loss_and_count(
        atomic_charge, sample, criterion, train_charge_ion=True,
        charge_scale=None):
    if atomic_charge is None or "fragment" not in sample or "fragment_charge" not in sample:
        return None, 0, False

    dtype = atomic_charge.dtype
    device = atomic_charge.device
    num_atom = sample["num_atom"].reshape(-1).to(device=device, dtype=torch.int64)
    fragment = sample["fragment"].reshape(-1).to(device=device, dtype=torch.int64)
    label = sample["fragment_charge"].reshape(-1).to(dtype=dtype, device=device)
    charge = atomic_charge.reshape(-1)

    valid = (~torch.isnan(label)) & (fragment >= 0)
    if not valid.any().item():
        return None, 0, False

    image_index = torch.repeat_interleave(
        torch.arange(num_atom.numel(), device=device, dtype=torch.int64),
        num_atom)
    max_fragment = torch.clamp(fragment[valid].max(), min=0) + 1
    global_fragment = image_index[valid] * max_fragment + fragment[valid]
    unique_fragment, inverse = torch.unique(global_fragment, sorted=True, return_inverse=True)

    pred = torch.zeros(unique_fragment.numel(), dtype=dtype, device=device)
    pred.index_add_(0, inverse, charge[valid])

    label_index = torch.full((unique_fragment.numel(),), label.numel(), dtype=torch.int64, device=device)
    atom_index = torch.arange(label.numel(), device=device, dtype=torch.int64)[valid]
    if hasattr(label_index, "scatter_reduce_"):
        label_index.scatter_reduce_(0, inverse, atom_index, reduce="amin", include_self=True)
    else:
        for idx, atom_idx in zip(inverse.tolist(), atom_index.tolist()):
            if atom_idx < label_index[idx]:
                label_index[idx] = atom_idx
    target = label[label_index]
    if train_charge_ion:
        if charge_scale is not None:
            pred = pred * charge_scale.to(dtype=dtype, device=device)
    else:
        neutral = torch.isclose(
            target, torch.zeros_like(target), rtol=0.0, atol=1e-8)
        if not neutral.any().item():
            return None, 0, True
        pred = pred[neutral]
        target = target[neutral]

    loss = criterion(pred.reshape(-1, 1), target.reshape(-1, 1))
    return loss, target.numel(), True


def _get_fragment_charge_loss(atomic_charge, sample, criterion):
    loss, _, _ = _get_fragment_charge_loss_and_count(
        atomic_charge, sample, criterion, train_charge_ion=True)
    return loss


def get_charge_loss_stats(
        charge_predict, sample, criterion, args:InputParam,
        atomic_charge=None, charge_scale=None):
    """Return optimizer loss, log loss, and the number of charge targets."""
    if not getattr(args.optimizer_param, "train_charge", False):
        return None, None, 0

    fragment_loss, fragment_count, has_fragment_labels = \
        _get_fragment_charge_loss_and_count(
            atomic_charge,
            sample,
            criterion,
            train_charge_ion=getattr(
                args.optimizer_param, "train_charge_ion", False),
            charge_scale=charge_scale,
        )
    if fragment_loss is not None:
        return fragment_loss, fragment_loss, fragment_count
    if has_fragment_labels:
        return None, None, 0

    if charge_predict is None or "charge" not in sample:
        return None, None, 0

    charge_label = sample["charge"].reshape(-1, 1).to(
        dtype=charge_predict.dtype, device=charge_predict.device)
    optimizer_loss = criterion(charge_predict, charge_label)
    num_atom = sample["num_atom"].reshape(-1, 1).to(
        dtype=charge_predict.dtype, device=charge_predict.device)
    log_loss = criterion(charge_predict / num_atom, charge_label / num_atom)
    return optimizer_loss, log_loss, charge_label.numel()


def get_charge_loss(
        charge_predict, sample, criterion, args:InputParam,
        atomic_charge=None, charge_scale=None):
    optimizer_loss, _, _ = get_charge_loss_stats(
        charge_predict, sample, criterion, args, atomic_charge, charge_scale)
    return optimizer_loss


def get_charge_loss_per_atom(
        charge_predict, sample, criterion, args:InputParam,
        atomic_charge=None, charge_scale=None):
    _, log_loss, _ = get_charge_loss_stats(
        charge_predict, sample, criterion, args, atomic_charge, charge_scale)
    return log_loss


def has_bec_label(sample, args:InputParam):
    if not getattr(args.optimizer_param, "train_bec", False) or "bec" not in sample:
        return False
    return (sample["bec"][:, 0] > -1e6).any().item()


def get_bec_loss(bec_predict, sample, criterion, args:InputParam):
    if not getattr(args.optimizer_param, "train_bec", False):
        return None, None
    if bec_predict is None or "bec" not in sample:
        return None, None
    bec_label = sample["bec"].to(dtype=bec_predict.dtype, device=bec_predict.device)
    bec_mask = bec_label[:, 0] > -1e6
    if not bec_mask.any().item():
        return None, None
    return criterion(bec_predict[bec_mask], bec_label[bec_mask]), bec_mask


def _has_bec_label_for_inference(sample):
    if "bec" not in sample:
        return False
    return (sample["bec"][:, 0] > -1e6).any().item()


def _split_tensor_by_num_atom(tensor, num_atom):
    if tensor is None:
        return []
    tensor = tensor.reshape(-1, *tensor.shape[1:])
    num_atom_list = num_atom.reshape(-1).detach().cpu().numpy().astype(int).tolist()
    chunks = torch.split(tensor, num_atom_list, dim=0)
    return [chunk.detach().cpu().numpy() for chunk in chunks]


def _charge_mode_enabled(model, args):
    module = get_model_module(model, args)
    return bool(getattr(module, "charge_mode", 0))


def _metric_value(loss, take_root=False):
    if loss is None:
        return np.nan
    value = loss ** 0.5 if take_root else loss
    return float(value)


def _build_predict_metric_row(
        image_index, etot_rmse, etot_atom_rmse, ei_rmse, force_rmse, args,
        charge_loss=None, bec_loss=None, egroup_loss=None,
        virial_loss=None, virial_per_atom_loss=None):
    row = {
        "img_idx": image_index,
        "RMSE_Etot": float(etot_rmse),
        "RMSE_Etot_per_atom": float(etot_atom_rmse),
        "RMSE_Ei": float(ei_rmse),
        "RMSE_F": float(force_rmse),
    }
    if args.optimizer_param.train_charge:
        row["RMSE_charge"] = _metric_value(charge_loss, take_root=True)
    if args.optimizer_param.train_bec:
        row["RMSE_BEC"] = _metric_value(bec_loss, take_root=True)
    if args.optimizer_param.train_egroup:
        row["RMSE_Egroup"] = _metric_value(egroup_loss)
    if args.optimizer_param.train_virial:
        row["RMSE_virial"] = _metric_value(virial_loss)
        row["RMSE_virial_per_atom"] = _metric_value(virial_per_atom_loss)
    return row


def _collect_charge_outputs_for_inference(
        atomic_charge, total_charge_predict, sample, train_charge_ion=False,
        charge_scale=None):
    if atomic_charge is None:
        return [], [], []

    num_atom = sample["num_atom"].reshape(-1).detach().cpu().numpy().astype(int).tolist()
    charge_predict_chunks = [
        chunk.detach().cpu().numpy().reshape(-1)
        for chunk in torch.split(atomic_charge.reshape(-1), num_atom, dim=0)
    ]
    charge_label_chunks = []
    charge_rmse_list = []
    scale = None
    if charge_scale is not None:
        scale = float(charge_scale.detach().cpu().reshape(-1)[0])

    if "fragment_charge" in sample and "fragment" in sample:
        fragment_charge_chunks = _split_tensor_by_num_atom(sample["fragment_charge"], sample["num_atom"])
        fragment_chunks = _split_tensor_by_num_atom(sample["fragment"], sample["num_atom"])
        for charge_predict, charge_label, fragment in zip(charge_predict_chunks, fragment_charge_chunks, fragment_chunks):
            charge_label = charge_label.reshape(-1)
            fragment = fragment.reshape(-1)
            valid = (fragment >= 0) & np.isfinite(charge_label)
            if not valid.any():
                charge_label_chunks.append(charge_label)
                continue

            pred_frag_charge = []
            label_frag_charge = []
            for frag in np.unique(fragment[valid]):
                frag_mask = fragment == frag
                valid_frag_mask = frag_mask & valid
                if not valid_frag_mask.any():
                    continue
                pred_frag_charge.append(np.sum(charge_predict[frag_mask]))
                label_frag_charge.append(charge_label[valid_frag_mask][0])
            if pred_frag_charge:
                pred_frag_charge = np.asarray(pred_frag_charge)
                label_frag_charge = np.asarray(label_frag_charge)
                if train_charge_ion:
                    if scale is not None:
                        pred_frag_charge = pred_frag_charge * scale
                else:
                    neutral = np.isclose(label_frag_charge, 0.0, rtol=0.0, atol=1e-8)
                    pred_frag_charge = pred_frag_charge[neutral]
                    label_frag_charge = label_frag_charge[neutral]
                if pred_frag_charge.size == 0:
                    charge_label_chunks.append(np.array([]))
                    continue
                charge_rmse_list.append(np.sqrt(np.mean((pred_frag_charge - label_frag_charge) ** 2)))
                charge_label_chunks.append(label_frag_charge)
            else:
                charge_label_chunks.append(charge_label)
    elif "charge" in sample:
        charge_label = sample["charge"].reshape(-1).detach().cpu().numpy()
        if total_charge_predict is not None:
            total_charge_predictions = total_charge_predict.reshape(-1).detach().cpu().numpy()
        else:
            physical_scale = 1.0 if scale is None else scale
            total_charge_predictions = np.asarray([
                np.sum(charge_predict) * physical_scale
                for charge_predict in charge_predict_chunks
            ])
        for charge_predict, total_charge, predicted_total_charge in zip(
                charge_predict_chunks, charge_label, total_charge_predictions):
            charge_label_chunks.append(np.asarray([total_charge]))
            charge_rmse_list.append(np.abs(predicted_total_charge - total_charge))
    else:
        charge_label_chunks = [np.array([]) for _ in charge_predict_chunks]

    return charge_label_chunks, charge_predict_chunks, charge_rmse_list


def _collect_bec_outputs_for_inference(bec_predict, sample):
    if bec_predict is None:
        return [], []

    bec_predict_chunks = _split_tensor_by_num_atom(bec_predict, sample["num_atom"])
    if "bec" in sample:
        bec_label_chunks = _split_tensor_by_num_atom(sample["bec"], sample["num_atom"])
    else:
        bec_label_chunks = [np.ones_like(bec_predict) * (-1e6) for bec_predict in bec_predict_chunks]
    return bec_label_chunks, bec_predict_chunks
def _get_model_output_requests(sample, args: InputParam, train_virial: bool):
    need_force = bool(getattr(args.optimizer_param, "train_force", True))
    need_bec = has_bec_label(sample, args)
    need_charge_energy = bool(
        getattr(args.optimizer_param, "train_energy", True) or
        need_force or
        train_virial)
    return {
        "need_force": need_force,
        "need_bec": need_bec,
        "need_charge_virial": train_virial,
        "need_charge_energy": need_charge_energy,
    }


def train(train_loader, model, criterion, optimizer, scheduler, epoch,
          optimizer_peak_lr, completed_updates, warmup_updates,
          device, args:InputParam):
    batch_time = AverageMeter("Time", ":6.3f", device=device, world_size=args.world_size)
    data_time = AverageMeter("Data", ":6.3f", device=device, world_size=args.world_size)
    learning_rate = AverageMeter("LR", ":.8e", Summary.AVERAGE, device=device, world_size=args.world_size)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE, device=device, world_size=args.world_size)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Charge = AverageMeter("Charge", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_BEC = AverageMeter("BEC", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    progress_meters = [
        batch_time,
        data_time,
        learning_rate,
        losses,
        loss_L1,
        loss_L2,
        loss_Etot,
        loss_Etot_per_atom,
        loss_Force,
        loss_Ei,
    ]
    if args.optimizer_param.train_egroup:
        progress_meters.append(loss_Egroup)
    if args.optimizer_param.train_charge:
        progress_meters.append(loss_Charge)
    if args.optimizer_param.train_bec:
        progress_meters.append(loss_BEC)
    if args.optimizer_param.train_virial:
        progress_meters.extend([loss_Virial, loss_Virial_per_atom])
    progress = ProgressMeter(
        len(train_loader),
        progress_meters,
        prefix=f"Epoch: [{epoch}]",
    )

    module = model.module if args.world_size > 1 else model
    model.train()
    end = time.time()
    for i, sample in enumerate(train_loader):
        sample = {key: value.to(device) for key, value in sample.items()}
        nn_radial, nn_angular = CalcOps.calculate_maxneigh(
            sample["num_atom"],
            sample["box"],
            sample["box_original"],
            sample["num_cell"],
            sample["position"],
            module.cutoff_radial,
            module.cutoff_angular,
            len(module.atom_type),
            sample["atom_type_map"],
            False
        )
        max_NN_radial = max(torch.max(nn_radial).item(), 10)
        max_NN_angular = max(torch.max(nn_angular).item(), 10)
        FFAtomType = torch.from_numpy(np.array(module.atom_type)).to(device=device, dtype=sample["atom_type_map"].dtype)
        # mem_3c = (int(sample['num_atom_sum'][-1])  *  model.max_NN_angular + int(sample['num_atom_sum'][-1]) ) * len(args.atom_type) *args.nep_param.basis_size[1] * args.nep_param.n_max[1] * 8 / 1024/ 1024/ 1024
        # line = f"Epoch {epoch} - iter {i}: Rank: {args.rank}, LocalRank: {args.local_rank} start: timeused {time.time() - end}"
        # check_cuda_memory(epoch, args.optimizer_param.epochs, line, False, args.rank)
        NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular = \
            CalcOps.calculate_neighbor(
                sample["num_atom"],
                sample["atom_type_map"],
                FFAtomType - 1,
                sample["box"],
                sample["box_original"],
                sample["num_cell"],
                sample["position"],
                module.cutoff_radial,
                module.cutoff_angular,
                max_NN_radial,
                max_NN_angular,
                True
            )
        Virial_label = sample["virial"]
        Etot_label = sample["energy"]
        Ei_label = sample["ei"]
        Egroup_label = None
        Force_label = sample["force"]

        data_time.update(time.time() - end)
        batch_size = sample["num_atom"].shape[0]
        avg_atom_number = (sample['num_atom_sum'][-1] / batch_size).item()
        global_update = optimizer_update_step(completed_updates, i)
        loss_weight_progress = calculate_loss_weight_progress(
            global_update, args.optimizer_param.stop_step)
        # 如果采用预热，则前n个epoch 学习率线性增加
        if global_update < warmup_updates:
            optimizer_lr = calculate_warmup_lr(
                global_update=global_update,
                warmup_updates=warmup_updates,
                start_lr=args.optimizer_param.stop_lr,
                optimizer_peak_lr=optimizer_peak_lr,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = optimizer_lr
            is_warmlr = True
        else:
            is_warmlr = False
            if scheduler is None: # 不启用周期性重启
                optimizer_lr = adjust_lr(
                    global_update, optimizer_peak_lr,
                    args.optimizer_param.stop_step, args.optimizer_param.decay_step, args.optimizer_param.stop_lr
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = optimizer_lr
            else: # 周期性重启
                if global_update == warmup_updates:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = optimizer_peak_lr
                optimizer_lr = optimizer.param_groups[0]["lr"]

        data_mask = Virial_label[:, 0] > -1e6
        train_virial = args.optimizer_param.train_virial and data_mask.any().item()
        output_requests = _get_model_output_requests(sample, args, train_virial)
        # check_cuda_memory(epoch, -1, f"before forword atomnums {Force_label.shape[0]}", False, args.rank)
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict, Charge_predict, Bec_predict = model(
            NN_radial, NL_radial, Ri_radial,
            NN_angular, NL_angular, Ri_angular,
            sample["num_atom"], sample["atom_type_map"], None, None,
            charge_label=sample.get("charge"),
            position=sample.get("position"),
            box_original=sample.get("box_original"),
            volume=sample.get("volume"),
            **output_requests
        )
        # check_cuda_memory(epoch, -1, "end forword", False, args.rank)
        optimizer.zero_grad()
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        if args.optimizer_param.train_force:
            loss_F_val = criterion(Force_predict, Force_label)
        else:
            loss_F_val = torch.zeros_like(loss_Etot_val)
        loss_Etot_per_atom_val = criterion(Etot_predict / sample["num_atom"], Etot_label / sample["num_atom"])
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        atomic_charge_for_loss = getattr(module, "atomic_charge_shifted", None)
        charge_scale_for_loss = getattr(module, "sqrt_epsilon_inf", None)
        loss_Charge_val, loss_Charge_per_atom_val, charge_target_count = get_charge_loss_stats(
            Charge_predict, sample, criterion, args,
            atomic_charge_for_loss, charge_scale_for_loss)
        loss_BEC_val, bec_mask = get_bec_loss(Bec_predict, sample, criterion, args)
        loss_Egroup_val = None
        loss_Virial_val = None

        if args.optimizer_param.train_egroup:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
        if train_virial:
            _Virial_label = Virial_label[:, [0, 1, 2, 4, 5, 8]][data_mask]
            if train_virial:
                loss_Virial_val = criterion(Virial_predict[data_mask][:, [0, 1, 2, 4, 5, 8]], _Virial_label)
                loss_Virial_per_atom_val = criterion(
                    Virial_predict[data_mask][:, [0, 1, 2, 4, 5, 8]] / sample["num_atom"][data_mask],
                    _Virial_label / sample["num_atom"][data_mask]
                )
                loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])

        loss = get_loss(
            args,
            optimizer_lr,
            avg_atom_number,
            loss_F_val,
            loss_Etot_val,
            loss_Virial_val,
            loss_Egroup_val,
            loss_Charge_val,
            loss_BEC_val,
            train_virial,
            loss_weight_progress=loss_weight_progress,
        )
        # check_cuda_memory(epoch, -1, "before backward", False, args.rank)
        loss.backward()
        torch.cuda.empty_cache() # 释放pytoch 缓存管理器持有的缓冲块，因为它对cuda算子不可见，导致算子内存不够用，这部分缓冲块 64batch下约10个G
        # check_cuda_memory(epoch, -1, "end backward", False, args.rank)

        if args.optimizer_param.norm_type is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.optimizer_param.max_norm, args.optimizer_param.norm_type)
        elif args.optimizer_param.clip_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), args.optimizer_param.clip_value)
        optimizer_lr = optimizer_step_with_lr(
            optimizer,
            scheduler if scheduler is not None and is_warmlr is False else None,
        )
        learning_rate.update(optimizer_lr)

        loss_val = loss
        L1, L2 = print_l1_l2(model)
        if args.optimizer_param.lambda_2:
            loss_val += L2

        losses.update(loss_val.item(), batch_size)
        loss_Etot.update(loss_Etot_val.item(), batch_size)
        loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
        loss_Ei.update(loss_Ei_val.item(), batch_size)
        loss_L1.update(L1.item(), batch_size)
        loss_L2.update(L2.item(), batch_size)
        if args.optimizer_param.train_egroup:
            loss_Egroup.update(loss_Egroup_val.item(), batch_size)
        if loss_Charge_per_atom_val is not None:
            loss_Charge.update(loss_Charge_per_atom_val.item(), charge_target_count)
        if loss_BEC_val is not None:
            loss_BEC.update(loss_BEC_val.item(), int(bec_mask.sum().item()))
        if args.optimizer_param.train_force:
            loss_Force.update(loss_F_val.item(), batch_size)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            if args.world_size > 1 and args.reduce_loss:
                progress.sync_meters()
            if args.rank == 0:
                progress.display(i + 1)

        if args.save_step is not None and i % args.save_step == 0:
            if args.rank == 0:
                save_step_checkpoint(
                    {
                        "json_file": args.to_dict(),
                        "epoch": epoch,
                        "state_dict": model.state_dict()
                        # "energy_shift": module.energy_shift,
                        # "max_neighbor": [model.max_NN_radial, model.max_NN_angular],
                        # "q_scaler": model.get_q_scaler(),
                        # "atom_type_order": args.atom_type
                    },
                    os.path.join(args.file_paths.model_store_dir, "saved_models"),
                    epoch,
                    i,
                    args.max_save_num
                )
            if args.world_size > 1:
                dist.barrier()

    if args.world_size > 1:
        progress.sync_meters()

    if args.rank == 0:
        progress.display_summary([
            "Training Set:",
            f"PeakLR {optimizer_peak_lr:.8e}",
            f"LastLRUsed {optimizer_lr:.8e}",
        ])

    return (
        losses.avg,
        loss_Etot.root,
        loss_Etot_per_atom.root,
        loss_Force.root,
        loss_Ei.root,
        loss_Egroup.root,
        loss_Virial.root,
        loss_Virial_per_atom.root,
        loss_Charge.root,
        loss_BEC.root,
        optimizer_lr,
        loss_L1.root,
        loss_L2.root
    )

def train_KF(train_loader, model, criterion, optimizer, epoch, device, args:InputParam):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)
    loss_Charge = AverageMeter("Charge", ":.4e", Summary.ROOT)
    loss_BEC = AverageMeter("BEC", ":.4e", Summary.ROOT)
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT)
    progress_meters = [
        batch_time,
        data_time,
        losses,
        loss_L1,
        loss_L2,
        loss_Etot,
        loss_Etot_per_atom,
        loss_Force,
        loss_Ei,
    ]
    if args.optimizer_param.train_egroup:
        progress_meters.append(loss_Egroup)
    if args.optimizer_param.train_charge:
        progress_meters.append(loss_Charge)
    if args.optimizer_param.train_bec:
        progress_meters.append(loss_BEC)
    if args.optimizer_param.train_virial:
        progress_meters.extend([loss_Virial, loss_Virial_per_atom])
    progress = ProgressMeter(
        len(train_loader),
        progress_meters,
        prefix="Epoch: [{}]".format(epoch),
    )

    KFOptWrapper = KFOptimizerWrapper(
        model, optimizer, args.optimizer_param.nselect, args.optimizer_param.groupsize, lambda_l1 = args.optimizer_param.lambda_1, lambda_l2 = args.optimizer_param.lambda_2
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        sample = {key: value.to(device) for key, value in sample.items()}
        nn_radial, nn_angular = CalcOps.calculate_maxneigh(
            sample["num_atom"],
            sample["box"],
            sample["box_original"],
            sample["num_cell"],
            sample["position"],
            model.cutoff_radial,
            model.cutoff_angular,
            len(model.atom_type),
            sample["atom_type_map"],
            False
        )
        max_NN_radial = max(torch.max(nn_radial).item(), 10)
        max_NN_angular = max(torch.max(nn_angular).item(), 10)
        FFAtomType = torch.from_numpy(np.array(model.atom_type)).to(device=device, dtype=sample["atom_type_map"].dtype)
        NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular = \
            CalcOps.calculate_neighbor(
            sample["num_atom"],
            sample["atom_type_map"],
            FFAtomType-1,
            sample["box"],
            sample["box_original"],
            sample["num_cell"],
            sample["position"],
            model.cutoff_radial,
            model.cutoff_angular,
            max_NN_radial,
            max_NN_angular,
            True #calculate_neighbor
        )
        kalman_inputs = [NN_radial, NL_radial, Ri_radial, NN_angular, NL_angular, Ri_angular, \
                            sample["num_atom"], sample["atom_type_map"], None, None,
                            sample.get("charge")]
        Virial_label = sample["virial"]
        Etot_label   = sample["energy"]
        Ei_label     = sample["ei"]
        Egroup_label = None
        Force_label  = sample["force"]
        Charge_label = sample.get("charge")
        BEC_label = sample.get("bec")
        batch_has_bec_label = has_bec_label(sample, args)
        Charge_predict = None
        Bec_predict = None
        if args.optimizer_param.train_virial is True:
            # check_cuda_memory(epoch, i, "train_virial start")
            Virial_predict = KFOptWrapper.update_virial(kalman_inputs, Virial_label, args.optimizer_param.pre_fac_virial, train_type = "NEP")
        if args.optimizer_param.train_energy is True:
            # check_cuda_memory(epoch, i, "update_energy start")
            Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, args.optimizer_param.pre_fac_etot, train_type = "NEP")
            # check_cuda_memory(-1, -1, "update_energy end")
        if args.optimizer_param.train_ei is True:
            Ei_predict = KFOptWrapper.update_ei(kalman_inputs, Ei_label, args.optimizer_param.pre_fac_ei, train_type = "NEP")

        if args.optimizer_param.train_egroup is True:
            Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, args.optimizer_param.pre_fac_egroup, train_type = "NEP")

        if args.optimizer_param.train_charge is True and Charge_label is not None:
            Charge_predict = KFOptWrapper.update_charge(kalman_inputs, Charge_label, args.optimizer_param.pre_fac_charge, train_type = "NEP")

        if batch_has_bec_label and BEC_label is not None:
            Bec_predict = KFOptWrapper.update_bec(kalman_inputs, BEC_label, args.optimizer_param.pre_fac_bec, train_type = "NEP")

        if args.optimizer_param.train_force is True:
            # check_cuda_memory(epoch, i, "update_force start")
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                kalman_inputs, Force_label, args.optimizer_param.pre_fac_force, train_type = "NEP")
                # check_cuda_memory(-1, -1, "update_force end")
        # Force_predict = Force_label
        # Ei_predict = Ei_label
        loss_F_val = criterion(Force_predict, Force_label)
        L1, L2 = print_l1_l2(model)

        # divide by natoms
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Etot_per_atom_val = criterion(Etot_predict/sample["num_atom"], Etot_label/sample["num_atom"])

        loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_Charge_val, loss_Charge_per_atom_val, charge_target_count = get_charge_loss_stats(
            Charge_predict, sample, criterion, args)
        loss_BEC_val, bec_mask = get_bec_loss(Bec_predict, sample, criterion, args)
        if args.optimizer_param.train_egroup is True:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

        loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                    args.optimizer_param.pre_fac_etot * loss_Etot_val
        if loss_Charge_val is not None:
            loss_val += args.optimizer_param.pre_fac_charge * loss_Charge_val
        if loss_BEC_val is not None:
            loss_val += args.optimizer_param.pre_fac_bec * loss_BEC_val

        if args.optimizer_param.train_virial is True:
            data_mask = Virial_label[:, 0] > -1e6
            _Virial_label = Virial_label[:, [0,1,2,4,5,8]][data_mask]
            if data_mask.any().item():
                loss_Virial_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]], _Virial_label)
                loss_Virial_per_atom_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]]/sample["num_atom"][data_mask], _Virial_label/sample["num_atom"][data_mask])
                loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
                loss_val += args.optimizer_param.pre_fac_virial * loss_Virial_val

        if args.optimizer_param.lambda_2 is not None:
            loss_val += L2
        if args.optimizer_param.lambda_1 is not None:
            loss_val += L1
        batch_size = sample["num_atom"].shape[0]
        # measure accuracy and record loss
        losses.update(loss_val.item(), batch_size)
        loss_L1.update(L1.item(), batch_size)
        loss_L2.update(L2.item(), batch_size)

        loss_Etot.update(loss_Etot_val.item(), batch_size)
        loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
        loss_Ei.update(loss_Ei_val.item(), Ei_predict.shape[0])
        if args.optimizer_param.train_egroup is True:
            loss_Egroup.update(loss_Egroup_val.item(), batch_size)
        if loss_Charge_per_atom_val is not None:
            loss_Charge.update(loss_Charge_per_atom_val.item(), charge_target_count)
        if loss_BEC_val is not None:
            loss_BEC.update(loss_BEC_val.item(), int(bec_mask.sum().item()))
        loss_Force.update(loss_F_val.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1)

    """
    if args.hvd:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Etot_per_atom.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()
        if args.optimizer_param.train_egroup is True:
            loss_Egroup.all_reduce()
        if args.optimizer_param.train_virial is True:
            loss_Virial.all_reduce()
            loss_Virial_per_atom.all_reduce()
        batch_time.all_reduce()
    """
    progress.display_summary(["Training Set:"])
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root, loss_Charge.root, loss_BEC.root, loss_L1.root, loss_L2.root

def valid(val_loader, model, criterion, device, args:InputParam):
    def run_validate(loader, base_progress=0):
        end = time.time()
        L1, L2 = print_l1_l2(model)
        for i, sample in enumerate(val_loader):
            sample = {key: value.to(device) for key, value in sample.items()}
            FFAtomType = torch.from_numpy(np.array(module.atom_type)).to(device=device, dtype=sample["atom_type_map"].dtype)

            nn_radial, nn_angular = CalcOps.calculate_maxneigh(
                sample["num_atom"],
                sample["box"],
                sample["box_original"],
                sample["num_cell"],
                sample["position"],
                module.cutoff_radial,
                module.cutoff_angular,
                len(module.atom_type),
                sample["atom_type_map"],
                False
            )
            max_NN_radial = max(torch.max(nn_radial).item(), 10)
            max_NN_angular = max(torch.max(nn_angular).item(), 10)
            NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular = \
                CalcOps.calculate_neighbor(
                sample["num_atom"],
                sample["atom_type_map"],
                FFAtomType-1,
                sample["box"],
                sample["box_original"],
                sample["num_cell"],
                sample["position"],
                module.cutoff_radial,
                module.cutoff_angular,
                max_NN_radial,
                max_NN_angular,
                True #calculate_neighbor
            )
            Virial_label = sample["virial"]
            Etot_label   = sample["energy"]
            Ei_label     = sample["ei"]
            Egroup_label = None
            Force_label  = sample["force"]

            # measure data loading time
            batch_size =  sample["num_atom"].shape[0]
            avg_atom_number = (sample['num_atom_sum'][-1] / batch_size).item()
            nr_batch_sample = sample["num_atom"].shape[0]

            # if args.optimizer_param.train_egroup is True:
            #     Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict, Charge_predict, Bec_predict = model(
            #         dR_neigh_list, ImageDR, dR_neigh_type_list, \
            #             dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
            #             atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider)

                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
            data_mask = Virial_label[:, 0] > -1e6
            need_charge_virial = args.optimizer_param.train_virial and data_mask.any().item()
            batch_has_bec_label = has_bec_label(sample, args)
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict, Charge_predict, Bec_predict = model(
                    NN_radial, NL_radial, Ri_radial,
                        NN_angular, NL_angular, Ri_angular,
                            sample["num_atom"], sample["atom_type_map"], None, None,
                            charge_label=sample.get("charge"),
                            position=sample.get("position"),
                            box_original=sample.get("box_original"),
                            volume=sample.get("volume"),
                            need_force=True,
                            need_bec=batch_has_bec_label,
                            need_charge_virial=need_charge_virial,
                            need_charge_energy=True)

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = criterion(Etot_predict/sample["num_atom"], Etot_label/sample["num_atom"])
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            atomic_charge_for_loss = getattr(module, "atomic_charge_shifted", None)
            charge_scale_for_loss = getattr(module, "sqrt_epsilon_inf", None)
            loss_Charge_val, loss_Charge_log_val, charge_target_count = get_charge_loss_stats(
                Charge_predict, sample, criterion, args,
                atomic_charge_for_loss, charge_scale_for_loss)
            loss_BEC_val, bec_mask = get_bec_loss(Bec_predict, sample, criterion, args)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                    args.optimizer_param.pre_fac_etot * loss_Etot_val

            if loss_Charge_val is not None:
                loss_val += args.optimizer_param.pre_fac_charge * loss_Charge_val
            if loss_BEC_val is not None:
                loss_val += args.optimizer_param.pre_fac_bec * loss_BEC_val

            if args.optimizer_param.train_virial is True:
                # loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))  #115.415137283393
                data_mask = Virial_label[:, 0] > -1e6
                _Virial_label = Virial_label[:, [0,1,2,4,5,8]][data_mask]
                if data_mask.any().item():
                    loss_Virial_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]], _Virial_label)
                    loss_Virial_per_atom_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]]/sample["num_atom"][data_mask], _Virial_label/sample["num_atom"][data_mask])
                    loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                    loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
                    loss_val += args.optimizer_param.pre_fac_virial * loss_Virial_val
                if args.optimizer_param.lambda_2 is not None:
                    loss_val += L2
                if args.optimizer_param.lambda_1 is not None:
                    loss_val += L1
            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            if loss_Charge_val is not None:
                loss_Charge.update(loss_Charge_log_val.item(), charge_target_count)
            if loss_BEC_val is not None:
                loss_BEC.update(loss_BEC_val.item(), int(bec_mask.sum().item()))
            loss_Force.update(loss_F_val.item(), batch_size)
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            if args.world_size > 1 and args.reduce_loss:
                progress.sync_meters()
            if args.rank == 0:
                progress.display(i + 1)

    batch_time = AverageMeter("Time", ":6.3f", device=device, world_size=args.world_size)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE, device=device, world_size=args.world_size)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Charge = AverageMeter("Charge", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_BEC = AverageMeter("BEC", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)

    progress_meters = [
        batch_time,
        losses,
        loss_Etot,
        loss_Etot_per_atom,
        loss_Force,
        loss_Ei,
    ]
    if args.optimizer_param.train_egroup:
        progress_meters.append(loss_Egroup)
    if args.optimizer_param.train_charge:
        progress_meters.append(loss_Charge)
    if args.optimizer_param.train_bec:
        progress_meters.append(loss_BEC)
    if args.optimizer_param.train_virial:
        progress_meters.extend([loss_Virial, loss_Virial_per_atom])
    progress = ProgressMeter(
        len(val_loader),
        progress_meters,
        prefix="Test: ",
    )
    module = model.module if args.world_size > 1 else model
    # switch to evaluate mode
    model.eval()

    run_validate(val_loader)

    """
    if args.hvd and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset)):
        aux_val_dataset = Subset(
            val_loader.dataset,
            range(len(val_loader.sampler) * hvd.size(), len(val_loader.dataset)),
        )
        aux_val_loader = torch.utils.data.DataLoader(
            aux_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )
        run_validate(aux_val_loader, len(val_loader))

    if args.hvd:
        losses.all_reduce()
        loss_Etot.all_reduce()
        loss_Etot_per_atom.all_reduce()
        loss_Force.all_reduce()
        loss_Ei.all_reduce()
        if args.optimizer_param.train_virial is True:
            loss_Virial.all_reduce()
            loss_Virial_per_atom.all_reduce()
    """

    if args.world_size > 1:
        progress.sync_meters()
    if args.rank == 0:
        progress.display_summary(["Test Set:"])

    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root, loss_Charge.root, loss_BEC.root

'''
description:
this function is used for inference:
the output is a pandas DataFrame object
param {*} val_loader
param {*} model
param {*} criterion
param {*} device
param {*} args
return {*}
author: wuxingxing
'''
def predict(val_loader, model, criterion, device, args:InputParam, isprofile=False):
    train_lists = ["img_idx"] #"Etot_lab", "Etot_pre", "Ei_lab", "Ei_pre", "Force_lab", "Force_pre"
    train_lists.extend(["RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F"])
    if args.optimizer_param.train_charge:
        train_lists.append("RMSE_charge")
    if args.optimizer_param.train_bec:
        train_lists.append("RMSE_BEC")
    if args.optimizer_param.train_egroup:
        train_lists.append("RMSE_Egroup")
    if args.optimizer_param.train_virial:
        train_lists.append("RMSE_virial")
        train_lists.append("RMSE_virial_per_atom")

    res_pd = pd.DataFrame(columns=train_lists)
    force_label_list = []
    force_predict_list = []
    ei_label_list = []
    ei_predict_list = []
    etot_label_list = []
    etot_predict_list = []
    virial_label_list = []
    virial_predict_list = []
    charge_label_list = []
    charge_predict_list = []
    charge_rmse_list = []
    bec_label_list = []
    bec_predict_list = []
    model.eval()
    virial_index = [0, 1, 2, 4, 5, 8]
    for i, sample in enumerate(val_loader):
        sample = {key: value.to(device) for key, value in sample.items()}
        FFAtomType = torch.from_numpy(np.array(model.atom_type)).to(device=device, dtype=sample["atom_type_map"].dtype)
        NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular = \
            CalcOps.calculate_neighbor(
            sample["num_atom"],
            sample["atom_type_map"],
            FFAtomType-1,
            sample["box"],
            sample["box_original"],
            sample["num_cell"],
            sample["position"],
            model.cutoff_radial,
            model.cutoff_angular,
            model.max_NN_radial,
            model.max_NN_angular,
            True #calculate_neighbor
        )
        Virial_label = sample["virial"]
        Etot_label   = sample["energy"]
        Ei_label     = sample["ei"]
        Egroup_label = None
        Force_label  = sample["force"]

        # measure data loading time
        data_mask = Virial_label[:, 0] > -1e6
        need_charge_virial = args.optimizer_param.train_virial and data_mask.any().item()
        inference_mode = getattr(args, "inference", False)
        batch_has_bec_label = (
            has_bec_label(sample, args) or
            (inference_mode and (_has_bec_label_for_inference(sample) or _charge_mode_enabled(model, args))))
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict, Charge_predict, Bec_predict = model(
                NN_radial, NL_radial, Ri_radial,
                    NN_angular, NL_angular, Ri_angular,
                        sample["num_atom"], sample["atom_type_map"], None, None,
                        charge_label=sample.get("charge"),
                        position=sample.get("position"),
                        box_original=sample.get("box_original"),
                        volume=sample.get("volume"),
                        need_force=True,
                        need_bec=batch_has_bec_label,
                        need_charge_virial=need_charge_virial,
                        need_charge_energy=True)

        # mse
        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Etot_per_atom_val = criterion(Etot_predict/sample["num_atom"], Etot_label/sample["num_atom"])
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        atomic_charge_for_loss = getattr(model, "atomic_charge_shifted", None)
        charge_scale_for_loss = getattr(model, "sqrt_epsilon_inf", None)
        loss_Charge_val = get_charge_loss(
            Charge_predict, sample, criterion, args,
            atomic_charge_for_loss, charge_scale_for_loss)
        loss_BEC_val, _ = get_bec_loss(Bec_predict, sample, criterion, args)
        loss_Egroup_val = None
        loss_Virial_val = None
        loss_Virial_per_atom_val = None
        if args.optimizer_param.train_egroup is True:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

        loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                args.optimizer_param.pre_fac_etot * loss_Etot_val

        if args.optimizer_param.train_virial is True:
            # loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))  #115.415137283393
            data_mask = Virial_label[:, 0] > -1e6
            _Virial_label = Virial_label[:, virial_index][data_mask]
            if data_mask.any().item():
                loss_Virial_val = criterion(Virial_predict[data_mask][:,virial_index], _Virial_label)
                loss_Virial_per_atom_val = criterion(Virial_predict[data_mask][:,virial_index]/sample["num_atom"][data_mask], _Virial_label/sample["num_atom"][data_mask])
                # loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                # loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
                loss_val += args.optimizer_param.pre_fac_virial * loss_Virial_val

        # rmse
        Etot_rmse = loss_Etot_val ** 0.5
        etot_atom_rmse = loss_Etot_per_atom_val**0.5
        Ei_rmse = loss_Ei_val ** 0.5
        F_rmse = loss_F_val ** 0.5

        res_row = _build_predict_metric_row(
            image_index=i,
            etot_rmse=Etot_rmse,
            etot_atom_rmse=etot_atom_rmse,
            ei_rmse=Ei_rmse,
            force_rmse=F_rmse,
            args=args,
            charge_loss=loss_Charge_val,
            bec_loss=loss_BEC_val,
            egroup_loss=loss_Egroup_val,
            virial_loss=loss_Virial_val,
            virial_per_atom_loss=loss_Virial_per_atom_val,
        )

        force_label_list.append(Force_label.flatten().cpu().numpy())
        force_predict_list.append(Force_predict.flatten().detach().cpu().numpy())
        ei_label_list.append(Ei_label.flatten().cpu().numpy())
        ei_predict_list.append(Ei_predict.flatten().detach().cpu().numpy())
        etot_label_list.append(float(Etot_label))
        etot_predict_list.append(float(Etot_predict))
        res_pd.loc[res_pd.shape[0]] = res_row
        virial_label_list.append(Virial_label[:,virial_index].flatten().detach().cpu().numpy())
        virial_predict_list.append(Virial_predict[:,virial_index].flatten().detach().cpu().numpy())
        if inference_mode:
            module = get_model_module(model, args)
            atomic_charge_for_output = getattr(module, "atomic_charge_shifted", None)
            charge_labels, charge_predicts, charge_rmses = _collect_charge_outputs_for_inference(
                atomic_charge_for_output,
                Charge_predict,
                sample,
                train_charge_ion=getattr(
                    args.optimizer_param, "train_charge_ion", False),
                charge_scale=getattr(module, "sqrt_epsilon_inf", None),
            )
            charge_label_list.extend(charge_labels)
            charge_predict_list.extend(charge_predicts)
            charge_rmse_list.extend(charge_rmses)
            if Bec_predict is not None:
                bec_labels, bec_predicts = _collect_bec_outputs_for_inference(Bec_predict, sample)
                bec_label_list.extend(bec_labels)
                bec_predict_list.extend(bec_predicts)
    return (
        res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list,
        force_label_list, force_predict_list, virial_label_list, virial_predict_list,
        charge_label_list, charge_predict_list, charge_rmse_list, bec_label_list, bec_predict_list)


def save_step_checkpoint(state, save_dir:str, epoch:int, iter:int, max_save_num:int=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get model
    ckpt_list = glob.glob(os.path.join(save_dir, "*.ckpt"))
    ckpt_list = sorted(ckpt_list, key=lambda x:tuple(map(int, os.path.basename(x).split('.')[0].split('_'))))
    if len(ckpt_list) >= max_save_num:
        for i in range(0, len(ckpt_list)-max_save_num):
            os.remove(ckpt_list[i])
    save_checkpoint(state, "{}_{}.ckpt".format(epoch, iter), save_dir)

def save_checkpoint(state, filename, prefix):
    filename = os.path.join(prefix, filename)
    torch.save(state, filename)
