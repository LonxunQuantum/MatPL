import os
import glob
import pandas as pd
import numpy as np
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from src.loss.dploss import calc_loss, adjust_lr, warmup_lr

from src.optimizer.KFWrapper import KFOptimizerWrapper
# import horovod.torch as hvd
# from torch.profiler import profile, record_function, ProfilerActivity
from src.user.input_param import InputParam
from src.utils.debug_operation import check_cuda_memory
from collections import defaultdict
from src.utils.train_log import AverageMeter, Summary, ProgressMeter

if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device

def get_model_module(model, args:InputParam):
    return model.module if getattr(args, "world_size", 1) > 1 else model


def get_charge_loss(model, sample, criterion, args:InputParam, real_lr=None, start_lr=None):
    if not getattr(args.optimizer_param, "train_charge", False):
        return None, None
    module = get_model_module(model, args)
    charge_predict = getattr(module, "charge_predict", None)
    if charge_predict is None or "charge" not in sample:
        return None, None
    charge_label = sample["charge"].reshape(-1, 1).to(dtype=charge_predict.dtype, device=charge_predict.device)
    atom_num = sample["num_atom"].reshape(-1, 1).to(dtype=charge_predict.dtype, device=charge_predict.device)
    loss_charge = criterion(charge_predict / atom_num, charge_label / atom_num)
    if real_lr is not None and start_lr is not None and hasattr(args.optimizer_param, "start_pre_fac_charge"):
        pref_charge = args.optimizer_param.end_pre_fac_charge + (
            args.optimizer_param.start_pre_fac_charge - args.optimizer_param.end_pre_fac_charge
        ) * real_lr / start_lr
    else:
        pref_charge = getattr(args.optimizer_param, "pre_fac_charge", 0.1)
    return loss_charge, pref_charge


def print_l1_l2(model):
    params = model.parameters()
    dtype = next(params).dtype
    device = next(params).device
    L1 = torch.tensor(0.0, device=device, dtype=dtype).detach().requires_grad_(False)
    L2 = torch.tensor(0.0, device=device, dtype=dtype).detach().requires_grad_(False)
    nums_param = 0
    for p in params:
        L1 += torch.sum(torch.abs(p))
        L2 += torch.sum(p**2)
        nums_param += p.nelement()
    L1 = L1 / nums_param
    L2 = L2 / nums_param
    return L1, L2

def train(train_loader, model, criterion, optimizer, scheduler, epoch, start_lr, device, args:InputParam):
    def scale_learning_rate(ngpus, nbatch, avg_atom_nums, scaling_method):
        if scaling_method == 'linear_gpu':
            return ngpus # 只乘以卡数
        elif scaling_method == 'sqrt_batch':
            return nbatch** 0.5
        elif scaling_method == 'sqrt_gpu':
            return ngpus** 0.5
        elif scaling_method == 'sqrt':
            return (nbatch * ngpus) ** 0.5
        elif scaling_method == 'sqrt_batch_gpu_atom':
            return (nbatch * ngpus * avg_atom_nums) ** 0.5
        return (avg_atom_nums) ** 0.5

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
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rate, losses, loss_L1, loss_L2, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Charge, loss_Virial, loss_Virial_per_atom],
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
        nr_batch_sample = sample["num_atom"].shape[0]
        # 如果采用预热，则前n个epoch 学习率线性增加
        if args.optimizer_param.warmup is not None and epoch <= args.optimizer_param.warmup: # epoch 从1计数
            start_lr = args.optimizer_param.stop_lr * scale_learning_rate(args.world_size, batch_size, avg_atom_number, args.optimizer_param.scaling_method)
            end_lr = args.optimizer_param.learning_rate * scale_learning_rate(args.world_size, batch_size, avg_atom_number, args.optimizer_param.scaling_method)
            real_lr = warmup_lr(iter=i,
                                iternum=len(train_loader),
                                cur_epoch=epoch,
                                warm_epochs=args.optimizer_param.warmup,
                                start_lr=start_lr,
                                end_lr=end_lr
                                )
            is_warmlr = True
        else:
            is_warmlr = False
            if scheduler is None: # 不启用周期性重启
                global_step = (epoch - 1) * len(train_loader) + i * nr_batch_sample
                real_lr = adjust_lr(
                    global_step, start_lr,
                    args.optimizer_param.stop_step, args.optimizer_param.decay_step, args.optimizer_param.stop_lr
                )
                real_lr = real_lr * scale_learning_rate(args.world_size, batch_size, avg_atom_number, args.optimizer_param.scaling_method) # rl * (sqrt(batch*gpu)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = real_lr
            else: # 周期性重启
                real_lr = optimizer.param_groups[0]["lr"] * scale_learning_rate(args.world_size, batch_size, avg_atom_number, args.optimizer_param.scaling_method)

        learning_rate.update(real_lr)
        # check_cuda_memory(epoch, -1, f"before forword atomnums {Force_label.shape[0]}", False, args.rank)
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
            NN_radial, NL_radial, Ri_radial,
            NN_angular, NL_angular, Ri_angular,
            sample["num_atom"], sample["atom_type_map"], None, None,
            charge_label=sample.get("charge"),
            position=sample.get("position"),
            box_original=sample.get("box_original"),
            volume=sample.get("volume")
        )
        # check_cuda_memory(epoch, -1, "end forword", False, args.rank)
        optimizer.zero_grad()
        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Etot_per_atom_val = criterion(Etot_predict / sample["num_atom"], Etot_label / sample["num_atom"])
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_Charge_val, pref_charge = get_charge_loss(model, sample, criterion, args, real_lr, start_lr)
        loss_val = torch.zeros_like(loss_F_val)

        if args.optimizer_param.train_egroup:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
        if args.optimizer_param.train_virial:
            data_mask = Virial_label[:, 0] > -1e6
            _Virial_label = Virial_label[:, [0, 1, 2, 4, 5, 8]][data_mask]
            if data_mask.any().item():
                loss_Virial_val = criterion(Virial_predict[data_mask][:, [0, 1, 2, 4, 5, 8]], _Virial_label)
                loss_Virial_per_atom_val = criterion(
                    Virial_predict[data_mask][:, [0, 1, 2, 4, 5, 8]] / sample["num_atom"][data_mask],
                    _Virial_label / sample["num_atom"][data_mask]
                )
                loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
                loss_val += args.optimizer_param.pre_fac_virial * loss_Virial_val

        w_f, w_e, w_v, w_eg, w_ei = 0, 0, 0, 0, 0
        if args.optimizer_param.train_force:
            w_f = 1.0
            loss_val += loss_F_val
        if args.optimizer_param.train_energy:
            w_e = 1.0
            loss_val += loss_Etot_val
        if args.optimizer_param.train_virial and data_mask.any().item():
            w_v = 1.0
            loss_val += loss_Virial_val
        if args.optimizer_param.train_egroup:
            w_eg = 1.0
            loss_val += loss_Egroup_val

        if args.optimizer_param.train_egroup and args.optimizer_param.train_virial:
            loss, _, _ = calc_loss(
                args, 0.001, real_lr, 1, w_f, loss_F_val, w_e, loss_Etot_val, w_v, loss_Virial_val, w_eg, loss_Egroup_val, w_ei, loss_Ei_val, avg_atom_number
            )
        elif args.optimizer_param.train_egroup and not args.optimizer_param.train_virial:
            loss, _, _ = calc_loss(
                args, 0.001, real_lr, 2, w_f, loss_F_val, w_e, loss_Etot_val, w_eg, loss_Egroup_val, w_ei, loss_Ei_val, avg_atom_number
            )
        elif not args.optimizer_param.train_egroup and args.optimizer_param.train_virial and data_mask.any().item():
            loss, _, _ = calc_loss(
                args, 0.001, real_lr, 3, w_f, loss_F_val, w_e, loss_Etot_val, w_v, loss_Virial_val, w_ei, loss_Ei_val, avg_atom_number
            )
        else:
            loss, _, _ = calc_loss(
                args, 0.001, real_lr, 4, w_f, loss_F_val, w_e, loss_Etot_val, w_ei, loss_Ei_val, avg_atom_number
            )
        if loss_Charge_val is not None:
            loss = loss + pref_charge * loss_Charge_val
        # check_cuda_memory(epoch, -1, "before backward", False, args.rank)
        loss.backward()
        torch.cuda.empty_cache() # 释放pytoch 缓存管理器持有的缓冲块，因为它对cuda算子不可见，导致算子内存不够用，这部分缓冲块 64batch下约10个G
        # check_cuda_memory(epoch, -1, "end backward", False, args.rank)

        if args.optimizer_param.norm_type is not None:
            nn.utils.clip_grad_norm_(model.parameters(), args.optimizer_param.max_norm, args.optimizer_param.norm_type)
        elif args.optimizer_param.clip_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), args.optimizer_param.clip_value)
        optimizer.step()

        if scheduler is not None and is_warmlr is False:
            scheduler.step()

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
        if loss_Charge_val is not None:
            loss_Charge.update(loss_Charge_val.item(), batch_size)
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
        progress.display_summary(["Training Set:"])

    return (
        losses.avg,
        loss_Etot.root,
        loss_Etot_per_atom.root,
        loss_Force.root,
        loss_Ei.root,
        loss_Egroup.root,
        loss_Virial.root,
        loss_Virial_per_atom.root,
        real_lr,
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
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, loss_L1, loss_L2, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
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
                            sample["num_atom"], sample["atom_type_map"], None, None]
        Virial_label = sample["virial"]
        Etot_label   = sample["energy"]
        Ei_label     = sample["ei"]
        Egroup_label = None
        Force_label  = sample["force"]
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
        if args.optimizer_param.train_egroup is True:
            loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

        loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                    args.optimizer_param.pre_fac_etot * loss_Etot_val

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
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root, loss_L1.root, loss_L2.root

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
            #     Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
            #         dR_neigh_list, ImageDR, dR_neigh_type_list, \
            #             dR_neigh_list_angular, ImageDR_angular, dR_neigh_type_list_angular, \
            #             atom_type_map[0], atom_type[0], 0, Egroup_weight, Divider)

                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
            Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    NN_radial, NL_radial, Ri_radial,
                        NN_angular, NL_angular, Ri_angular,
                            sample["num_atom"], sample["atom_type_map"], None, None,
                            charge_label=sample.get("charge"),
                            position=sample.get("position"),
                            box_original=sample.get("box_original"),
                            volume=sample.get("volume"))

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = criterion(Etot_predict/sample["num_atom"], Etot_label/sample["num_atom"])
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            loss_Charge_val, pref_charge = get_charge_loss(model, sample, criterion, args)
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                    args.optimizer_param.pre_fac_etot * loss_Etot_val

            if loss_Charge_val is not None:
                loss_val += pref_charge * loss_Charge_val

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
                loss_Charge.update(loss_Charge_val.item(), batch_size)
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
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT, device=device, world_size=args.world_size)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Charge, loss_Virial, loss_Virial_per_atom],
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

    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root

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
        Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                NN_radial, NL_radial, Ri_radial,
                    NN_angular, NL_angular, Ri_angular,
                        sample["num_atom"], sample["atom_type_map"], None, None,
                        charge_label=sample.get("charge"),
                        position=sample.get("position"),
                        box_original=sample.get("box_original"),
                        volume=sample.get("volume"))

        # mse
        loss_F_val = criterion(Force_predict, Force_label)
        loss_Etot_val = criterion(Etot_predict, Etot_label)
        loss_Etot_per_atom_val = criterion(Etot_predict/sample["num_atom"], Etot_label/sample["num_atom"])
        loss_Ei_val = criterion(Ei_predict, Ei_label)
        loss_Charge_val, pref_charge = get_charge_loss(model, sample, criterion, args)
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

        res_list = [i, float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]
        if loss_Charge_val is not None:
            res_list.append(float(loss_Charge_val ** 0.5))
        #float(Etot_predict), float(Ei_label.abs().mean()), float(Ei_predict.abs().mean()), float(Force_label.abs().mean()), float(Force_predict.abs().mean()),\
        if args.optimizer_param.train_egroup:
            res_list.append(float(loss_Egroup_val))
        if args.optimizer_param.train_virial:
            res_list.append(float(loss_Virial_val))
            res_list.append(float(loss_Virial_per_atom_val))

        force_label_list.append(Force_label.flatten().cpu().numpy())
        force_predict_list.append(Force_predict.flatten().detach().cpu().numpy())
        ei_label_list.append(Ei_label.flatten().cpu().numpy())
        ei_predict_list.append(Ei_predict.flatten().detach().cpu().numpy())
        etot_label_list.append(float(Etot_label))
        etot_predict_list.append(float(Etot_predict))
        res_pd.loc[res_pd.shape[0]] = res_list
        virial_label_list.append(Virial_label[:,virial_index].flatten().detach().cpu().numpy())
        virial_predict_list.append(Virial_predict[:,virial_index].flatten().detach().cpu().numpy())
    return res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list, virial_label_list, virial_predict_list


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
