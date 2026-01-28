import os
import glob
import pandas as pd
import numpy as np
import time
from enum import Enum
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch.autograd import Variable
from src.loss.dploss import dp_loss, adjust_lr
from src.optimizer.KFWrapper import KFOptimizerWrapper
# import horovod.torch as hvd
from torch.profiler import profile, record_function, ProfilerActivity
from src.user.input_param import InputParam
from collections import defaultdict
from utils.debug_operation import check_cuda_memory
from utils.train_log import AverageMeter, Summary, ProgressMeter


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
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rate = AverageMeter("LR", ":.8e", Summary.AVERAGE)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT)
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rate, losses, loss_L1, loss_L2, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Epoch: [{}]".format(epoch),
    )
    
    # switch to train mode
    model.train()

    end = time.time()
    Sij_max = 0.0   # max Rij before davg and dstd cacled
    # check_cuda_memory(epoch, -1, "start train", False)
    for i, sample_batches in enumerate(train_loader):
        # if i < 1360:
        #     continue
        # measure data loading time
        data_time.update(time.time() - end)

        nr_batch_sample = sample_batches["Ei"].shape[0]
        # global_step = (epoch - 1) * len(train_loader) + i * nr_batch_sample
        # real_lr = adjust_lr(global_step, start_lr, \
        #                     args.optimizer_param.stop_step, args.optimizer_param.decay_step, args.optimizer_param.stop_lr) #  stop_step, decay_step

        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = real_lr * (nr_batch_sample**0.5)

        if scheduler is None:
            global_step = (epoch - 1) * len(train_loader) + i * nr_batch_sample
            real_lr = adjust_lr(global_step, start_lr, \
                            args.optimizer_param.stop_step, args.optimizer_param.decay_step, args.optimizer_param.stop_lr) #  stop_step, decay_step
            for param_group in optimizer.param_groups:
                param_group["lr"] = real_lr * (nr_batch_sample**0.5)
        else:
            real_lr = optimizer.param_groups[0]["lr"]
            # adjusted_lr = real_lr * (avg_atom_number ** 0.5)
            # for param_group in optimizer.param_groups:
            #     param_group["lr"] = adjusted_lr
        learning_rate.update(real_lr)

        if args.precision == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()
            Sij_max_cpu = sample_batches["max_ri"].double()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            # Ri_cpu = sample_batches["Ri"].double()
            # Ri_d_cpu = sample_batches["Ri_d"].double()

        elif args.precision == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()
            Sij_max_cpu = sample_batches["max_ri"].float()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            # Ri_cpu = sample_batches["Ri"].float()
            # Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        if max(Sij_max_cpu) > Sij_max:
            Sij_max = max(Sij_max_cpu)

        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int().squeeze(-1)
        atom_type_cpu = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms = natoms_img[0]
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if args.optimizer_param.train_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if args.optimizer_param.train_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = len(batch_indexs)
            if args.profiling:
                print("=" * 60, "Start profiling model inference", "=" * 60)
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    with record_function("model_inference"):
                        Etot_predict, Ei_predict, Force_predict, _ = model(
                            dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, None, None
                        )   # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT

                print(prof.key_averages().table(sort_by="cuda_time_total"))
                print("=" * 60, "Profiling model inference end", "=" * 60)
                prof.export_chrome_trace("model_infer.json")
            else:
                if args.optimizer_param.train_egroup is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, Egroup_weight, Divider)
                else:
                    # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, None, None)
                    
            optimizer.zero_grad()

            loss_F_val = criterion(Force_predict, Force_label)
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natoms/natoms
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            loss_val = torch.zeros_like(loss_F_val)

            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)
            if args.optimizer_param.train_virial is True:
                # loss_Virial_val = criterion(Virial_predict, Virial_label.squeeze(1))  #115.415137283393
                data_mask = Virial_label[:, 9] > 0  # 判断最后一列是否大于 0
                _Virial_label = Virial_label[:, :9][data_mask][:,[0,1,2,4,5,8]]
                if data_mask.any().item():
                    loss_Virial_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]], _Virial_label)
                    loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms
                    loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                    loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
            
            w_f, w_e, w_v, w_eg, w_ei = 0, 0, 0, 0, 0

            if args.optimizer_param.train_force is True:
                w_f = 1.0 
                loss_val += loss_F_val
            
            if args.optimizer_param.train_energy is True:
                w_e = 1.0
                loss_val += loss_Etot_val
            
            if args.optimizer_param.train_virial is True and data_mask.any().item():
                w_v = 1.0 
                loss_val += loss_Virial_val
            
            if args.optimizer_param.train_egroup is True:
                w_eg = 1.0 
                loss_val += loss_Egroup_val

            if args.optimizer_param.train_egroup is True and args.optimizer_param.train_virial is True:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    1,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_v,
                    loss_Virial_val,
                    w_eg,
                    loss_Egroup_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            elif args.optimizer_param.train_egroup is True and args.optimizer_param.train_virial is False:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    2,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_eg,
                    loss_Egroup_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            elif args.optimizer_param.train_egroup is False \
                and args.optimizer_param.train_virial is True and data_mask.any().item():
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    3,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_v,
                    loss_Virial_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            else:
                loss, _, _ = dp_loss(
                    args,
                    0.001,
                    real_lr,
                    4,
                    w_f,
                    loss_F_val,
                    w_e,
                    loss_Etot_val,
                    w_ei,
                    loss_Ei_val,
                    natoms_img[0].item(),
                )
            # import ipdb;ipdb.set_trace()
            loss.backward()
            if args.optimizer_param.norm_type is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.optimizer_param.max_norm, args.optimizer_param.norm_type)
            elif args.optimizer_param.clip_value is not None:
                nn.utils.clip_grad_value_(model.parameters(), args.optimizer_param.clip_value)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()
                
            loss_val = loss
            L1, L2 = print_l1_l2(model)
            if args.optimizer_param.lambda_2 is not None:
                loss_val += L2

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)

            loss_L1.update(L1.item(), batch_size)
            loss_L2.update(L2.item(), batch_size)

            if args.optimizer_param.train_egroup is True:
                loss_Egroup.update(loss_Egroup_val.item(), batch_size)
            loss_Force.update(loss_F_val.item(), batch_size)
            # check_cuda_memory(epoch, i, "after train update", True)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1)

        if args.save_step is not None and i % args.save_step == 0:
            save_step_checkpoint(
                    {
                    "json_file":args.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "davg":model.davg,
                    "dstd":model.dstd,
                    "energy_shift":model.energy_shift,
                    "atom_type_order": np.array(args.atom_type),    #atom type order of davg/dstd/energy_shift
                    "sij_max":Sij_max
                    },
                    os.path.join(args.file_paths.model_store_dir, "saved_models"),
                    epoch,
                    i,
                    args.max_save_num
                )

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
        Sij_max,   
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
    loss_L1 = AverageMeter("Loss_L1", ":.4e", Summary.ROOT)
    loss_L2 = AverageMeter("Loss_L2", ":.4e", Summary.ROOT)    
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)
    
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
    Sij_max = 0.0   # max Rij before davg and dstd cacled
    for i, sample_batches in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # load data to cpu
        if args.precision == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()
            Sij_max_cpu = sample_batches["max_ri"].double()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()

        elif args.precision == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()
            Sij_max_cpu = sample_batches["max_ri"].float()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            if args.optimizer_param.train_virial is True:
                Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()

        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        if max(Sij_max_cpu) > Sij_max:
            Sij_max = max(Sij_max_cpu)

        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int().squeeze(-1)
        atom_type_cpu = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))

        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms = natoms_img[0]
            
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if args.optimizer_param.train_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            if args.optimizer_param.train_virial is True:
                Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

            batch_size = len(batch_indexs)
            if args.optimizer_param.train_egroup is True:
                kalman_inputs = [dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, Egroup_weight, Divider]
            else:
                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                kalman_inputs = [dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, None, None]

            if args.profiling:
                print("=" * 60, "Start profiling KF update energy", "=" * 60)
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    with record_function("kf_update_energy"):
                        Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label)
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                print("=" * 60, "Profiling KF update energy end", "=" * 60)
                prof.export_chrome_trace("kf_update_energy.json")

                print("=" * 60, "Start profiling KF update force", "=" * 60)
                with profile(
                    activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                    record_shapes=True,
                ) as prof:
                    with record_function("kf_update_force"):
                        Etot_predict, Ei_predict, Force_predict = KFOptWrapper.update_force(
                            kalman_inputs, Force_label, 2
                        )
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                print("=" * 60, "Profiling KF update force end", "=" * 60)
                
                prof.export_chrome_trace("kf_update_force.json")
            else:
                if args.optimizer_param.train_virial is True:
                    Virial_predict = KFOptWrapper.update_virial(kalman_inputs, Virial_label, args.optimizer_param.pre_fac_virial)
                    
                if args.optimizer_param.train_energy is True: 
                    Etot_predict = KFOptWrapper.update_energy(kalman_inputs, Etot_label, args.optimizer_param.pre_fac_etot)
                
                if args.optimizer_param.train_ei is True:
                    Ei_predict = KFOptWrapper.update_ei(kalman_inputs, Ei_label, args.optimizer_param.pre_fac_ei)

                if args.optimizer_param.train_egroup is True:
                    Egroup_predict = KFOptWrapper.update_egroup(kalman_inputs, Egroup_label, args.optimizer_param.pre_fac_egroup)

                if args.optimizer_param.train_force is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = KFOptWrapper.update_force(
                        kalman_inputs, Force_label, args.optimizer_param.pre_fac_force)

            loss_F_val = criterion(Force_predict, Force_label)
            L1, L2 = print_l1_l2(model)

            # divide by natoms 
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_Etot_per_atom_val = loss_Etot_val/natoms/natoms
            
            loss_Ei_val = criterion(Ei_predict, Ei_label)   
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                        args.optimizer_param.pre_fac_etot * loss_Etot_val

            if args.optimizer_param.train_virial is True:
                data_mask = Virial_label[:, 9] > 0
                _Virial_label = Virial_label[:, :9][data_mask][:,[0,1,2,4,5,8]]
                if data_mask.any().item():
                    loss_Virial_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]], _Virial_label)
                    loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms
                    loss_Virial.update(loss_Virial_val.item(), _Virial_label.shape[0])
                    loss_Virial_per_atom.update(loss_Virial_per_atom_val.item(), _Virial_label.shape[0])
                    loss_val += args.optimizer_param.pre_fac_virial * loss_Virial_val

            if args.optimizer_param.lambda_2 is not None:
                loss_val += L2
            if args.optimizer_param.lambda_1 is not None:
                loss_val += L1

            # measure accuracy and record loss
            losses.update(loss_val.item(), batch_size)
            loss_L1.update(L1.item(), batch_size)
            loss_L2.update(L2.item(), batch_size)

            loss_Etot.update(loss_Etot_val.item(), batch_size)
            loss_Etot_per_atom.update(loss_Etot_per_atom_val.item(), batch_size)
            loss_Ei.update(loss_Ei_val.item(), batch_size)
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
    return losses.avg, loss_Etot.root, loss_Etot_per_atom.root, loss_Force.root, loss_Ei.root, loss_Egroup.root, loss_Virial.root, loss_Virial_per_atom.root, Sij_max, loss_L1.root, loss_L2.root

def _classify_batchs(atom_type_map, atom_types:int):
    mask = atom_type_map != -1
    all_atom_types = torch.arange(atom_types)
    atom_counts = []
    for idx, row in enumerate(atom_type_map):
        count = torch.bincount(row[mask[idx]], minlength=len(all_atom_types))
        atom_counts.append(count)
    atom_counts_tuples = [tuple(count.tolist()) for count in atom_counts]
    class_dict = defaultdict(list)
    for idx, count_tuple in enumerate(atom_counts_tuples):
        class_dict[count_tuple].append(idx)
    return list(class_dict.values())

def valid(val_loader, model, criterion, device, args:InputParam):
    def run_validate(loader, base_progress=0):
        end = time.time()
        L1, L2 = print_l1_l2(model)
        for i, sample_batches in enumerate(loader):
            i = base_progress + i
            if args.precision == "float64":
                Ei_label_cpu = sample_batches["Ei"].double()
                Etot_label_cpu = sample_batches["Etot"].double()
                Force_label_cpu = sample_batches["Force"][:, :, :].double()

                if args.optimizer_param.train_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].double()
                    Divider_cpu = sample_batches["Divider"].double()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].double()

                ImageDR_cpu = sample_batches["ImageDR"].double()
                # Ri_cpu = sample_batches["Ri"].double()
                # Ri_d_cpu = sample_batches["Ri_d"].double()

            elif args.precision == "float32":
                Ei_label_cpu = sample_batches["Ei"].float()
                Etot_label_cpu = sample_batches["Etot"].float()
                Force_label_cpu = sample_batches["Force"][:, :, :].float()

                if args.optimizer_param.train_egroup is True:
                    Egroup_label_cpu = sample_batches["Egroup"].float()
                    Divider_cpu = sample_batches["Divider"].float()
                    Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

                if args.optimizer_param.train_virial is True:
                    Virial_label_cpu = sample_batches["Virial"].float()

                ImageDR_cpu = sample_batches["ImageDR"].float()
                # Ri_cpu = sample_batches["Ri"].float()
                # Ri_d_cpu = sample_batches["Ri_d"].float()
            else:
                raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
            
            dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
            natoms_img_cpu = sample_batches["ImageAtomNum"].int().squeeze(-1)
            atom_type_cpu = sample_batches["AtomType"].int()
            atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
            # classify batchs according to their atom type and atom nums
            batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))
            for batch_indexs in batch_clusters:
                # transport data to GPU
                natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
                natoms = natoms_img[0]
                
                dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
                # atom list of image
                atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
                atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
                Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
                Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
                Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

                if args.optimizer_param.train_egroup is True:
                    Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                    Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                    Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

                if args.optimizer_param.train_virial is True:
                    Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
                
                ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
                # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
                # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))

                batch_size = len(batch_indexs)
                """
                    Dim of Ri [bs, natoms, ntype*max_neigh_num, 4] 
                """ 
                if args.optimizer_param.train_egroup is True:
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, Egroup_weight, Divider)
                else:
                    # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                    Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                        dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, None, None)
                
                #return 
                loss_F_val = criterion(Force_predict, Force_label)
                loss_Etot_val = criterion(Etot_predict, Etot_label)
                loss_Etot_per_atom_val = loss_Etot_val/natoms/natoms
                loss_Ei_val = criterion(Ei_predict, Ei_label)
                if args.optimizer_param.train_egroup is True:
                    loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

                loss_val = args.optimizer_param.pre_fac_force * loss_F_val + \
                        args.optimizer_param.pre_fac_etot * loss_Etot_val

                if args.optimizer_param.train_virial is True:
                    data_mask = Virial_label[:, 9] > 0  # 判断最后一列是否大于 0
                    _Virial_label = Virial_label[:, :9][data_mask][:,[0,1,2,4,5,8]]
                    if data_mask.any().item():
                        loss_Virial_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]], _Virial_label)
                        loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms
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
                loss_Force.update(loss_F_val.item(), batch_size)
            # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.optimizer_param.print_freq == 0:
            progress.display(i + 1) 
        

    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    losses = AverageMeter("Loss", ":.4e", Summary.AVERAGE)
    loss_Etot = AverageMeter("Etot", ":.4e", Summary.ROOT)
    loss_Etot_per_atom = AverageMeter("Etot_per_atom", ":.4e", Summary.ROOT)
    loss_Force = AverageMeter("Force", ":.4e", Summary.ROOT)
    loss_Ei = AverageMeter("Ei", ":.4e", Summary.ROOT)
    loss_Egroup = AverageMeter("Egroup", ":.4e", Summary.ROOT)
    loss_Virial = AverageMeter("Virial", ":.4e", Summary.ROOT)
    loss_Virial_per_atom = AverageMeter("Virial_per_atom", ":.4e", Summary.ROOT)

    progress = ProgressMeter(
        len(val_loader),
        # + (
        #     args.hvd
        #     and (len(val_loader.sampler) * hvd.size() < len(val_loader.dataset))
        # ),
        [batch_time, losses, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_Egroup, loss_Virial, loss_Virial_per_atom],
        prefix="Test: ",    
    )
    
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
    train_lists.extend(["RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F", "RMSE_Virial", "RMSE_Virial_per_atom"])
    atom_num_list = []
    if args.optimizer_param.train_egroup:
        train_lists.append("RMSE_Egroup")
    res_pd = pd.DataFrame(columns=train_lists)
    virial_label_list = []
    virial_predict_list = []
    force_label_list = []
    force_predict_list = []
    ei_label_list = []
    ei_predict_list = []
    etot_label_list = []
    etot_predict_list = []
    model.eval()
    for i, sample_batches in enumerate(val_loader):
        # measure data loading time
        # load data to cpu
        if args.precision == "float64":
            Ei_label_cpu = sample_batches["Ei"].double()
            Etot_label_cpu = sample_batches["Etot"].double()
            Force_label_cpu = sample_batches["Force"][:, :, :].double()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].double()
                Divider_cpu = sample_batches["Divider"].double()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].double()

            Virial_label_cpu = sample_batches["Virial"].double()

            ImageDR_cpu = sample_batches["ImageDR"].double()
            # Ri_cpu = sample_batches["Ri"].double()
            # Ri_d_cpu = sample_batches["Ri_d"].double()

        elif args.precision == "float32":
            Ei_label_cpu = sample_batches["Ei"].float()
            Etot_label_cpu = sample_batches["Etot"].float()
            Force_label_cpu = sample_batches["Force"][:, :, :].float()

            if args.optimizer_param.train_egroup is True:
                Egroup_label_cpu = sample_batches["Egroup"].float()
                Divider_cpu = sample_batches["Divider"].float()
                Egroup_weight_cpu = sample_batches["Egroup_weight"].float()

            Virial_label_cpu = sample_batches["Virial"].float()

            ImageDR_cpu = sample_batches["ImageDR"].float()
            # Ri_cpu = sample_batches["Ri"].float()
            # Ri_d_cpu = sample_batches["Ri_d"].float()
        else:
            raise Exception("Error! Please specify floating point type: float32 or float64 by the parameter --datatype! ")
        
        dR_neigh_list_cpu = sample_batches["ListNeighbor"].int()
        natoms_img_cpu = sample_batches["ImageAtomNum"].int().squeeze(-1)
        atom_type_cpu = sample_batches["AtomType"].int()
        atom_type_map_cpu = sample_batches["AtomTypeMap"].int()
        # classify batchs according to their atom type and atom nums
        batch_clusters = _classify_batchs(atom_type_map_cpu, len(args.atom_type))
        virial_index = [0, 1, 2, 4, 5, 8]
        for batch_indexs in batch_clusters:
            # transport data to GPU
            natoms_img = Variable(natoms_img_cpu[batch_indexs].int().to(device))
            natoms = natoms_img[0]
            atom_num_list.append(int(natoms))
            dR_neigh_list = Variable(dR_neigh_list_cpu[batch_indexs, :natoms].int().to(device))
            # atom list of image
            atom_type = Variable(atom_type_cpu[batch_indexs].to(device))
            atom_type_map = Variable(atom_type_map_cpu[batch_indexs, :natoms].to(device))
            Ei_label = Variable(Ei_label_cpu[batch_indexs, :natoms].to(device))
            Etot_label = Variable(Etot_label_cpu[batch_indexs].to(device))
            Force_label = Variable(Force_label_cpu[batch_indexs, :natoms].to(device))  # [40,108,3]

            if args.optimizer_param.train_egroup is True:
                Egroup_label = Variable(Egroup_label_cpu[batch_indexs, :natoms].to(device))
                Divider = Variable(Divider_cpu[batch_indexs, :natoms].to(device))
                Egroup_weight = Variable(Egroup_weight_cpu[batch_indexs, :natoms, :natoms].to(device))

            Virial_label = Variable(Virial_label_cpu[batch_indexs].to(device))
            
            ImageDR = Variable(ImageDR_cpu[batch_indexs, :natoms].to(device))
            # Ri = Variable(Ri_cpu[batch_indexs, :natoms].to(device), requires_grad=True)
            # Ri_d = Variable(Ri_d_cpu[batch_indexs, :natoms].to(device))
            # batch_size = len(batch_indexs)
            if args.optimizer_param.train_egroup is True:
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, Egroup_weight, Divider)
            else:
                # atom_type_map: we only need the first element, because it is same for each image of MOVEMENT
                Etot_predict, Ei_predict, Force_predict, Egroup_predict, Virial_predict = model(
                    dR_neigh_list, atom_type_map[0], atom_type[0], ImageDR, 0, None, None 
                )
            # mse
            loss_Etot_val = criterion(Etot_predict, Etot_label)
            loss_F_val = criterion(Force_predict, Force_label)
            loss_Ei_val = criterion(Ei_predict, Ei_label)
            # loss_val = loss_F_val + loss_Etot_val
            if args.optimizer_param.train_egroup is True:
                loss_Egroup_val = criterion(Egroup_predict, Egroup_label)

            data_mask = Virial_label[:, 9] > 0  # 判断最后一列是否大于 0
            _Virial_label = Virial_label[:, :9][data_mask][:,[0,1,2,4,5,8]]
            if data_mask.any().item():
                loss_Virial_val = criterion(Virial_predict[data_mask][:,[0,1,2,4,5,8]], _Virial_label)
                loss_Virial_per_atom_val = loss_Virial_val/natoms/natoms
            # rmse
            Etot_rmse = loss_Etot_val ** 0.5
            etot_atom_rmse = Etot_rmse / natoms
            Ei_rmse = loss_Ei_val ** 0.5
            F_rmse = loss_F_val ** 0.5

            res_list = [i, float(Etot_rmse), float(etot_atom_rmse), float(Ei_rmse), float(F_rmse)]
            if data_mask.any().item():
                res_list.append(float(loss_Virial_val))
                res_list.append(float(loss_Virial_per_atom_val))
            else:
                res_list.append(-1e6)
                res_list.append(-1e6)                
            virial_label_list.append(Virial_label[:,virial_index].squeeze().cpu().numpy())
            virial_predict_list.append(Virial_predict.detach()[:,virial_index].squeeze().cpu().numpy())
            force_label_list.append(Force_label.squeeze().cpu().numpy())
            force_predict_list.append(Force_predict.detach().squeeze().cpu().numpy())
            ei_label_list.append(Ei_label.flatten().cpu().numpy().tolist())
            ei_predict_list.append(Ei_predict.flatten().detach().cpu().numpy().tolist())

            etot_label_list.append(float(Etot_label))
            etot_predict_list.append(float(Etot_predict))
            res_pd.loc[res_pd.shape[0]] = res_list
    
    return atom_num_list, res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list, virial_label_list, virial_predict_list

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

