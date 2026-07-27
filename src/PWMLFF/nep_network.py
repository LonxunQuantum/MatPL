import os
import sys
import pathlib
import random
import torch
import time
import torch.nn as nn
import torch.distributed as dist
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from src.feature.nep_find_neigh.findneigh import FindNeigh
import numpy as np
import pandas as pd
from src.model.nep_net import NEP
from src.pre_data.nep_data_loader import calculate_neighbor_num_max_min, calculate_neighbor_scaler, UniDataset, variable_length_collate_fn, variable_length_collate_fn_nolimit, calculate_batch, type_map, NepTestData
from src.PWMLFF.nep_mods.nep_trainer import train_KF, train, valid, save_checkpoint, predict
from src.user.input_param import InputParam
from src.utils.file_operation import write_arrays_to_file, write_force_ei
from src.utils.nep_to_gpumd import extract_model
from src.aux.inference_plot import inference_plot
import concurrent.futures
import multiprocessing
from src.utils.debug_operation import check_cuda_memory, check_cpu_memory
from src.utils.learning_rate import is_epoch_before_restart
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer

# 动态添加路径
codepath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(codepath)
sys.path.append(codepath + '/../model')
sys.path.append(codepath + '/..')
sys.path.append(codepath + '/../aux')
sys.path.append(codepath + '/../..')

def _get_image_total_charge(image, default_total_charge=0.0):
    charge = getattr(image, "charge", None)
    fragment = getattr(image, "fragment", None)
    if charge is not None:
        charge_array = np.asarray(charge, dtype=float)
        if fragment is not None and charge_array.size == getattr(image, "atom_nums", charge_array.size):
            fragment_array = np.asarray(fragment).reshape(-1)
            charge_array = charge_array.reshape(-1)
            if fragment_array.size == charge_array.size:
                valid_fragment = fragment_array >= 0
                if valid_fragment.any():
                    fragment_charge = charge_array[valid_fragment]
                    fragment_id = fragment_array[valid_fragment]
                    if np.isfinite(fragment_charge).all():
                        total_charge = 0.0
                        for frag in np.unique(fragment_id):
                            total_charge += fragment_charge[fragment_id == frag][0]
                        return float(total_charge)
                    total_charge = np.asarray(getattr(image, "total_charge", default_total_charge), dtype=float).reshape(-1)
                    if total_charge.size and np.isfinite(total_charge[0]):
                        return float(total_charge[0])
                    return float(default_total_charge)
        if charge_array.size == 1 and np.isfinite(charge_array.reshape(-1)[0]):
            return float(charge_array.reshape(-1)[0])
    total_charge = getattr(image, "total_charge", default_total_charge)
    total_charge = np.asarray(total_charge, dtype=float).reshape(-1)
    if total_charge.size and np.isfinite(total_charge[0]):
        return float(total_charge[0])
    return float(default_total_charge)


def _get_fragment_charge_rmse_and_label(image, charge_predict):
    charge_label = getattr(image, "charge", None)
    fragment = getattr(image, "fragment", None)
    atom_nums = getattr(image, "atom_nums", len(charge_predict))
    if charge_label is None or fragment is None:
        return None
    charge_label = np.asarray(charge_label, dtype=float).reshape(-1)
    fragment = np.asarray(fragment).reshape(-1)
    if charge_label.size != atom_nums or fragment.size != atom_nums:
        return None
    valid = (fragment >= 0) & np.isfinite(charge_label)
    if not valid.any():
        return None
    pred_frag_charge = []
    label_frag_charge = []
    for frag in np.unique(fragment[valid]):
        frag_mask = fragment == frag
        valid_frag_mask = frag_mask & valid
        if not valid_frag_mask.any():
            continue
        pred_frag_charge.append(np.sum(charge_predict[frag_mask]))
        label_frag_charge.append(charge_label[valid_frag_mask][0])
    if len(label_frag_charge) == 0:
        return None
    pred_frag_charge = np.asarray(pred_frag_charge)
    label_frag_charge = np.asarray(label_frag_charge)
    return np.sqrt(np.mean((pred_frag_charge - label_frag_charge) ** 2)), charge_label


def _init_nep_txt_calculator(nep_txt_path, device_type="cpu", gpu_id=0, print_info=0):
    if device_type == "cuda":
        torch.cuda.set_device(gpu_id)
        from src.feature.NEP_GPU.build.nep_gpu import NEP as NEP_GPU
        calc_obj = NEP_GPU()
        calc_obj.init_from_file(nep_txt_path, print_info, gpu_id)
    else:
        calc_obj = FindNeigh()
        calc_obj.init_model(nep_txt_path)
    return calc_obj


def _calculate_nep_image_result(idx, image, input_atom_types, calc_obj, kspace_method="ewald"):
    atom_nums = image.atom_nums
    atom_types_struc = image.atom_types_image
    input_atom_types = np.array(input_atom_types)
    atom_types = image.atom_type
    img_max_types = len(input_atom_types)
    try:
        ntypes = len(atom_types)
    except TypeError:
        ntypes = 1

    if hasattr(image, "cartesian") and image.cartesian is False:
        image._set_cartesian()

    if ntypes > img_max_types:
        raise Exception("Error! the atom types in structure file is larger than the max atom types in model!")
    type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)

    inference_result = calc_obj.inference(
        list(type_maps[0]),
        list(np.array(image.lattice).transpose(1, 0).reshape(-1)),
        np.array(image.position).transpose(1, 0).reshape(-1),
        kspace_method,
        _get_image_total_charge(image)
    )
    ei_predict, force_predict, virial_predict = inference_result[:3]
    charge_predict = inference_result[3] if len(inference_result) > 3 else []
    bec_predict = inference_result[4] if len(inference_result) > 4 else []

    ei_predict = np.array(ei_predict).reshape(atom_nums)
    etot_predict = np.sum(ei_predict)
    etot_rmse = np.abs(etot_predict - image.Ep)
    etot_atom_rmse = etot_rmse / atom_nums
    ei_rmse = np.sqrt(np.mean((ei_predict - image.atomic_energy) ** 2))
    force_predict = np.array(force_predict).reshape(3, atom_nums).transpose(1, 0)
    force_rmse = np.sqrt(np.mean((force_predict - image.force) ** 2))
    result = {
        "idx": idx,
        "etot_rmse": etot_rmse,
        "etot_atom_rmse": etot_atom_rmse,
        "ei_rmse": ei_rmse,
        "force_rmse": force_rmse,
        "etot_label": image.Ep,
        "etot_predict": etot_predict,
        "ei_label": image.atomic_energy,
        "ei_predict": ei_predict,
        "force_label": image.force,
        "force_predict": force_predict
    }
    virial_predict = np.array(virial_predict)
    if image.virial is not None:
        virial_label = image.virial.flatten()
        virial_rmse = np.sqrt(np.mean((virial_predict[[0,1,2,4,5,8]] - virial_label[[0,1,2,4,5,8]]) ** 2))
        virial_atom_rmse = virial_rmse / atom_nums
    else:
        virial_rmse = -1e6
        virial_atom_rmse = -1e6
        virial_label = np.ones_like(virial_predict) * (-1e6)
    result["virial_rmse"] = virial_rmse
    result["virial_atom_rmse"] = virial_atom_rmse
    result["virial_label"] = virial_label
    result["virial_predict"] = virial_predict

    charge_predict = np.array(charge_predict)
    if charge_predict.size:
        charge_predict = charge_predict.reshape(atom_nums)
        fragment_charge_result = _get_fragment_charge_rmse_and_label(image, charge_predict)
        if fragment_charge_result is not None:
            charge_rmse, charge_label = fragment_charge_result
        else:
            charge_label = getattr(image, "charge", None)
            if charge_label is None:
                charge_label = _get_image_total_charge(image)
            charge_label = np.asarray(charge_label, dtype=float)
            if charge_label.size == atom_nums:
                charge_label = charge_label.reshape(atom_nums)
                charge_rmse = np.sqrt(np.mean((charge_predict - charge_label) ** 2))
            else:
                charge_label = float(charge_label.reshape(-1)[0]) if charge_label.size else _get_image_total_charge(image)
                charge_rmse = np.abs(np.sum(charge_predict) - charge_label)
        result["charge_rmse"] = charge_rmse
        result["charge_label"] = charge_label
        result["charge_predict"] = charge_predict
    else:
        result["charge_rmse"] = -1e6
        result["charge_label"] = np.array([])
        result["charge_predict"] = np.array([])

    bec_predict = np.array(bec_predict)
    if bec_predict.size:
        bec_predict = bec_predict.reshape(9, atom_nums).transpose(1, 0)
        bec_label = getattr(image, "bec", None)
        if bec_label is not None:
            bec_label = np.asarray(bec_label).reshape(-1, 9)
            bec_rmse = np.sqrt(np.mean((bec_predict - bec_label) ** 2))
        else:
            bec_rmse = -1e6
            bec_label = np.ones_like(bec_predict) * (-1e6)
        result["bec_rmse"] = bec_rmse
        result["bec_label"] = bec_label
        result["bec_predict"] = bec_predict
    else:
        result["bec_rmse"] = -1e6
        result["bec_label"] = np.array([])
        result["bec_predict"] = np.array([])

    return result


def _split_indexed_images(indexed_images, worker_count):
    worker_count = max(1, min(worker_count, len(indexed_images)))
    chunks = [[] for _ in range(worker_count)]
    loads = [0 for _ in range(worker_count)]
    for indexed_image in sorted(indexed_images, key=lambda item: getattr(item[1], "atom_nums", 1), reverse=True):
        worker_id = min(range(worker_count), key=lambda item: loads[item])
        chunks[worker_id].append(indexed_image)
        loads[worker_id] += getattr(indexed_image[1], "atom_nums", 1)
    return [chunk for chunk in chunks if chunk]


def _run_nep_txt_inference_worker(nep_txt_path, indexed_images, input_atom_types, device_type="cpu", gpu_id=0, kspace_method="ewald", print_info=0):
    calc_obj = _init_nep_txt_calculator(nep_txt_path, device_type=device_type, gpu_id=gpu_id, print_info=print_info)
    return [
        _calculate_nep_image_result(idx, image, input_atom_types, calc_obj, kspace_method=kspace_method)
        for idx, image in indexed_images
    ]


class nep_network:
    def __init__(self, nep_param:InputParam):
        self.input_param = nep_param
        torch.set_printoptions(precision = 12)

        if self.input_param.seed is not None:
            random.seed(self.input_param.seed)
            torch.manual_seed(self.input_param.seed)

        self.is_rank_0 = True if self.input_param.rank == 0 else False
        # 初始化 DDP 环境
        if self.input_param.multi_gpus:
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{self.input_param.master_addr}:{self.input_param.master_port}",
                rank=self.input_param.rank,
                world_size=self.input_param.world_size
            )
            torch.cuda.set_device(self.input_param.local_rank)
            self.device = torch.device(f"cuda:{self.input_param.local_rank}")
            print(f'Rank {self.input_param.rank}: LocalRank: {self.input_param.local_rank}, device {self.device} for training, Master IP: {self.input_param.master_addr} Free Port {self.input_param.master_port}')
        else: # single gpu
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

        if self.input_param.precision == "float32":
            self.training_type = torch.float32
        else:
            self.training_type = torch.float64

        self.criterion = nn.MSELoss().to(self.device)

    def load_data(self):
        if self.input_param.inference:# 只在debug ckpt 推理时启用
            test_dataset = UniDataset(self.input_param.file_paths.test_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            cal_energy=False)

            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset,
                num_replicas=1,
                rank=0,
                shuffle=False
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,  # DistributedSampler 控制 shuffle
                sampler=test_sampler,
                collate_fn=variable_length_collate_fn, 
                num_workers=self.input_param.workers,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            energy_shift = test_dataset.get_energy_shift()

            forscaler_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=128,
                shuffle=False,  # DistributedSampler 控制 shuffle
                sampler=test_sampler,
                collate_fn=variable_length_collate_fn_nolimit, 
                num_workers=self.input_param.workers,
                drop_last=False,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            return energy_shift, test_loader, None, forscaler_loader
        else:
            train_dataset = UniDataset(self.input_param.file_paths.train_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            batch_max_types=self.input_param.max_allow_atom_type,
                                            cal_energy=True,
                                            fill_metal_bec=self.input_param.optimizer_param.train_bec)

            valid_dataset = UniDataset(self.input_param.file_paths.valid_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            cal_energy=False,
                                            fill_metal_bec=self.input_param.optimizer_param.train_bec
                                            )
            energy_shift = train_dataset.get_energy_shift()
            # 使用 DistributedSampler
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.input_param.world_size,
                rank=self.input_param.rank,
                shuffle=self.input_param.data_shuffle
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.input_param.optimizer_param.batch_size,
                shuffle=False,  # DistributedSampler 控制 shuffle
                sampler=train_sampler,
                collate_fn=variable_length_collate_fn, 
                num_workers=self.input_param.workers,
                drop_last=True,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            max_batch = calculate_batch(train_dataset.max_atom_nums, 400) # 按照最大默认400个近邻取batchsize
            forscaler_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=max_batch,
                shuffle=False,  # DistributedSampler 控制 shuffle
                sampler=train_sampler,
                collate_fn=variable_length_collate_fn_nolimit, 
                num_workers=self.input_param.workers,
                drop_last=False,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                valid_dataset,
                num_replicas=self.input_param.world_size,
                rank=self.input_param.rank,
                shuffle=False
            )
            val_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.input_param.optimizer_param.batch_size,
                shuffle=False,
                sampler=valid_sampler,
                collate_fn=variable_length_collate_fn,
                num_workers=self.input_param.workers,
                pin_memory=True,
                drop_last=True,
                prefetch_factor=2,
                persistent_workers=True
            )
            return energy_shift, train_loader, val_loader, forscaler_loader
    
    '''
    description:
        if davg, dstd and energy_shift not from load_data, get it from model_load_file no use code
    return {*} 
    author: wuxingxing
    '''
    def load_model_optimizer(self, energy_shift, avg_atom_num=1, iterations=1, q_scaler = None, max_NN_radial = -1, max_NN_angular = -1):
        def _adjust_ckpt_keys(ckpt, new_ckpt):
            keys = list(ckpt['state_dict'].keys())
            new_dict = {}
            
            if 'q_scaler' in keys: # ckpt from single GPU training
                if self.is_rank_0:
                    print("The checkpoint file from single gpu training!")
                for key in keys:
                    if self.input_param.world_size > 1: # current is multi gpus
                        new_dict[f'{module}{key}'] = ckpt['state_dict'][key]
                if self.input_param.world_size == 1: # current is single gpus
                    new_dict = ckpt['state_dict']
                
                new_dict[f'{module}q_scaler'] = torch.tensor(list(ckpt['state_dict']['q_scaler']),  # set q_scaler
                                                    dtype=new_ckpt.state_dict()[f'{module}c_param_2'].dtype, 
                                                    device=new_ckpt.state_dict()[f'{module}c_param_2'].device)
                for key in ["C3B", "C4B", "C5B", "atom_type_device", "max_NN_radial", "max_NN_angular"]:
                    new_dict[f'{module}{key}'] = new_ckpt.state_dict()[f'{module}{key}'] # these parameters are fixed values
                    
            else: # ckpt from multi-train version
                if ("module." in keys[0] and self.input_param.world_size > 1) or ("module." not in keys[0] and self.input_param.world_size == 1): # ckpt from multi-gpu
                    new_dict = ckpt['state_dict']
                else:
                    for key in keys:
                        if "module." in keys[0] and self.input_param.world_size == 1: # ckpt from multi-gpu and current work use single cpu remove the module key
                            new_dict[key.replace("module.", "")] = ckpt['state_dict'][key]
                        else: # ckpt from single train but current is multi training
                            new_dict[f'module.{key}'] = ckpt['state_dict'][key]
            ckpt['state_dict'] = new_dict
            return ckpt

        model = NEP(self.input_param, 
                        energy_shift,
                        q_scaler = q_scaler, 
                        max_NN_radial = max_NN_radial, 
                        max_NN_angular = max_NN_angular,
                        dtype = self.training_type, 
                        device = self.device
                        ).to(self.training_type).to(self.device)
        # 包装模型为 DDP
        if torch.cuda.is_available() and self.input_param.world_size > 1:
            model = nn.parallel.DistributedDataParallel(model, 
                                            device_ids=[self.input_param.local_rank], 
                                            output_device=self.input_param.local_rank,
                                            find_unused_parameters=True)
        checkpoint = None
        model_path = None
        # inference 用于debug，直接走的nepcpu or nepgpu
        if self.input_param.inference:
            model_path = self.input_param.file_paths.model_load_path
        elif self.input_param.recover_train and self.input_param.file_paths.model_load_path and \
           os.path.exists(self.input_param.file_paths.model_load_path):
            model_path = self.input_param.file_paths.model_load_path
        
        else:
            if self.input_param.nep_param.model_wb is None:
                if self.input_param.file_paths.model_load_path and \
                   os.path.exists(self.input_param.file_paths.model_load_path):
                    model_path = self.input_param.file_paths.model_load_path
                else:
                    model_path = self.input_param.file_paths.model_save_path
            else:
                model_path = None

        module = 'module.' if self.input_param.world_size > 1 else ''
        if model_path and os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            checkpoint = _adjust_ckpt_keys(checkpoint, model) # 适配旧版本以及单卡多卡版本
            model.load_state_dict(checkpoint["state_dict"])
            if "epoch" in checkpoint:
                if self.input_param.optimizer_param.reset_epoch and self.input_param.inference is False:
                    if checkpoint["epoch"] != 1:
                        print(f"Rank {self.input_param.rank}: Resetting epoch to 1 from {checkpoint['epoch']}")
                    self.input_param.optimizer_param.start_epoch = 1
                else:
                    self.input_param.optimizer_param.start_epoch = checkpoint["epoch"] + 1
            if self.input_param.world_size > 1:
                print(f"Reload ckpt: Rank {self.input_param.rank}, LocalRank {self.input_param.local_rank}, start_epoch: {self.input_param.optimizer_param.start_epoch}")
                dist.barrier()

        # optimizer, and learning rate scheduler
        scheduler = None
        if self.input_param.optimizer_param.opt_name in ["ADAM", "ADAMW", "SGD"]:
            if self.input_param.optimizer_param.warmup is not None:# 如果采用预热，则前n个epoch 学习率线性增加,一般前5% epochs，从最小增加
                init_lr = self.input_param.optimizer_param.stop_lr 
            else:
                init_lr = self.input_param.optimizer_param.learning_rate

            if self.input_param.optimizer_param.opt_name == "ADAM":
                optimizer = optim.Adam(
                    model.parameters(),
                    lr=init_lr,
                    weight_decay=self.input_param.optimizer_param.lambda_2 or 0
                )
            elif self.input_param.optimizer_param.opt_name == "ADAMW":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=init_lr,
                    weight_decay=self.input_param.optimizer_param.lambda_2 or 0
                )
            elif self.input_param.optimizer_param.opt_name == "SGD":
                optimizer = optim.SGD(
                    model.parameters(),
                    lr=init_lr,
                    momentum=self.input_param.optimizer_param.momentum,
                    weight_decay=self.input_param.optimizer_param.weight_decay
                )
            # 初始化学习率调度器
            if self.input_param.optimizer_param.t_0 and self.input_param.optimizer_param.opt_name not in ["LKF", "GKF"]:
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.input_param.optimizer_param.t_0 * iterations,
                T_mult=self.input_param.optimizer_param.t_mult,
                eta_min=self.input_param.optimizer_param.stop_lr,
                last_epoch=-1
            )
        elif self.input_param.optimizer_param.opt_name == "LKF":
            optimizer = LKFOptimizer(
                model.parameters(),
                self.input_param.optimizer_param.kalman_lambda,
                self.input_param.optimizer_param.kalman_nue,
                self.input_param.optimizer_param.block_size,
                self.input_param.optimizer_param.p0_weight
            )
        elif self.input_param.optimizer_param.opt_name == "GKF":
            optimizer = GKFOptimizer(
                model.parameters(),
                self.input_param.optimizer_param.kalman_lambda,
                self.input_param.optimizer_param.kalman_nue
            )
        else:
            raise Exception("Error: Unsupported optimizer!")

        return model, optimizer, scheduler


    def reset_lr(self, model, iterations, optimizer, scheduler):
        # 初始化优化器
        init_lr = self.input_param.optimizer_param.learning_rate
        if self.input_param.optimizer_param.opt_name == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=init_lr,
                weight_decay=self.input_param.optimizer_param.lambda_2 or 0
            )
        elif self.input_param.optimizer_param.opt_name == "ADAMW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=init_lr,
                weight_decay=self.input_param.optimizer_param.lambda_2 or 0
            )
        elif self.input_param.optimizer_param.opt_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(),
                lr=init_lr,
                momentum=self.input_param.optimizer_param.momentum,
                weight_decay=self.input_param.optimizer_param.weight_decay
            )
        else:
            raise Exception("Error: Unsupported optimizer!")
        # 初始化学习率调度器
        scheduler = None
        if self.input_param.optimizer_param.t_0 and self.input_param.optimizer_param.opt_name not in ["LKF", "GKF"]:
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.input_param.optimizer_param.t_0 * iterations,
                T_mult=self.input_param.optimizer_param.t_mult,
                eta_min=self.input_param.optimizer_param.stop_lr,
                last_epoch=-1
            )
        return optimizer, scheduler

    def train(self):
        energy_shift, train_loader, val_loader, forscaler_loader = self.load_data()
        if len(train_loader) < 1:
            print(f"ERROR! The training set size {len(train_loader)} is too small, please adjust the number of GPU or batch_size: training_set_size >= batch_size * gpu_nums")
        max_NN_radial, min_NN_radial, max_NN_angular, min_NN_angular, q_scaler = None, None, None, None, None
        # max_NN 训练集计算，之后如果存在nep.txt，取最大值作为max_nn,用于模型初始化。初始化后，如果存在ckpt文件（recover）,则更新为ckpt中的值。 
    
        # print(f"======= rank {self.input_param.rank} len forscaler_loader {len(forscaler_loader)} ======")
        local_global_max, local_global_min, local_max_NN_radial, local_min_NN_radial, local_max_NN_angular, local_min_NN_angular = calculate_neighbor_scaler(
                    forscaler_loader,
                    self.input_param.nep_param.n_max[0],      # model.n_max_radial,
                    self.input_param.nep_param.basis_size[0], # model.n_base_radial,
                    self.input_param.nep_param.n_max[1],      # model.n_max_angular,
                    self.input_param.nep_param.basis_size[1], # model.n_base_angular,
                    self.input_param.nep_param.l_max[0],      # model.l_max_3b,
                    self.input_param.nep_param.l_max[1],      # model.l_max_4b,
                    self.input_param.nep_param.l_max[2],      # model.l_max_5b,
                    self.device,
                    num_workers=self.input_param.workers)
        if self.input_param.world_size > 1:
            # 汇总 global_max
            local_global_max_tensor = local_global_max.clone().detach().to(self.device)
            dist.all_reduce(local_global_max_tensor, op=dist.ReduceOp.MAX)
            global_max = local_global_max_tensor
            
            # 汇总 global_min
            local_global_min_tensor = local_global_min.clone().detach().to(self.device)
            dist.all_reduce(local_global_min_tensor, op=dist.ReduceOp.MIN)
            global_min = local_global_min_tensor
            
            # 汇总 max_NN_radial
            max_radial_tensor = torch.tensor([local_max_NN_radial], dtype=torch.int64, device=self.device)
            dist.all_reduce(max_radial_tensor, op=dist.ReduceOp.MAX)
            max_NN_radial = max_radial_tensor.item()
            
            # 汇总 max_NN_angular
            max_angular_tensor = torch.tensor([local_max_NN_angular], dtype=torch.int64, device=self.device)
            dist.all_reduce(max_angular_tensor, op=dist.ReduceOp.MAX)
            max_NN_angular = max_angular_tensor.item()
        else:
            # 单卡情况
            global_max = local_global_max
            global_min = local_global_min
            max_NN_radial = local_max_NN_radial
            max_NN_angular = local_max_NN_angular
            # 计算最终的 q_scaler
        if self.input_param.nep_param.q_scaler is None:
            q_scaler = 1.0 / (global_max - global_min)
            q_scaler = q_scaler.tolist()
        else:
            # 如果提供了预定义的 q_scaler
            q_scaler = self.input_param.nep_param.q_scaler
            if self.input_param.nep_param.max_nn_from_txt:
                max_NN_radial  = max(self.input_param.nep_param.max_NN_radial, max_NN_radial)
                max_NN_angular = max(self.input_param.nep_param.max_NN_angular, max_NN_angular)

        # print(f"INIT: Rank: {self.input_param.rank}, LocalRank: {self.input_param.local_rank},  Max neighbor numbers: radial={max_NN_radial}, angular={max_NN_angular}, scaler[-1]:{q_scaler[-1]} lendata {len(train_loader)}")
        if self.input_param.world_size > 1:
            dist.barrier()

        model, optimizer, scheduler = self.load_model_optimizer(energy_shift, 
                                                                avg_atom_num=1, 
                                                                iterations=len(train_loader), 
                                                                q_scaler = q_scaler, 
                                                                max_NN_radial = max_NN_radial, 
                                                                max_NN_angular = max_NN_angular)

        if self.is_rank_0 and not os.path.exists(self.input_param.file_paths.model_store_dir):
            os.makedirs(self.input_param.file_paths.model_store_dir)
        if self.input_param.world_size > 1:
            dist.barrier()

        train_lists = ["epoch", "loss"]
        valid_lists = ["epoch", "loss"]
        
        if self.input_param.optimizer_param.lambda_1 is not None:
            train_lists.append("Loss_l1")
        if self.input_param.optimizer_param.lambda_2 is not None:
            train_lists.append("Loss_l2")

        if self.input_param.optimizer_param.train_energy:
            train_lists.append("RMSE_Etot(eV/atom)")
            valid_lists.append("RMSE_Etot(eV/atom)")
        if self.input_param.optimizer_param.train_ei:
            train_lists.append("RMSE_Ei")
            valid_lists.append("RMSE_Ei")
        if self.input_param.optimizer_param.train_egroup:
            train_lists.append("RMSE_Egroup")
            valid_lists.append("RMSE_Egroup")
        if self.input_param.optimizer_param.train_force:
            train_lists.append("RMSE_F(eV/Å)")
            valid_lists.append("RMSE_F(eV/Å)")
        if self.input_param.optimizer_param.train_charge:
            train_lists.append("RMSE_charge")
            valid_lists.append("RMSE_charge")
        if self.input_param.optimizer_param.train_bec:
            train_lists.append("RMSE_BEC")
            valid_lists.append("RMSE_BEC")
        if self.input_param.optimizer_param.train_virial:
            train_lists.append("RMSE_virial(eV/atom)")
            valid_lists.append("RMSE_virial(eV/atom)")
        if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
            train_lists.extend(["time(s)"])
        else:
            train_lists.extend(["real_lr", "time(s)"])

        train_print_width = {
            "epoch": 5,
            "loss": 18,
            "RMSE_Etot(eV)": 18,
            "RMSE_Etot(eV/atom)": 21,
            "RMSE_Ei": 18,
            "RMSE_Egroup": 18,
            "RMSE_F(eV/Å)": 21,
            "RMSE_charge": 18,
            "RMSE_BEC": 18,
            "RMSE_virial(eV)": 18,
            "RMSE_virial(eV/atom)": 23,
            "Loss_l1": 18,
            "Loss_l2": 18,
            "real_lr": 18,
            "time(s)": 15,
        }

        train_format = "".join(["%{}s".format(train_print_width[i]) for i in train_lists])
        valid_format = "".join(["%{}s".format(train_print_width[i]) for i in valid_lists])
        train_log = os.path.join(self.input_param.file_paths.model_store_dir, "epoch_train.dat")
        valid_log = os.path.join(self.input_param.file_paths.model_store_dir, "epoch_valid.dat")
        if self.is_rank_0:
            write_mode = "a" if os.path.exists(train_log) else "w"
            with open(train_log, write_mode) as f_train_log:
                if write_mode == "w":
                    f_train_log.write("# %s\n" % (train_format % tuple(train_lists)))
            if val_loader and len(val_loader) > 0:
                with open(valid_log, write_mode) as f_valid_log:
                    if write_mode == "w":
                        f_valid_log.write("# %s\n" % (valid_format % tuple(valid_lists)))

        for epoch in range(self.input_param.optimizer_param.start_epoch, self.input_param.optimizer_param.epochs + 1):
            time_start = time.time()
            if self.input_param.optimizer_param.warmup is not None and self.input_param.optimizer_param.warmup + 1 == epoch: # epoch 从1计数
                optimizer, scheduler = self.reset_lr(model, len(train_loader), optimizer, scheduler)
            # 设置 sampler 的 epoch 以确保 shuffle 一致
            if hasattr(train_loader, 'sampler') and isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, loss_charge, loss_bec, loss_l1, loss_l2 = train_KF(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.input_param
                )
            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, loss_charge, loss_bec, real_lr, loss_l1, loss_l2 = train(
                    train_loader, model, self.criterion, optimizer, scheduler, epoch,
                        self.input_param.optimizer_param.learning_rate, self.device, self.input_param
                )

            time_end = time.time()
            # self.convert_to_gpumd(model)

            # evaluate on validation set
            if val_loader and len(val_loader) > 0:
                vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom, val_loss_charge, val_loss_bec = valid(
                    val_loader, model, self.criterion, self.device, self.input_param
                )

            if self.is_rank_0:
                with open(train_log, "a") as f_train_log:
                    train_log_line = f"{epoch:5d}{loss:20.10e}"
                    if self.input_param.optimizer_param.lambda_1:
                        train_log_line += f"{loss_l1:18.10e}"
                    if self.input_param.optimizer_param.lambda_2:
                        train_log_line += f"{loss_l2:18.10e}"
                    if self.input_param.optimizer_param.train_energy:
                        train_log_line += f"{loss_Etot_per_atom:21.10e}"
                    if self.input_param.optimizer_param.train_ei:
                        train_log_line += f"{loss_Ei:18.10e}"
                    if self.input_param.optimizer_param.train_egroup:
                        train_log_line += f"{loss_egroup:18.10e}"
                    if self.input_param.optimizer_param.train_force:
                        train_log_line += f"{loss_Force:21.10e}"
                    if self.input_param.optimizer_param.train_charge:
                        train_log_line += f"{loss_charge:18.10e}"
                    if self.input_param.optimizer_param.train_bec:
                        train_log_line += f"{loss_bec:18.10e}"
                    if self.input_param.optimizer_param.train_virial:
                        train_log_line += f"{loss_virial_per_atom:23.10e}"
                    if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                        train_log_line += "%15.4f" % (time_end - time_start)
                    else:
                        train_log_line += f"{real_lr:18.10e}{(time_end - time_start):15.4f}"
                    f_train_log.write(f"{train_log_line}\n")

                if val_loader and len(val_loader) > 0:
                    with open(valid_log, "a") as f_valid_log:
                        valid_log_line = f"{epoch:5d}{vld_loss:20.10e}"
                        if self.input_param.optimizer_param.train_energy:
                            valid_log_line += f"{vld_loss_Etot_per_atom:21.10e}"
                        if self.input_param.optimizer_param.train_ei:
                            valid_log_line += f"{vld_loss_Ei:18.10e}"
                        if self.input_param.optimizer_param.train_egroup:
                            valid_log_line += f"{val_loss_egroup:18.10e}"
                        if self.input_param.optimizer_param.train_force:
                            valid_log_line += f"{vld_loss_Force:21.10e}"
                        if self.input_param.optimizer_param.train_charge:
                            valid_log_line += f"{val_loss_charge:18.10e}"
                        if self.input_param.optimizer_param.train_bec:
                            valid_log_line += f"{val_loss_bec:18.10e}"
                        if self.input_param.optimizer_param.train_virial:
                            valid_log_line += f"{val_loss_virial_per_atom:23.10e}"
                        f_valid_log.write(f"{valid_log_line}\n")
            # 保存检查点
            if self.is_rank_0:
                checkpoint_dict = {
                    "json_file": self.input_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict()
                    # "energy_shift": energy_shift,
                    # "max_neighbor": [model.module.max_NN_radial, model.module.max_NN_angular],
                    # "atom_type_order": self.input_param.atom_type
                    # "q_scaler": model.module.get_q_scaler(),
                }
                if self.input_param.optimizer_param.opt_name in ["LKF", "GKF"] and self.input_param.file_paths.save_p_matrix:
                    checkpoint_dict["optimizer"] = optimizer.state_dict()
                save_checkpoint(
                    checkpoint_dict,
                    self.input_param.file_paths.model_name,
                    self.input_param.file_paths.model_store_dir,
                )
                self.convert_to_gpumd()

                if self.input_param.optimizer_param.t_0 is not None and \
                    is_epoch_before_restart(self.input_param.optimizer_param.t_0, self.input_param.optimizer_param.t_mult, epoch):
                    save_checkpoint(checkpoint_dict,
                                    f'epoch_{epoch}_{self.input_param.file_paths.model_name}',
                                    self.input_param.file_paths.model_store_dir,
                                    )
                    self.convert_to_gpumd(prefix=f"epoch_{epoch}_")

        # 清理 DDP 环境
        if self.input_param.world_size > 1:
            dist.destroy_process_group()
            
    '''
    description: 
        delete nep.in file, this file not used
    param {*} self
    param {NEP} model
    param {str} save_dir
    return {*}
    author: wuxingxing
    '''
    def convert_to_gpumd(self, prefix=""):
        ckpt_path = os.path.join(self.input_param.file_paths.model_store_dir, self.input_param.file_paths.model_name)
        # extract parameters
        nep_content, model_atom_type, atom_names = extract_model(ckpt_path)
        first_line = nep_content.splitlines()[0] if nep_content else ""
        nep_file_name = "nep4.txt" if first_line.startswith("nep4") else "nep5.txt"
        save_nep_txt_path = os.path.join(self.input_param.file_paths.model_store_dir, f"{prefix}{nep_file_name}")
        with open(save_nep_txt_path, 'w') as wf:
                wf.writelines(nep_content)

    # mulit cpu, code has error
    def process_image(self, idx, image, calc_obj=None, kspace_method="ewald"):
        global calc
        if calc_obj is None:
            calc_obj = calc
        return _calculate_nep_image_result(idx, image, self.input_param.atom_type, calc_obj, kspace_method=kspace_method)

    def multi_cpus_nep_inference(self, nep_txt_path, kspace_method="ewald"):
        time0 = time.time()
        images = NepTestData(self.input_param).image_list
        indexed_images = list(enumerate(images))
        results = []
        if len(indexed_images) == 0:
            raise Exception("Error! No images found for NEP test inference.")

        if self.device.type == "cuda" and torch.cuda.is_available():
            gpu_count = min(torch.cuda.device_count(), len(indexed_images))
            print("The GPUs: {}".format(gpu_count))
            chunks = _split_indexed_images(indexed_images, gpu_count)
            if gpu_count == 1:
                results = _run_nep_txt_inference_worker(
                    nep_txt_path,
                    chunks[0],
                    self.input_param.atom_type,
                    device_type="cuda",
                    gpu_id=0,
                    kspace_method=kspace_method,
                    print_info=1
                )
            else:
                mp_context = multiprocessing.get_context("spawn")
                with concurrent.futures.ProcessPoolExecutor(max_workers=gpu_count, mp_context=mp_context) as executor:
                    futures = [
                        executor.submit(
                            _run_nep_txt_inference_worker,
                            nep_txt_path,
                            chunk,
                            self.input_param.atom_type,
                            "cuda",
                            gpu_id,
                            kspace_method,
                            1 if gpu_id == 0 else 0
                        )
                        for gpu_id, chunk in enumerate(chunks)
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        results.extend(future.result())
        else:
            cpu_count = multiprocessing.cpu_count()
            print("The CPUs: {}".format(cpu_count))
            global calc
            calc = FindNeigh()
            calc.init_model(nep_txt_path)
            if cpu_count == 1:
                for idx, image in indexed_images:
                    results.append(self.process_image(idx, image, kspace_method=kspace_method))
            else:
                with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                    futures = [
                        executor.submit(self.process_image, idx, image, None, kspace_method)
                        for idx, image in indexed_images
                    ]
                    results = [future.result() for future in concurrent.futures.as_completed(futures)]
        # Collecting results
        etot_rmse, etot_atom_rmse, ei_rmse, force_rmse = [], [], [], []
        etot_label_list, etot_predict_list = [], []
        ei_label_list, ei_predict_list = [], []
        force_label_list, force_predict_list = [], []
        virial_rmse, virial_atom_rmse = [], []
        virial_label_list, virial_predict_list = [], []
        charge_label_list, charge_predict_list, charge_rmse_list = [], [], []
        bec_label_list, bec_predict_list = [], []
        atom_num_list = []
        virial_index = [0, 1, 2, 4, 5, 8]
        results = sorted(results, key=lambda x: x['idx'])
        has_charge = any(np.asarray(result["charge_predict"]).size for result in results)
        has_bec = any(np.asarray(result["bec_predict"]).size for result in results)
        for result in results:
            etot_rmse.append(result["etot_rmse"])
            etot_atom_rmse.append(result["etot_atom_rmse"])
            ei_rmse.append(result["ei_rmse"])
            force_rmse.append(result["force_rmse"])
            etot_label_list.append(result["etot_label"])
            etot_predict_list.append(result["etot_predict"])
            ei_label_list.append(result["ei_label"])
            ei_predict_list.append(result["ei_predict"])
            force_label_list.append(result["force_label"])
            force_predict_list.append(result["force_predict"])
            atom_num_list.append(images[result["idx"]].atom_nums)
            
            if result["virial_atom_rmse"] > -1e6:
                virial_rmse.append(result["virial_rmse"])
                virial_atom_rmse.append(result["virial_atom_rmse"])
            virial_label_list.append(result["virial_label"][virial_index])
            virial_predict_list.append(result["virial_predict"][virial_index])
            if has_charge:
                charge_label_list.append(result["charge_label"])
                charge_predict_list.append(result["charge_predict"])
                if result["charge_rmse"] > -1e6:
                    charge_rmse_list.append(result["charge_rmse"])
            if has_bec:
                bec_label_list.append(result["bec_label"])
                bec_predict_list.append(result["bec_predict"])

        inference_path = self.input_param.file_paths.test_dir
        if os.path.exists(inference_path) is False:
            os.makedirs(inference_path)

        # Saving results
        write_arrays_to_file(os.path.join(inference_path, "image_atom_nums.txt"), atom_num_list)
        write_arrays_to_file(os.path.join(inference_path, "dft_total_energy.txt"), etot_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_total_energy.txt"), etot_predict_list)
        write_arrays_to_file(os.path.join(inference_path, "dft_force.txt"), force_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_force.txt"), force_predict_list)
        write_arrays_to_file(os.path.join(inference_path, "dft_atomic_energy.txt"), ei_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_atomic_energy.txt"), ei_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_virial.txt"), virial_label_list, head_line="#\txx\txy\txz\tyy\tyz\tzz")
        write_arrays_to_file(os.path.join(inference_path, "inference_virial.txt"), virial_predict_list, head_line="#\txx\txy\txz\tyy\tyz\tzz")
        if has_charge:
            write_arrays_to_file(os.path.join(inference_path, "dft_charge.txt"), charge_label_list)
            write_arrays_to_file(os.path.join(inference_path, "inference_charge.txt"), charge_predict_list)
        if has_bec:
            write_arrays_to_file(os.path.join(inference_path, "dft_bec.txt"), bec_label_list, head_line="#\txx\txy\txz\tyx\tyy\tyz\tzx\tzy\tzz")
            write_arrays_to_file(os.path.join(inference_path, "inference_bec.txt"), bec_predict_list, head_line="#\txx\txy\txz\tyx\tyy\tyz\tzx\tzy\tzz")

        rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2, plot_rmse_charge, charge_r2, rmse_bec, bec_r2 = inference_plot(inference_path, return_extra=True)
        rmse_charge = np.mean(charge_rmse_list) if len(charge_rmse_list) else plot_rmse_charge
        inference_cout = ""
        inference_cout += "For {} images: \n".format(len(images))
        inference_cout += "Average RMSE of Etot per atom: {} R2: {}\n".format(rmse_E, e_r2)
        inference_cout += "Average RMSE of Force: {} R2: {}\n".format(rmse_F, f_r2)
        if rmse_charge is not None:
            inference_cout += "Average RMSE of Charge: {} R2: {}\n".format(rmse_charge, charge_r2)
        if rmse_bec is not None:
            inference_cout += "Average RMSE of BEC: {} R2: {}\n".format(rmse_bec, bec_r2)
        inference_cout += "Average RMSE of Virial per atom: {} R2: {}\n".format(rmse_V, v_r2)
        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(os.path.realpath(self.input_param.file_paths.test_dir))
        print(inference_cout)
        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)

        time2 = time.time()
        print("The test work finished, cost time {} s".format(time2 - time0))

    '''
    description: 
    has been replaced by multi_process_nep_inference
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def inference(self):
        # do inference
        self.input_param.world_size
        energy_shift, train_loader, val_loader, forscaler_loader = self.load_data()
        local_global_max, local_global_min, local_max_NN_radial, local_min_NN_radial, local_max_NN_angular, local_min_NN_angular = calculate_neighbor_scaler(
                    forscaler_loader,
                    self.input_param.nep_param.n_max[0],      # model.n_max_radial,
                    self.input_param.nep_param.basis_size[0], # model.n_base_radial,
                    self.input_param.nep_param.n_max[1],      # model.n_max_angular,
                    self.input_param.nep_param.basis_size[1], # model.n_base_angular,
                    self.input_param.nep_param.l_max[0],      # model.l_max_3b,
                    self.input_param.nep_param.l_max[1],      # model.l_max_4b,
                    self.input_param.nep_param.l_max[2],      # model.l_max_5b,
                    self.device,
                    num_workers=self.input_param.workers)

        # model.max_NN_radial  = max(model.max_NN_radial, max_NN_radial) # for single gpu
        # model.max_NN_angular = max(model.max_NN_angular, max_NN_angular)
        q_scaler = 1.0 / (local_global_max - local_global_min)
        model, optimizer,_ = self.load_model_optimizer(energy_shift, 
                                                    avg_atom_num=1, 
                                                    iterations=len(train_loader), 
                                                    q_scaler = q_scaler, 
                                                    max_NN_radial = local_max_NN_radial, 
                                                    max_NN_angular = local_max_NN_angular)


        start = time.time()
        res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list, virial_label_list, virial_predict_list\
        = predict(train_loader, model, self.criterion, self.device, self.input_param)
        end = time.time()
        print("fitting time:", end - start, 's')

        inference_path = self.input_param.file_paths.test_dir
        if os.path.exists(inference_path) is False:
            os.makedirs(inference_path)
        write_arrays_to_file(os.path.join(inference_path, "image_atom_nums.txt"), [int(len(_)/3) for _ in force_predict_list])
        write_arrays_to_file(os.path.join(inference_path, "dft_total_energy.txt"), etot_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_total_energy.txt"), etot_predict_list)
        # for force
        write_arrays_to_file(os.path.join(inference_path, "dft_force.txt"), [_.reshape(-1,3) for _ in force_label_list])
        write_arrays_to_file(os.path.join(inference_path, "inference_force.txt"), [_.reshape(-1,3) for _ in force_predict_list])
        # ei
        write_arrays_to_file(os.path.join(inference_path, "dft_atomic_energy.txt"), ei_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_atomic_energy.txt"), ei_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_virial.txt"), virial_label_list, head_line="#\txx\txy\txz\tyy\tyz\tzz")
        write_arrays_to_file(os.path.join(inference_path, "inference_virial.txt"), virial_predict_list, head_line="#\txx\txy\txz\tyy\tyz\tzz")

        # res_pd.to_csv(os.path.join(inference_path, "inference_loss.csv"))

        rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2 = inference_plot(inference_path)

        inference_cout = ""
        inference_cout += "For {} images: \n".format(res_pd.shape[0])
        inference_cout += "Average RMSE of Etot per atom: {} \n".format(rmse_E)
        inference_cout += "Average RMSE of Force: {} \n".format(rmse_F)
        inference_cout += "Average RMSE of Virial per atom: {} \n".format(rmse_V)
        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(os.path.realpath(self.input_param.file_paths.test_dir))
        print(inference_cout)
        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)
