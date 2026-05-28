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
from src.pre_data.nep_data_loader import calculate_neighbor_num_max_min, calculate_neighbor_scaler, UniDataset, variable_length_collate_fn, variable_length_collate_fn_nolimit, calculate_batch, type_map, NepTestData, SameNlocBatchSampler, DistributedSameNlocBatchSampler
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
from src.optimizer.model_ema import ModelEMA

# 动态添加路径
codepath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(codepath)
sys.path.append(codepath + '/../model')
sys.path.append(codepath + '/..')
sys.path.append(codepath + '/../aux')
sys.path.append(codepath + '/../..')

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
                                            cal_energy=True)

            valid_dataset = UniDataset(self.input_param.file_paths.valid_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            cal_energy=False
                                            )

            energy_shift = train_dataset.get_energy_shift()
            # 使用 DistributedSampler
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self.input_param.world_size,
                rank=self.input_param.rank,
                shuffle=self.input_param.data_shuffle
            )
            if self.input_param.same_nloc_sampler:
                train_batch_sampler = DistributedSameNlocBatchSampler(
                    train_dataset,
                    batch_size=self.input_param.optimizer_param.batch_size,
                    num_replicas=self.input_param.world_size,
                    rank=self.input_param.rank,
                    shuffle=self.input_param.data_shuffle,
                    drop_last=True,
                    seed=self.input_param.seed,
                )
                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_sampler=train_batch_sampler,
                    collate_fn=variable_length_collate_fn,
                    num_workers=self.input_param.workers,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True
                )
            else:
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
            if self.input_param.same_nloc_sampler:
                val_batch_sampler = DistributedSameNlocBatchSampler(
                    valid_dataset,
                    batch_size=self.input_param.optimizer_param.batch_size,
                    num_replicas=self.input_param.world_size,
                    rank=self.input_param.rank,
                    shuffle=False,
                    drop_last=True,
                    seed=self.input_param.seed,
                )
                val_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_sampler=val_batch_sampler,
                    collate_fn=variable_length_collate_fn,
                    num_workers=self.input_param.workers,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True
                )
            else:
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
                    weight_decay=self.input_param.optimizer_param.lambda_2 or 0,
                    fused=torch.cuda.is_available(),  # Phase 1.5: fused Adam
                )
            elif self.input_param.optimizer_param.opt_name == "ADAMW":
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=init_lr,
                    weight_decay=self.input_param.optimizer_param.lambda_2 or 0,
                    fused=torch.cuda.is_available(),  # Phase 1.5: fused AdamW
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

        # Phase 2.4: opt-in EMA. Only valid on the Adam/AdamW/SGD path; KF
        # optimizers have their own averaging behavior and are left untouched.
        ema = None
        if (
            self.input_param.optimizer_param.ema_decay is not None
            and self.input_param.optimizer_param.opt_name in ["ADAM", "ADAMW", "SGD"]
        ):
            ema = ModelEMA(model, decay=self.input_param.optimizer_param.ema_decay)
            if checkpoint is not None and "ema_state" in checkpoint:
                ema.load_state_dict(checkpoint["ema_state"])

        return model, optimizer, scheduler, ema


    def reset_lr(self, model, iterations, optimizer, scheduler):
        # 初始化优化器
        init_lr = self.input_param.optimizer_param.learning_rate
        if self.input_param.optimizer_param.opt_name == "ADAM":
            optimizer = optim.Adam(
                model.parameters(),
                lr=init_lr,
                weight_decay=self.input_param.optimizer_param.lambda_2 or 0,
                fused=torch.cuda.is_available(),  # Phase 1.5
            )
        elif self.input_param.optimizer_param.opt_name == "ADAMW":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=init_lr,
                weight_decay=self.input_param.optimizer_param.lambda_2 or 0,
                fused=torch.cuda.is_available(),  # Phase 1.5
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
        # Phase 1.3: enable TF32 for fp32 matmul (A100/H100/4090) + cuDNN autotune
        if self.input_param.precision == "float32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

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

        model, optimizer, scheduler, ema = self.load_model_optimizer(energy_shift,
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
            if hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)
            elif isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, loss_l1, loss_l2 = train_KF(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.input_param
                )
            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, real_lr, loss_l1, loss_l2 = train(
                    train_loader, model, self.criterion, optimizer, scheduler, epoch,
                        self.input_param.optimizer_param.learning_rate, self.device, self.input_param,
                        ema=ema,
                )

            time_end = time.time()
            # self.convert_to_gpumd(model)

            # Phase 2.4: validate / save with EMA weights swapped in (DPA4 convention).
            if ema is not None:
                ema.apply_shadow()
            try:
                # evaluate on validation set
                if val_loader and len(val_loader) > 0:
                    vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom = valid(
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
                            if self.input_param.optimizer_param.train_virial:
                                valid_log_line += f"{val_loss_virial_per_atom:23.10e}"
                            f_valid_log.write(f"{valid_log_line}\n")

                # 保存检查点
                if self.is_rank_0:
                    checkpoint_dict = {
                        "json_file": self.input_param.to_dict(),
                        "epoch": epoch,
                        "state_dict": model.state_dict()
                    }
                    if self.input_param.optimizer_param.opt_name in ["LKF", "GKF"] and self.input_param.file_paths.save_p_matrix:
                        checkpoint_dict["optimizer"] = optimizer.state_dict()
                    if ema is not None:
                        checkpoint_dict["ema_state"] = ema.state_dict()
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
            finally:
                if ema is not None:
                    ema.restore()

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
        save_nep_txt_path = os.path.join(self.input_param.file_paths.model_store_dir, f"{prefix}{self.input_param.file_paths.nep_model_file}")
        # extract parameters
        nep_content, model_atom_type, atom_names = extract_model(ckpt_path)
        with open(save_nep_txt_path, 'w') as wf:
                wf.writelines(nep_content)

    # mulit cpu, code has error
    def process_image(self, idx, image):
        global calc
        atom_nums = image.atom_nums
        atom_types_struc = image.atom_types_image
        input_atom_types = np.array(self.input_param.atom_type)
        atom_types = image.atom_type
        img_max_types = len(self.input_param.atom_type)
        if isinstance(atom_types.tolist(), list):
            ntypes = atom_types.shape[0]
        else:
            ntypes = 1

        if ntypes > img_max_types:
            raise Exception("Error! the atom types in structure file is larger than the max atom types in model!")
        type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)

        ei_predict, force_predict, virial_predict = calc.inference(
            list(type_maps[0]), 
            list(np.array(image.lattice).transpose(1, 0).reshape(-1)), 
            np.array(image.position).transpose(1, 0).reshape(-1)
        )

        ei_predict = np.array(ei_predict).reshape(atom_nums)
        etot_predict = np.sum(ei_predict)
        etot_rmse = np.abs(etot_predict - image.Ep)
        # etot_rmse = np.sqrt(np.mean((etot_predict - image.Ep)**2)) because the images is 1
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

        return result

    def multi_cpus_nep_inference(self, nep_txt_path):
        cpu_count = multiprocessing.cpu_count()
        print("The CPUs: {}".format(cpu_count))
        # cpu_count = 10 if cpu_count > 10 else cpu_count
        time0 = time.time()
        train_lists = ["img_idx", "RMSE_Etot", "RMSE_Etot_per_atom", "RMSE_Ei", "RMSE_F", "RMSE_Virial", "RMSE_Virial_per_atom"]
        images = NepTestData(self.input_param).image_list
        # img_max_types = len(self.input_param.atom_type)
        res_pd = pd.DataFrame(columns=train_lists)
        # Use ProcessPoolExecutor to run the processes in parallel
        global calc
        calc = FindNeigh()
        calc.init_model(nep_txt_path)
        results = []
        if cpu_count == 1:
            for idx, image in enumerate(images):
                result = self.process_image(idx, image)
                results.append(result)
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = [
                    executor.submit(self.process_image, idx, image)
                    for idx, image in enumerate(images)
                ]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
        # Collecting results
        etot_rmse, etot_atom_rmse, ei_rmse, force_rmse = [], [], [], []
        etot_label_list, etot_predict_list = [], []
        ei_label_list, ei_predict_list = [], []
        force_label_list, force_predict_list = [], []
        virial_rmse, virial_atom_rmse = [], []
        virial_label_list, virial_predict_list = [], []
        atom_num_list = []
        virial_index = [0, 1, 2, 4, 5, 8]
        results = sorted(results, key=lambda x: x['idx'])
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
            res_pd.loc[res_pd.shape[0]] = [
                result["idx"], result["etot_rmse"], result["etot_atom_rmse"],
                result["ei_rmse"], result["force_rmse"],
                result["virial_rmse"], result["virial_atom_rmse"]]

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

        # res_pd.to_csv(os.path.join(inference_path, "inference_loss.csv"))
        rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2 = inference_plot(inference_path)
        inference_cout = ""
        inference_cout += "For {} images: \n".format(len(images))
        inference_cout += "Average RMSE of Etot per atom: {} R2: {}\n".format(rmse_E, e_r2)
        inference_cout += "Average RMSE of Force: {} R2: {}\n".format(rmse_F, f_r2)
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
        model, optimizer, _, _ = self.load_model_optimizer(energy_shift,
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


