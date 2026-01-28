import os,sys
import pathlib

codepath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(codepath)

#for model.mlff 
sys.path.append(codepath+'/../model')

#for default_para, data_loader_2type dfeat_sparse dp_mlff
sys.path.append(codepath+'/../pre_data')

#for optimizer
sys.path.append(codepath+'/..')
sys.path.append(codepath+'/../aux')
sys.path.append(codepath+'/../lib')
sys.path.append(codepath+'/../..')

import random
import torch
import time
import torch.nn as nn
# import horovod.torch as hvd
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

from src.feature.nep_find_neigh.findneigh import FindNeigh
import numpy as np
import pandas as pd

from src.model.nep_net import NEP
# from src.model.dp_dp_typ_emb_Gk5 import TypeDP as Gk5TypeDP # this is Gk5 type embedding of dp
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer

from src.pre_data.nep_data_loader import calculate_neighbor_num_max_min, calculate_neighbor_scaler, UniDataset, variable_length_collate_fn, type_map, NepTestData
from src.PWMLFF.nep_mods.nep_trainer import train_KF, train, valid, save_checkpoint, predict
from src.user.input_param import InputParam
from utils.file_operation import write_arrays_to_file
from utils.nep_to_gpumd import extract_model
from utils.learning_rate import is_epoch_before_restart
from src.aux.inference_plot import inference_plot
import concurrent.futures
import multiprocessing
from utils.debug_operation import check_cuda_memory, check_cpu_memory
class nep_network:
    def __init__(self, nep_param:InputParam):
        self.input_param = nep_param
        # self.config = self.nep_params.get_dp_net_dict()
        self.davg_dstd_energy_shift = None # davg/dstd/energy_shift from training data
        torch.set_printoptions(precision = 12)
        if self.input_param.seed is not None:
            random.seed(self.input_param.seed)
            torch.manual_seed(self.input_param.seed)

        # if self.dp_params.hvd:
        #     hvd.init()
        #     self.dp_params.gpu = hvd.local_rank()

        if torch.cuda.is_available():
            if self.input_param.gpu:
                print("Use GPU: {} for training".format(self.input_param.gpu))
                self.device = torch.device("cuda:{}".format(self.input_param.gpu))
            else:
                self.device = torch.device("cuda")
        #elif torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.input_param.precision == "float32":
            self.training_type = torch.float32  # training type is weights type
        else:
            self.training_type = torch.float64

        self.criterion = nn.MSELoss().to(self.device)

    def load_data(self):
        if self.input_param.inference:
            test_dataset = UniDataset(self.input_param.file_paths.test_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            cal_energy=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn= variable_length_collate_fn, 
                num_workers=self.input_param.workers,   
                drop_last=True,
                pin_memory=True,
            )
            energy_shift = test_dataset.get_energy_shift()
            return energy_shift, test_loader, None, test_dataset
        else:
            train_dataset = UniDataset(self.input_param.file_paths.train_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            cal_energy=True)

            valid_dataset = UniDataset(self.input_param.file_paths.valid_data_path, 
                                            self.input_param.file_paths.format, 
                                            self.input_param.atom_type,
                                            cutoff_radial = self.input_param.nep_param.cutoff[0],
                                            cutoff_angular= self.input_param.nep_param.cutoff[1],
                                            cal_energy=False
                                            )

            energy_shift = train_dataset.get_energy_shift()

            # should add a collate function for padding
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.input_param.optimizer_param.batch_size,
                shuffle=self.input_param.data_shuffle,
                collate_fn= variable_length_collate_fn, 
                num_workers=self.input_param.workers,   
                drop_last=True,
                pin_memory=True,
            )
            
            if self.input_param.inference:
                val_loader = None
            else:
                val_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=self.input_param.optimizer_param.batch_size,
                    shuffle=False,
                    collate_fn= variable_length_collate_fn, 
                    num_workers=self.input_param.workers,
                    pin_memory=True,
                    drop_last=True
                )
            return energy_shift, train_loader, val_loader, train_dataset
    
    '''
    description:
        if davg, dstd and energy_shift not from load_data, get it from model_load_file no use code
    return {*} 
    author: wuxingxing
    '''
    def load_model_optimizer(self, energy_shift, avg_atom_num=1, iterations=1):
        # create model 
        # when running evaluation, nothing needs to be done with davg.npy
        # if does not use CosineAnnealingWarmRestarts, avg_atom_num = 1
        model = NEP(self.input_param, energy_shift)
        model = model.to(self.training_type)

        # optionally resume from a checkpoint
        checkpoint = None

        if self.input_param.inference: # recover from user input ckpt file for inference work
            model_path = self.input_param.file_paths.model_load_path
        else: # for training 
            model_path = None 
            if self.input_param.recover_train:
                if self.input_param.file_paths.model_load_path is not None and \
                    os.path.exists(self.input_param.file_paths.model_load_path): # Prioritize loading models from model_load_path
                    model_path = self.input_param.file_paths.model_load_path
                elif os.path.exists(self.input_param.file_paths.model_save_path):   
                    model_path = self.input_param.file_paths.model_save_path  # recover from last training
            
            if model_path is not None and \
                    self.input_param.nep_param.model_wb is not None and len(self.input_param.nep_param.model_wb) > 0:
                raise Exception(f"ERROR! Multiple source model paths detected, please check the {model_path} directory, parameter 'model_load_file or 'nep_txt_file'!")

        if model_path is not None and os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            if not torch.cuda.is_available():
                checkpoint = torch.load(model_path,map_location=torch.device('cpu'), weights_only=False)
            elif self.input_param.gpu is None:
                checkpoint = torch.load(model_path, weights_only=False)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(self.input_param.gpu)
                checkpoint = torch.load(model_path, map_location=loc, weights_only=False)
            # start afresh
            if self.input_param.optimizer_param.reset_epoch:
                if checkpoint["epoch"] != 1:
                    print("The loaded model has been trained for {} epochs. Reset the starting epoch to 1. To disable it, please set 'reset_epoch' in the JSON file to false".format(checkpoint["epoch"]))
                self.input_param.optimizer_param.start_epoch = 1
            else:
                self.input_param.optimizer_param.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            
            if "max_neighbor" in checkpoint.keys():
                model.max_NN_radial, model.max_NN_angular = checkpoint["max_neighbor"]
            # scheduler.load_state_dict(checkpoint["scheduler"])
            print("=> loaded checkpoint '{}' (epoch {})"\
                    .format(model_path, checkpoint["epoch"]))
            if "compress" in checkpoint.keys():
                model.set_comp_tab(checkpoint["compress"])
        else:
            print("=> no checkpoint file found at '{}'".format(model_path))

        if not torch.cuda.is_available():
            print("using CPU")
            '''
        elif self.input_param.hvd:
            if torch.cuda.is_available():
                if self.input_param.gpu is not None:
                    torch.cuda.set_device(self.input_param.gpu)
                    model.cuda(self.input_param.gpu)
                    self.input_param.optimizer_param.batch_size = int(self.input_param.optimizer_param.batch_size / hvd.size())
            '''
        elif self.input_param.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.input_param.gpu)
            model = model.cuda(self.input_param.gpu)
        else:
            model = model.cuda()
            # if model.compress_tab is not None:
            #     model.compress_tab.to(device=self.device)
        # optimizer, and learning rate scheduler
        if self.input_param.optimizer_param.opt_name == "LKF":
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
        elif self.input_param.optimizer_param.opt_name == "ADAM":
            if self.input_param.optimizer_param.lambda_2 is None:
                optimizer = optim.Adam(model.parameters(), 
                                    lr=self.input_param.optimizer_param.learning_rate * avg_atom_num**0.5)
            else:
                optimizer = optim.Adam(model.parameters(), 
                                    lr=self.input_param.optimizer_param.learning_rate * avg_atom_num**0.5, 
                                        weight_decay=self.input_param.optimizer_param.lambda_2)
            
        elif self.input_param.optimizer_param.opt_name == "ADAMW":
            if self.input_param.optimizer_param.lambda_2 is None:
                optimizer = optim.AdamW(model.parameters(), 
                                    lr=self.input_param.optimizer_param.learning_rate * avg_atom_num**0.5)
            else:
                optimizer = optim.AdamW(model.parameters(), 
                                    lr=self.input_param.optimizer_param.learning_rate * avg_atom_num**0.5, 
                                        weight_decay=self.input_param.optimizer_param.lambda_2)
        
        elif self.input_param.optimizer_param.opt_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(), 
                self.input_param.optimizer_param.learning_rate * avg_atom_num**0.5,
                momentum=self.input_param.optimizer_param.momentum,
                weight_decay=self.input_param.optimizer_param.weight_decay
            )
        else:
            raise Exception("Error: Unsupported optimizer!")

        if self.input_param.optimizer_param.t_0 is not None and \
            self.input_param.optimizer_param.opt_name not in  ["LKF", "GKF"]: 
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                    T_0= self.input_param.optimizer_param.t_0*iterations, 
                                                                    T_mult=self.input_param.optimizer_param.t_mult,
                                                                    eta_min=self.input_param.optimizer_param.stop_lr,
                                                                    last_epoch=-1,
                                                                    verbose=self.input_param.optimizer_param.verbose)

        else:
            scheduler = None
        q_scaler = None
        if checkpoint is not None and "q_scaler" in checkpoint.keys(): # from model ckpt file
            q_scaler = checkpoint["q_scaler"]
        elif self.input_param.nep_param.q_scaler is not None:
            q_scaler = self.input_param.nep_param.q_scaler
        if self.input_param.optimizer_param.opt_name in ["LKF", "GKF", "ADAM", "ADAMW", "SGD"]:
            if checkpoint is not None and "optimizer" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint["optimizer"])
            if checkpoint is not None and self.input_param.optimizer_param.opt_name in ["LKF"] and "optimizer" in checkpoint.keys() and 'P' in checkpoint["optimizer"]['state'][0].keys():
                load_p = checkpoint["optimizer"]['state'][0]['P']
                optimizer.set_kalman_P(load_p, checkpoint["optimizer"]['state'][0]['kalman_lambda'])
        
        model.set_nep_param_device(q_scaler)
        '''
        if self.input_param.hvd:
            # after hvd.DistributedOptimizer, the matrix P willed be reset to Identity matrix
            # its because hvd.DistributedOptimizer will initialize a new object of Optimizer Class
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters()
            )
            # Broadcast parameters from rank 0 to all other processes.
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        '''
        # self.device = optimizer._state["P"][0].device
        # set params device
        return model, optimizer, scheduler

    def train(self):
        energy_shift, train_loader, val_loader, train_datset = self.load_data()
        #energy_shift is same as energy_shift of upper; atom_map is the user input order
        model, optimizer, scheduler = self.load_model_optimizer(energy_shift, avg_atom_num=1, iterations=len(train_loader)) #train_datset.avg_image_atom
        

        max_NN_radial, min_NN_radial, max_NN_angular, min_NN_angular = \
                        calculate_neighbor_num_max_min(dataset=train_datset, device = self.device, num_workers=self.input_param.workers)
        
        if self.input_param.nep_param.max_nn_from_txt:
            model.max_NN_radial  = max(self.input_param.nep_param.max_NN_radial, max_NN_radial)
            model.max_NN_angular = max(self.input_param.nep_param.max_NN_angular, max_NN_angular)
        else:
            model.max_NN_radial = max(model.max_NN_radial, max_NN_radial)
            model.max_NN_angular = max(model.max_NN_angular, max_NN_angular)

        if model.q_scaler is None:
            q_scaler = calculate_neighbor_scaler(
                            train_datset,
                            model.max_NN_radial,
                            model.max_NN_angular,
                            model.n_max_radial,
                            model.n_base_radial,
                            model.n_max_angular,
                            model.n_base_angular,
                            model.l_max_3b,
                            model.l_max_4b,
                            model.l_max_5b,
                            self.device,
                            num_workers=self.input_param.workers)

            model.reset_scaler(q_scaler, self.training_type, self.device)

        if not os.path.exists(self.input_param.file_paths.model_store_dir):
            os.makedirs(self.input_param.file_paths.model_store_dir)

        # Define the lists based on the training type
        train_lists = ["epoch", "loss"]
        valid_lists = ["epoch", "loss"]
        
        if self.input_param.optimizer_param.lambda_1 is not None:
            train_lists.append("Loss_l1")
        if self.input_param.optimizer_param.lambda_2 is not None:
            train_lists.append("Loss_l2")

        if self.input_param.optimizer_param.train_energy:
            # train_lists.append("RMSE_Etot")
            # valid_lists.append("RMSE_Etot")
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
        write_mode = "a" if os.path.exists(train_log) else "w"
        if write_mode == "w":
            f_train_log = open(train_log, "w")
            f_train_log.write("# %s\n" % (train_format % tuple(train_lists)))
            if len(val_loader) > 0:
                f_valid_log = open(valid_log, "w")
                f_valid_log.write("# %s\n" % (valid_format % tuple(valid_lists)))

        for epoch in range(self.input_param.optimizer_param.start_epoch, self.input_param.optimizer_param.epochs + 1):
            time_start = time.time()
            if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, loss_l1, loss_l2 = train_KF(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.input_param
                )
            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, real_lr, loss_l1, loss_l2 = train(
                    train_loader, model, self.criterion, optimizer, scheduler, epoch, \
                        self.input_param.optimizer_param.learning_rate, self.device, self.input_param
                )
            time_end = time.time()
            # self.convert_to_gpumd(model)

            # evaluate on validation set
            if len(val_loader) > 0:
                vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom = valid(
                        val_loader, model, self.criterion, self.device, self.input_param
                    )

            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):

            f_train_log = open(train_log, "a")

            # Write the log line to the file based on the training mode
            train_log_line = "%5d%20.10e" % (
                epoch,
                loss,
            )
            if len(val_loader) > 0: #valid log
                f_valid_log = open(valid_log, "a")
                valid_log_line = "%5d%20.10e" % (
                    epoch,
                    vld_loss,
                )
                if self.input_param.optimizer_param.train_energy:
                    # valid_log_line += "%18.10e" % (vld_loss_Etot)
                    valid_log_line += "%21.10e" % (vld_loss_Etot_per_atom)
                if self.input_param.optimizer_param.train_ei:
                    valid_log_line += "%18.10e" % (vld_loss_Ei)
                if self.input_param.optimizer_param.train_egroup:
                    valid_log_line += "%18.10e" % (val_loss_egroup)
                if self.input_param.optimizer_param.train_force:
                    valid_log_line += "%21.10e" % (vld_loss_Force)
                if self.input_param.optimizer_param.train_virial:
                    # valid_log_line += "%18.10e" % (val_loss_virial)
                    valid_log_line += "%23.10e" % (val_loss_virial_per_atom)
                f_valid_log.write("%s\n" % (valid_log_line))
                f_valid_log.close()

            # train log
            if self.input_param.optimizer_param.lambda_1 is not None:
                train_log_line += "%18.10e" % (loss_l1)
            if self.input_param.optimizer_param.lambda_2 is not None:
                train_log_line += "%18.10e" % (loss_l2)
            if self.input_param.optimizer_param.train_energy:
                # train_log_line += "%18.10e" % (loss_Etot)
                train_log_line += "%21.10e" % (loss_Etot_per_atom)
            if self.input_param.optimizer_param.train_ei:
                train_log_line += "%18.10e" % (loss_Ei)
            if self.input_param.optimizer_param.train_egroup:
                train_log_line += "%18.10e" % (loss_egroup)
            if self.input_param.optimizer_param.train_force:
                train_log_line += "%21.10e" % (loss_Force)
            if self.input_param.optimizer_param.train_virial:
                # train_log_line += "%18.10e" % (loss_virial)
                train_log_line += "%23.10e" % (loss_virial_per_atom)
            if self.input_param.optimizer_param.opt_name == "LKF" or self.input_param.optimizer_param.opt_name == "GKF":
                train_log_line += "%15.4f" % (time_end - time_start)
            else:
                train_log_line += "%18.10e%15.4f" % (real_lr , time_end - time_start)

            f_train_log.write("%s\n" % (train_log_line))
            f_train_log.close()
            
            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
            if self.input_param.optimizer_param.opt_name in ["LKF", "GKF"] and \
                self.input_param.file_paths.save_p_matrix:
                save_checkpoint(
                    {
                    "json_file":self.input_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "energy_shift":energy_shift,
                    "max_neighbor": [model.max_NN_radial, model.max_NN_angular],
                    "atom_type_order": self.input_param.atom_type,    #atom type order of davg/dstd/energy_shift, the user input order
                    # "sij_max":Sij_max,
                    "q_scaler": model.get_q_scaler(),
                    "optimizer":optimizer.state_dict()
                    },
                    self.input_param.file_paths.model_name,
                    self.input_param.file_paths.model_store_dir,
                )
            else: # ADAM or SGD optimizer
                save_checkpoint(
                    {
                    "json_file":self.input_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "energy_shift":energy_shift,
                    "max_neighbor": [model.max_NN_radial, model.max_NN_angular],
                    "q_scaler": model.get_q_scaler(),
                    "atom_type_order": self.input_param.atom_type    #atom type order of davg/dstd/energy_shift
                    # "optimizer":optimizer.state_dict()
                    # "sij_max":Sij_max
                    },
                    self.input_param.file_paths.model_name,
                    self.input_param.file_paths.model_store_dir,
                )
                # if use CosineAnnealingWarmRestarts, save the model before restarting learning rate
                if self.input_param.optimizer_param.t_0 is not None and \
                    is_epoch_before_restart(self.input_param.optimizer_param.t_0, self.input_param.optimizer_param.t_mult, epoch):
                    save_path = os.path.join(self.input_param.file_paths.model_store_dir, "saved_models")
                    if os.path.exists(save_path) is not True:
                        os.makedirs(save_path)
                    save_checkpoint({
                                    "json_file":self.input_param.to_dict(),
                                    "epoch": epoch,
                                    "state_dict": model.state_dict(),
                                    "energy_shift":energy_shift,
                                    "max_neighbor": [model.max_NN_radial, model.max_NN_angular],
                                    "q_scaler": model.get_q_scaler(),
                                    "atom_type_order": self.input_param.atom_type    #atom type order of davg/dstd/energy_shift
                                    # "optimizer":optimizer.state_dict()
                                    # "sij_max":Sij_max
                                    },
                                    f'epoch_{epoch}_{self.input_param.file_paths.model_name}',
                                    save_path,
                        )
                    self.convert_to_gpumd_saved_models(f"epoch_{epoch}_", save_path)

            self.convert_to_gpumd()
            
    '''
    description: 
        delete nep.in file, this file not used
    param {*} self
    param {NEP} model
    param {str} save_dir
    return {*}
    author: wuxingxing
    '''
    def convert_to_gpumd(self):
        ckpt_path = os.path.join(self.input_param.file_paths.model_store_dir, self.input_param.file_paths.model_name)
        save_nep_txt_path = os.path.join(self.input_param.file_paths.model_store_dir, self.input_param.file_paths.nep_model_file)            
        # extract parameters
        nep_content, model_atom_type, atom_names = extract_model(ckpt_path)
        with open(save_nep_txt_path, 'w') as wf:
                wf.writelines(nep_content)
        # print("Successfully convert to nep.in and nep.txt file.") 

    def convert_to_gpumd_saved_models(self, prefix="", save_path=None):
        ckpt_path = os.path.join(self.input_param.file_paths.model_store_dir, self.input_param.file_paths.model_name)
        save_nep_txt_path = os.path.join(save_path, f"{prefix}{self.input_param.file_paths.nep_model_file}")            
        # extract parameters
        nep_content, model_atom_type, atom_names = extract_model(ckpt_path)
        with open(save_nep_txt_path, 'w') as wf:
                wf.writelines(nep_content)
        # print("Successfully convert to nep.in and nep.txt file.") 
        
    def evaluate(self,num_thread = 1):
        """
            evaluate a model against AIMD
            put a MOVEMENT in /MD and run MD100 
        """
        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found")
        import md100
        md100.run_md100(imodel = 5, atom_type = self.input_param.atom_type, num_process = num_thread) 
        
    def plot_evaluation(self):
        if not os.path.exists("MOVEMENT"):
            raise Exception("MOVEMENT not found. It should be force field MD result")
        import src.aux.plot_evaluation as plot_evaluation
        plot_evaluation.plot()

    def run_md(self, init_config = "atom.config", md_details = None, num_thread = 1,follow = False):
        import subprocess 

        # remove existing MOVEMENT file for not 
        if follow == False:
            os.system('rm -f MOVEMENT')     
        
        if md_details is None:
            raise Exception("md detail is missing")
        
        md_detail_line = str(md_details)[1:-1]+"\n"
        
        if os.path.exists(init_config) is not True: 
            raise Exception("initial config for MD is not found")
        
        # preparing md.input 
        f = open('md.input', 'w')
        f.write(init_config+"\n")

        f.write(md_detail_line) 
        f.write('F\n')
        f.write("5\n")     # imodel=1,2,3.    {1:linear;  2:VV;   3:NN, 5: dp 
        f.write('1\n')               # interval for MOVEMENT output
        f.write('%d\n' % len(self.input_param.atom_type)) 
        
        for i in range(len(self.input_param.atom_type)):
            f.write('%d %d\n' % (self.input_param.atom_type[i], 2*self.input_param.atom_type[i]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 

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
        # t1 = time.time()
        # results = []
        # for idx, image in enumerate(images):
        #     result = self.process_image(idx, image)
        #     results.append(result)
        t2 = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = [
                executor.submit(self.process_image, idx, image)
                for idx, image in enumerate(images)
            ]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        t3 = time.time()
        # print("{} {}".format(t3-t2, t2-t1))
        time1 = time.time()

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
    replaced by multi_process_nep_inference
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def inference(self):
        # do inference
        energy_shift, train_loader, val_loader, train_datset = self.load_data()
        model, optimizer,_ = self.load_model_optimizer(energy_shift)
        max_NN_radial, min_NN_radial, max_NN_angular, min_NN_angular = \
                        calculate_neighbor_num_max_min(dataset=train_datset, device = self.device)
        
        model.max_NN_radial  = max(model.max_NN_radial, max_NN_radial)
        model.max_NN_angular = max(model.max_NN_angular, max_NN_angular)
    
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
        inference_cout += "Average RMSE of Etot per atom: {} R2: {}\n".format(rmse_E, e_r2)
        inference_cout += "Average RMSE of Force: {} R2: {}\n".format(rmse_F, f_r2)
        inference_cout += "Average RMSE of Virial per atom: {} R2: {}\n".format(rmse_V, v_r2)
        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(os.path.realpath(self.input_param.file_paths.test_dir))
        print(inference_cout)
        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)

