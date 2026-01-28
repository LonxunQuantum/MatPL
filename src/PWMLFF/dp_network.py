import os,sys
import glob
import pathlib
import random
import numpy as np
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

import torch
    
import time
import torch.nn as nn
# import horovod.torch as hvd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed

"""
    customized modules 
"""
from src.model.dp_dp_typ_emb import TypeDP
# from src.model.dp_dp_typ_emb_Gk5 import TypeDP as Gk5TypeDP # this is Gk5 type embedding of dp
from src.model.dp_dp import DP
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer
import src.pre_data.dp_mlff as dp_mlff
# from src.pre_data.dp_data_loader import MovementDataset
from src.pre_data.dpuni_data_loader import UniDataset, type_map, variable_length_collate_fn, calculate_neighbor_num_max_min

from src.PWMLFF.dp_mods.dp_trainer import train_KF, train, valid, save_checkpoint, predict
from src.PWMLFF.dp_param_extract import load_davg_dstd_from_checkpoint, load_davg_dstd_from_feature_path
from src.user.input_param import InputParam
from utils.learning_rate import is_epoch_before_restart
from utils.file_operation import write_arrays_to_file, copy_movements_to_work_dir, smlink_file
#from data_loader_2type_dp import MovementDataset, get_torch_data
from src.aux.inference_plot import inference_plot

class dp_network:
    def __init__(self, dp_param:InputParam):
        self.dp_params = dp_param
        self.config = self.dp_params.get_dp_net_dict()
        self.davg_dstd_energy_shift = None # davg/dstd/energy_shift from training data
        torch.set_printoptions(precision = 12)
        if self.dp_params.seed is not None:
            random.seed(self.dp_params.seed)
            torch.manual_seed(self.dp_params.seed)

        # if self.dp_params.hvd:
        #     hvd.init()
        #     self.dp_params.gpu = hvd.local_rank()

        if torch.cuda.is_available():
            if self.dp_params.gpu:
                print("Use GPU: {} for training".format(self.dp_params.gpu))
                self.device = torch.device("cuda:{}".format(self.dp_params.gpu))
            else:
                self.device = torch.device("cuda")
        #elif torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.dp_params.precision == "float32":
            self.training_type = torch.float32  # training type is weights type
        else:
            self.training_type = torch.float64

        self.criterion = nn.MSELoss().to(self.device)
    
    # delete in 2025
    def generate_data(self):    
        """
        Generate training data for MLFF model.

        Returns:
            list: list of labels path
        """
        data_file_config = self.dp_params.get_data_file_dict()
        raw_data_path = self.dp_params.file_paths.raw_path
        datasets_path = os.path.join(self.dp_params.file_paths.json_dir, data_file_config["trainSetDir"])
        train_ratio = data_file_config['ratio']
        train_data_path = data_file_config["trainDataPath"]
        valid_data_path = data_file_config["validDataPath"]
        labels_path = dp_mlff.gen_train_data(train_ratio, raw_data_path, datasets_path, 
                               train_data_path, valid_data_path, 
                               self.dp_params.valid_shuffle, self.dp_params.seed, self.dp_params.format)
        return labels_path

    def load_davg_from_ckpt(self):
        if self.dp_params.inference:
            if os.path.exists(self.dp_params.file_paths.model_load_path):
                # load davg, dstd from checkpoint of model
                davg, dstd, atom_map, energy_shift = load_davg_dstd_from_checkpoint(self.dp_params.file_paths.model_load_path)
            elif os.path.exists(self.dp_params.file_paths.model_save_path):
                davg, dstd, atom_map, energy_shift = load_davg_dstd_from_checkpoint(self.dp_params.file_paths.model_save_path)
            else:
                raise Exception("Erorr! Loading model for inference can not find checkpoint: \
                                \nmodel load path: {} \n or model at work path: {}\n"\
                                .format(self.dp_params.file_paths.model_load_path, self.dp_params.file_paths.model_save_path))
            return davg, dstd, atom_map, energy_shift
        else:
            if os.path.exists(self.dp_params.file_paths.model_save_path):
                davg, dstd, atom_map, energy_shift = load_davg_dstd_from_checkpoint(self.dp_params.file_paths.model_save_path)
                return davg, dstd, atom_map, energy_shift
        return None, None, None, None

    def load_data(self, dp_config:dict):
        if self.dp_params.inference:
            test_dataset = UniDataset(dp_config,
                                    self.dp_params.file_paths.test_data_path, 
                                    self.dp_params.file_paths.format,
                                    calculate_maxnn=True)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn= variable_length_collate_fn, 
                num_workers=self.dp_params.workers,   
                drop_last=True,
                pin_memory=True,
            )
            return test_loader, None, test_dataset
        else:
            train_dataset = UniDataset(dp_config,
                                self.dp_params.file_paths.train_data_path, 
                                self.dp_params.file_paths.format,
                                calculate_maxnn=True)

            valid_dataset = UniDataset(dp_config,
                                self.dp_params.file_paths.valid_data_path, 
                                self.dp_params.file_paths.format,
                                calculate_maxnn=True
                                )
            dp_config["maxNeighborNum"] = max(train_dataset.m_neigh, valid_dataset.m_neigh)
            if len(valid_dataset) > 0:
                valid_dataset.m_neigh = dp_config["maxNeighborNum"]
                valid_dataset.config = dp_config
            train_dataset.m_neigh = dp_config["maxNeighborNum"] 
            train_dataset.config = dp_config

            # should add a collate function for padding
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.dp_params.optimizer_param.batch_size,
                shuffle=self.dp_params.data_shuffle,
                collate_fn= variable_length_collate_fn, 
                num_workers=self.dp_params.workers,   
                drop_last=True,
                pin_memory=True,
            )
            
            if self.dp_params.inference:
                val_loader = None
            else:
                val_loader = torch.utils.data.DataLoader(
                    valid_dataset,
                    batch_size=self.dp_params.optimizer_param.batch_size,
                    shuffle=False,
                    collate_fn= variable_length_collate_fn, 
                    num_workers=self.dp_params.workers,
                    pin_memory=True,
                    drop_last=True
                )
            return train_loader, val_loader, train_dataset

    '''
    description:
        if davg, dstd and energy_shift not from load_data, get it from model_load_file
    return {*}
    author: wuxingxing
    '''
    def get_stat(self):
        if self.davg_dstd_energy_shift is None:
            if os.path.exists(self.dp_params.file_paths.model_load_path) is False:
                raise Exception("ERROR! {} is not exist when get energy shift !".format(self.dp_params.file_paths.model_load_path))
            davg_dstd_energy_shift = load_davg_dstd_from_checkpoint(self.dp_params.file_paths.model_load_path)
        else:
            davg_dstd_energy_shift = self.davg_dstd_energy_shift
        return davg_dstd_energy_shift
    
    def load_model_optimizer(self, davg, dstd, energy_shift, iterations=1):
        # create model 
        # when running evaluation, nothing needs to be done with davg.npy
        if self.dp_params.descriptor.type_embedding:
            model = TypeDP(self.config, davg, dstd, energy_shift)
        else:
            model = DP(self.config, davg, dstd, energy_shift)
        model = model.to(self.training_type)

        # optionally resume from a checkpoint
        checkpoint = None
        if self.dp_params.recover_train and \
            (self.dp_params.file_paths.model_load_path is not None and os.path.exists(self.dp_params.file_paths.model_load_path)):
            model_path = self.dp_params.file_paths.model_load_path
        elif self.inference: 
            model_path = self.dp_params.file_paths.model_load_path
        else:
            model_path = self.dp_params.file_paths.model_save_path  #recover from last training for training

        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            if not torch.cuda.is_available():
                checkpoint = torch.load(model_path,map_location=torch.device('cpu') , weights_only=False)
            elif self.dp_params.gpu is None:
                checkpoint = torch.load(model_path, weights_only=False)
            elif torch.cuda.is_available():
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(self.dp_params.gpu)
                checkpoint = torch.load(model_path, map_location=loc, weights_only=False)
            # start afresh
            if self.dp_params.optimizer_param.reset_epoch:
                self.dp_params.optimizer_param.start_epoch = 1
            else:
                self.dp_params.optimizer_param.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["state_dict"])
            
            # scheduler.load_state_dict(checkpoint["scheduler"])
            print("=> loaded checkpoint '{}' (epoch {})"\
                    .format(model_path, checkpoint["epoch"]))
            if "compress" in checkpoint.keys():
                model.set_comp_tab(checkpoint["compress"])
        else:
            print("=> no checkpoint found at '{}'".format(model_path))

        if not torch.cuda.is_available():
            print("using CPU")
            '''
        elif self.dp_params.hvd:
            if torch.cuda.is_available():
                if self.dp_params.gpu is not None:
                    torch.cuda.set_device(self.dp_params.gpu)
                    model.cuda(self.dp_params.gpu)
                    self.dp_params.optimizer_param.batch_size = int(self.dp_params.optimizer_param.batch_size / hvd.size())
            '''
        elif self.dp_params.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.dp_params.gpu)
            model = model.cuda(self.dp_params.gpu)
        else:
            model = model.cuda()
            if model.compress_tab is not None:
                model.compress_tab.to(device=self.device)
        # optimizer, and learning rate scheduler
        if self.dp_params.optimizer_param.opt_name == "LKF":
            optimizer = LKFOptimizer(
                model.parameters(),
                self.dp_params.optimizer_param.kalman_lambda,
                self.dp_params.optimizer_param.kalman_nue,
                self.dp_params.optimizer_param.block_size,
                self.dp_params.optimizer_param.p0_weight
            )
        elif self.dp_params.optimizer_param.opt_name == "GKF":
            optimizer = GKFOptimizer(
                model.parameters(),
                self.dp_params.optimizer_param.kalman_lambda,
                self.dp_params.optimizer_param.kalman_nue
            )
        elif self.dp_params.optimizer_param.opt_name == "ADAM":
            if self.dp_params.optimizer_param.lambda_2 is None:
                optimizer = optim.Adam(model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate)
            else:
                optimizer = optim.Adam(model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate, 
                                        weight_decay=self.dp_params.optimizer_param.lambda_2)

        elif self.dp_params.optimizer_param.opt_name == "ADAMW":
            if self.dp_params.optimizer_param.lambda_2 is None:
                optimizer = optim.AdamW(model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate)
            else:
                optimizer = optim.AdamW(model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate, 
                                        weight_decay=self.dp_params.optimizer_param.lambda_2)

        elif self.dp_params.optimizer_param.opt_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(), 
                self.dp_params.optimizer_param.learning_rate,
                momentum=self.dp_params.optimizer_param.momentum,
                weight_decay=self.dp_params.optimizer_param.weight_decay
            )
        else:
            raise Exception("Error: Unsupported optimizer!")
        
        if self.dp_params.optimizer_param.t_0 is not None and \
            self.dp_params.optimizer_param.opt_name not in  ["LKF", "GKF"]: 
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                    T_0= self.dp_params.optimizer_param.t_0*iterations, 
                                                                    T_mult=self.dp_params.optimizer_param.t_mult,
                                                                    eta_min=self.dp_params.optimizer_param.stop_lr,
                                                                    last_epoch=-1,
                                                                    verbose=self.dp_params.optimizer_param.verbose)

        else:
            scheduler = None

        if checkpoint is not None and "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            load_p = checkpoint["optimizer"]['state'][0]['P']
            optimizer.set_kalman_P(load_p, checkpoint["optimizer"]['state'][0]['kalman_lambda'])
                
        '''
        if self.dp_params.hvd:
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

        return model, optimizer, scheduler

    def load_model_script(self, davg, dstd, energy_shift):
        # create model 
        # when running evaluation, nothing needs to be done with davg.npy
        if self.dp_params.descriptor.type_embedding:
            from src.model.dp_dp_typ_emb_script import TypeDP
            model = TypeDP(self.config, davg, dstd, energy_shift)
        else:
            from src.model.dp_dp_script import DP
            model = DP(self.config, davg, dstd, energy_shift)
        model = model.to(self.training_type)

        # optionally resume from a checkpoint
        checkpoint = None
        if self.dp_params.recover_train:
            if self.inference and os.path.exists(self.dp_params.file_paths.model_load_path): # recover from user input ckpt file for inference work
                model_path = self.dp_params.file_paths.model_load_path
            else: # resume model specified by user
                model_path = self.dp_params.file_paths.model_save_path  #recover from last training for training
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if not torch.cuda.is_available():
                    checkpoint = torch.load(model_path,map_location=torch.device('cpu') , weights_only=False)
                elif self.dp_params.gpu is None:
                    checkpoint = torch.load(model_path, weights_only=False)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.dp_params.gpu)
                    checkpoint = torch.load(model_path, map_location=loc, weights_only=False)
                # start afresh
                if self.dp_params.optimizer_param.reset_epoch:
                    self.dp_params.optimizer_param.start_epoch = 1
                else:
                    self.dp_params.optimizer_param.start_epoch = checkpoint["epoch"] + 1
                model.load_state_dict(checkpoint["state_dict"])
                
                # scheduler.load_state_dict(checkpoint["scheduler"])
                print("=> loaded checkpoint '{}' (epoch {})"\
                      .format(model_path, checkpoint["epoch"]))
                if "compress" in checkpoint.keys():
                    model.set_comp_tab(checkpoint["compress"])
            else:
                print("=> no checkpoint found at '{}'".format(model_path))

        elif self.dp_params.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.dp_params.gpu)
            model = model.cuda(self.dp_params.gpu)
        else:
            model = model.cuda()
            if model.compress_tab is not None:
                model.compress_tab.to(device=self.device)
        return model

    def inference(self):
        # do inference
        dp_config = self.dp_params.get_data_file_dict()
        davg, dstd, atom_map, energy_shift = self.load_davg_from_ckpt()

        test_loader, _, test_dataset = self.load_data(dp_config) #davg, dstd, energy_shift, atom_map
        self.dp_params.max_neigh_num = max(self.dp_params.max_neigh_num, test_dataset.m_neigh)
        self.dp_params.print_input_params(json_file_save_name="std_input.json")

        model, optimizer, _ = self.load_model_optimizer(davg, dstd, energy_shift)
        start = time.time()
        atom_num_list, res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list, virial_label_list, virial_predict_list\
        = predict(test_loader, model, self.criterion, self.device, self.dp_params)
        end = time.time()
        print("fitting time:", end - start, 's')

        # print infos
        inference_path = self.dp_params.file_paths.test_dir
        if os.path.exists(inference_path) is False:
            os.makedirs(inference_path)

        write_arrays_to_file(os.path.join(inference_path, "image_atom_nums.txt"), atom_num_list)
        write_arrays_to_file(os.path.join(inference_path, "dft_total_energy.txt"), etot_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_total_energy.txt"), etot_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_atomic_energy.txt"), ei_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_atomic_energy.txt"), ei_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_force.txt"), force_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_force.txt"), force_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_virial.txt"), virial_label_list, head_line="#\txx\txy\txz\tyy\tyz\tzz")
        write_arrays_to_file(os.path.join(inference_path, "inference_virial.txt"), virial_predict_list, head_line="#\txx\txy\txz\tyy\tyz\tzz")

        # res_pd.to_csv(os.path.join(inference_path, "inference_loss.csv"))
        rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2 = inference_plot(inference_path)

        inference_cout = ""
        inference_cout += "For {} images: \n".format(len(test_loader))
        inference_cout += "Average RMSE of Etot per atom: {} R2: {}\n".format(rmse_E, e_r2)
        inference_cout += "Average RMSE of Force: {} R2: {}\n".format(rmse_F, f_r2)
        inference_cout += "Average RMSE of Virial per atom: {} R2: {}\n".format(rmse_V, v_r2)
        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(os.path.realpath(self.dp_params.file_paths.test_dir))
        print(inference_cout)
        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)

        return 

    def train(self):
        dp_config = self.dp_params.get_data_file_dict()
        davg, dstd, atom_map, energy_shift = self.load_davg_from_ckpt()
        train_loader, val_loader, train_dataset = self.load_data(dp_config) #davg, dstd, energy_shift, atom_map
        
        self.dp_params.max_neigh_num = max(self.dp_params.max_neigh_num, train_dataset.m_neigh)
        self.dp_params.print_input_params(json_file_save_name="std_input.json")

        if davg is None:
            energy_shift = train_dataset.get_energy_shift()
            davg, dstd = train_dataset.get_davg_dstd()
        model, optimizer, scheduler = self.load_model_optimizer(davg, dstd, energy_shift, iterations=len(train_loader))
        if not os.path.exists(self.dp_params.file_paths.model_store_dir):
            os.makedirs(self.dp_params.file_paths.model_store_dir)
        if self.dp_params.model_num == 1:
            smlink_file(self.dp_params.file_paths.model_store_dir, os.path.join(self.dp_params.file_paths.json_dir, os.path.basename(self.dp_params.file_paths.model_store_dir)))
        
        # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
        # Define the lists based on the training type
        train_lists = ["epoch", "loss"]
        valid_lists = ["epoch", "loss"]

        if self.dp_params.optimizer_param.lambda_1 is not None:
            train_lists.append("Loss_l1")
        if self.dp_params.optimizer_param.lambda_2 is not None:
            train_lists.append("Loss_l2")

        if self.dp_params.optimizer_param.train_energy:
            # train_lists.append("RMSE_Etot")
            # valid_lists.append("RMSE_Etot")
            train_lists.append("RMSE_Etot(eV/atom)")
            valid_lists.append("RMSE_Etot(eV/atom)")
        if self.dp_params.optimizer_param.train_ei:
            train_lists.append("RMSE_Ei")
            valid_lists.append("RMSE_Ei")
        if self.dp_params.optimizer_param.train_egroup:
            train_lists.append("RMSE_Egroup")
            valid_lists.append("RMSE_Egroup")
        if self.dp_params.optimizer_param.train_force:
            train_lists.append("RMSE_F(eV/Å)")
            valid_lists.append("RMSE_F(eV/Å)")
        if self.dp_params.optimizer_param.train_virial:
            # train_lists.append("RMSE_virial")
            # valid_lists.append("RMSE_virial")
            train_lists.append("RMSE_virial(eV/atom)")
            valid_lists.append("RMSE_virial(eV/atom)")
        if self.dp_params.optimizer_param.opt_name == "LKF" or self.dp_params.optimizer_param.opt_name == "GKF":
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
        train_log = os.path.join(self.dp_params.file_paths.model_store_dir, "epoch_train.dat")
        valid_log = os.path.join(self.dp_params.file_paths.model_store_dir, "epoch_valid.dat")
        write_mode = "a" if os.path.exists(train_log) else "w"
        if write_mode == "w":
            f_train_log = open(train_log, "w")
            f_train_log.write("# %s\n" % (train_format % tuple(train_lists)))
            if len(val_loader) > 0:
                f_valid_log = open(valid_log, "w")
                f_valid_log.write("# %s\n" % (valid_format % tuple(valid_lists)))

        for epoch in range(self.dp_params.optimizer_param.start_epoch, self.dp_params.optimizer_param.epochs + 1):
            # if self.dp_params.hvd: # this code maybe error, check when add multi GPU training. wu
            #     self.train_sampler.set_epoch(epoch)

            # train for one epoch
            time_start = time.time()
            if self.dp_params.optimizer_param.opt_name == "LKF" or self.dp_params.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, Sij_max, loss_l1, loss_l2 = train_KF(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.dp_params
                )
            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, real_lr, Sij_max, loss_l1, loss_l2 = train(
                    train_loader, model, self.criterion, optimizer, scheduler, epoch, \
                        self.dp_params.optimizer_param.learning_rate, self.device, self.dp_params
                )
            time_end = time.time()

            # evaluate on validation set
            if len(val_loader) > 0:
                vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom = valid(
                    val_loader, model, self.criterion, self.device, self.dp_params
                    )
                f_valid_log = open(valid_log, "a")
                valid_log_line = "%5d%20.10e" % (
                    epoch,
                    vld_loss,
                    )
                if self.dp_params.optimizer_param.train_energy:
                    valid_log_line += "%21.10e" % (vld_loss_Etot_per_atom)
                if self.dp_params.optimizer_param.train_ei:
                    valid_log_line += "%18.10e" % (vld_loss_Ei)
                if self.dp_params.optimizer_param.train_egroup:
                    valid_log_line += "%18.10e" % (val_loss_egroup)
                if self.dp_params.optimizer_param.train_force:
                    valid_log_line += "%21.10e" % (vld_loss_Force)
                if self.dp_params.optimizer_param.train_virial:
                    valid_log_line += "%23.10e" % (val_loss_virial_per_atom)
                f_valid_log.write("%s\n" % (valid_log_line))
                f_valid_log.close()

                
            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
            f_train_log = open(train_log, "a")
            # Write the log line to the file based on the training mode
            train_log_line = "%5d%20.10e" % (
                epoch,
                loss,
            )
            if self.dp_params.optimizer_param.lambda_1 is not None:
                train_log_line += "%18.10e" % (loss_l1)
            if self.dp_params.optimizer_param.lambda_2 is not None:
                train_log_line += "%18.10e" % (loss_l2)
            if self.dp_params.optimizer_param.train_energy:
                train_log_line += "%21.10e" % (loss_Etot_per_atom)
            if self.dp_params.optimizer_param.train_ei:
                train_log_line += "%18.10e" % (loss_Ei)
            if self.dp_params.optimizer_param.train_egroup:
                train_log_line += "%18.10e" % (loss_egroup)
            if self.dp_params.optimizer_param.train_force:
                train_log_line += "%21.10e" % (loss_Force)
            if self.dp_params.optimizer_param.train_virial:
                # train_log_line += "%18.10e" % (loss_virial)
                # valid_log_line += "%18.10e" % (val_loss_virial)
                train_log_line += "%23.10e" % (loss_virial_per_atom)

            if self.dp_params.optimizer_param.opt_name == "LKF" or self.dp_params.optimizer_param.opt_name == "GKF":
                train_log_line += "%15.4f" % (time_end - time_start)
            else:
                train_log_line += "%18.10e%15.4f" % (real_lr, time_end - time_start)

            f_train_log.write("%s\n" % (train_log_line))
            f_train_log.close()
            
            # should include dstd.npy and davg.npy 
            
            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
            if self.dp_params.file_paths.save_p_matrix:
                save_checkpoint(
                    {
                    "json_file":self.dp_params.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "davg":davg,
                    "dstd":dstd,
                    "energy_shift":energy_shift,
                    "atom_type_order": np.array(self.dp_params.atom_type),    #atom type order of davg/dstd/energy_shift
                    "sij_max":Sij_max,
                    "optimizer":optimizer.state_dict()
                    },
                    self.dp_params.file_paths.model_name,
                    self.dp_params.file_paths.model_store_dir,
                )
            else: 
                save_checkpoint(
                    {
                    "json_file":self.dp_params.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "davg":davg,
                    "dstd":dstd,
                    "energy_shift":energy_shift,
                    "atom_type_order": np.array(self.dp_params.atom_type),    #atom type order of davg/dstd/energy_shift
                    "sij_max":Sij_max
                    },
                    self.dp_params.file_paths.model_name,
                    self.dp_params.file_paths.model_store_dir,
                )
                # if use CosineAnnealingWarmRestarts, save the model before restarting learning rate
                if self.dp_params.optimizer_param.t_0 is not None and \
                    is_epoch_before_restart(self.dp_params.optimizer_param.t_0, self.dp_params.optimizer_param.t_mult, epoch):
                    save_path = os.path.join(self.dp_params.file_paths.model_store_dir, "saved_models")
                    if os.path.exists(save_path) is not True:
                        os.makedirs(save_path)
                    save_checkpoint({
                                    "json_file":self.dp_params.to_dict(),
                                    "epoch": epoch,
                                    "state_dict": model.state_dict(),
                                    "davg":davg,
                                    "dstd":dstd,
                                    "energy_shift":energy_shift,
                                    "atom_type_order": np.array(self.dp_params.atom_type),    #atom type order of davg/dstd/energy_shift
                                    "sij_max":Sij_max
                                    },
                                    f'epoch_{epoch}_{self.dp_params.file_paths.model_name}',
                                    save_path,
                                    )


    def load_model_with_ckpt(self, davg, dstd, energy_shift):
        model, optimizer, _ = self.load_model_optimizer(davg, dstd, energy_shift)
        return model

    def evaluate(self,num_thread = 1):
        """
            evaluate a model against AIMD
            put a MOVEMENT in /MD and run MD100 
        """
        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found")
        import md100
        md100.run_md100(imodel = 5, atom_type = self.dp_params.atom_type, num_process = num_thread) 
        
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
        f.write('%d\n' % len(self.dp_params.atom_type)) 
        
        for i in range(len(self.dp_params.atom_type)):
            f.write('%d %d\n' % (self.dp_params.atom_type[i], 2*self.dp_params.atom_type[i]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 

