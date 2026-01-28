import os,sys
import shutil
import subprocess
import pathlib

codepath = str(pathlib.Path(__file__).parent.resolve())

#for model.mlff 
sys.path.append(codepath+'/../model')

#for default_para, data_loader_2type dfeat_sparse
sys.path.append(codepath+'/../pre_data')

#for optimizer
sys.path.append(codepath+'/..')
sys.path.append(codepath+'/../aux')
sys.path.append(codepath+'/../lib')
sys.path.append(codepath+'/../..')

import torch
# import horovod.torch as hvd

import time
import numpy as np
import pandas as pd
import torch.nn as nn
import math
import random
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import time
from src.optimizer.KFWrapper import KFOptimizerWrapper
from sklearn.preprocessing import MinMaxScaler
from joblib import dump, load

import pickle

from src.model.MLFF import MLFFNet
import default_para as pm

# get_torch_data for nn single system training
from src.pre_data.data_loader_2type import get_torch_data
#get_torch_data_hybrid for nn multi hybrid systems training
from src.pre_data.data_loader_2type_nn_hybrid import MovementHybridDataset, get_torch_data_hybrid
from src.pre_data.dfeat_sparse import dfeat_raw 
from src.pre_data.nn_mlff_hybrid import get_cluster_dirs, make_work_dir, mv_featrues, copy_file

from src.PWMLFF.nn_param_extract import load_scaler_from_checkpoint, load_dfeat_input

from utils.file_operation import write_line_to_file, smlink_file
from utils.debug_operation import check_cuda_memory

from src.user.input_param import InputParam
# from optimizer.kalmanfilter import GKalmanFilter, LKalmanFilter, SKalmanFilter
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer

from src.PWMLFF.nn_mods.nn_trainer import train_KF, train, valid, predict

"""
    class that wraps the training model and the data
    network/network parameters 
    training/validation data
    training method 
    Using default_para is purely a legacy problem
"""  
class nn_network:
    def __init__(self, dp_params: InputParam):
        self.workers = 1
        # Global parameters of the class.  
        # Some member data are only "declared" here, and assignment requires further operations 
        # Frequently used control parameter can be reset by class.set_xx() functions
        
        # Frequently-used control parameters that are directly passed in as arguments:        
        # initializing the opt class 
        # self.opts = opt_values()
        self.dp_params = dp_params
        # passing some input arguments to default_para.py
        # at the end of the day, one no longer needs parameters.py
        pm.atomType = self.dp_params.atom_type
        self.atom_type = self.dp_params.atom_type
        pm.maxNeighborNum = self.dp_params.max_neigh_num 
        pm.nodeDim = self.dp_params.model_param.fitting_net.network_size
        pm.atomTypeNum = len(pm.atomType)       #this step is important. Unexpected error
        pm.ntypes = len(pm.atomType)
        #number of layer
        pm.nLayer = len(pm.nodeDim)
        # pm.is_dfeat_sparse is True as default
        pm.is_dfeat_sparse = self.dp_params.is_dfeat_sparse 
        # network for each type of element
        tmp = []
        for layer in range(pm.nLayer):
            tmp_layer = [pm.nodeDim[layer] for atom in pm.atomType]
            tmp.append(tmp_layer.copy()) 
        pm.nNodes = np.array(tmp) 
        # feature set feature type 
        pm.use_Ftype = sorted(self.dp_params.descriptor.feature_type)
        for ftype  in pm.use_Ftype:
            ftype_key = "{}".format(ftype)
            if ftype_key == '1':
                pm.Ftype1_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '2':
                pm.Ftype2_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '3':
                pm.Ftype3_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '4':
                pm.Ftype4_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '5':
                pm.Ftype5_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '6':
                pm.Ftype6_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '7':
                pm.Ftype7_para = self.dp_params.descriptor.feature_dict[ftype_key]
            elif ftype_key == '8':
                pm.Ftype8_para = self.dp_params.descriptor.feature_dict[ftype_key]

        # update nfeat_type 
        pm.nfeat_type = len(pm.use_Ftype)
        # self.feat_mod = feat_modifier() 

        pm.feature_dtype = self.dp_params.precision
        # set training precision. Overarching 
        if (self.dp_params.precision == 'float64'):
            print("Training: set default dtype to double")
            torch.set_default_dtype(torch.float64)
        elif (self.dp_params.precision == 'float32'):
            print("Training: set default dtype to single")
            torch.set_default_dtype(torch.float32)
        else:
            raise RuntimeError("Training: unsupported dtype: %s" %self.dp_params.precision)  

        # if self.dp_params.hvd:
        #     hvd.init()
        #     self.dp_params.gpu = hvd.local_rank()

        # set training device
        if torch.cuda.is_available():
            if self.dp_params.gpu:
                print("Use GPU: {} for training".format(self.dp_params.gpu))
                self.device = torch.device("cuda:{}".format(self.dp_params.gpu))
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print ("device used",self.device)
        # set random seed
        if self.dp_params.seed is not None:
            random.seed(self.dp_params.seed)
            torch.manual_seed(self.dp_params.seed)
            torch.cuda.manual_seed(self.dp_params.seed)
        # set print precision
        torch.set_printoptions(precision = 12)
        self.criterion = nn.MSELoss()
        # anything other than dfeat
        self.train_loader = None
        self.val_loader = None

        #sparse dfeat from fortran 
        self.dfeat = None 
        self.scaler = None

        self.optimizer = None 
        self.model = None
        self.scheduler = None

        # R cut for feature 
        pm.Rc_M = self.dp_params.descriptor.Rmax
        pm.Rc_min = self.dp_params.descriptor.Rmin

        # # load custom feature 
        # if custom_feat_7 is not None:
        #     pm.Ftype7_para = custom_feat_7.copy()
        # if custom_feat_1 is not None: 
        #     pm.Ftype1_para = custom_feat_1.copy()
        # if custom_feat_2 is not None:
        #     pm.Ftype2_para = custom_feat_2.copy()
            
    def generate_data(self, chunk_size=10, shuffle=False, movement_path:list[str]=None, feature_type="train_feature"):
        gen_feature_data = os.path.abspath(os.path.join(self.dp_params.file_paths.nn_work, feature_type))
        # movement_path = self.dp_params.file_paths.train_data_path
        if os.path.exists(gen_feature_data) is True: # work_dir/feature dir
            shutil.rmtree(gen_feature_data)

        # classify all MOVEMENT files based on atom type and atom type nums
        work_dir_list, atom_num = get_cluster_dirs(movement_path)
        # copy the movement files to the working path by cluster result
        sub_work_dir = make_work_dir(gen_feature_data, 
                                    self.dp_params.file_paths.trainSetDir, #'PWdata'
                                    self.dp_params.file_paths.movement_name, #"MOVEMENT"
                                    work_dir_list)
        
        for index, sub_dir in enumerate(sub_work_dir):
            os.chdir(sub_dir)
            # reset pm parameters
            atom_type = work_dir_list[os.path.basename(sub_dir)]['types']
            self._reset_pm_params(sub_dir, atom_type)
            # calculating feature 
            from src.pre_data.mlff import calc_feat
            calc_feat()
            # seperate training set and valid set
            import src.pre_data.seper as seper 
            seper.seperate_data(chunk_size = chunk_size, shuffle = shuffle, atom_num = atom_num, 
                                alive_atomic_energy = self.dp_params.file_paths.alive_atomic_energy, 
                                train_egroup = self.dp_params.optimizer_param.train_egroup)
            # save as .npy
            import src.pre_data.gen_data as gen_data
            gen_data.write_data(alive_atomic_energy = self.dp_params.file_paths.alive_atomic_energy, 
                                train_egroup = self.dp_params.optimizer_param.train_egroup)
    
        # copy feat_info and vdw_fitB.ntype
        fread_dfeat_dir = os.path.join(gen_feature_data, "fread_dfeat") #target dir
        if os.path.exists(fread_dfeat_dir):
            shutil.rmtree(fread_dfeat_dir)
        os.makedirs(fread_dfeat_dir)
        copy_file(os.path.join(sub_work_dir[0], "fread_dfeat/feat.info"), os.path.join(fread_dfeat_dir, "feat.info"))
        copy_file(os.path.join(sub_work_dir[0], "fread_dfeat/vdw_fitB.ntype"), os.path.join(fread_dfeat_dir, "vdw_fitB.ntype"))
        # copy input dir 
        input_dir = os.path.join(gen_feature_data, "input") #target dir
        source_input_dir = os.path.join(sub_work_dir[0], "input")
        if os.path.exists(input_dir) is False:
            shutil.copytree(source_input_dir, input_dir)
        # copy output dir, grid must be used for feature 1&2
        output_dir = os.path.join(gen_feature_data, "output")
        source_output_dir = os.path.join(sub_work_dir[0], "output")
        if os.path.exists(output_dir) is False:
            shutil.copytree(source_output_dir, output_dir)
        os.chdir(self.dp_params.file_paths.nn_work)
        self._reset_pm_params(self.dp_params.file_paths.nn_work, self.atom_type)
        
        return gen_feature_data


    '''
    description: reset PM paramters
    param {*} self
    param {*} work_dir, new work_dir
    param {*} atom_type, new atom type
    return {*}
    author: wuxingxing
    '''
    def _reset_pm_params(self, work_dir, atom_type):
        pm.sourceFileList = []
        # pm.atomType = atom_type
        # pm.atomTypeNum = len(pm.atomType)       #this step is important. Unexpected error
        # pm.ntypes = len(pm.atomType)
        # reset dirs
        pm.train_data_path = os.path.join(work_dir, r'train_data/final_train')
        pm.test_data_path =os.path.join(work_dir,  r'train_data/final_test')
        pm.prefix = work_dir
        pm.trainSetDir = os.path.join(work_dir, r'PWdata')
        # fortranFitSourceDir=r'/home/liuliping/program/nnff/git_version/src/fit'
        pm.fitModelDir = os.path.join(work_dir, r'fread_dfeat')
        pm.dRneigh_path = pm.trainSetDir + r'dRneigh.dat'

        pm.trainSetDir=os.path.abspath(pm.trainSetDir)
        #genFeatDir=os.path.abspath(genFeatDir)
        pm.fbinListPath=os.path.join(pm.trainSetDir,'location')
        sourceFileList=[]
        pm.InputPath=os.path.abspath('./input/')
        pm.OutputPath=os.path.abspath('./output/')
        pm.Ftype1InputPath=os.path.join('./input/',pm.Ftype_name[1]+'.in')
        pm.Ftype2InputPath=os.path.join('./input/',pm.Ftype_name[2]+'.in')

        pm.featCollectInPath=os.path.join(pm.fitModelDir,'feat_collect.in')
        pm.fitInputPath_lin=os.path.join(pm.fitModelDir,'fit_linearMM.input')
        pm.fitInputPath2_lin=os.path.join(pm.InputPath,'fit_linearMM.input')
        pm.featCollectInPath2=os.path.join(pm.InputPath,'feat_collect.in')

        pm.linModelCalcInfoPath=os.path.join(pm.fitModelDir,'linear_feat_calc_info.txt')
        pm.linFitInputBakPath=os.path.join(pm.fitModelDir,'linear_fit_input.txt')

        pm.dir_work = os.path.join(pm.fitModelDir,'NN_output/')

        pm.f_train_feat = os.path.join(pm.dir_work,'feat_train.csv')
        pm.f_test_feat = os.path.join(pm.dir_work,'feat_test.csv')

        pm.f_train_natoms = os.path.join(pm.dir_work,'natoms_train.csv')
        pm.f_test_natoms = os.path.join(pm.dir_work,'natoms_test.csv')        

        pm.f_train_dfeat = os.path.join(pm.dir_work,'dfeatname_train.csv')
        pm.f_test_dfeat  = os.path.join(pm.dir_work,'dfeatname_test.csv')

        pm.f_train_dR_neigh = os.path.join(pm.dir_work,'dR_neigh_train.csv')
        pm.f_test_dR_neigh  = os.path.join(pm.dir_work,'dR_neigh_test.csv')

        pm.f_train_force = os.path.join(pm.dir_work,'force_train.csv')
        pm.f_test_force  = os.path.join(pm.dir_work,'force_test.csv')

        pm.f_train_egroup = os.path.join(pm.dir_work,'egroup_train.csv')
        pm.f_test_egroup  = os.path.join(pm.dir_work,'egroup_test.csv')

        pm.f_train_ep = os.path.join(pm.dir_work,'ep_train.csv')
        pm.f_test_ep  = os.path.join(pm.dir_work,'ep_test.csv')

        pm.d_nnEi  = os.path.join(pm.dir_work,'NNEi/')
        pm.d_nnFi  = os.path.join(pm.dir_work,'NNFi/')

        pm.f_Einn_model   = pm.d_nnEi+'allEi_final.ckpt'
        pm.f_Finn_model   = pm.d_nnFi+'Fi_final.ckpt'

        pm.f_data_scaler = pm.d_nnFi+'data_scaler.npy'
        pm.f_Wij_np  = pm.d_nnFi+'Wij.npy'

    def scale_data(self, input_data:MovementHybridDataset, data_num:int):
        feat_data = None
        shape_data = [0]
        for data_index in range(data_num):
            feat_data = input_data.feat[data_index] if feat_data is None else np.concatenate((feat_data,input_data.feat[data_index]),0)
            shape_data.append(input_data.feat[data_index].shape[0] + sum(shape_data))            
        # scale feat
        if  self.scaler is not None:
            feat_data = self.scaler.transform(feat_data)
        else:
            self.scaler = MinMaxScaler()
            if os.path.exists(self.dp_params.file_paths.model_store_dir) is False:
                os.makedirs(self.dp_params.file_paths.model_store_dir)
            pickle.dump(self.scaler, open(os.path.join(self.dp_params.file_paths.model_store_dir, "scaler.pkl"),'wb'))
            print ("scaler.pkl saved to:",self.dp_params.file_paths.model_store_dir)
            feat_data = self.scaler.fit_transform(feat_data)

        for i in range(len(shape_data)-1):
            input_data.feat[data_index] = feat_data[shape_data[i]:shape_data[i+1]]
        # scale dfeat
        if pm.is_dfeat_sparse == False:
            for data_index in range(data_num):
                trans = lambda x : x.transpose(0, 1, 3, 2) 
                input_data.dfeat[data_index] = trans(trans(input_data.dfeat[data_index]) * self.scaler.scale_) 
        return input_data

    def load_data(self, feature_paths:list[str], data_shuffle=True, batch_size=1, workers=1, alive_atomic_energy=False, train_egroup = False, save_flag=False):
        def search_data_paths(feature_paths:list[str]):
            data_list = []
            for feature_path in feature_paths:
                data_dirs = os.listdir(feature_path)
                for data_dir in data_dirs:
                    if os.path.exists(os.path.join(feature_path, data_dir, "train_data", "final_train")):
                        data_list.append(os.path.join(feature_path, data_dir, "train_data"))
            return data_list
        self.load_nFeature(feature_paths)
        self.dfread_dfeat_input = load_dfeat_input(os.path.join(feature_paths[0], "fread_dfeat"),
                                                         os.path.join(feature_paths[0], "input"),
                                                         os.path.join(feature_paths[0], "output"))
        data_list = search_data_paths(feature_paths)
        if len(data_list) < 1:
            return None
        torch_data = get_torch_data_hybrid(data_list, "final_train", alive_atomic_energy, train_egroup, \
                                                 atom_type = pm.atomType, is_dfeat_sparse=pm.is_dfeat_sparse)
        if save_flag:
            self.energy_shift = torch_data.energy_shift

        if self.dp_params.inference:
            if os.path.exists(self.dp_params.file_paths.model_load_path):
                self.scaler = load_scaler_from_checkpoint(self.dp_params.file_paths.model_load_path)
            elif os.path.exists(self.dp_params.file_paths.model_save_path):
                self.scaler = load_scaler_from_checkpoint(self.dp_params.file_paths.model_save_path)
            else:
                raise Exception("Error! Load scaler from checkpoint: {}".format(self.dp_params.file_paths.model_load_path))
        
        torch_data = self.scale_data(torch_data, len(data_list))
        train_sampler = None
        data_loader = torch.utils.data.DataLoader(
            torch_data,
            batch_size=batch_size,
            shuffle=data_shuffle, #(train_sampler is None)
            num_workers=workers, 
            pin_memory=True,
            sampler=train_sampler
        )
        # load sparse dfeat. This is the default setting 
        if save_flag is True and pm.is_dfeat_sparse == True:     
            self.dfeat = dfeat_raw( input_dfeat_record_path_train = pm.f_train_dfeat, 
                                    input_feat_path_train = pm.f_train_feat,
                                    input_natoms_path_train = pm.f_train_natoms,
                                    input_dfeat_record_path_valid = pm.f_test_dfeat, 
                                    input_feat_path_valid = pm.f_test_feat,
                                    input_natoms_path_valid = pm.f_test_natoms,
                                    scaler = self.scaler)
            self.dfeat.load() 
        return data_loader

    def set_model_optimizer(self, start_epoch = 1, model_name = None):
        # initialize model 
        pm.itype_Ei_mean = self.energy_shift
        self.model = MLFFNet(device = self.device)
        self.model.to(self.device)

        # initialize optimzer 
        opt_optimizer = self.dp_params.optimizer_param.opt_name
        # if self.use_GKalman or self.use_LKalman or self.use_SKalman:
        if opt_optimizer == "LKF":
            self.optimizer = LKFOptimizer(
                self.model.parameters(),
                self.dp_params.optimizer_param.kalman_lambda,
                self.dp_params.optimizer_param.kalman_nue,
                self.dp_params.optimizer_param.block_size,
                self.dp_params.optimizer_param.p0_weight
            )
        elif opt_optimizer == "GKF":
            self.optimizer = GKFOptimizer(
                self.model.parameters(),
                self.dp_params.optimizer_param.kalman_lambda,
                self.dp_params.optimizer_param.kalman_nue
            )
        elif self.dp_params.optimizer_param.opt_name == "ADAM":
            if self.dp_params.optimizer_param.lambda_2 is None:
                self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate)
            else:
                self.optimizer = optim.Adam(self.model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate, 
                                        weight_decay=self.dp_params.optimizer_param.lambda_2)

        elif self.dp_params.optimizer_param.opt_name == "ADAMW":
            if self.dp_params.optimizer_param.lambda_2 is None:
                self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate)
            else:
                self.optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.dp_params.optimizer_param.learning_rate, 
                                        weight_decay=self.dp_params.optimizer_param.lambda_2)
        elif opt_optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                self.dp_params.optimizer_param.learning_rate,
                momentum=self.dp_params.optimizer_param.momentum,
                weight_decay=self.dp_params.optimizer_param.weight_decay
            )
        else:
            print("Unsupported optimizer!")

        # optionally resume from a checkpoint
        if self.dp_params.recover_train:
            if os.path.exists(self.dp_params.file_paths.model_load_path): # recover from user input ckpt file
                model_path = self.dp_params.file_paths.model_load_path
            else: # resume model specified by user
                model_path = self.dp_params.file_paths.model_save_path  #recover from last training
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if not torch.cuda.is_available():
                    checkpoint = torch.load(model_path,map_location=torch.device('cpu'), weights_only=False )
                elif self.dp_params.gpu is None:
                    checkpoint = torch.load(model_path, weights_only=False)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.dp_params.gpu)
                    checkpoint = torch.load(model_path, map_location=loc, weights_only=False)
                # start afresh
                self.dp_params.optimizer_param.start_epoch = checkpoint["epoch"] + 1
                self.model.load_state_dict(checkpoint["state_dict"])
                if "optimizer" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer"])
                # scheduler.load_state_dict(checkpoint["scheduler"])
                print("=> loaded checkpoint '{}' (epoch {})"\
                      .format(model_path, checkpoint["epoch"]))
            else:
                print("=> no checkpoint found at '{}'".format(model_path))
        print("network initialized")
   
    def train(self):
        self.train_loader = self.load_data(
                        feature_paths = self.dp_params.file_paths.train_feature_path,
                        data_shuffle=self.dp_params.data_shuffle, 
                        batch_size=self.dp_params.optimizer_param.batch_size,
                        workers=self.dp_params.workers,
                        alive_atomic_energy=self.dp_params.file_paths.alive_atomic_energy, 
                        train_egroup = self.dp_params.optimizer_param.train_egroup,
                        save_flag=True)
        if len(self.dp_params.file_paths.valid_feature_path) > 0:
            self.val_loader  = self.load_data(
                            feature_paths = self.dp_params.file_paths.valid_feature_path,
                            data_shuffle=False, 
                            batch_size=1,
                            workers=self.dp_params.workers,
                            alive_atomic_energy=self.dp_params.file_paths.alive_atomic_energy, 
                            train_egroup = self.dp_params.optimizer_param.train_egroup,
                            save_flag=False)
        else:
            self.val_loader = None
        self.set_model_optimizer()

        """    
            trianing method for the class 
        """ 
        smlink_file(self.dp_params.file_paths.model_store_dir, \
                    os.path.join(self.dp_params.file_paths.json_dir, os.path.basename(self.dp_params.file_paths.model_store_dir)))
        # set the log files
        train_log = os.path.join(self.dp_params.file_paths.model_store_dir, "epoch_train.dat")
        f_train_log = open(train_log, "w")

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

        f_train_log.write("# %s\n" % (train_format % tuple(train_lists)))
        if self.val_loader is not None:
            valid_log = os.path.join(self.dp_params.file_paths.model_store_dir, "epoch_valid.dat")
            f_valid_log = open(valid_log, "w")
            f_valid_log.write("# %s\n" % (valid_format % tuple(valid_lists)))

        for epoch in range(self.dp_params.optimizer_param.start_epoch, self.dp_params.optimizer_param.epochs + 1):
            # train for one epoch
            time_start = time.time()
            # check_cuda_memory(epoch, self.dp_params.optimizer_param.epochs, "before train")
            if self.dp_params.optimizer_param.opt_name == "LKF" or self.dp_params.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, loss_l1, loss_l2 = train_KF(
                    self.train_loader, self.model, self.criterion, self.optimizer, epoch, self.device, self.dp_params
                )
            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, real_lr, loss_l1, loss_l2 = train(
                    self.train_loader, self.model, self.criterion, self.optimizer, epoch, \
                        self.dp_params.optimizer_param.learning_rate, self.device, self.dp_params
                )
            time_end = time.time()
            # check_cuda_memory(epoch, self.dp_params.optimizer_param.epochs, "after train")
            # check_cuda_memory(epoch, self.dp_params.optimizer_param.epochs, "after valid")
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
                train_log_line += "%23.10e" % (loss_virial_per_atom)
            if self.dp_params.optimizer_param.opt_name == "LKF" or self.dp_params.optimizer_param.opt_name == "GKF":
                train_log_line += "%15.4f" % (time_end - time_start)
            else:
                train_log_line += "%18.10e%15.4f" % (real_lr, time_end - time_start)

            f_train_log.write("%s\n" % (train_log_line))
            f_train_log.close()

            if self.val_loader is not None:
            # evaluate on validation set
                vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom = valid(
                    self.val_loader, self.model, self.criterion, self.device, self.dp_params
                )
                f_valid_log = open(valid_log, "a")
                valid_log_line = "%5d%20.10e" % (epoch, vld_loss)
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

            # save model
            # if not self.dp_params.hvd or (self.dp_params.hvd and hvd.rank() == 0):
            if self.dp_params.file_paths.save_p_matrix:
                self.save_checkpoint(
                    {
                    "json_file":self.dp_params.to_dict(),
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer":self.optimizer.state_dict(),
                    "scaler": self.scaler,
                    "dfread_dfeat_input": self.dfread_dfeat_input,
                    },
                    self.dp_params.file_paths.model_name,
                    self.dp_params.file_paths.model_store_dir,
                )
            else: 
                self.save_checkpoint(
                    {
                    "json_file":self.dp_params.to_dict(),
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "scaler": self.scaler,
                    "dfread_dfeat_input": self.dfread_dfeat_input,
                    },
                    self.dp_params.file_paths.model_name,
                    self.dp_params.file_paths.model_store_dir,
                )

    def save_checkpoint(self, state, filename, prefix):
        filename = os.path.join(prefix, filename)
        torch.save(state, filename)

    def inference(self):
        self.test_loader  = self.load_data(
                        feature_paths = self.dp_params.file_paths.test_feature_path,
                        data_shuffle=False, 
                        batch_size=1,
                        workers=self.dp_params.workers,
                        alive_atomic_energy=self.dp_params.file_paths.alive_atomic_energy, 
                        train_egroup = self.dp_params.optimizer_param.train_egroup,
                        save_flag=True)
        self.set_model_optimizer()
        predict(self.test_loader, self.model, self.criterion, self.device, self.dp_params)

    """ 
        ============================================================
        ====================MD related functions====================
        ============================================================ 
    """
    def gen_lammps_config(self):
        """
            generate lammps config from POSCAR
            Default name is POSCAR
        """ 
        
        import poscar2lammps
        poscar2lammps.p2l() 
        
    def run_md(self, init_config = "atom.config", md_details = None, num_thread = 1,follow = False):
        
        mass_table = {  1:1.007,2:4.002,3:6.941,4:9.012,5:10.811,6:12.011,
                        7:14.007,8:15.999,9:18.998,10:20.18,11:22.99,12:24.305,
                        13:26.982,14:28.086,15:30.974,16:32.065,17:35.453,
                        18:39.948,19:39.098,20:40.078,21:44.956,22:47.867,
                        23:50.942,24:51.996,25:54.938,26:55.845,27:58.933,
                        28:58.693,29:63.546,30:65.38,31:69.723,32:72.64,33:74.922,
                        34:78.96,35:79.904,36:83.798,37:85.468,38:87.62,39:88.906,
                        40:91.224,41:92.906,42:95.96,43:98,44:101.07,45:102.906,46:106.42,
                        47:107.868,48:112.411,49:114.818,50:118.71,51:121.76,52:127.6,
                        53:126.904,54:131.293,55:132.905,56:137.327,57:138.905,58:140.116,
                        59:140.908,60:144.242,61:145,62:150.36,63:151.964,64:157.25,65:158.925,
                        66:162.5,67:164.93,68:167.259,69:168.934,70:173.054,71:174.967,72:178.49,
                        73:180.948,74:183.84,75:186.207,76:190.23,77:192.217,78:195.084,
                        79:196.967,80:200.59,81:204.383,82:207.2,83:208.98,84:210,85:210,86:222,
                        87:223,88:226,89:227,90:232.038,91:231.036,92:238.029,93:237,94:244,
                        95:243,96:247,97:247,98:251,99:252,100:257,101:258,102:259,103:262,104:261,105:262,106:266}
        
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
        f.write("3\n")     # imodel=1,2,3.    {1:linear;  2:VV;   3:NN;}
        f.write('1\n')               # interval for MOVEMENT output
        f.write('%d\n' % len(pm.atomType)) 
            
        # write mass 
        for i in range(len(pm.atomType)):
            #f.write('%d %f\n' % (pm.atomType[i], 2*pm.atomType[i]))
            f.write('%d %f\n' % (pm.atomType[i], mass_table[pm.atomType[i]]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 
    
    def load_nFeature(self, feature_path):
        # obtain number of feature from fread_dfeat/feat.info
        feat_info = os.path.join(feature_path[0], "fread_dfeat/feat.info")
        f = open(feat_info,'r')
        raw = f.readlines()[-1].split()
        pm.nFeatures = sum([int(item) for item in raw])
        print("number of features:",pm.nFeatures)
    
    # calculate loss 
    def get_loss_func(self,start_lr, real_lr, has_fi, lossFi, has_etot, loss_Etot, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum):
        start_pref_egroup =self.dp_params.optimizer_param.start_pre_fac_egroup  # 0.02
        start_pref_F =self.dp_params.optimizer_param.start_pre_fac_force  # 1000  #1000
        start_pref_etot =self.dp_params.optimizer_param.start_pre_fac_etot # 0.02   
        start_pref_ei =self.dp_params.optimizer_param.start_pre_fac_ei # 0.02

        limit_pref_egroup =self.dp_params.optimizer_param.end_pre_fac_egroup  # 1.0
        limit_pref_F =self.dp_params.optimizer_param.end_pre_fac_force # 1.0
        limit_pref_etot =self.dp_params.optimizer_param.end_pre_fac_etot # 1.0
        limit_pref_ei =self.dp_params.optimizer_param.end_pre_fac_ei # 2.0

        pref_fi = has_fi * (limit_pref_F + (start_pref_F - limit_pref_F) * real_lr / start_lr)
        pref_etot = has_etot * (limit_pref_etot + (start_pref_etot - limit_pref_etot) * real_lr / start_lr)
        pref_egroup = has_egroup * (limit_pref_egroup + (start_pref_egroup - limit_pref_egroup) * real_lr / start_lr)
        pref_ei = has_ei * (limit_pref_ei + (start_pref_ei - limit_pref_ei) * real_lr / start_lr)

        l2_loss = 0
        
        if has_fi==1:
            l2_loss += pref_fi * lossFi      # * 108
        if has_etot==1:
            l2_loss += 1./natoms_sum * pref_etot * loss_Etot  # 1/108 = 0.009259259259, 1/64=0.015625
        if has_egroup==1:
            l2_loss += pref_egroup * loss_Egroup
        if has_ei==1:
            l2_loss += pref_ei * loss_Ei
        
        return l2_loss, pref_fi, pref_etot, pref_egroup

    #update learning rate at iter_num noused
    def adjust_lr(self,iter_num):
        stop_lr= self.dp_params.optimizer_param.stop_lr #3.51e-8
        start_lr = self.dp_params.optimizer_param.learning_rate 
        stop_step = self.dp_params.optimizer_param.stop_step # 1000000
        decay_step= self.dp_params.optimizer_param.decay_step # 5000
        decay_rate = np.exp(np.log(stop_lr/start_lr) / (stop_step/decay_step)) #0.9500064099092085
        real_lr = start_lr * np.power(decay_rate, (iter_num//decay_step))
        return real_lr  

    # implement a linear scheduler noused
    def LinearLR(self,optimizer, base_lr, target_lr, total_epoch, cur_epoch):
        lr = base_lr - (base_lr - target_lr) * (float(cur_epoch) / float(total_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def evaluate(self,num_thread = 1):
        """
            evaluate a model w.r.t AIMD
            put a MOVEMENT in /MD and run MD100 
        """
        if not os.path.exists("input"):
            os.mkdir("input")
        import prepare as pp
        pp.writeGenFeatInput()
        os.system('rm -f MOVEMENT')
        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found")
        import md100
        md100.run_md100(imodel = 3, atom_type = pm.atomType, num_process = num_thread)

    def plot_evaluation(self, plot_elem, save_data):
        if not os.path.exists("MOVEMENT"):
            raise Exception("MOVEMENT not found. It should be force field MD result")
        import src.aux.plot_evaluation as plot_evaluation
        # plot_evaluation.plot()
        if self.dp_params.optimizer_param.train_ei or self.dp_params.optimizer_param.train_egroup:
            plot_ei = True
        else:
            plot_ei = False
        plot_evaluation.plot_new(atom_type = pm.atomType, plot_elem = plot_elem, save_data = save_data, plot_ei = plot_ei)

