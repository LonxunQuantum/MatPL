import os
import random
import torch
import time
import numpy as np
import torch.nn as nn
# import horovod.torch as hvd
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from sklearn.preprocessing import MinMaxScaler

from src.model.cheby_net import ChebyNet
from src.optimizer.GKF import GKFOptimizer
from src.optimizer.LKF import LKFOptimizer
import src.pre_data.dp_mlff as dp_mlff
from src.user.input_param import InputParam
from src.pre_data.cheby_data_loader import MovementDataset
from src.PWMLFF.cheby_mods.cheby_trainer import train_KF, train, valid, save_checkpoint, predict
from utils.file_operation import write_arrays_to_file, smlink_file
from numpy.ctypeslib import ndpointer
import ctypes
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_1 = ctypes.CDLL(os.path.join(lib_path, 'feature/chebyshev/build/lib/libneighborList.so')) # multi-neigh-list
lib_2 = ctypes.CDLL(os.path.join(lib_path, 'feature/chebyshev/build/lib/libdescriptor.so')) # multi-descriptor
lib_1.CreateNeighbor.argtypes = [ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                               ndpointer(ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"), 
                               ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS"), 
                               ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")]
lib_2.CreateDescriptor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, 
                                  ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                  ndpointer(ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, ndim=3, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, ndim=4, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, ndim=5, flags="C_CONTIGUOUS")]
                                  
lib_1.CreateNeighbor.restype = ctypes.c_void_p
lib_2.CreateDescriptor.restype = ctypes.c_void_p

# lib_1.ShowNeighbor.argtypes = [ctypes.c_void_p]
lib_1.DestroyNeighbor.argtypes = [ctypes.c_void_p]
# lib_2.show.argtypes = [ctypes.c_void_p]
lib_2.DestroyDescriptor.argtypes = [ctypes.c_void_p]

lib_1.GetNumNeighAll.argtypes = [ctypes.c_void_p]
lib_1.GetNumNeighAll.restype = ctypes.POINTER(ctypes.c_int)
lib_1.GetNeighborsListAll.argtypes = [ctypes.c_void_p]
lib_1.GetNeighborsListAll.restype = ctypes.POINTER(ctypes.c_int)
lib_1.GetDRNeighAll.argtypes = [ctypes.c_void_p]
lib_1.GetDRNeighAll.restype = ctypes.POINTER(ctypes.c_double)

lib_2.get_feat.argtypes = [ctypes.c_void_p]
lib_2.get_feat.restype = ctypes.POINTER(ctypes.c_double)
# lib_2.get_dfeat.argtypes = [ctypes.c_void_p]
# lib_2.get_dfeat.restype = ctypes.POINTER(ctypes.c_double)
# lib_2.get_dfeat2c.argtypes = [ctypes.c_void_p]
# lib_2.get_dfeat2c.restype = ctypes.POINTER(ctypes.c_double)

class cheby_network:
    def __init__(self, cheby_param: InputParam):
        self.cheby_param = cheby_param
        self.davg_dstd_energy_shift = None # scaler/energy_shift from training data
        torch.set_printoptions(precision = 12)
        if self.cheby_param.seed is not None:
            random.seed(self.cheby_param.seed)
            torch.manual_seed(self.cheby_param.seed)

        # if self.cheby_param.hvd:
        #     hvd.init()
        #     self.cheby_param.gpu = hvd.local_rank()

        if torch.cuda.is_available():
            if self.cheby_param.gpu:
                print("Use GPU: {} for training".format(self.cheby_param.gpu))
                self.device = torch.device("cuda:{}".format(self.cheby_param.gpu))
            else:
                self.device = torch.device("cuda")
        #elif torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        if self.cheby_param.precision == "float32":
            self.training_type = torch.float32  # training type is weights type
        else:
            self.training_type = torch.float64

        self.criterion = nn.MSELoss().to(self.device)
        
    def generate_data(self):
        """
        Generate training data for MLFF model.

        Returns:
            list: list of labels path
        """
        raw_data_path = self.cheby_param.file_paths.raw_path
        datasets_path = os.path.join(self.cheby_param.file_paths.json_dir, self.cheby_param.file_paths.trainSetDir)
        train_ratio = self.cheby_param.train_valid_ratio
        train_data_path = self.cheby_param.file_paths.trainDataPath
        valid_data_path = self.cheby_param.file_paths.validDataPath
        labels_path = dp_mlff.gen_train_data(train_ratio, raw_data_path, datasets_path, 
                                             train_data_path, valid_data_path, 
                                             self.cheby_param.valid_shuffle, self.cheby_param.seed, self.cheby_param.format)
        return labels_path
    
    def _get_stat(self):
        if self.cheby_param.inference:
            if os.path.exists(self.cheby_param.file_paths.model_load_path):
                # load scaler from checkpoint of model
                scaler, atom_map, energy_shift = self.load_stat_from_checkpoint(self.cheby_param.file_paths.model_load_path)
            elif os.path.exists(self.cheby_param.file_paths.model_save_path):
                scaler, atom_map, energy_shift = self.load_stat_from_checkpoint(self.cheby_param.file_paths.model_save_path)
            else:
                raise Exception("Erorr! Loading model for inference can not find checkpoint: \
                                \nmodel load path: {} \n or model at work path: {}\n"\
                                .format(self.cheby_param.file_paths.model_load_path, self.cheby_param.file_paths.model_save_path))
            stat_add = [scaler, atom_map, energy_shift]
        else:
            stat_add = None
        
        train_data_path = self.cheby_param.file_paths.trainDataPath
        ntypes = len(self.cheby_param.atom_type)
        # input_atom_type = np.array([(_['type']) for _ in self.cheby_param.atom_type_dict])   # input atom type order
        input_atom_type = np.array(self.cheby_param.atom_type)   # input atom type order
        if stat_add is not None:
            print("input_atom_type and energy_shift are read from model checkpoint")
            scaler, input_atom_type, energy_shift = stat_add
        else:
            energy_shift = None

        max_atom_nums = 0
        valid_chunk = False
        for dataset_path in self.cheby_param.file_paths.datasets_path:
            atom_types_image = np.load(os.path.join(dataset_path, train_data_path, "image_type.npy"))
            max_atom_nums = max(max_atom_nums, atom_types_image.shape[1])
            if energy_shift is None:
                _atom_types = np.load(os.path.join(dataset_path, train_data_path, "atom_type.npy"))
                if _atom_types.shape[1] != ntypes:
                    continue
                # the energy_shift atom order are the same --> movement's atom order
                lattice = np.load(os.path.join(dataset_path, train_data_path, "lattice.npy"))
                img_per_mvmt = lattice.shape[0]
                if img_per_mvmt < self.cheby_param.chunk_size:
                    continue
                valid_chunk = True
                position = np.load(os.path.join(dataset_path, train_data_path, "position.npy"))
                _Ei = np.load(os.path.join(dataset_path, train_data_path, "ei.npy"))
                type_maps = np.array(dp_mlff.type_map(atom_types_image[0], input_atom_type))
                types, type_incides, atom_types_nums = np.unique(type_maps, return_index=True, return_counts=True)
                atom_types_nums = atom_types_nums[np.argsort(type_incides)]
                scaler = cheby_network.calculate_scaler(self.cheby_param, type_maps, lattice, position)
                energy_shift = dp_mlff.calculate_energy_shift(self.cheby_param.chunk_size, _Ei, atom_types_nums)
                energy_shift = cheby_network.adjust_order_same_as_user_input(energy_shift, _atom_types[0].tolist(), input_atom_type)

        if not valid_chunk and energy_shift is None:
            raise ValueError("Invalid chunk size, the number of images (include all atom types) in the movement is too small, \nPlease set a smaller chunk_size (default: 10) or add more images in the datasets")
        
        return scaler, energy_shift, max_atom_nums
       
    @staticmethod
    def calculate_scaler(cheby_param:InputParam, type_maps:np.ndarray, lattice:np.ndarray, position:np.ndarray):
        """
        Calculate the scaler for the dataset

        Args:
        cheby_param: the input parameter
        type_maps: the type map of the dataset
        lattice: the lattice of the dataset
        position: the position of the dataset

        Returns:
        list: the scaler of the dataset
        """
        # Calculate the scaler
        image_num = cheby_param.chunk_size
        ntypes = len(cheby_param.atom_type)
        max_neigh_num = cheby_param.max_neigh_num
        rcut_max = cheby_param.descriptor.Rmax
        rcut_smooth = cheby_param.descriptor.Rmin
        coords_all = position[:image_num].flatten()
        box_all = lattice[:image_num].flatten()
        type_maps = type_maps.astype(np.int32)
        natoms = type_maps.shape[0]

        beta = cheby_param.descriptor.cheby_order
        m1 = cheby_param.descriptor.radial_num1
        m2 = cheby_param.descriptor.radial_num2
        nfeat = m1 * m2

        # Create neighbor list
        mnl = lib_1.CreateNeighbor(image_num, rcut_max, max_neigh_num, ntypes, natoms, type_maps, coords_all, box_all)
        # lib_1.ShowNeighbor(mnl)
        num_neigh_all = lib_1.GetNumNeighAll(mnl)
        list_neigh_all = lib_1.GetNeighborsListAll(mnl)
        dr_neigh_all = lib_1.GetDRNeighAll(mnl)
        num_neigh_all = np.ctypeslib.as_array(num_neigh_all, (image_num, natoms, ntypes))
        list_neigh_all = np.ctypeslib.as_array(list_neigh_all, (image_num, natoms, ntypes, max_neigh_num))
        dr_neigh_all = np.ctypeslib.as_array(dr_neigh_all, (image_num, natoms, ntypes, max_neigh_num, 4))   # rij, delx, dely, delz
        descriptor = lib_2.CreateDescriptor(image_num, beta, m1, m2, rcut_max, rcut_smooth, natoms, ntypes, max_neigh_num, type_maps, num_neigh_all, list_neigh_all, dr_neigh_all, None)
        # lib_2.show(descriptor)
        feat = lib_2.get_feat(descriptor)
        # dfeat = lib_2.get_dfeat(descriptor)
        # dfeat2c = lib_2.get_dfeat2c(descriptor)
        feat = np.ctypeslib.as_array(feat, (image_num, natoms, nfeat))
        # dfeat = np.ctypeslib.as_array(dfeat, (image_num, natoms, nfeat, max_neigh_num, 3))

        # Calculate the scaler
        scaler = MinMaxScaler()
        scaler.fit_transform(feat.reshape(-1, feat.shape[-1]))
        lib_1.DestroyNeighbor(mnl)
        lib_2.DestroyDescriptor(descriptor)
        return scaler   

    @staticmethod
    def adjust_order_same_as_user_input(energy_shift:list, atom_type_order:list, atom_type_list:list):
        """ 
        adjust the order of energy shift to the same as user input order

        Args:
        energy_shift: the energy shift of the dataset
        atom_type_order: the order of atom type in dataset
        atom_type_list: the order of atom type in user input

        Returns:
        list: the energy shift of the dataset with the same order as user input
        """
        energy_shift_res = []
        for i, atom in enumerate(atom_type_list):
            energy_shift_res.append(energy_shift[atom_type_order.index(atom)])
        return energy_shift_res          
    
    def load_data(self, max_atom_nums):
        # Create dataset
        if self.cheby_param.inference:
            train_dataset = MovementDataset("train", self.cheby_param, max_atom_nums)
            valid_dataset = None
        else:            
            train_dataset = MovementDataset("train", self.cheby_param, max_atom_nums)
            valid_dataset = MovementDataset("valid", self.cheby_param, max_atom_nums)
        
        atom_map = train_dataset.get_stat()

        # if self.cheby_param.hvd:
        #     train_sampler = torch.utils.data.distributed.DistributedSampler(
        #         train_dataset, num_replicas=hvd.size(), rank=hvd.rank()
        #     )
        #     val_sampler = torch.utils.data.distributed.DistributedSampler(
        #         valid_dataset, num_replicas=hvd.size(), rank=hvd.rank(), drop_last=True
        #     )
        # else:
        train_sampler = None
        val_sampler = None

        # should add a collate function for padding
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cheby_param.optimizer_param.batch_size,
            shuffle=self.cheby_param.data_shuffle,
            num_workers=self.cheby_param.workers,   
            pin_memory=True,
            sampler=train_sampler,
        )
        
        if self.cheby_param.inference:
            val_loader = None
        else:
            val_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=self.cheby_param.optimizer_param.batch_size,
                shuffle=False,
                num_workers=self.cheby_param.workers,
                pin_memory=True,
                sampler=val_sampler,
            )
        return atom_map, train_loader, val_loader
    
    '''
    description:
        if scaler and energy_shift not from load_data, get it from model_load_file
    return {*}
    author: wuxingxing
    '''
    def get_stat(self):
        if self.davg_dstd_energy_shift is None:
            if os.path.exists(self.cheby_param.file_paths.model_load_path) is False:
                raise Exception("ERROR! {} is not exist when get energy shift !".format(self.cheby_param.file_paths.model_load_path))
            davg_dstd_energy_shift = self.load_stat_from_checkpoint(self.cheby_param.file_paths.model_load_path)
        else:
            davg_dstd_energy_shift = self.davg_dstd_energy_shift
        return davg_dstd_energy_shift
    
    def load_model_optimizer(self, scaler, energy_shift):
        # create model        
        model = ChebyNet(self.cheby_param, scaler, energy_shift)
        model = model.to(self.training_type)

        # optionally resume from a checkpoint
        checkpoint = None
        if self.cheby_param.recover_train:
            if self.inference and os.path.exists(self.cheby_param.file_paths.model_load_path): # recover from user input ckpt file for inference work
                model_path = self.cheby_param.file_paths.model_load_path
            else: # resume model specified by user
                model_path = self.cheby_param.file_paths.model_save_path  #recover from last training for training
            if os.path.isfile(model_path):
                print("=> loading checkpoint '{}'".format(model_path))
                if not torch.cuda.is_available():
                    checkpoint = torch.load(model_path,map_location=torch.device('cpu') , weights_only=False)
                elif self.cheby_param.gpu is None:
                    checkpoint = torch.load(model_path, weights_only=False)
                elif torch.cuda.is_available():
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(self.cheby_param.gpu)
                    checkpoint = torch.load(model_path, map_location=loc, weights_only=False)
                # start afresh
                if self.cheby_param.optimizer_param.reset_epoch:
                    self.cheby_param.optimizer_param.start_epoch = 1
                else:
                    self.cheby_param.optimizer_param.start_epoch = checkpoint["epoch"] + 1
                model.load_state_dict(checkpoint["state_dict"])
                
                # scheduler.load_state_dict(checkpoint["scheduler"])
                print("=> loaded checkpoint '{}' (epoch {})"\
                      .format(model_path, checkpoint["epoch"]))
                if "compress" in checkpoint.keys():
                    model.set_comp_tab(checkpoint["compress"])
            else:
                print("=> no checkpoint found at '{}'".format(model_path))

        if not torch.cuda.is_available():
            print("using CPU, this will be slow")
            '''
        elif self.cheby_param.hvd:
            if torch.cuda.is_available():
                if self.cheby_param.gpu is not None:
                    torch.cuda.set_device(self.cheby_param.gpu)
                    model.cuda(self.cheby_param.gpu)
                    self.cheby_param.optimizer_param.batch_size = int(self.cheby_param.optimizer_param.batch_size / hvd.size())
            '''
        elif self.cheby_param.gpu is not None and torch.cuda.is_available():
            torch.cuda.set_device(self.cheby_param.gpu)
            model = model.cuda(self.cheby_param.gpu)
        else:
            model = model.cuda()
            # if model.compress_tab is not None:
            #     model.compress_tab.to(device=self.device)
        # optimizer, and learning rate scheduler
        if self.cheby_param.optimizer_param.opt_name == "LKF":
            optimizer = LKFOptimizer(
                model.parameters(),
                self.cheby_param.optimizer_param.kalman_lambda,
                self.cheby_param.optimizer_param.kalman_nue,
                self.cheby_param.optimizer_param.block_size
            )
        elif self.cheby_param.optimizer_param.opt_name == "GKF":
            optimizer = GKFOptimizer(
                model.parameters(),
                self.cheby_param.optimizer_param.kalman_lambda,
                self.cheby_param.optimizer_param.kalman_nue
            )
        elif self.cheby_param.optimizer_param.opt_name == "ADAM":
            optimizer = optim.Adam(model.parameters(), 
                                   self.cheby_param.optimizer_param.learning_rate)
        elif self.cheby_param.optimizer_param.opt_name == "SGD":
            optimizer = optim.SGD(
                model.parameters(), 
                self.cheby_param.optimizer_param.learning_rate,
                momentum=self.cheby_param.optimizer_param.momentum,
                weight_decay=self.cheby_param.optimizer_param.weight_decay
            )
        else:
            raise Exception("Error: Unsupported optimizer!")
        
        if checkpoint is not None and "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            load_p = checkpoint["optimizer"]['state'][0]['P']
            optimizer.set_kalman_P(load_p, checkpoint["optimizer"]['state'][0]['kalman_lambda'])
                
        '''
        if self.cheby_param.hvd:
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

        return model, optimizer

    def inference(self, scaler, energy_shift, max_atom_nums):
        # do inference
        atom_map, train_loader, val_loader = self.load_data(max_atom_nums)
        model, optimizer = self.load_model_optimizer(scaler, energy_shift)
        start = time.time()
        res_pd, etot_label_list, etot_predict_list, ei_label_list, ei_predict_list, force_label_list, force_predict_list\
        = predict(train_loader, model, self.criterion, self.device, self.cheby_param)
        end = time.time()
        print("fitting time:", end - start, 's')

        # print infos
        inference_cout = ""
        inference_cout += "For {} images: \n".format(res_pd.shape[0])
        inference_cout += "Avarage REMSE of Etot: {} \n".format(res_pd['RMSE_Etot'].mean())
        inference_cout += "Avarage REMSE of Etot per atom: {} \n".format(res_pd['RMSE_Etot_per_atom'].mean())
        inference_cout += "Avarage REMSE of Ei: {} \n".format(res_pd['RMSE_Ei'].mean())
        inference_cout += "Avarage REMSE of RMSE_F: {} \n".format(res_pd['RMSE_F'].mean())
        if self.cheby_param.optimizer_param.train_egroup:
            inference_cout += "Avarage REMSE of RMSE_Egroup: {} \n".format(res_pd['RMSE_Egroup'].mean())
        if self.cheby_param.optimizer_param.train_virial:
            inference_cout += "Avarage REMSE of RMSE_virial: {} \n".format(res_pd['RMSE_virial'].mean())
            inference_cout += "Avarage REMSE of RMSE_virial_per_atom: {} \n".format(res_pd['RMSE_virial_per_atom'].mean())
        
        inference_cout += "\nMore details can be found under the file directory:\n{}\n".format(os.path.realpath(self.cheby_param.file_paths.test_dir))
        print(inference_cout)

        inference_path = self.cheby_param.file_paths.test_dir
        if os.path.exists(inference_path) is False:
            os.makedirs(inference_path)
        write_arrays_to_file(os.path.join(inference_path, "dft_total_energy.txt"), etot_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_total_energy.txt"), etot_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_atomic_energy.txt"), ei_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_atomic_energy.txt"), ei_predict_list)

        write_arrays_to_file(os.path.join(inference_path, "dft_force.txt"), force_label_list)
        write_arrays_to_file(os.path.join(inference_path, "inference_force.txt"), force_predict_list)

        res_pd.to_csv(os.path.join(inference_path, "inference_loss.csv"))

        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)
        return 

    def train(self, scaler, energy_shift, max_atom_nums):
        atom_map, train_loader, val_loader = self.load_data(max_atom_nums)
        model, optimizer = self.load_model_optimizer(scaler, energy_shift)
        if not os.path.exists(self.cheby_param.file_paths.model_store_dir):
            os.makedirs(self.cheby_param.file_paths.model_store_dir)
        if self.cheby_param.model_num == 1:
            smlink_file(self.cheby_param.file_paths.model_store_dir, os.path.join(self.cheby_param.file_paths.json_dir, os.path.basename(self.cheby_param.file_paths.model_store_dir)))
        
        # if not self.cheby_param.hvd or (self.cheby_param.hvd and hvd.rank() == 0):
        train_log = os.path.join(self.cheby_param.file_paths.model_store_dir, "epoch_train.dat")
        f_train_log = open(train_log, "w")
        valid_log = os.path.join(self.cheby_param.file_paths.model_store_dir, "epoch_valid.dat")
        f_valid_log = open(valid_log, "w")
        # Define the lists based on the training type
        train_lists = ["epoch", "loss"]
        valid_lists = ["epoch", "loss"]
        if self.cheby_param.optimizer_param.train_energy:
            # train_lists.append("RMSE_Etot")
            # valid_lists.append("RMSE_Etot")
            train_lists.append("RMSE_Etot_per_atom")
            valid_lists.append("RMSE_Etot_per_atom")
        if self.cheby_param.optimizer_param.train_ei:
            train_lists.append("RMSE_Ei")
            valid_lists.append("RMSE_Ei")
        if self.cheby_param.optimizer_param.train_egroup:
            train_lists.append("RMSE_Egroup")
            valid_lists.append("RMSE_Egroup")
        if self.cheby_param.optimizer_param.train_force:
            train_lists.append("RMSE_F")
            valid_lists.append("RMSE_F")
        if self.cheby_param.optimizer_param.train_virial:
            # train_lists.append("RMSE_virial")
            # valid_lists.append("RMSE_virial")
            train_lists.append("RMSE_virial_per_atom")
            valid_lists.append("RMSE_virial_per_atom")
        if self.cheby_param.optimizer_param.opt_name == "LKF" or self.cheby_param.optimizer_param.opt_name == "GKF":
            train_lists.extend(["time"])
        else:
            train_lists.extend(["real_lr", "time"])

        train_print_width = {
            "epoch": 5,
            "loss": 18,
            "RMSE_Etot": 18,
            "RMSE_Etot_per_atom": 21,
            "RMSE_Ei": 18,
            "RMSE_Egroup": 18,
            "RMSE_F": 18,
            "RMSE_virial": 18,
            "RMSE_virial_per_atom": 23,
            "real_lr": 18,
            "time": 10,
        }

        train_format = "".join(["%{}s".format(train_print_width[i]) for i in train_lists])
        valid_format = "".join(["%{}s".format(train_print_width[i]) for i in valid_lists])

        f_train_log.write("%s\n" % (train_format % tuple(train_lists)))
        f_valid_log.write("%s\n" % (valid_format % tuple(valid_lists)))

        for epoch in range(self.cheby_param.optimizer_param.start_epoch, self.cheby_param.optimizer_param.epochs + 1):
            # if self.cheby_param.hvd: # this code maybe error, check when add multi GPU training. wu
            #     self.train_sampler.set_epoch(epoch)

            # train for one epoch
            time_start = time.time()
            if self.cheby_param.optimizer_param.opt_name == "LKF" or self.cheby_param.optimizer_param.opt_name == "GKF":
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, Sij_max = train_KF(
                    train_loader, model, self.criterion, optimizer, epoch, self.device, self.cheby_param
                )

            else:
                loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, real_lr, Sij_max = train(
                    train_loader, model, self.criterion, optimizer, epoch, \
                        self.cheby_param.optimizer_param.learning_rate, self.device, self.cheby_param
                )
            time_end = time.time()

            # evaluate on validation set
            vld_loss, vld_loss_Etot, vld_loss_Etot_per_atom, vld_loss_Force, vld_loss_Ei, val_loss_egroup, val_loss_virial, val_loss_virial_per_atom = valid(
                val_loader, model, self.criterion, self.device, self.cheby_param
            )

            # if not self.cheby_param.hvd or (self.cheby_param.hvd and hvd.rank() == 0):

            f_train_log = open(train_log, "a")
            f_valid_log = open(valid_log, "a")

            # Write the log line to the file based on the training mode
            train_log_line = "%5d%18.10e" % (
                epoch,
                loss,
            )
            valid_log_line = "%5d%18.10e" % (
                epoch,
                vld_loss,
            )

            if self.cheby_param.optimizer_param.train_energy:
                # train_log_line += "%18.10e" % (loss_Etot)
                # valid_log_line += "%18.10e" % (vld_loss_Etot)
                train_log_line += "%21.10e" % (loss_Etot_per_atom)
                valid_log_line += "%21.10e" % (vld_loss_Etot_per_atom)
            if self.cheby_param.optimizer_param.train_ei:
                train_log_line += "%18.10e" % (loss_Ei)
                valid_log_line += "%18.10e" % (vld_loss_Ei)
            if self.cheby_param.optimizer_param.train_egroup:
                train_log_line += "%18.10e" % (loss_egroup)
                valid_log_line += "%18.10e" % (val_loss_egroup)
            if self.cheby_param.optimizer_param.train_force:
                train_log_line += "%18.10e" % (loss_Force)
                valid_log_line += "%18.10e" % (vld_loss_Force)
            if self.cheby_param.optimizer_param.train_virial:
                # train_log_line += "%18.10e" % (loss_virial)
                # valid_log_line += "%18.10e" % (val_loss_virial)
                train_log_line += "%23.10e" % (loss_virial_per_atom)
                valid_log_line += "%23.10e" % (val_loss_virial_per_atom)

            if self.cheby_param.optimizer_param.opt_name == "LKF" or self.cheby_param.optimizer_param.opt_name == "GKF":
                train_log_line += "%10.4f" % (time_end - time_start)
            else:
                train_log_line += "%18.10e%10.4f" % (real_lr, time_end - time_start)

            f_train_log.write("%s\n" % (train_log_line))
            f_valid_log.write("%s\n" % (valid_log_line))
        
            f_train_log.close()
            f_valid_log.close()
            
           
            # if not self.cheby_param.hvd or (self.cheby_param.hvd and hvd.rank() == 0):
            if self.cheby_param.file_paths.save_p_matrix:
                save_checkpoint(
                    {
                    "json_file":self.cheby_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "scaler": scaler,
                    "energy_shift":energy_shift,
                    "atom_type_order": atom_map,
                    "sij_max":Sij_max,
                    "optimizer":optimizer.state_dict()
                    },
                    self.cheby_param.file_paths.model_name,
                    self.cheby_param.file_paths.model_store_dir,
                )
            else: 
                save_checkpoint(
                    {
                    "json_file":self.cheby_param.to_dict(),
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "scaler": scaler,
                    "energy_shift":energy_shift,
                    "atom_type_order": atom_map,
                    "sij_max":Sij_max
                    },
                    self.cheby_param.file_paths.model_name,
                    self.cheby_param.file_paths.model_store_dir,
                )

    def load_model_with_ckpt(self, scaler, energy_shift):
        model, optimizer = self.load_model_optimizer(scaler, energy_shift)
        return model
    
    def load_stat_from_checkpoint(self, model_path):
        model_checkpoint = torch.load(model_path,map_location=torch.device("cpu"), weights_only=False)
        scaler = model_checkpoint['scaler']
        atom_type_order = model_checkpoint['atom_type_order']
        energy_shift = model_checkpoint['energy_shift']
        if atom_type_order.size == 1:
            atom_type_order = [atom_type_order.tolist()]
        return scaler, atom_type_order, energy_shift
