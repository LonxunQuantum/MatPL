import numpy as np
import os
import math
from torch.utils.data import Dataset
import torch
import yaml
from src.user.input_param import InputParam
from pwdata import Config
from pwdata.image import Image
from utils.debug_operation import check_cuda_memory
import time
# from src.feature.nep_find_neigh.findneigh import FindNeigh
import random
from typing import Union, Optional
from tqdm import tqdm
if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device

def get_det(box: np.array):
    matrix = box.reshape((3, 3))
    return np.linalg.det(matrix)

def get_area(a: np.array, b: np.array):
    s1 = a[1] * b[2] - a[2] * b[1]
    s2 = a[2] * b[0] - a[0] * b[2]
    s3 = a[0] * b[1] - a[1] * b[0]
    return math.sqrt(s1 * s1 + s2 * s2 + s3 * s3)

def variable_length_collate_fn(batch):
    keys = batch[0].keys()
    res = {}

    def extract_items(tensors, key):
        return [x[key] for x in tensors]

    for key in keys:
        if key in ["position", "force", "atom_type_map", "ei"]:
            res[key] = torch.concat(extract_items(batch, key), dim=0)
        else:
            res[key] = torch.stack(extract_items(batch, key), dim=0)
    res["num_atom_sum"] = res["num_atom"].cumsum(0).to(res["num_atom"].dtype)
    return res

class NepTestData():
    def __init__(self, input_param:InputParam):
        self.image_list = []
        self.input_param = input_param

        if len(self.input_param.file_paths.test_data_path) > 0:
            for config in self.input_param.file_paths.test_data_path:
                image_read = Config(data_path=config, format=self.input_param.file_paths.format).images
                if isinstance(image_read, list):
                    self.image_list.extend(image_read)
                else:
                    self.image_list.append(image_read)

        for image in self.image_list:
            if image.cartesian is False:
                image._set_cartesian()
                # image.lattice = image.lattice.T
            # image.atom_types_image = np.array([self.atom_types.index(_) for _ in image.atom_types_image])
        # return image_list

class UniDataset(Dataset):
    def __init__(self, 
                data_paths, 
                format, 
                atom_types, 
                cutoff_radial=0, 
                cutoff_angular=0,
                cal_energy=False,
                dtype: Union[torch.dtype, str] = torch.float64, 
                index_type: Union[torch.dtype, str] = torch.int64,
                use_cartesian=True):
        super(UniDataset, self).__init__()
        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.dirs = data_paths  # include all movement data path
        self.format = format
        self.image_list = []
        self.atom_types = atom_types
        self.cal_energy = cal_energy
        self.cutoff_radial = cutoff_radial
        self.cutoff_angular = cutoff_angular
        self.max_NN_radial = 100
        self.max_NN_angular= 100
        self.max_atom_nums = 1
        self.avg_image_atom= None
        self.use_cartesian = use_cartesian
        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.index_type = (
            index_type
            if isinstance(index_type, torch.dtype)
            else getattr(torch, index_type)
        )
        self.image_list, self.total_images = self.__concatenate_data()

        if self.total_images > 0:
            data = self.__load_data(0)
        # print()

    def __concatenate_data(self):
        if len(self.dirs) > 0:
            for config in self.dirs:
                image_read = Config(data_path=config, format=self.format).images
                if isinstance(image_read, list):
                    self.image_list.extend(image_read)
                else:
                    self.image_list.append(image_read)
        class_dict = {}
        atom_avg_list = []
        for image in self.image_list:
            if self.use_cartesian:
                if image.cartesian is False:
                    image._set_cartesian()
            else:
                if image.cartesian is True:
                    image._set_fractional()
            self.max_atom_nums = max(self.max_atom_nums, image.position.shape[0])
            atom_avg_list.append(image.position.shape[0])
            if not hasattr(image, 'atom_type_map'):
                image.atom_type_map = np.array([self.atom_types.index(_) for _ in image.atom_types_image])

                atom_types = image.atom_types_image
                cout_type, cout_num = np.unique(atom_types, return_counts=True)
                image_key = "_".join(["{}:{}".format(i1, i2) for i1, i2 in zip(cout_type, cout_num)])
                if image_key in class_dict.keys():
                    class_dict[image_key]['Ep'].append(image.Ep)
                else:
                    class_dict[image_key] = {}
                    class_dict[image_key]['Ep'] = [image.Ep]
                    class_dict[image_key]['cout_num'] = cout_num
                    class_dict[image_key]['cout_type'] = cout_type

        if self.cal_energy:
            self.energy_shift = self.set_energy_shift_2(class_dict) # self.set_energy_shift()
        else:
            self.energy_shift = [0 for _ in self.atom_types] # for valid or test the energy shift is reused from models
        
        self.avg_image_atom = np.mean(atom_avg_list)        
        return self.image_list, len(self.image_list)

    def set_energy_shift_2(self, class_dict):
        energy_dict = {}
        for atom in self.atom_types:
            energy_dict[atom] = []
        energy_dict['E'] = []
        for key in class_dict.keys():
            Ep = np.mean(class_dict[key]['Ep'])
            cout_num  = class_dict[key]['cout_num'] 
            cout_type = class_dict[key]['cout_type']

            for element in self.atom_types:
                if element in cout_type:
                    indices = np.where(cout_type == element)[0]
                    energy_dict[element].append(cout_num[indices][0])
                else:
                    energy_dict[element].append(0)
            energy_dict['E'].append(Ep)
        _num_matrix = []
        for key in energy_dict.keys():
            if key != 'E':
                _num_matrix.append(energy_dict[key])
        x, residuals, rank, s = np.linalg.lstsq(np.array(_num_matrix).T, energy_dict['E'], rcond=None)
        energy_shift = x.tolist()
        return energy_shift # array([-7.85248589, -8.80939613, -5.12775169, -3.13588943])

    def get_energy_shift(self):
        return self.energy_shift

    def __getitem__(self, index):
        data = self.__load_data(index)
        # if self.train_hybrid is True:
        #     data = self.__completing_tensor_rows(data)
        return data
    
    def __len__(self): 
        return self.total_images

    def __load_data(self, index):
        data = {}
        num_cell = np.zeros(3, dtype=int)
        box = np.zeros(18, dtype=float) 
        volume = self.expand_box(self.image_list[index].lattice.T.flatten(), self.cutoff_radial, num_cell, box)
        data["box"] = torch.from_numpy(box).to(self.dtype)
        data["box_original"] = torch.from_numpy(self.image_list[index].lattice.T.flatten()).to(self.dtype)
        data["num_cell"] = torch.from_numpy(num_cell).to(self.index_type)
        data["volume"] = torch.from_numpy(np.array([volume])).to(self.dtype)
        # data["atom_type"] = torch.from_numpy(self.image_list[index].atom_type).to(self.index_type)
        data["atom_type_map"] = torch.from_numpy(self.image_list[index].atom_type_map).to(self.index_type)
        data["num_atom"] = torch.from_numpy(np.array([len(data["atom_type_map"])])).to(self.index_type)
        data["force"] = torch.from_numpy(self.image_list[index].force).to(self.dtype)
        data["ei"] = torch.from_numpy(self.image_list[index].atomic_energy).to(self.dtype)
        data["energy"] = torch.from_numpy(np.array([self.image_list[index].Ep])).to(self.dtype)
        data["position"] = torch.from_numpy(self.image_list[index].position).to(self.dtype)
        data["virial"] = torch.from_numpy(np.ones([9]) * -1e6).to(self.dtype) if self.image_list[index].virial is None \
                            else torch.from_numpy(self.image_list[index].virial.flatten()).to(self.dtype)
        return data
        # for key in list(data.keys()):
        #     print(key)
        #     print(data[key].shape)
    def expand_box(self, lattice, cutoff_radial, num_cell, box):
        a = lattice[0::3]
        b = lattice[1::3]
        c = lattice[2::3]
        det = get_det(lattice)
        volume = abs(det)
        num_cell[0] = int(
            math.ceil(2.0 * cutoff_radial / (volume / get_area(b, c)))
        )
        num_cell[1] = int(
            math.ceil(2.0 * cutoff_radial / (volume / get_area(c, a)))
        )
        num_cell[2] = int(
            math.ceil(2.0 * cutoff_radial / (volume / get_area(a, b)))
        )

        box[0:9:3] = lattice[0::3] * num_cell[0]
        box[1:9:3] = lattice[1::3] * num_cell[1]
        box[2:9:3] = lattice[2::3] * num_cell[2]

        box[9] = box[4] * box[8] - box[5] * box[7]
        box[10] = box[2] * box[7] - box[1] * box[8]
        box[11] = box[1] * box[5] - box[2] * box[4]
        box[12] = box[5] * box[6] - box[3] * box[8]
        box[13] = box[0] * box[8] - box[2] * box[6]
        box[14] = box[2] * box[3] - box[0] * box[5]
        box[15] = box[3] * box[7] - box[4] * box[6]
        box[16] = box[1] * box[6] - box[0] * box[7]
        box[17] = box[0] * box[4] - box[1] * box[3]

        det *= num_cell[0] * num_cell[1] * num_cell[2]
        for n in range(9, 18):
            box[n] /= det
        return volume    

def type_map(atom_types_image, atom_type):
    """
    Maps the atom types to their corresponding indices in the atom_type array.

    Args:
    atom_types_image (numpy.ndarray): Array of atom types to be mapped.
    atom_type (numpy.ndarray): Array of integers representing the atom type of each atom in the system.

    Returns:
    list: List of indices corresponding to the atom types in the atom_type array.

    Raises:
    AssertionError: If no atom types in atom_types_image are found in atom_type.

    Examples: CH4 molecule
    >>> atom_types_image = array([6, 1, 1, 1, 1])
    >>> atom_type = array([6, 1])
    >>> type_map(atom_types_image, atom_type)
    [0, 1, 1, 1, 1]
    """
    atom_type_map = []
    if isinstance(atom_types_image.tolist(), int):
        atom_types_image = [atom_types_image.tolist()]
    for elem in atom_types_image:
        if elem in atom_type:
            atom_type_map.append(np.where(atom_type == elem)[0][0])
    assert len(atom_type_map) != 0, "this atom type didn't found"
    return atom_type_map

'''
description: 
 nouse
param {*} chunk_size
param {*} _Ei
param {*} atom_types_nums
return {*}
author: wuxingxing
'''
def calculate_energy_shift(chunk_size, _Ei, atom_types_nums):
    Ei = _Ei[:chunk_size]
    res = []
    current_type = 0
    for atom_type_num in atom_types_nums:
        current_type_indices = current_type + atom_type_num
        avg_Ei = np.mean(Ei[:, current_type:current_type_indices])
        res.append(avg_Ei)
        current_type = current_type_indices
    return res

"""
Calculate the maximum and minimum neighbor numbers for radial and angular distributions.

:param dataloader: DataLoader object containing the dataset.
:param device: The device (CPU or GPU) to which the data should be moved.
:return: None
"""
def calculate_neighbor_num_max_min(
                dataset: UniDataset, 
                device: torch.device,
                num_workers:int=0) -> None:
    max_radial = -1e10
    min_radial = 1e10
    max_angular = -1e10
    min_angular = 1e10
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=variable_length_collate_fn,
        num_workers=num_workers,
    )
    
    for _, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating max neighbors"):
        sample = {key: value.to(device) for key, value in sample.items()}
        nn_radial, nn_angular = CalcOps.calculate_maxneigh(
            sample["num_atom"],
            sample["box"],
            sample["box_original"],
            sample["num_cell"],
            sample["position"],
            dataset.cutoff_radial,
            dataset.cutoff_angular,
            len(dataset.atom_types),
            sample["atom_type_map"],
            False
        )
        # print(_, torch.sum(nn_radial))
        max_radial = max(max_radial, nn_radial.max().item())
        min_radial = min(min_radial, nn_radial.min().item())
        max_angular = max(max_angular, nn_angular.max().item())
        min_angular = min(min_angular, nn_angular.min().item())
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("the max neighbor of train data: {} {}".format(max_radial, max_angular))
    return max_radial, min_radial, max_angular, min_angular

def calculate_neighbor_scaler(
                dataset: UniDataset,
                max_NN_radial,
                max_NN_angular,
                n_max_radial,
                basis_size_radial,
                n_max_angular,
                basis_size_angular,
                lmax_3,
                lmax_4,
                lmax_5,
                device: torch.device,
                num_workers:int=0):
    t1 = time.time()
    batch_size_num = calculate_batch(dataset.max_atom_nums, max_NN_radial)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_num,
        shuffle=False,
        collate_fn=variable_length_collate_fn,
        num_workers=num_workers,
    )

    dtype = dataset.dtype
    weight_radial = 1 * torch.ones(
        len(dataset.atom_types),
        len(dataset.atom_types),
        n_max_radial + 1,
        basis_size_radial + 1,
        dtype=dtype,
        device=device,
    )

    weight_angular = 1 * torch.ones(
        len(dataset.atom_types),
        len(dataset.atom_types),
        n_max_angular + 1,
        basis_size_angular + 1,
        dtype=dtype,
        device=device,
    )
    
    # check_cuda_memory(-1, -1, "before calculate_neighbor_scaler ", empty=False)
    FFAtomType = torch.from_numpy(np.array(dataset.atom_types)).to(device=device, dtype=dataset.index_type)
    # tmp_desc = []
    first_batch = True
    global_max = None
    global_min = None    
    for _, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating scaler"):
        sample = {key: value.to(device) for key, value in sample.items()}
        NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular = \
            CalcOps.calculate_neighbor(
            sample["num_atom"],
            sample["atom_type_map"],
            FFAtomType-1,
            sample["box"],
            sample["box_original"],
            sample["num_cell"],
            sample["position"],
            dataset.cutoff_radial,
            dataset.cutoff_angular,
            max_NN_radial,
            max_NN_angular,
            False # with_rij
        )

        # check_cuda_memory(_, 0, "after calculate_neighbor",empty=False)
        descriptor = CalcOps.calculate_descriptor(
            weight_radial,
            weight_angular,
            Ri_radial,
            NL_radial,
            sample["atom_type_map"],
            dataset.cutoff_radial,
            dataset.cutoff_angular,
            max_NN_radial,
            lmax_3,
            lmax_4,
            lmax_5
        )[0]
        if first_batch:
            global_max = descriptor.amax(dim=0)
            global_min = descriptor.amin(dim=0)
            first_batch = False
        else:
            global_max = torch.maximum(global_max, descriptor.amax(dim=0))
            global_min = torch.minimum(global_min, descriptor.amin(dim=0))

        # check_cuda_memory(_, 1, "in calculate_scaler ",empty=False)
        del descriptor, NN_radial, NN_angular, NL_radial, NL_angular, Ri_radial, Ri_angular

        torch.cuda.empty_cache()

        # tmp_desc.append(descriptor.detach())
        # check_cuda_memory(_, 2, "after calculate_scaler ",empty=False)

    # desc = torch.concat(tmp_desc, dim=0)
    # qscaler_radial = 1.0 / (desc.amax(dim=0) - desc.amin(dim=0))
    qscaler_radial = 1.0 / (global_max - global_min)
    qscaler = qscaler_radial.tolist()
    t2 = time.time()
    # print("scaler time {}", t2 - t1)
    # qscaler = []
    # qscaler.extend(qscaler_radial.tolist()) 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return qscaler

def calculate_batch(N, MN):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        memory_total_b = torch.cuda.get_device_properties(device).total_memory
        all = 0
        batch = 0
        while all < memory_total_b:
            batch += 1
            all += (2 * (N + N * MN + N * MN * 4) + N * 4 + N * 100) * 8 # (NN + NL + Rij + po) * 2 + N * 3 
    else:
        batch = 512 # for cpu
    return batch

def main():
    input_json = "/data/home/wuxingxing/codespace/PWMLFF_nep/pwmat_mlff_workdir/hfo2/nep_train/train.json"
    nep_param = InputParam(input_json, "train")
    #     print("Read Config successful")
    # import ipdb; ipdb.set_trace()
    # load_type, nep_param:InputParam, energy_shift, max_atom_nums
    energy_shift = [-6.052877426125001, -12.10575485225]
    max_atom_nums = 96
    dataset = UniDataset("train", nep_param, energy_shift, max_atom_nums)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    for i, sample_batches in enumerate(dataloader):
        # import ipdb;ipdb.set_trace()

        print(sample_batches["Force"].shape)
        print(sample_batches["Virial"].shape)
        print(sample_batches["Ei"].shape)
        print(sample_batches["energy"].shape)
        #print(sample_batches["Egroup"].shape)
        #print(sample_batches["Divider"].shape)
        #print(sample_batches["Egroup_weight"].shape)

        print(sample_batches["ListNeighbor"].shape)
        print(sample_batches["Position"].shape)
        # print(sample_batches["NeighborType"].shape)
        # print(sample_batches["Ri"].shape)
        # print(sample_batches["ImageDR"].shape)
        # print(sample_batches["Ri_d"].shape)
        print(sample_batches["ImageAtomNum"].shape)


if __name__ == "__main__":
    main()
