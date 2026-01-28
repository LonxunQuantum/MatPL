import numpy as np
import os
import math
from torch.utils.data import Dataset
import torch
import yaml
from src.user.input_param import InputParam
from pwdata import Config
from pwdata.image import Image
# from src.feature.nep_find_neigh.findneigh import FindNeigh
import random
from typing import Union, Optional
from src.lib.NeighConst import neighconst
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
    def extract_items(tensors, key, max_atom, value):
        return [pad_atom(x[key], max_atom, value) for x in tensors]

    def extract_items2(tensors, key):
        return [x[key] for x in tensors]

    def pad_atom(tensor, target_size, value):
        padding_size = target_size - tensor.size(0)
        if padding_size > 0:
            tensor = torch.cat([tensor, value*torch.ones(padding_size, *tensor.shape[1:])], dim=0)
        return tensor
        
    keys = batch[0].keys()
    res = {}
    max_atom = 0
    for _batch in batch:
        max_atom = max(max_atom, _batch["Position"].shape[0])
    # need to append with 0:
    # Force Ei Position ListNeighbor ImageDR 
    # if has Egroup_weight, Divider, Egroup
    # need to append AtomType with 0
    for key in keys:
        if key in ["Position", "Force", "Ei", "ListNeighbor", "ImageDR"]:
            res[key] = torch.stack(extract_items(batch, key, max_atom, 0.0), dim=0)
        # for egroup divider egroup_weight not realized
        elif key in ["AtomTypeMap"]:
            res[key] = torch.stack(extract_items(batch, key, max_atom, -1), dim=0)
        else:
            res[key] = torch.stack(extract_items2(batch, key), dim=0)
    # res["num_atom_sum"] = res["num_atom"].cumsum(0).to(res["num_atom"].dtype)
    return res

def variable_length_collate_fn_mn(batch):
    keys = batch[0].keys()
    res = {}

    def extract_items(tensors, key):
        return [x[key] for x in tensors]

    for key in keys:
        if key in ["Position", "AtomTypeMap"]:
            res[key] = torch.concat(extract_items(batch, key), dim=0)
        elif key in ["ImageAtomNum", "box", "Lattice", "num_cell"]:
            res[key] = torch.stack(extract_items(batch, key), dim=0)
    res["num_atom_sum"] = res["ImageAtomNum"].cumsum(0).to(res["ImageAtomNum"].dtype)
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
            # image.atom_types_image = np.array([self.atom_types.index(_) for _ in image.atom_types_image])
        # return image_list

class UniDataset(Dataset):
    def __init__(self, 
                config:dict,
                data_paths, 
                format, 
                dtype: Union[torch.dtype, str] = torch.float64, 
                index_type: Union[torch.dtype, str] = torch.int64,
                calculate_maxnn:bool=False
                ):
        super(UniDataset, self).__init__()
        self.m_neigh = config['maxNeighborNum']
        self.img_max_types = len(config["atomType"])
        self.atom_types = np.array([(_['type']) for _ in config["atomType"]])   # input atom type order
        self.atom_types_list = list(self.atom_types)
        self.Rc_M = config['Rc_M']
        self.Rc_type = np.array([(_['Rc']) for _ in config["atomType"]])
        self.Rm_type = np.array([(_['Rm']) for _ in config["atomType"]])
        self.Egroup = config['train_egroup']
        self.use_fractional = True
        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.dirs = data_paths  # include all movement data path
        self.format = format
        self.image_list = []
        self.config = config
        self.davg = []
        self.dstd = []
        self.energy_shift = []
        self.index_type = (
            index_type
            if isinstance(index_type, torch.dtype)
            else getattr(torch, index_type)
        )
        self.image_list, self.total_images = self.__concatenate_data()
        
        if calculate_maxnn:
            self.calculate_maxneighbor()
        # if self.total_images > 0:
        #     data = self.__load_data(0)
        # print()

    def __concatenate_data(self):
        if len(self.dirs) > 0:
            for config in self.dirs:
                image_read = Config(data_path=config, format=self.format).images
                if isinstance(image_read, list):
                    self.image_list.extend(image_read)
                else:
                    self.image_list.append(image_read)

        for image in self.image_list:
            # if image.cartesian is True:
            #         image._set_fractional()
            #         image.lattice = image.lattice.flatten()
            # if isinstance(image.atom_type.tolist(), int): # resconstructed the pwdata to ensure the shape of atom_type must be (N, )
            #     image.atom_type = image.atom_type.reshape([1])
            if not hasattr(image, 'atom_type_map'):
                image.atom_type_map = np.array([self.atom_types_list.index(_) for _ in image.atom_types_image])
        return self.image_list, len(self.image_list)

    def calculate_maxneighbor(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        maxneigh = calculate_neighbor_num_max_min(self, device)
        self.m_neigh = max(maxneigh, self.m_neigh)
        self.config['maxNeighborNum'] = self.m_neigh

    def get_davg_dstd(self):
        if len(self.davg) < 1:
            self.davg, self.dstd = self.set_davg_dstd()
        return self.davg, self.dstd

    def get_energy_shift(self):
        if len(self.energy_shift) < 1:
            self.energy_shift = self.set_energy_shift_2()
        return self.energy_shift

    def set_davg_dstd(self):
        # calculate davg and dstd
        searched_atom = set()
        repeat_num = 0
        davg = []
        dstd = []
        davg_dict = {}
        dstd_dict = {}
        ntypes = len(self.atom_types)
        for type in self.atom_types:
            davg_dict[type] = []
            dstd_dict[type] = []
        shuffled_list = random.sample(self.image_list, len(self.image_list))
        for image in shuffled_list:
            atom_types = image.atom_types_image
            cout_type, indices = np.unique(atom_types, return_index=True)
            sorted_indices = np.argsort(indices)
            cout_type = cout_type[sorted_indices]
            cout_num = np.bincount(atom_types)[cout_type]
            _davg, _dstd = calculate_davg_dstd(self.config, 
                                                    lattice=image.lattice, 
                                                    position = image.position, 
                                                    _atom_types = cout_type, 
                                                    input_atom_type = self.atom_types, 
                                                    type_maps = np.array(type_map(atom_types, self.atom_types))
                                                )
            for idx, _type in enumerate(cout_type):
                if _type not in searched_atom:
                    searched_atom.add(_type)
                    davg_dict[_type].append(np.tile(_davg[idx], self.config["maxNeighborNum"]*ntypes).reshape(-1,4))
                    dstd_dict[_type].append(np.tile(_dstd[idx], self.config["maxNeighborNum"]*ntypes).reshape(-1,4))
            if len(searched_atom) == len(self.atom_types):
                repeat_num += 1
                searched_atom.clear()
            if repeat_num > 10:
                break
        # for atom in self.atom_types:
        #     self.dstd.append(np.maximum.reduce(np.array(dstd_dict[atom])).reshape(ntypes, -1))
        #     self.davg.append(np.minimum.reduce(np.array(davg_dict[atom])).reshape(ntypes, -1))
        for atom in self.atom_types:
            dstd.append(np.mean(np.array(dstd_dict[atom]), axis=0).flatten())
            davg.append(np.mean(np.array(davg_dict[atom]), axis=0).flatten())
        return np.array(davg), np.array(dstd)

    def set_energy_shift_2(self):
        class_dict = {}
        for image in self.image_list:
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
        return np.array(energy_shift) # array([-7.85248589, -8.80939613, -5.12775169, -3.13588943])

    # def set_energy_shift(self):
    #     energy_dict = {}
    #     atom_type_searched = set()
    #     repeat_num = 0
    #     for atom in self.atom_types:
    #         energy_dict[atom] = []
    #     energy_dict['E'] = []
    #     shuffled_list = random.sample(self.image_list, len(self.image_list))
    #     for image in shuffled_list:
    #         atom_types = image.atom_types_image
    #         cout_type, cout_num = np.unique(atom_types, return_counts=True)
    #         atom_types_image_dict = dict(zip(cout_type, cout_num))
    #         for element in self.atom_types:
    #             if element in atom_types_image_dict.keys():
    #                 energy_dict[element].append(atom_types_image_dict[element])
    #             else:
    #                 energy_dict[element].append(0)
    #         energy_dict['E'].append(image.Ep)
    #         for element in atom_types_image_dict.keys():
    #             atom_type_searched.add(element)
    #         if len(atom_type_searched) == len(self.atom_types):
    #             repeat_num += 1
    #             atom_type_searched.clear()
    #         if repeat_num > 5:
    #             break
    #     _num_matrix = []
    #     for key in energy_dict.keys():
    #         if key != 'E':
    #             _num_matrix.append(energy_dict[key])
    #     x, residuals, rank, s = np.linalg.lstsq(np.array(_num_matrix).T, energy_dict['E'], rcond=None)
    #     energy_shift = x.tolist()
    #     return np.array(energy_shift)

    def __getitem__(self, index):
        if self.use_fractional:
            data = self.__load_data(index)
        else:
            data = self.__load_data_mn(index)
        # if self.train_hybrid is True:
        #     data = self.__completing_tensor_rows(data)
        return data
    
    def __len__(self): 
        return self.total_images

    def __load_data_mn(self, index):
        data = {}
        num_cell = np.zeros(3, dtype=int)
        box = np.zeros(18, dtype=float) 
        volume = self.expand_box(self.image_list[index].lattice.T.flatten(), self.Rc_type[0], num_cell, box)
        data["box"] = torch.from_numpy(box).to(self.dtype)
        data["box_original"] = torch.from_numpy(self.image_list[index].lattice.T.flatten()).to(self.dtype)
        data["num_cell"] = torch.from_numpy(num_cell).to(self.index_type)
        data["volume"] = torch.from_numpy(np.array([volume])).to(self.dtype)
        if self.image_list[index].cartesian is False:
            self.image_list[index]._set_cartesian()
        data["Position"] = torch.from_numpy(self.image_list[index].position).to(self.dtype)
        data["Lattice"] = torch.from_numpy(self.image_list[index].lattice.flatten()).to(self.dtype)

        atom_in = np.zeros_like(self.atom_types)
        for i in range(0, len(self.image_list[index].atom_type)):
            atom_in[i] = self.image_list[index].atom_type[i]
        data["AtomTypeMap"] = torch.from_numpy(self.image_list[index].atom_type_map).to(self.index_type)
        data["ImageAtomNum"] = torch.from_numpy(np.array([len(data["AtomTypeMap"])])).to(self.index_type)
        return data

    def __load_data(self, index):
        data = {}
        # num_cell = np.zeros(3, dtype=int)
        # box = np.zeros(18, dtype=float) 
        # volume = self.expand_box(self.image_list[index].lattice.flatten(), self.Rc_type[0], num_cell, box)
        # data["box"] = torch.from_numpy(box).to(self.dtype)
        # data["num_cell"] = torch.from_numpy(num_cell).to(self.index_type)
        # data["volume"] = torch.from_numpy(np.array([volume])).to(self.dtype)
        data["Force"] = torch.from_numpy(self.image_list[index].force).to(self.dtype)
        data["Ei"] = torch.from_numpy(self.image_list[index].atomic_energy).to(self.dtype)
        data["Etot"] = torch.from_numpy(np.array([self.image_list[index].Ep])).to(self.dtype)
        if self.image_list[index].cartesian is True:
            self.image_list[index]._set_fractional()
        data["Position"] = torch.from_numpy(self.image_list[index].position).to(self.dtype)
        data["Lattice"] = torch.from_numpy(self.image_list[index].lattice.flatten()).to(self.dtype)

        atom_in = np.zeros_like(self.atom_types)
        for i in range(0, len(self.image_list[index].atom_type)):
            atom_in[i] = self.image_list[index].atom_type[i]
        data["AtomType"] = torch.from_numpy(atom_in).to(self.index_type)
        data["AtomTypeMap"] = torch.from_numpy(self.image_list[index].atom_type_map).to(self.index_type)
        data["ImageAtomNum"] = torch.from_numpy(np.array([len(data["AtomTypeMap"])])).to(self.index_type)
        list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = \
            find_neighbore(self.image_list[index].atom_type_map, 
                self.image_list[index].position, 
                self.image_list[index].lattice, 
                self.image_list[index].position.shape[0], 
                self.image_list[index].atomic_energy, 
                self.img_max_types, 
                self.Rc_type, 
                self.Rm_type, 
                self.m_neigh, 
                self.Rc_M, 
                self.Egroup
                )
        data["ListNeighbor"] = torch.from_numpy(list_neigh).to(self.dtype)
        data["ImageDR"] = torch.from_numpy(dR_neigh).to(self.dtype)
        data["max_ri"] = max_ri
        if self.Egroup:
            data["Egroup_weight"] = torch.from_numpy(Egroup_weight).to(self.dtype)
            data["Divider"] = torch.from_numpy(Divider).to(self.dtype)
            data["Egroup"] = torch.from_numpy(Egroup).to(self.dtype)

        if self.image_list[index].virial is None:
            none_virial = -1e6 * np.ones([9])
            ones_column = np.insert(none_virial, 9, 0)
            data["Virial"] = torch.from_numpy(ones_column).to(self.dtype) 
        else:
            data["Virial"] = torch.from_numpy(np.insert(self.image_list[index].virial.flatten(), 9, 1)).to(self.dtype)
        return data

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


def calculate_davg_dstd(config, lattice, position, _atom_types, input_atom_type, type_maps):
    """
    Calculate the average and standard deviation of the pairwise distances between atoms.
    neighconst is a fortran module, which is used to calculate the pairwise distances between atoms.
    Args:
        config (dict): Configuration parameters.
        lattice (ndarray): Lattice vectors.
        position (ndarray): Atomic positions.
        chunk_size (int): Number of images in each chunk.
        _atom_types (list): List of atom types in the movement.
        input_atom_type (ndarray): Atom types in the input file.
        ntypes (int): Number of atom types.
        type_maps (ndarray): Mapping of atom types.
    Returns:
        tuple: A tuple containing the average (davg) and standard deviation (dstd) of the pairwise distances,
            as well as the number of atoms for each atom type (atom_types_nums).
    """
    ntypes = len(input_atom_type)
    Rc_m = config["Rc_M"]
    m_neigh = config["maxNeighborNum"]
    # input_atom_type_nums = []       # the number of each atom type in input_atom_type
    # for itype, iatom in enumerate(input_atom_type):
    #     input_atom_type_nums.append(np.sum(itype == type_maps))
    types, type_incides, atom_types_nums = np.unique(type_maps, return_index=True, return_counts=True)
    atom_types_nums = atom_types_nums[np.argsort(type_incides)]
    Rc_type = np.asfortranarray(np.array([(_['Rc']) for _ in config["atomType"]]))
    type_maps = np.asfortranarray(type_maps + 1)
    lattice = np.asfortranarray(lattice.reshape(1, 3, 3))
    position = np.asfortranarray(position.reshape(1, -1, 3))
    natoms = position.shape[1]
    neighconst.find_neighbore(1, lattice, position, ntypes, natoms, m_neigh, Rc_m, Rc_type, type_maps)
    _list_neigh = neighconst.list_neigh
    _dR_neigh = neighconst.dr_neigh
    list_neigh = np.transpose(_list_neigh, (3, 2, 1, 0))
    dR_neigh = np.transpose(_dR_neigh, (4, 3, 2, 1, 0))
    atom_type_list = []
    for atom in list(_atom_types):
        atom_type_list.append(list(input_atom_type).index(atom))
    atom_type_list = sorted(atom_type_list)
    # dR_neigh = dR_neigh[:, :, atom_type_list, :,:]
    # list_neigh = list_neigh[:,:,atom_type_list,:]
    davg, dstd = calc_stat(config, np.copy(dR_neigh[:, :, atom_type_list, :,:]), np.copy(list_neigh[:,:,atom_type_list,:]), m_neigh, natoms, len(atom_type_list), atom_types_nums)
    neighconst.dealloc()
    return davg, dstd

def calc_stat(config, dR_neigh, list_neigh, m_neigh, natoms, ntypes, atom_types_nums):
    def compute_std(sum2, sum, sumn):
        if sumn == 0:
            return 1e-2
        val = np.sqrt(sum2 / sumn - np.multiply(sum / sumn, sum / sumn))
        if np.abs(val) < 1e-2:
            val = 1e-2
        return val
    davg = []
    dstd = []
    image_dR = np.reshape(dR_neigh, (-1, natoms, ntypes * m_neigh, 3))
    list_neigh = np.reshape(list_neigh, (-1, natoms, ntypes * m_neigh))
    image_dR = torch.tensor(image_dR, dtype=torch.float64)
    list_neigh = torch.tensor(list_neigh, dtype=torch.int)

    mask = list_neigh > 0
    dR2 = torch.zeros_like(list_neigh, dtype=torch.float64)
    Rij = torch.zeros_like(list_neigh, dtype=torch.float64)
    dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
    Rij[mask] = torch.sqrt(dR2[mask])

    nr = torch.zeros_like(dR2)
    inr = torch.zeros_like(dR2)

    dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
    Ri_xyz = torch.zeros_like(dR2_copy)

    nr[mask] = dR2[mask] / Rij[mask]
    Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
    inr[mask] = 1 / Rij[mask]

    davg_tensor = torch.zeros((ntypes, m_neigh * ntypes, 4), dtype=torch.float64)
    dstd_tensor = torch.ones((ntypes, m_neigh * ntypes, 4), dtype=torch.float64)
    Ri, _, _ = smooth(
                    config,
                    image_dR, 
                    nr, 
                    Ri_xyz, 
                    mask, 
                    inr, 
                    davg_tensor, 
                    dstd_tensor, 
                    atom_types_nums)
    Ri2 = Ri * Ri
    atom_sum = 0
    for i in range(ntypes):
        Ri_ntype = Ri[:, atom_sum : atom_sum + atom_types_nums[i]].reshape(-1, 4)
        Ri2_ntype = Ri2[:, atom_sum : atom_sum + atom_types_nums[i]].reshape(-1, 4)
        sum_Ri = Ri_ntype.sum(axis=0).tolist()
        sum_Ri_r = sum_Ri[0]
        sum_Ri_a = np.average(sum_Ri[1:])
        sum_Ri2 = Ri2_ntype.sum(axis=0).tolist()
        sum_Ri2_r = sum_Ri2[0]
        sum_Ri2_a = np.average(sum_Ri2[1:])
        sum_n = Ri_ntype.shape[0]

        davg_unit = [sum_Ri[0] / (sum_n + 1e-15), 0, 0, 0]
        dstd_unit = [
            compute_std(sum_Ri2_r, sum_Ri_r, sum_n),
            compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
            compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
            compute_std(sum_Ri2_a, sum_Ri_a, sum_n),
        ]
            
        # davg.append(
        #     np.tile(davg_unit, m_neigh * ntypes).reshape(-1, 4)
        # )
        # dstd.append(
        #     np.tile(dstd_unit, m_neigh * ntypes).reshape(-1, 4)
        # )
        davg.append(davg_unit)
        dstd.append(dstd_unit)

        atom_sum = atom_sum + atom_types_nums[i]

    # davg = np.array(davg).reshape(ntypes, -1)
    # dstd = np.array(dstd).reshape(ntypes, -1)
    return davg, dstd

def smooth(config, image_dR, x, Ri_xyz, mask, inr, davg, dstd, atom_types_nums):

    batch_size = image_dR.shape[0]
    ntypes = len(atom_types_nums)

    uu = torch.zeros_like(x)
    vv = torch.zeros_like(x)
    # dvv = torch.zeros_like(x)

    res = torch.zeros_like(x)

    # x < rcut_min vv = 1;
    mask_min = x < config["atomType"][0]["Rm"]
    mask_1 = mask & mask_min  # [2,108,100]
    vv[mask_1] = 1
    # dvv[mask_1] = 0

    # rcut_min< x < rcut_max;
    mask_max = x < config["atomType"][0]["Rc"]
    mask_2 = ~mask_min & mask_max & mask
    # uu = (xx - rmin) / (rmax - rmin);
    uu[mask_2] = (x[mask_2] - config["atomType"][0]["Rm"]) / (
        config["atomType"][0]["Rc"] - config["atomType"][0]["Rm"]
    )
    vv[mask_2] = (
        uu[mask_2]
        * uu[mask_2]
        * uu[mask_2]
        * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10)
        + 1
    )

    mask_3 = ~mask_max & mask
    vv[mask_3] = 0
    # dvv[mask_3] = 0

    res[mask] = 1.0 / x[mask]
    Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)
    
    vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
    Ri[mask] *= vv_copy[mask]

    davg_res, dstd_res = None, None
    # 0 is that the atom nums is zero, for example, CH4 system in CHO system hybrid training, O atom nums is zero.\
    # beacuse the dstd or davg does not contain O atom, therefore, special treatment is needed here for atoms with 0 elements
    # atom_types_nums = [_ for _ in atom_types_nums if _ != 0]
    # ntypes = len(atom_types_nums)
    for ntype in range(ntypes):
        atom_num_ntype = atom_types_nums[ntype]
        davg_ntype = (
            davg[ntype].reshape(-1, 4).repeat(batch_size, atom_num_ntype, 1, 1)
        )  # [32,100,4]
        dstd_ntype = (
            dstd[ntype].reshape(-1, 4).repeat(batch_size, atom_num_ntype, 1, 1)
        )  # [32,100,4]
        davg_res = davg_ntype if davg_res is None else torch.concat((davg_res, davg_ntype), dim=1)
        dstd_res = dstd_ntype if dstd_res is None else torch.concat((dstd_res, dstd_ntype), dim=1)

    max_ri = torch.max(Ri[:,:,:,0])
    Ri = (Ri - davg_res) / dstd_res
    # dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
    # Ri_d = Ri_d / dstd_res
    return Ri, None, max_ri
        

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

def find_neighbore(AtomTypeMap, Position, Lattice, ImageAtomNum, Ei, 
                   img_max_types, Rc_type, Rm_type, m_neigh, Rc_M, train_egroup):
    """
    Call the Fortran subroutine that finds the neighbors for each atom in the system.

    Args:
        AtomTypeMap (numpy.ndarray): List of atom types to index.
        Position (numpy.ndarray): List of atomic positions.
        Lattice (numpy.ndarray): List of lattice vectors.
        ImageAtomNum (int): The number of atoms in the system.
        Ei (numpy.ndarray): List of atomic energies.
        img_max_types (int): The maximum number of atom types in the system.
        Rc_type (numpy.ndarray): List of cutoff radii for each atom type.
        Rm_type (numpy.ndarray): List of minimum cutoff radii for each atom type.
        m_neigh (int): The maximum number of neighbors for each atom.
        Rc_M (float): The maximum cutoff radius for the system.
        train_egroup (bool): Whether to train the energy group.

    Returns:
        tuple: A tuple containing list_neigh, ImageDR, and max_ri.
            - list_neigh (numpy.ndarray): The list of neighbors.
            - ImageDR (numpy.ndarray): The displacement vectors for each neighbor.
            - max_ri (float): The maximum value of Ri.
    """
    images = 1
    ntypes = img_max_types
    natoms = ImageAtomNum
    Rc_type = np.asfortranarray(Rc_type)
    type_maps = np.asfortranarray(AtomTypeMap[:natoms] + 1)
    lattice = np.asfortranarray(Lattice.reshape(-1, 3, 3))
    position = np.asfortranarray(Position.reshape(1, -1, 3))

    neighconst.find_neighbore(images, lattice, position, ntypes, natoms, 
                                m_neigh, Rc_M, Rc_type, type_maps)
    _list_neigh = neighconst.list_neigh
    _dR_neigh = neighconst.dr_neigh
    list_neigh = np.transpose(_list_neigh.copy(), (3, 2, 1, 0)).reshape(images, natoms, ntypes*m_neigh)
    dR_neigh = np.transpose(_dR_neigh.copy(), (4, 3, 2, 1, 0)).reshape(images, natoms, ntypes*m_neigh, 3)
    if train_egroup:
        Ei = np.asfortranarray(Ei[:natoms].reshape(images, natoms))
        neighconst.calc_egroup(images, lattice, position, natoms, Rc_M, type_maps, Ei)
        _Egroup_weight = neighconst.fact
        _Divider = neighconst.divider
        _Egroup = neighconst.energy_group
        Egroup_weight = np.transpose(_Egroup_weight.copy(), (2, 1, 0)).squeeze(0)
        Divider = np.transpose(_Divider.copy(), (1, 0)).squeeze(0)
        Egroup = np.transpose(_Egroup.copy(), (1, 0)).squeeze(0)
    else:
        Egroup_weight = None
        Divider = None
        Egroup = None
    neighconst.dealloc()
    
    max_ri, Rij = compute_Ri(list_neigh, dR_neigh, Rc_type, Rm_type)
    ImageDR = np.concatenate((Rij, dR_neigh), axis=-1)
    return list_neigh.squeeze(0), ImageDR.squeeze(0), max_ri, Egroup_weight, Divider, Egroup

def compute_Ri(list_neigh, dR_neigh, Rc_type, Rm_type):
    """
    Compute the Ri values for a given list of neighbors and their displacement vectors.

    Args:
        list_neigh (list): List of neighbor indices.
        dR_neigh (list): List of displacement vectors for each neighbor.

    Returns:
        tuple: A tuple containing max_ri, and Rij.
            - max_ri (torch.Tensor): The maximum value of Ri.
            - Rij (numpy.ndarray): The squared root of the sum of the squared displacement vectors.
    """
    device = torch.device("cpu")
    image_dR = torch.tensor(dR_neigh, device=device, dtype=torch.float64)
    list_neigh = torch.tensor(list_neigh, device=device, dtype=torch.int64)

    mask = list_neigh > 0
    dR2 = torch.zeros_like(list_neigh, dtype=torch.float64)
    Rij = torch.zeros_like(list_neigh, dtype=torch.float64)
    dR2[mask] = torch.sum(image_dR[mask] * image_dR[mask], -1)
    Rij[mask] = torch.sqrt(dR2[mask])

    nr = torch.zeros_like(dR2)
    inr = torch.zeros_like(dR2)

    dR2_copy = dR2.unsqueeze(-1).repeat(1, 1, 1, 3)
    Ri_xyz = torch.zeros_like(dR2_copy)

    nr[mask] = dR2[mask] / Rij[mask]
    Ri_xyz[mask] = image_dR[mask] / dR2_copy[mask]
    inr[mask] = 1 / Rij[mask]

    uu = torch.zeros_like(nr)
    vv = torch.zeros_like(nr)
    res = torch.zeros_like(nr)

    # x < rcut_min vv = 1;
    mask_min = nr < Rm_type[0] # why just use the first element of Rm_type?
    mask_1 = mask & mask_min
    vv[mask_1] = 1

    # rcut_min< x < rcut_max;
    mask_max = nr < Rc_type[0]
    mask_2 = ~mask_min & mask_max & mask
    # uu = (xx - rmin) / (rmax - rmin);
    uu[mask_2] = (nr[mask_2] - Rm_type[0]) / (Rc_type[0] - Rm_type[0])
    vv[mask_2] = (
        uu[mask_2]
        * uu[mask_2]
        * uu[mask_2]
        * (-6 * uu[mask_2] * uu[mask_2] + 15 * uu[mask_2] - 10)
        + 1
    )
    mask_3 = ~mask_max & mask
    vv[mask_3] = 0

    res[mask] = 1.0 / nr[mask] 
    Ri = torch.cat((res.unsqueeze(-1), Ri_xyz), dim=-1)

    vv_copy = vv.unsqueeze(-1).repeat(1, 1, 1, 4)
    Ri[mask] *= vv_copy[mask]
    max_ri = torch.max(Ri[:,:,:,0])
    Rij = Rij.unsqueeze(-1).numpy()
    return max_ri, Rij

def calculate_neighbor_num_max_min(
                dataset: UniDataset, 
                device: torch.device) -> None:
    if dataset.total_images == 0:
        return -1
    dataset.use_fractional = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1024,
        shuffle=False,
        collate_fn=variable_length_collate_fn_mn,
        num_workers=4,
    )
    global_max = None
    global_min = None

    for _, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating max neighbor"):
        sample = {key: value.to(device) for key, value in sample.items()}
        nn_radial, nn_angular = CalcOps.calculate_maxneigh(
            sample["ImageAtomNum"],
            sample["box"],
            sample["Lattice"],
            sample["num_cell"],
            sample["Position"],
            dataset.Rc_type[0],
            dataset.Rc_type[0],
            len(dataset.atom_types_list),
            sample["AtomTypeMap"],
            True
        )

        batch_max, _ = torch.max(nn_radial, dim=0)
        batch_min, _ = torch.min(nn_radial, dim=0)

        if global_max is None:
            global_max = batch_max
        else:
            global_max = torch.max(global_max, batch_max)

        if global_min is None:
            global_min = batch_min
        else:
            global_min = torch.min(global_min, batch_min)
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dataset.use_fractional = True
    return global_max.max().item()

# def main():
#     input_json = "/data/home/wuxingxing2/data/pwmlff/Si-SiO2-La2O3-HfO2-TiN/batch_version/ADAM_novirial/test2/input.json"
#     nep_param = InputParam(input_json, "train")
#     #     print("Read Config successful")
#     # import ipdb; ipdb.set_trace()
#     # load_type, nep_param:InputParam, energy_shift, max_atom_nums
#     energy_shift = [-6.052877426125001, -12.10575485225]
#     max_atom_nums = 96
#     dataset = UniDataset("train", nep_param, energy_shift, max_atom_nums)

#     dataloader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=4,
#         shuffle=False,
#         num_workers=4,
#         pin_memory=True,
#     )
    
#     for i, sample_batches in enumerate(dataloader):
#         # import ipdb;ipdb.set_trace()

#         print(sample_batches["Force"].shape)
#         print(sample_batches["Virial"].shape)
#         print(sample_batches["Ei"].shape)
#         print(sample_batches["energy"].shape)
#         #print(sample_batches["Egroup"].shape)
#         #print(sample_batches["Divider"].shape)
#         #print(sample_batches["Egroup_weight"].shape)

#         print(sample_batches["ListNeighbor"].shape)
#         print(sample_batches["Position"].shape)
#         # print(sample_batches["NeighborType"].shape)
#         # print(sample_batches["Ri"].shape)
#         # print(sample_batches["ImageDR"].shape)
#         # print(sample_batches["Ri_d"].shape)
#         print(sample_batches["ImageAtomNum"].shape)


# if __name__ == "__main__":
#     main()
