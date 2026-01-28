import numpy as np
import os
import math
from torch.utils.data import Dataset
import torch
from pwdata import Config
from pwdata.image import Image
# from src.feature.nep_find_neigh.findneigh import FindNeigh
import random
from typing import Union, Optional
from tqdm import tqdm

if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
    device = torch.cuda.current_device()
    memory_total = torch.cuda.get_device_properties(device).total_memory
    memory_total_gb = memory_total / (1024 ** 3)
    if memory_total_gb < 13:
        load_batch_size = 1024
    else:
        load_batch_size = 2048
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device
    load_batch_size = 1024

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

class UniDataset(Dataset):
    def __init__(self, 
                data_paths, 
                format, 
                atom_types, 
                cutoff_radial, 
                cutoff_angular,
                dtype: Union[torch.dtype, str] = torch.float64, 
                index_type: Union[torch.dtype, str] = torch.int64
                ):
        super(UniDataset, self).__init__()
        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.dirs = data_paths  # include all movement data path
        self.format = format
        self.image_list = []
        self.atom_types = atom_types
        self.cutoff_radial = cutoff_radial
        self.cutoff_angular = cutoff_angular
        self.max_NN_radial = 100
        self.max_NN_angular= 100

        self.dtype = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
        self.index_type = (
            index_type
            if isinstance(index_type, torch.dtype)
            else getattr(torch, index_type)
        )
        self.image_list, self.total_images = self.__concatenate_data()

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
            if image.cartesian is False:
                image._set_cartesian()
            if not hasattr(image, 'atom_type_map'):
                image.atom_type_map = np.array([self.atom_types.index(_) for _ in image.atom_types_image])

        return self.image_list, len(self.image_list)

    def __getitem__(self, index):
        data = self.__load_data(index)
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
        data["position"] = torch.from_numpy(self.image_list[index].position).to(self.dtype)
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

"""
Calculate the maximum and minimum neighbor numbers for radial and angular distributions.

:param dataloader: DataLoader object containing the dataset.
:param device: The device (CPU or GPU) to which the data should be moved.
:return: None
"""
def calculate_neighbor_num_max_min(
                dataset: UniDataset, 
                device: torch.device,
                with_type=False) -> None:

    max_radial = -1e10
    min_radial = 1e10
    max_angular = -1e10
    min_angular = 1e10
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=load_batch_size,
        shuffle=False,
        collate_fn=variable_length_collate_fn,
        num_workers=4,
    )
    
    for _, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Calculating max neighbor"):
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
            True
        )
        # print(_, torch.sum(nn_radial))
        if with_type:
            max_radial = max(max_radial, nn_radial.max().item())
            min_radial = min(min_radial, nn_radial.min().item())
            max_angular = max(max_angular, nn_angular.max().item())
            min_angular = min(min_angular, nn_angular.min().item())
        else:
            max_radial = max(max_radial, nn_radial.sum(dim=1).max().item())
            min_radial = min(min_radial, nn_radial.sum(dim=1).min().item())
            max_angular = max(max_angular, nn_angular.sum(dim=1).max().item())
            min_angular = min(min_angular, nn_angular.sum(dim=1).min().item())
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return max_radial, min_radial, max_angular, min_angular

def get_max_neighbor(
            data_paths:list[str],
            format:str,
            atom_types:list[int],
            cutoff_radial:float,
            cutoff_angular:float=4.0,
            with_type:bool=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dataset = UniDataset(
                data_paths=data_paths, 
                format = format, 
                atom_types = atom_types, 
                cutoff_radial=cutoff_radial,
                cutoff_angular=cutoff_angular
            )
    max_radial, min_radial, max_angular, min_angular = calculate_neighbor_num_max_min(dataset=dataset, device = device, with_type=with_type)
    return max_radial, min_radial, max_angular, min_angular, dataset
    
if __name__ == "__main__":
    data_paths = [
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag10Au14",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag12Au36",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag24Au24",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag28Au4",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag30Au18",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag3Au21",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag4Au20",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag7Au17",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag8Au8",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Au32",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag12Au12",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag22Au10",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag27Au21",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag29Au19",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag32Au16",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag45Au3",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag5Au43",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag8Au40",
        "/data/home/wuxingxing2/codespace/PWMLFF_grad_batch/example/Ag-Au-D3/PWdata_cpu/Ag9Au15"
    ]
    atom_types = [47, 79]
    format = "pwmlff/npy"

    data_paths= ["/data/home/wuxingxing2/data/pwmlff/Si-SiO2-La2O3-HfO2-TiN/batch_version/ADAM_novirial/nepxyz/train.xyz"]
    format = "extxyz"
    atom_types = [1, 14, 8, 13, 72, 57, 22, 7]
    
    cutoff_radial=6.0
    max_radial, min_radial, max_angular, min_angular = get_max_neighbor(
            data_paths,
            format,
            atom_types,
            cutoff_radial,
            with_type=False        
    )
    print(max_radial, min_radial, max_angular, min_angular)
