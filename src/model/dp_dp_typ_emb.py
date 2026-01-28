import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
from typing import List, Tuple, Optional, Dict
sys.path.append(os.getcwd())

from src.model.dp_embedding_typ_emb import EmbeddingNet, FittingNet
# from src.model.calculate_force import CalculateCompress, CalculateForce, CalculateVirialForce
from utils.atom_type_emb_dict import get_normalized_data_list

if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device
'''
# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.DPFF')

def dump(msg, *args, **kwargs):
    logger.log(logging_level_DUMP, msg, *args, **kwargs)
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)
def summary(msg, *args, **kwargs):
    logger.log(logging_level_SUMMARY, msg, *args, **kwargs)
def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)
def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)
def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs, exc_info=True)
'''
class TypeDP(nn.Module):
    def __init__(self, config, davg, dstd, energy_shift, magic=False):
        super(TypeDP, self).__init__()
        self.config = config
        self.model_type = config['model_type']
        self.ntypes = len(config['atomType'])
        self.atom_type = [_['type'] for _ in config['atomType']] #this value in used in forward for hybrid Training
        self.M2 = config["M2"]
        self.Rmax = config["Rc_M"]
        self.Rmin = config['atomType'][0]['Rm']
        self.maxNeighborNum = config["maxNeighborNum"]
        self.physical_property = config["net_cfg"]["type_embedding_net"]["physical_property"]
        if self.config["training_type"] == "float64":
            self.dtype = torch.double
        elif self.config["training_type"] == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")
        self.davg = torch.tensor(davg, dtype=self.dtype)
        self.dstd = torch.tensor(dstd, dtype=self.dtype)
        # self.energy_shift = torch.tensor(energy_shift, dtype=self.dtype)

        self.embedding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()
        
        # initial bias for fitting net? 
        # initial type embedding net
        if len(self.config["net_cfg"]["type_embedding_net"]["network_size"]) > 0:
            # self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["type_embedding_net"], type_feat_num=None, is_type_emb=True))
            self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["type_embedding_net"]["network_size"], 
                                                   self.config["net_cfg"]["type_embedding_net"]["bias"], 
                                                   self.config["net_cfg"]["type_embedding_net"]["resnet_dt"],
                                                   self.config["net_cfg"]["type_embedding_net"]["activation"],
                                                   type_feat_num = None, is_type_emb = True))
            type_feat_num = self.config["net_cfg"]["type_embedding_net"]["network_size"][-1]
        else:
            type_feat_num = len(self.config["net_cfg"]["type_embedding_net"]["physical_property"])
        # initial embedding net
        # self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"], 
        #                                        type_feat_num= type_feat_num,# if type_emb_net exists, type_feat_num is last layer of net work, otherwise, is type num of physical_property
        #                                        is_type_emb=False))
        self.embedding_net.append(EmbeddingNet(self.config["net_cfg"]["embedding_net"]["network_size"], 
                                                   self.config["net_cfg"]["embedding_net"]["bias"], 
                                                   self.config["net_cfg"]["embedding_net"]["resnet_dt"],
                                                   self.config["net_cfg"]["embedding_net"]["activation"],
                                                   type_feat_num = type_feat_num,   # if type_emb_net exists, type_feat_num is last layer of net work, otherwise, is type num of physical_property
                                                   is_type_emb = False))
        # initial fitting net
        fitting_net_input_dim = self.config["net_cfg"]["embedding_net"]["network_size"][-1]
        for i in range(self.ntypes):
            # self.fitting_net.append(FittingNet(config["net_cfg"]["fitting_net"], self.M2 * fitting_net_input_dim, self.energy_shift[i], magic))
            self.fitting_net.append(FittingNet(self.config["net_cfg"]["fitting_net"]["network_size"],
                                               self.config["net_cfg"]["fitting_net"]["bias"],
                                               self.config["net_cfg"]["fitting_net"]["resnet_dt"],
                                               self.config["net_cfg"]["fitting_net"]["activation"],
                                               self.M2 * fitting_net_input_dim, energy_shift[i], magic))

        self.compress_tab = None #for dp compress

        # set type embedding physical infos
        self.type_vector = get_normalized_data_list(self.atom_type, self.physical_property)
        # torch.tensor(compress_dict["table"], dtype=self.dtype)
        
    def get_egroup(self,
                   Ei: torch.Tensor,
                   Egroup_weight: Optional[torch.Tensor] = None,
                   divider: Optional[torch.Tensor] = None)-> Optional[torch.Tensor]:
        # commit by wuxing and replace by the under line code
        # batch_size = Ei.shape[0]
        # Egroup = torch.zeros_like(Ei)

        # for i in range(batch_size):
        #     Etot1 = Ei[i]
        #     weight_inner = Egroup_weight[i]
        #     E_inner = torch.matmul(weight_inner, Etot1)
        #     Egroup[i] = E_inner
        if Egroup_weight is not None and divider is not None:       # Egroup_out is not defined in the false branch:
            Egroup = torch.matmul(Egroup_weight, Ei)
            Egroup_out = torch.divide(Egroup.squeeze(-1), divider)
        else:
            Egroup_out = None
        
        return Egroup_out

    def set_comp_tab(self, compress_dict:dict):
        self.compress_tab = torch.tensor(compress_dict["table"], dtype=self.dtype)
        self.compress_tab_shape = self.compress_tab.shape
        self.compress_tab = self.compress_tab.reshape(self.compress_tab_shape[0]*self.compress_tab_shape[1],
                                                      self.compress_tab_shape[2],
                                                      self.compress_tab_shape[3])
        self.dx = compress_dict["dx"]
        # self.davg = compress_dict["davg"]
        # self.dstd = compress_dict["dstd"]
        self.sij_min = compress_dict["sij_min"]
        self.sij_max = compress_dict["sij_max"]
        self.sij_out_max = compress_dict["sij_out_max"]
        self.sij_len = compress_dict["sij_len"]
        self.sij_out_len = compress_dict["sij_out_len"]
        self.order = compress_dict["order"] if "order" in compress_dict.keys() else 5 #the default compress order is 5
        # self.type_vector = compress_dict["type_vector"] if "type_vector" in compress_dict.keys() else None #the default compress order is 5

    '''
    description: 
    return the embeding net index list and type nums of the image
    for example: 
        when the user input atom_type is [3, 14]:
            the atom_type_data is [14, 3], the index of user atom_type is [2, 1], then return:
                [[[1, 1], [1, 0]], [[0, 1], [0, 0]]], 2

            the atom_type_data is [14, 0], the index of user atom_type is [2, 1], then return:
                [[[1, 1]]], 1
            
        attention: 1. '0' is used in hybrid multi-batch training for completing tensor dimensions
                    2. in this user atom_type [3, 14]: the [1, 1] is the Si-Si embeding net index, [0, 1] is the Li-Si embeding net index
    
    param {*} self
    param {*} atom_type_data: the atom type list of image from dataloader
    return {*}
    author: wuxingxing
    '''
    def get_index(self, user_input_order: List[int], key:torch.Tensor):
        for i, v in enumerate(user_input_order):
            if v == key:
                return i
        return -1
    def get_train_2body_type(self, type_map: torch.Tensor) -> Tuple[List[List[List[int]]], int]:
        type_2body_list: List[List[List[int]]] = []         
        type_2body_index: List[int] = []
        for i, atom in enumerate(type_map):
            # if self.atom_type.eq(atom).any():
            #     type_2body_index.append(i)
            index = self.get_index(self.atom_type, atom)
            if index != -1:
                type_2body_index.append(self.get_index(self.atom_type, atom))
        for atom in type_2body_index:
            type_2body: List[List[int]] = []        # 整数列表的列表
            for atom2 in type_2body_index:
                type_2body.append([atom, atom2])        # 整数列表
            type_2body_list.append(type_2body)
        pair_indices = len(type_2body_index)
        return type_2body_list, pair_indices

    def forward(self, 
                list_neigh: torch.Tensor, 
                Imagetype_map: torch.Tensor, 
                type_map: torch.Tensor, 
                ImageDR: torch.Tensor, 
                nghost: int, 
                Egroup_weight: Optional[torch.Tensor] = None,
                divider: Optional[torch.Tensor] = None, 
                is_calc_f: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            list_neigh (torch.Tensor): Tensor representing the neighbor list. Shape: (batch_size, natoms_sum, max_neighbor * ntypes).
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.. Shape: (natoms_sum).
            type_map (torch.Tensor): Tensor representing the image's atom types. Shape: (ntypes).
            ImageDR (torch.Tensor): Tensor representing the image DRneigh. Shape: (batch_size, natoms_sum, max_neighbor * ntypes, 4).
            nghost (int): Number of ghost atoms.
            Egroup_weight (Optional[torch.Tensor], optional): Tensor representing the Egroup weight. Defaults to None.
            divider (Optional[torch.Tensor], optional): Tensor representing the divider. Defaults to None.
            is_calc_f (Optional[bool], optional): Flag indicating whether to calculate forces and virial. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: Tuple containing the total energy (Etot), atomic energies (Ei), forces (Force), energy group (Egroup), and virial (Virial).
        """
        device = ImageDR.device
        dtype = ImageDR.dtype
        batch_size = list_neigh.shape[0]
        natoms_sum = list_neigh.shape[1]
        max_neighbor_type = list_neigh.shape[2]  # ntype * max_neighbor_num
        emb_list, type_nums =  self.get_train_2body_type(type_map)
        Ri, Ri_d = self.calculate_Ri(natoms_sum, batch_size, max_neighbor_type, Imagetype_map, ImageDR, device, dtype)
        Ri.requires_grad_()
        Ei = self.calculate_Ei(Imagetype_map, Ri, batch_size, emb_list, type_nums, device, dtype)
        assert Ei is not None
        Etot = torch.sum(Ei, 1)  
        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        Ei = torch.squeeze(Ei, 2)
        if is_calc_f is False:
            Force, Virial = None, None
        else:
            Force, Virial = self.calculate_force_virial(Ri, Ri_d, Etot, natoms_sum, batch_size, list_neigh, ImageDR, nghost, device, dtype)
        return Etot, Ei, Force, Egroup, Virial

    
    def calculate_Ri(self,
                     natoms_sum: int,
                     batch_size: int,
                     max_neighbor_type: int,
                     Imagetype_map: torch.Tensor,
                     ImagedR: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the Ri and dfeat tensors.

        Args:
            natoms_sum (int): The total number of atoms.
            batch_size (int): The batch size.
            max_neighbor_type (int): The maximum number of neighbor types.
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.
            ImagedR (torch.Tensor): The tensor containing the atom's distances and Δ(position vectors).
            device (torch.device): The device to perform the calculations on.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The Ri and dfeat tensors.
        """
        # t1 = time.time()
        R = ImagedR[:, :, :, 0]
        R.requires_grad_()

        Srij = torch.zeros(batch_size, natoms_sum, max_neighbor_type, dtype=dtype, device=device)
        Srij[(R > 0) & (R < self.Rmin)] = 1 / R[(R > 0) & (R < self.Rmin)]
        r = R[(R > self.Rmin) & (R < self.Rmax)]
        Srij[(R > self.Rmin) & (R < self.Rmax)] = 1 / r * (((r - self.Rmin) / (self.Rmax - self.Rmin)) ** 3 * (
                -6 * ((r - self.Rmin) / (self.Rmax - self.Rmin)) ** 2 + 15 * (
                   (r - self.Rmin) / (self.Rmax - self.Rmin)) - 10) + 1)
        mask = R.abs() > 1e-5

        # Ri[batch_size, natoms_sum, max_neighbor * ntypes, 4] 4-->[Srij, Srij * xij / rij, Srij * yij / rij, Srij * yij / rij]
        Ri = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, dtype=dtype, device=device)
        Ri[:, :, :, 0] = Srij
        Ri[:, :, :, 1][mask] = Srij[mask] * ImagedR[:, :, :, 1][mask] / ImagedR[:, :, :, 0][mask]
        Ri[:, :, :, 2][mask] = Srij[mask] * ImagedR[:, :, :, 2][mask] / ImagedR[:, :, :, 0][mask]
        Ri[:, :, :, 3][mask] = Srij[mask] * ImagedR[:, :, :, 3][mask] / ImagedR[:, :, :, 0][mask]
        # t2 = time.time()
        # d_Srij[batch_size, natoms_sum, max_neighbor * ntypes]  d_Srij = dSrij/drij
        mask_Srij: List[Optional[torch.Tensor]] = [torch.ones_like(R)]
        d_Srij = torch.autograd.grad([Srij], [R], grad_outputs=mask_Srij, retain_graph=True, create_graph=True)[0]
        assert d_Srij is not None
        # feat [batch_size, natoms_sum, max_neighbor * ntypes, 4]
        # dfeat [batch_size, natoms_sum, max_neighbor * ntypes, 4, 3] 3-->[dx, dy, dz]
        # feat = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, dtype=dtype, device=device)
        dfeat = torch.zeros(batch_size, natoms_sum, max_neighbor_type, 4, 3, dtype=dtype, device=device)
        rij = ImagedR[:, :, :, 0][mask]
        xij = ImagedR[:, :, :, 1][mask]
        yij = ImagedR[:, :, :, 2][mask]
        zij = ImagedR[:, :, :, 3][mask]
        Srij_temp = Srij[mask]
        common = (d_Srij[mask] - Srij_temp / rij) / rij ** 2
        # dSrij / dxij = d_Srij * xij / rij (x-->x,y,z)
        # d(Srij * xij / rij) / dxij = xij * xij * (d_Srij - Srij / rij) / rij ** 2 + Srij / rij * ( dxij / dxij)
        dfeat[:, :, :, 0, 0][mask] = d_Srij[mask] * xij / rij
        dfeat[:, :, :, 1, 0][mask] = common * xij * xij + Srij_temp / rij
        dfeat[:, :, :, 2, 0][mask] = common * yij * xij
        dfeat[:, :, :, 3, 0][mask] = common * zij * xij
        # dy
        dfeat[:, :, :, 0, 1][mask] = d_Srij[mask] * yij / rij
        dfeat[:, :, :, 1, 1][mask] = common * xij * yij
        dfeat[:, :, :, 2, 1][mask] = common * yij * yij + Srij_temp / rij
        dfeat[:, :, :, 3, 1][mask] = common * zij * yij
        # dz
        dfeat[:, :, :, 0, 2][mask] = d_Srij[mask] * zij / rij
        dfeat[:, :, :, 1, 2][mask] = common * xij * zij
        dfeat[:, :, :, 2, 2][mask] = common * yij * zij
        dfeat[:, :, :, 3, 2][mask] = common * zij * zij + Srij_temp / rij

        # t3 = time.time()
        # 0 is that the atom nums is zero, for example, CH4 system in CHO system hybrid training, O atom nums is zero.\
        # beacuse the dstd or davg does not contain O atom, therefore, special treatment is needed here for atoms with 0 elements
        davg_res = self.davg.to(device)[Imagetype_map]
        dstd_res = self.dstd.to(device)[Imagetype_map]
        davg_res = davg_res.reshape(-1, natoms_sum, max_neighbor_type, 4).repeat(batch_size, 1, 1, 1)
        dstd_res = dstd_res.reshape(-1, natoms_sum, max_neighbor_type, 4).repeat(batch_size, 1, 1, 1) 
        Ri = (Ri - davg_res) / dstd_res
        dstd_res = dstd_res.unsqueeze(-1).repeat(1, 1, 1, 1, 3)
        dfeat = - dfeat / dstd_res
        # t4 = time.time()
        # print("\nRi:", t2-t1, "\ndfeat:", t3-t2, "\ndavg_res:", t4-t3, "\n********************")
        return Ri, dfeat
    
    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     Ri: torch.Tensor,
                     batch_size: int,
                     emb_list: List[List[List[int]]],
                     type_nums: int, 
                     device: torch.device, 
                     dtype: torch.dtype) -> Optional[torch.Tensor]:
        """
        Calculate the energy Ei for each type of atom in the system.

        Args:
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.
            Ri (torch.Tensor): A tensor representing the atomic descriptors.
            batch_size (int): The size of the batch.
            emb_list (List[List[List[int]]]): A list of embedded atom types.
            type_nums (int): The number of atom types.
            device (torch.device): The device to perform the calculations on.

        Returns:
            Optional[torch.Tensor]: The calculated energy Ei for each type of atom, or None if the calculation fails.
        """
        Ei : Optional[torch.Tensor] = None
        fit_net_dict = {idx: fit_net for idx, fit_net in enumerate(self.fitting_net)}
        for type_emb in emb_list:
            # t1 = time.time()
            xyz_scater_a, xyz_scater_b = self.calculate_xyz_scater(Imagetype_map, Ri, type_emb, type_nums, device, dtype)
            if xyz_scater_a.any() == 0:
                continue
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(batch_size, xyz_scater_a.shape[1], -1)
            # t2 = time.time()
            Ei_ntype: Optional[torch.Tensor] = None
            fit_net = fit_net_dict.get(type_emb[0][0]) 
            assert fit_net is not None
            Ei_ntype = fit_net.forward(DR_ntype)
            # t3 = time.time()
            assert Ei_ntype is not None
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
            # t4 = time.time()
            # print("calculate_xyz_scater:", t2-t1, "\nfitting_net:", t3-t2, "\nconcat:", t4-t3, "\n********************")
        return Ei
    
    def calculate_xyz_scater(self, 
                             Imagetype_map: torch.Tensor,
                             Ri: torch.Tensor,
                             type_emb: List[List[int]],
                             type_nums: int, 
                             device: torch.device, 
                             dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        type_emb_feat: Optional[torch.Tensor] = None
        # vector_tensor: Optional[torch.Tensor] = None
        i_jtype = torch.tensor(type_emb, device=device)
        mask = (Imagetype_map == i_jtype[0][0])
        if not mask.any():
            return torch.zeros(1, device=device), torch.zeros(1, device=device)
        indices = torch.arange(len(Imagetype_map.flatten()),device=device)[mask]
        neigh_index: List[int] = []
        for i in range(type_nums):
            jtype = i_jtype[i, 1].item()
            start = int(jtype * self.maxNeighborNum)
            end = int((jtype + 1) * self.maxNeighborNum)
            neigh_index.extend(list(range(start, end)))
            type_emb_ = torch.tensor(self.type_vector[self.atom_type[jtype]], dtype=dtype, device=device).repeat(self.maxNeighborNum,1)
            type_emb_feat = type_emb_ if type_emb_feat is None else torch.concat((type_emb_feat, type_emb_), dim=0)
        assert type_emb_feat is not None
        # For each neighbor of the central atom 'ntype', obtain the type code by passing it through the type embedding net
        #-------------------------------------------------------#
        # this code is for type embedding that physical vectors pass to a small net work
        # if len(self.embedding_net) > 1:  
        #     type_emb_encoded = self.embedding_net[0](type_emb_feat)
        #     S_Rij_type = torch.concat((S_Rij, type_emb_encoded.unsqueeze(0).unsqueeze(0).expand(S_Rij.shape[0], S_Rij.shape[1], -1, -1)), dim=3)
        # else:
        #     S_Rij_type = torch.concat((S_Rij, type_emb_feat.unsqueeze(0).unsqueeze(0).expand(S_Rij.shape[0], S_Rij.shape[1], -1, -1)), dim=3)
        #-------------------------------------------------------#
        if self.compress_tab is None:
            S_Rij = Ri[:, indices][:, :, neigh_index, 0].unsqueeze(-1)
            tmp_a = Ri[:, indices][:, :, neigh_index]
            tmp_a = tmp_a.transpose(-2, -1)
            S_Rij_type = torch.concat((S_Rij, type_emb_feat.unsqueeze(0).unsqueeze(0).expand(S_Rij.shape[0], S_Rij.shape[1], -1, -1)), dim=3)
            G = self.embedding_net[-1](S_Rij_type)
            xyz_scater_a = torch.matmul(tmp_a, G)
        else:
            neigh_index = torch.tensor(neigh_index, device=device).reshape(type_nums, -1)
            S_Rij = Ri[:, indices][:, :, neigh_index, 0]
            tmp_a = Ri[:, indices][:, :, neigh_index].transpose(1, 2).transpose(-1,-2)
            if self.compress_tab.device != device:
                self.compress_tab = self.compress_tab.to(device)
            G = self.calc_compress_polynomial(S_Rij, i_jtype[:, 1], type_nums, device)
            xyz_scater_a = torch.matmul(tmp_a, G).sum(1)
        # attention: for hybrid training, the division should be done based on \
        #   the number of element types in the current image, because the images may from different systems.
        xyz_scater_a = xyz_scater_a / (self.maxNeighborNum * type_nums)
        xyz_scater_b = xyz_scater_a[:, :, :, :self.M2]
        return xyz_scater_a, xyz_scater_b
    
    def calculate_force_virial(self, 
                               Ri: torch.Tensor,
                               Ri_d: torch.Tensor,
                               Etot: torch.Tensor,
                               natoms_sum: int,
                               batch_size: int,
                               list_neigh: torch.Tensor,
                               ImageDR: torch.Tensor, 
                               nghost: int,
                               device: torch.device,
                               dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Etot)]
        dE = torch.autograd.grad([Etot], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        assert dE is not None
        dE = torch.unsqueeze(dE, dim=-1)
        if device.type == "cpu":
            # dE * Ri_d [batch, natom, max_neighbor * len(type_map),4,1] * [batch, natom, max_neighbor * len(type_map), 4, 3]
            # dE_Rid [batch, natom, max_neighbor * len(type_map), 3]
            dE_Rid = torch.mul(dE, Ri_d).sum(dim=-2)
            Force = torch.zeros((batch_size, natoms_sum + nghost + 1, 3), device=device, dtype=dtype)
            Force[:, 1:natoms_sum + 1, :] = -1 * dE_Rid.sum(dim=-2)
            Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
            for batch_idx in range(batch_size):
                indice = list_neigh[batch_idx].view(-1).unsqueeze(-1).expand(-1, 3).to(torch.int64) # list_neigh's index start from 1, so the Force's dimension should be natoms_sum + 1
                values = dE_Rid[batch_idx].view(-1, 3)
                Force[batch_idx].scatter_add_(0, indice, values).view(natoms_sum + nghost + 1, 3)
                Virial[batch_idx, 0] = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 0]).view(-1).sum(dim=0) # xx
                Virial[batch_idx, 1] = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 1]).view(-1).sum(dim=0) # xy
                Virial[batch_idx, 2] = (ImageDR[batch_idx, :, :, 1] * dE_Rid[batch_idx, :, :, 2]).view(-1).sum(dim=0) # xz
                Virial[batch_idx, 4] = (ImageDR[batch_idx, :, :, 2] * dE_Rid[batch_idx, :, :, 1]).view(-1).sum(dim=0) # yy
                Virial[batch_idx, 5] = (ImageDR[batch_idx, :, :, 2] * dE_Rid[batch_idx, :, :, 2]).view(-1).sum(dim=0) # yz
                Virial[batch_idx, 8] = (ImageDR[batch_idx, :, :, 3] * dE_Rid[batch_idx, :, :, 2]).view(-1).sum(dim=0) # zz
                Virial[batch_idx, [3, 6, 7]] = Virial[batch_idx, [1, 2, 5]]
            Force = Force[:, 1:, :]
        else:
            Ri_d = Ri_d.view(batch_size, natoms_sum, -1, 3)
            dE = dE.view(batch_size, natoms_sum, 1, -1)
            Force = -1 * torch.matmul(dE, Ri_d).squeeze(-2)
            ImageDR = ImageDR[:,:,:,1:].clone()
            nghost_tensor = torch.tensor(nghost, device=device, dtype=torch.int64)
            list_neigh = torch.unsqueeze(list_neigh,2)
            list_neigh = (list_neigh - 1).type(torch.int)
            Force = CalcOps.calculateForce(list_neigh, dE, Ri_d, Force, nghost_tensor)[0]
            Virial = CalcOps.calculateVirial(list_neigh, dE, ImageDR, Ri_d, nghost_tensor)[0]
        return Force, Virial
    
    def calc_compress_polynomial(self, S_Rij:torch.Tensor, type_indices:torch.Tensor, type_nums: int, device: torch.device):
        _S_Rij = S_Rij.transpose(-3,-2) #Sji [1, 6, 500] ->[1, 6, 5, 100] -> [1, 5, 6, 100]
        sij = _S_Rij.flatten()
        
        mask = sij < self.sij_max

        x = torch.zeros_like(sij)
        x[mask] = (sij[mask]-self.sij_min)/self.dx
        x[~mask] = (sij[~mask]-self.sij_max)/(10*self.dx)
        index_k1 = x.type(torch.long) # get floor
        xk = torch.zeros_like(sij, dtype=torch.float32) # the index * dx + sij_min is a float type data
        xk[mask] = index_k1[mask]*self.dx + self.sij_min
        xk[~mask] = self.sij_max + index_k1[~mask]*self.dx*10
        f2 = (sij - xk).flatten().unsqueeze(-1)
        # f2 = (sij - xk).reshape(type_nums, -1, 1)
        
        index_k1 = index_k1 + (type_indices*self.compress_tab_shape[1]).repeat(S_Rij.shape[1]*self.maxNeighborNum, 1).transpose(0,1).flatten() # real index if compress table
        coefficient = self.compress_tab[index_k1, :]
        if device.type == "cpu":
            if self.order == 3:
                G = f2**3 *coefficient[:, :, 0] + f2**2 * coefficient[:, :, 1] + \
                    f2 * coefficient[:, :, 2] + coefficient[:, :, 3]
            else:
                G = f2**5 *coefficient[:, :, 0] + f2**4 * coefficient[:, :, 1] + \
                    f2**3 * coefficient[:, :, 2] + f2**2 * coefficient[:, :, 3] + \
                    f2 * coefficient[:, :, 4] + coefficient[:, :, 5]
        else:# calucluate by GPU
            G = CalcOps.calculateCompress(f2, coefficient)[0]
        G2 = G.reshape(_S_Rij.shape[0],_S_Rij.shape[1],_S_Rij.shape[2],_S_Rij.shape[3],self.compress_tab_shape[2])
        return G2
        """
        Example:
        max_neighbor = 5;
        type_nums = 2;
        batch_size = 1;
        natoms_ntype = 1;
        self.compress_tab_shape: compress_tab before reshape; --> torch.Size([4, 511, 25, 4]): [type_nums**2, grid_tab, network_nodes, 4], 4 here is the const (a, b, c, d) of the 3-order polynomial;
        self.compress_tab.shape: compress_tab after reshape; --> torch.Size([2044, 25, 4])
        f2.shape: [batch_size * natoms_ntype * max_neighbor * type_nums] -> [type_nums * natoms_ntype * max_neighbor, 1]
        index_k1.shape: [batch_size * natoms_ntype * max_neighbor * type_nums]
        coefficient.shape: torch.Size([10, 25, 4]) --> [type_nums * natoms_ntype * max_neighbor, network_nodes, 4]
        G.shape: torch.Size([10, 25]) --> torch.Size([1, 2, 1, 5, 25]) : [type_nums * natoms_ntype * max_neighbor, network_nodes] --> [batch_size, type_nums, natoms_ntype, max_neighbor, network_nodes]
        """
    """
    def calc_compress_5order(self, S_Rij:torch.Tensor, table_type:int, device: torch.device):
        sij = S_Rij.flatten()

        x = (sij-self.sij_min)/self.dx
        index_k1 = x.type(torch.long) # get floor
        xk = self.sij_min + index_k1*self.dx
        f2 = (sij - xk).flatten().unsqueeze(-1)
    
        coefficient = self.compress_tab[table_type, index_k1, :]

        # G = CalculateCompress.apply(f2, coefficient)
        if device.type == "cpu":
            G = f2**5 *coefficient[:, :, 0] + f2**4 * coefficient[:, :, 1] + \
                f2**3 * coefficient[:, :, 2] + f2**2 * coefficient[:, :, 3] + \
                f2 * coefficient[:, :, 4] + coefficient[:, :, 5]
        else:
            G = CalcOps.calculateCompress(f2, coefficient)[0]
        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        return G

    def calc_compress_3order(self, S_Rij:torch.Tensor, table_type:int, device: torch.device):
        sij = S_Rij.flatten()

        mask = sij < self.sij_max
        x = torch.zeros_like(sij)
        x[mask] = (sij[mask]-self.sij_min)/self.dx
        x[~mask] = (sij[~mask]-self.sij_max)/(10*self.dx)
        index_k1 = x.type(torch.long) # get floor

        xk = torch.zeros_like(sij, dtype=torch.float32) # the index * dx + sij_min is a float type data
        xk[mask] = index_k1[mask]*self.dx + self.sij_min
        xk[~mask] = self.sij_max + index_k1[~mask]*self.dx*10
        f2 = (sij - xk).flatten().unsqueeze(-1)
        
        coefficient = self.compress_tab[table_type, index_k1, :]
        # G = CalculateCompress.apply(f2, coefficient)
        if device.type == "cpu":
            G = f2**3 *coefficient[:, :, 0] + f2**2 * coefficient[:, :, 1] + \
                f2 * coefficient[:, :, 2] + coefficient[:, :, 3]
        else:
            G = CalcOps.calculateCompress(f2, coefficient)[0]

        G = G.reshape(S_Rij.shape[0], S_Rij.shape[1], S_Rij.shape[2], G.shape[1])
        
        return G
    """
    