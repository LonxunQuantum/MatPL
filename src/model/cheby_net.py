import sys, os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional
from src.user.input_param import InputParam
import time

sys.path.append(os.getcwd())
from src.model.dp_embedding import FittingNet
from numpy.ctypeslib import ndpointer
import ctypes
lib_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib = ctypes.CDLL(os.path.join(lib_path, 'feature/chebyshev/build/lib/libdescriptor.so')) # multi-descriptor
lib.CreateDescriptor.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, 
                                  ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                  ndpointer(ctypes.c_int, ndim=1, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, ndim=3, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_int, ndim=4, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, ndim=5, flags="C_CONTIGUOUS"),
                                  ndpointer(ctypes.c_double, ndim=4, flags="C_CONTIGUOUS")]
                                  
lib.CreateDescriptor.restype = ctypes.c_void_p

lib.show.argtypes = [ctypes.c_void_p]
lib.DestroyDescriptor.argtypes = [ctypes.c_void_p]

lib.get_feat.argtypes = [ctypes.c_void_p]
lib.get_feat.restype = ctypes.POINTER(ctypes.c_double)
lib.get_dfeat.argtypes = [ctypes.c_void_p]
lib.get_dfeat.restype = ctypes.POINTER(ctypes.c_double)
lib.get_dfeat2c.argtypes = [ctypes.c_void_p]
lib.get_dfeat2c.restype = ctypes.POINTER(ctypes.c_double)
lib.get_neighbor_list.argtypes = [ctypes.c_void_p]
lib.get_neighbor_list.restype = ctypes.POINTER(ctypes.c_int)

if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device

class ChebyNet(nn.Module):
    def __init__(self, input_param: InputParam, scaler, energy_shift):
        super(ChebyNet, self).__init__()
        self.input_param = input_param
        self.atom_types = input_param.atom_type
        self.ntypes = len(self.atom_types)
        self.m_neigh = input_param.max_neigh_num
        self.Rc_M = input_param.descriptor.Rmax
        self.rcut_smooth = input_param.descriptor.Rmin
        self.beta = input_param.descriptor.cheby_order
        self.m1 = input_param.descriptor.radial_num1
        self.m2 = input_param.descriptor.radial_num2
        self.nfeat = self.m1 * self.m2
        if self.input_param.precision == "float64":
            self.dtype = torch.double
        elif self.input_param.precision == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")
        
        self.scaler = scaler
        self.fitting_net = nn.ModuleList()
        for i in range(self.ntypes):
            self.fitting_net.append(FittingNet(network_size = input_param.model_param.fitting_net.network_size,
                                               bias         = input_param.model_param.fitting_net.bias,
                                               resnet_dt    = input_param.model_param.fitting_net.resnet_dt,
                                               activation   = input_param.model_param.fitting_net.activation,
                                               input_dim    = self.nfeat,
                                               ener_shift   = energy_shift[i],
                                               magic        = False))
        self.set_cparam()

    def set_cparam(self):
        size = self.ntypes * self.ntypes * self.m1 * self.beta
        r_k = torch.normal(mean=0, std=1, size=(size,), dtype=self.dtype)
        m = torch.rand(size, dtype=self.dtype) - 0.5
        s = torch.full_like(m, 0.1)
        c_param = m + s*r_k

        self.c_param = nn.Parameter(c_param.reshape(self.ntypes, self.ntypes, self.m1, self.beta), requires_grad=False)
        # self.c_param = c_param.reshape(self.ntypes, self.ntypes, self.m1, self.beta)
                
    def get_egroup(self,
                   Ei: torch.Tensor,
                   Egroup_weight: Optional[torch.Tensor] = None,
                   divider: Optional[torch.Tensor] = None)-> Optional[torch.Tensor]:
        if Egroup_weight is not None and divider is not None:       # Egroup_out is not defined in the false branch:
            Egroup = torch.matmul(Egroup_weight, Ei)
            Egroup_out = torch.divide(Egroup.squeeze(-1), divider)
        else:
            Egroup_out = None
        
        return Egroup_out
    
    def get_index(self, user_input_order: List[int], key:torch.Tensor):
        for i, v in enumerate(user_input_order):
            if v == key:
                return i
        return -1

    def get_fitnet_index(self, atom_type: torch.Tensor) -> Tuple[List[int]]:
        fitnet_index: List[int] = []
        for i, atom in enumerate(atom_type):
            index = self.get_index(self.atom_types, atom)
            fitnet_index.append(index)
        return fitnet_index
    
    def forward(self,
                list_neigh: torch.Tensor,   # int32
                Imagetype_map: torch.Tensor,    # int32
                atom_type: torch.Tensor,    # int32
                ImageDR: torch.Tensor,      # float64
                num_neigh: torch.Tensor,    # int32
                nghost: int, 
                Egroup_weight: Optional[torch.Tensor] = None, 
                divider: Optional[torch.Tensor] = None, 
                is_calc_f: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            list_neigh (torch.Tensor): Tensor representing the neighbor list. Shape: (batch_size, natoms_sum, max_neighbor * ntypes).
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.. Shape: (natoms_sum).
            atom_type (torch.Tensor): Tensor representing the image's atom types. Shape: (ntypes).
            num_neigh (torch.Tensor): Tensor representing the number of neighbors for each atom. Shape: (batch_size, natoms_sum, ntypes).
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
        ntypes = list_neigh.shape[2]
        m_neigh = list_neigh.shape[3]
        nfeat = self.m1 * self.m2
        fitnet_index = self.get_fitnet_index(atom_type)
        # t1 = time.time()
        feat, dfeat, dfeat2c, list_neigh_alltype = self.calculate_feat(batch_size, natoms_sum, ntypes, Imagetype_map, nfeat, num_neigh, list_neigh, ImageDR, device, dtype)
        feat.requires_grad_()
        # t2 = time.time()
        Ei = self.calculate_Ei(Imagetype_map, feat, batch_size, fitnet_index, device)
        # t3 = time.time()
        assert Ei is not None
        Etot = torch.sum(Ei, 1)
        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        Ei = torch.squeeze(Ei, 2)
        if is_calc_f is False:
            Force, Virial = None, None
        else:
            # t4 = time.time()
            Force, Virial, dEi_dc = self.calculate_force_virial(feat, dfeat, dfeat2c, Ei, natoms_sum, Imagetype_map, m_neigh, batch_size, list_neigh_alltype, ImageDR, nghost, device, dtype)
            # print("Force, Etot \n", Force, Etot)
            # t5 = time.time()
        return Etot, Ei, Force, Egroup, Virial, dEi_dc
    
    def calculate_feat(self,
                     batch_size: int,
                     natoms_sum: int,
                     ntypes: int,
                     Imagetype_map: torch.Tensor,
                     nfeat: int,
                     num_neigh: torch.Tensor,
                     list_neigh: torch.Tensor,
                     ImagedR: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the feat and dfeat tensors.

        Args:
            natoms_sum (int): The total number of atoms.
            batch_size (int): The batch size.
            num_neigh (torch.Tensor): The tensor representing the number of neighbors for each atom.
            list_neigh (torch.Tensor): The tensor representing the neighbor list.
            ImagedR (torch.Tensor): The tensor containing the atom's distances and Δ(position vectors).
            device (torch.device): The device to perform the calculations on.
            dtype (torch.dtype): The data type of the tensors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The feat and dfeat tensors.
        """
        c = self.c_param.cpu().numpy()
        # c[0,1,0,0] = c[0,1,0,0] + 0.001
        # c[0,1,1,0] = c[0,1,1,0] + 0.001
        descriptor = lib.CreateDescriptor(batch_size, self.beta, self.m1, self.m2, self.Rc_M, self.rcut_smooth, natoms_sum, ntypes, self.m_neigh, Imagetype_map.cpu().numpy(), num_neigh.cpu().numpy(), list_neigh.cpu().numpy(), ImagedR.cpu().numpy(), c)
        # lib.show(descriptor)
        feat = lib.get_feat(descriptor)
        dfeat = lib.get_dfeat(descriptor)
        dfeat2c = lib.get_dfeat2c(descriptor)
        list_neigh_alltype = lib.get_neighbor_list(descriptor)

        feat = np.ctypeslib.as_array(feat, (batch_size * natoms_sum, nfeat))
        dfeat = np.ctypeslib.as_array(dfeat, (batch_size, natoms_sum, nfeat, self.m_neigh, 3))
        dfeat2c = np.ctypeslib.as_array(dfeat2c, (batch_size, natoms_sum, nfeat, ntypes, self.m1, self.beta))
        list_neigh_alltype = np.ctypeslib.as_array(list_neigh_alltype, (batch_size, natoms_sum, self.m_neigh))

        feat = self.scaler.transform(feat)
        feat = torch.tensor(feat, dtype=dtype, device=device).reshape(batch_size, natoms_sum, nfeat)
        dfeat = torch.tensor(dfeat, dtype=dtype, device=device).transpose(2, 3)
        dfeat2c = torch.tensor(dfeat2c, dtype=dtype, device=device)
        list_neigh_alltype = torch.tensor(list_neigh_alltype, dtype=torch.int32, device=device)
        lib.DestroyDescriptor(descriptor)

        scaler = torch.tensor(self.scaler.scale_, dtype=dtype, device=device)
        dfeat = (dfeat.transpose(3, 4) * scaler).transpose(3, 4)
        dfeat2c = (dfeat2c.transpose(2, 5) * scaler).transpose(2, 5)
        return feat, dfeat, dfeat2c, list_neigh_alltype
        
    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     feat: torch.Tensor,
                     batch_size: int,
                     fit_index: List[int],
                     device: torch.device) -> Optional[torch.Tensor]:
        """
        Calculate the energy Ei for each type of atom in the system.

        Args:
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.
            feat (torch.Tensor): A tensor representing the atomic descriptors.
            batch_size (int): The size of the batch.
            fit_index (List[int]): The index of the fitting network to use for each atom type.
            type_nums (int): The number of atom types.
            device (torch.device): The device to perform the calculations on.

        Returns:
            Optional[torch.Tensor]: The calculated energy Ei for each type of atom, or None if the calculation fails.
        """
        Ei : Optional[torch.Tensor] = None
        fit_net_dict = {idx: fit_net for idx, fit_net in enumerate(self.fitting_net)}
        for net in fit_index:
            fit_net = fit_net_dict.get(net) 
            assert fit_net is not None
            mask = (Imagetype_map == net).flatten()
            if not mask.any():
                continue
            indices = torch.arange(len(Imagetype_map.flatten()),device=device)[mask]
            Ei_ntype = fit_net.forward(feat[:, indices])
            # 打印权重和偏置
            # for name, param in fit_net.named_parameters():
            #     if 'weight' in name:
            #         print('Weight:', name, param.data)
            #     if 'bias' in name:
            #         print('Bias:', name, param.data)
            Ei = Ei_ntype if Ei is None else torch.concat((Ei, Ei_ntype), dim=1)
        return Ei

    def calculate_force_virial(self, 
                               feat: torch.Tensor,
                               dfeat: torch.Tensor,
                               dfeat2c: torch.Tensor,
                               Ei: torch.Tensor,
                               natoms_sum: int,
                               Imagetype_map: torch.Tensor,
                               m_neigh: int,
                               batch_size: int,
                               list_neigh_alltype: torch.Tensor,
                               ImageDR: torch.Tensor, 
                               nghost: int,
                               device: torch.device,
                               dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [feat], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        # from utils.debug_operation import check_cuda_memory
        # print("000000000000000000000000000")
        # check_cuda_memory(1, 1, 1)
        assert dE is not None
        dE_tmp = dE.repeat(1, 1, m_neigh).reshape(batch_size, natoms_sum, m_neigh, -1).unsqueeze(-2)
        dE_dfeat = torch.matmul(dE_tmp, dfeat).sum(-2)  # partial E / partial dfeat = (partial E / partial feat) * (partial feat / partial delta(x,y,z) )
        with torch.no_grad():
            dE_tmp2c = (dfeat2c * dE.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(2)
        dEi_dc = torch.zeros((batch_size, self.ntypes, self.ntypes, self.m1, self.beta), device=device, dtype=dtype)
        dEi_dc.index_add_(1, Imagetype_map, dE_tmp2c)
        # print(dEi_dc.grad_fn is not None)
        list_neigh_map = torch.zeros_like(list_neigh_alltype)
        mask_index = list_neigh_alltype == -1
        list_index = list_neigh_alltype + 1
        list_neigh_alltype[mask_index] = 0.0
        list_neigh_map.scatter_add_(2, list_neigh_alltype.type(torch.int64), list_index)
        Force = torch.zeros((batch_size, natoms_sum + nghost + 1, 3), device=device, dtype=dtype)
        Force[:, 1:natoms_sum + 1, :] = dE_dfeat.sum(-2)
        # dE_tmp_list = torch.index_select(dE_tmp.reshape(batch_size*(natoms_sum+1), -1), 0, list_neigh_map.flatten()).reshape(batch_size, natoms_sum, m_neigh, 1, -1)
        Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
        for batch_idx in range(batch_size):
            indices = list_neigh_map[batch_idx].flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64)
            values = - dE_dfeat[batch_idx].view(-1, 3)
            Force[batch_idx].scatter_add_(0, indices, values).view(natoms_sum + nghost + 1, 3)
        Force = Force[:, 1:, :]
        return Force, Virial, dEi_dc.sum(0) 
