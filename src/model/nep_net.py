import sys, os
import time
import random
from math import pi as PI
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_ as normal

from typing import List, Tuple, Optional
from src.user.input_param import InputParam
from src.user.nep_param import NepParam
from src.utils.debug_operation import check_cuda_memory
sys.path.append(os.getcwd())
from src.model.nep_fitting import FittingNet, QNEPFittingNet
if torch.cuda.is_available():
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind.so")
    torch.ops.load_library(lib_path)
    CalcOps = torch.ops.CalcOps_cuda
else:
    lib_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "op/build/lib/libCalcOps_bind_cpu.so")
    torch.ops.load_library(lib_path)    # load the custom op, no use for cpu version
    CalcOps = torch.ops.CalcOps_cpu     # only for compile while no cuda device
   
class NEP(nn.Module):
    def __init__(self, input_param:InputParam=None, energy_shift=None, rank=0, q_scaler = None, max_NN_radial = -1, max_NN_angular = -1, dtype=None, device=None):
        super(NEP, self).__init__()
        self.input_param = input_param
        if self.input_param.seed is not None:
            random.seed(self.input_param.seed)
            np.random.seed(self.input_param.seed)
            torch.manual_seed(self.input_param.seed)
            torch.cuda.manual_seed_all(self.input_param.seed)  # 为所有 GPU 设置种子
        self.dtype = dtype
        self.device = device
        self.Pi = PI
        self.half_Pi = self.Pi/2
        self.model_type = input_param.model_type.upper()
        self.set_init_nep_param(input_param)
        self.charge_mode = getattr(input_param.nep_param, "charge_mode", 0) or 0
        self.charge_output_num = 2 if self.charge_mode else 1
        self.use_analytical_nep_grad = bool(getattr(input_param.nep_param, "use_analytical_nep_grad", False))
        self.gpumd_nep4 = bool(getattr(input_param.nep_param, "gpumd_nep4", False))
        self.zbl = input_param.nep_param.zbl
        self.zbl_factor = input_param.nep_param.use_typewise_cutoff_zbl
        if self.input_param.precision == "float64":
            self.dtype = torch.double
        elif self.input_param.precision == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training data type")
        self.energy_shift = energy_shift        
        self.set_cparam(np.mean(energy_shift))
        
        # 注册缓冲区
        self.register_buffer('q_scaler', None)
        self.register_buffer('C3B', None)
        self.register_buffer('C4B', None)
        self.register_buffer('C5B', None)
        self.register_buffer('atom_type_device', None)
        self.register_buffer('max_NN_radial', torch.tensor(max_NN_radial, dtype=torch.int64))
        self.register_buffer('max_NN_angular', torch.tensor(max_NN_angular, dtype=torch.int64))

        # 初始化缓冲区
        self._initialize_buffers(q_scaler = q_scaler)
        self.fitting_net = nn.ModuleList()
        self.common_bias = None
        self.charge_predict = None
        self.atomic_charge = None
        self.atomic_charge_shifted = None
        self.atomic_bec = None
        fitting_network_size = list(self.neuron[:-1]) + [1]
        qnep_network_size = list(self.neuron[:-1])
        if self.charge_mode:
            common_bias_value = float(np.mean(energy_shift))
            sqrt_epsilon_inf_value = 2.0
            if input_param.nep_param.c2_param is not None:
                bias_lastlayer = np.asarray(input_param.nep_param.bias_lastlayer)
                if bias_lastlayer.ndim > 1:
                    common_bias_value = float(np.mean(bias_lastlayer[:, 0]))
                else:
                    common_bias_value = float(np.mean(bias_lastlayer))
                if getattr(input_param.nep_param, "sqrt_epsilon_inf", None) is not None:
                    sqrt_epsilon_inf_value = float(input_param.nep_param.sqrt_epsilon_inf)
            self.common_bias = torch.nn.Parameter(torch.tensor(common_bias_value, dtype=self.dtype), requires_grad=True)
            self.sqrt_epsilon_inf = torch.nn.Parameter(torch.tensor(sqrt_epsilon_inf_value, dtype=self.dtype), requires_grad=True)
        else:
            self.sqrt_epsilon_inf = None

        for i in range(self.ntypes):
            nep_txt_param = None
            if input_param.nep_param.c2_param is not None:
                nep_txt_param = [input_param.nep_param.model_wb[i*3+0], input_param.nep_param.model_wb[i*3+1], input_param.nep_param.model_wb[i*3+2], input_param.nep_param.bias_lastlayer[i]]
            if self.charge_mode:
                self.fitting_net.append(QNEPFittingNet(network_size = qnep_network_size,
                                                    bias      = True,
                                                    resnet_dt = False,
                                                    activation= "tanh",
                                                    input_dim = self.feature_nums,
                                                    ener_shift= energy_shift[i],
                                                    charge_mode = self.charge_mode,
                                                    magic     = False,
                                                    nep_txt_param = nep_txt_param,
                                                    last_bias= not self.gpumd_nep4,
                                                    ))
            else:
                self.fitting_net.append(FittingNet(network_size   = fitting_network_size, #[50, output_num]
                                                    bias      = True,
                                                    resnet_dt = False,
                                                    activation= "tanh",
                                                    input_dim = self.feature_nums,
                                                    ener_shift= energy_shift[i],
                                                    magic     = False,
                                                    nep_txt_param = nep_txt_param,
                                                    last_bias= True,
                                                    #    self.nep_param["net_cfg"]["fitting_net"]["resnet_dt"],
                                                    #    self.nep_param["net_cfg"]["fitting_net"]["activation"], 
                                                    ))

    def _initialize_buffers(self, q_scaler = None):
        """初始化缓冲区，设置 q_max, q_min, C3B, C4B, C5B, atom_type_device 和 q_scaler（如果 nep.txt 提供）。"""
        dtype = self.dtype
        device = self.device  # 默认设备
        if isinstance(q_scaler, torch.Tensor):
            self.q_scaler = q_scaler.clone().detach().to(dtype=dtype, device=device)
        else:
            self.q_scaler = torch.tensor(q_scaler, dtype=dtype, device=device)
        self.C3B = torch.tensor([0.238732414637843, 0.238732414637843, 0.238732414637843, #c10, c11, 12
                0.099471839432435, 1.1936620731892151, 1.1936620731892151, 0.2984155182973038, 0.2984155182973038, #c20,c21=c22,c23=c24
                0.139260575205408, 0.20889086280811264, 0.20889086280811264, 2.088908628081126, 2.088908628081126, 0.34815143801352105, 0.34815143801352105, #c30, c31=c32,c33=c34
                0.01119058193614889, 0.44762327744595565, 0.44762327744595565, 0.22381163872297782, 0.22381163872297782, # c40, c41=c42, c43=c44
                3.1333629421216895, 3.1333629421216895, 0.3916703677652112, 0.3916703677652112 #c45=c46, c47=c48
                ], dtype=dtype, device=device)

        self.C4B = torch.tensor([-0.007499480826664, -0.134990654879954, 0.067495327439977, 0.404971964639861, -0.809943929279723], 
                                dtype=dtype, device=device)
        
        self.C5B = torch.tensor([0.026596810706114, 0.053193621412227, 0.026596810706114], 
                                dtype=dtype, device=device)

        # zbl
        self.K_C_SP = 14.399645 # 1/(4*PI*epsilon_0)
        self.zbl_para = [0.18175, 3.1998, 0.50986, 0.94229, 0.28022, 0.4029, 0.02817, 0.20162]
        self.atom_type_device = torch.tensor(self.atom_type, dtype=torch.int64, device=device)
        if self.zbl_factor is not None:
            self.COVALENT_RADIUS = torch.tensor(
                [0.0, 0.426667, 0.613333, 1.6,     1.25333, 1.02667, 1.0,     0.946667, 0.84,    0.853333,
                0.893333, 1.86667,  1.66667, 1.50667, 1.38667, 1.46667, 1.36,     1.32,    1.28,
                2.34667,  2.05333,  1.77333, 1.62667, 1.61333, 1.46667, 1.42667,  1.38667, 1.33333,
                1.32,     1.34667,  1.45333, 1.49333, 1.45333, 1.53333, 1.46667,  1.52,    1.56,
                2.52,     2.22667,  1.96,    1.85333, 1.76,    1.65333, 1.53333,  1.50667, 1.50667,
                1.44,     1.53333,  1.64,    1.70667, 1.68,    1.68,    1.64,     1.76,    1.74667,
                2.78667,  2.34667,  2.16,    1.96,    2.10667, 2.09333, 2.08,     2.06667, 2.01333,
                2.02667,  2.01333,  2.0,     1.98667, 1.98667, 1.97333, 2.04,     1.94667, 1.82667,
                1.74667,  1.64,     1.57333, 1.54667, 1.48,    1.49333, 1.50667,  1.76,    1.73333,
                1.73333,  1.81333,  1.74667, 1.84,    1.89333, 2.68,    2.41333,  2.22667, 2.10667,
                2.02667,  2.04,     2.05333, 2.06667], dtype=dtype, device=device) # 0.0 只用于占位
        else:
            self.COVALENT_RADIUS = None
    '''
    description: 
        for nep.txt 
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_nn_params(self):
        nn_params = []
        type_bias = []
        for i in range(self.ntypes):
            params, last_bias = self.fitting_net[i].get_param_list()
            nn_params.extend(params)
            if self.charge_mode and not self.gpumd_nep4:
                nn_params.extend(last_bias)
            elif len(last_bias) > 0:
                type_bias.extend(last_bias)
        if self.charge_mode:
            nn_params.append(float(self.sqrt_epsilon_inf.cpu().detach().numpy()))
            nn_params.append(float(-self.common_bias.cpu().detach().numpy()) if self.gpumd_nep4 else 0.0)
        else:
            nn_params.extend(type_bias) # for new nep.txt test
        nn_params.extend(list(self.c_param_2.permute(2, 3, 0, 1).flatten().cpu().detach().numpy()))
        if self.l_max_3b > 0:
            nn_params.extend(list(self.c_param_3.permute(2, 3, 0, 1).flatten().cpu().detach().numpy()))
        nn_params.extend(list(self.q_scaler.flatten().cpu().detach().numpy()))
        return nn_params
        
    '''
    description: 
    maybe these params could be get from model, descriptor and optimizor object
    param {*} self
    param {InputParam} input_param
    return {*}
    author: wuxingxing
    '''
    def set_init_nep_param(self, input_param:InputParam):
        nep_param = input_param.nep_param
        self.atom_type = input_param.atom_type
        self.ntypes = len(input_param.atom_type)
        self.ntypes_sq = self.ntypes * self.ntypes
        self.train_2b = input_param.nep_param.train_2b
        self.cutoff_radial  = float(nep_param.cutoff[0])
        self.cutoff_angular = float(nep_param.cutoff[1])
        self.rcinv_radial   = 1.0/self.cutoff_radial
        self.rcinv_angular  = 1.0/self.cutoff_angular
        self.neuron         = nep_param.neuron
        
        self.n_max_radial  = nep_param.n_max[0]
        self.n_max_angular = nep_param.n_max[1]
        
        self.n_base_radial = nep_param.basis_size[0]
        self.n_base_angular= nep_param.basis_size[1]

        self.l_max_3b = nep_param.l_max[0]
        self.l_max_4b = nep_param.l_max[1]
        self.l_max_5b = nep_param.l_max[2]
        # feature nums
        if self.train_2b:
            self.two_feat_num   = self.n_max_radial + 1
        else:
            self.two_feat_num  = 0
        self.three_feat_num = (self.n_max_angular + 1) * self.l_max_3b
        self.four_feat_num  = (self.n_max_angular + 1) if self.l_max_4b > 0 else 0
        self.five_feat_num  = (self.n_max_angular + 1) if self.l_max_5b > 0 else 0
        self.multi_feat_num = self.three_feat_num + self.four_feat_num + self.five_feat_num
        if self.l_max_3b > 0:
            self.feature_nums   = self.two_feat_num + self.multi_feat_num
        # c param nums, the 4-body and 5-body use the same c param of 3-body, their N_base_a the same
        else:
            self.feature_nums   = self.two_feat_num
        if self.feature_nums == 0:
            raise Exception("ERROR! The two body features and multi body features are both zero, please check the param!")
        self.two_c_num   = self.ntypes_sq * (self.n_max_radial+1)  * (self.n_base_radial+1)
        self.three_c_num = self.ntypes_sq * (self.n_max_angular+1) * (self.n_base_angular+1)
        self.c_num       = self.two_c_num + self.three_c_num

    def get_q_scaler(self):
        return self.q_scaler.cpu().detach().numpy()


    '''
    description: 
        c_params is init from randly or c_params if init from checkpoint
        or c_params is init from nep.txt
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def set_cparam(self, energy_shift:float):
        if self.input_param.nep_param.c2_param is not None: #load from nep.txt
            self.c_param_2 = torch.nn.Parameter(torch.tensor(self.input_param.nep_param.c2_param).contiguous(), requires_grad=True)
            self.c_param_3 = torch.nn.Parameter(torch.tensor(self.input_param.nep_param.c3_param).contiguous(), requires_grad=True) if self.l_max_3b > 0 else None
        else: # init by randly (for first training) or checkpoint
            r_k = torch.normal(mean=0, std=1, size=(self.c_num,), dtype=self.dtype)
            m = torch.rand(self.c_num, dtype=self.dtype) - 0.5
            s = torch.full_like(m, 0.1)
            c_param = m + s*r_k
            self.c_param_2 = torch.nn.Parameter(c_param[:self.two_c_num].reshape(self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)), requires_grad=True)
            self.c_param_3 = torch.nn.Parameter(c_param[self.two_c_num : ].reshape(self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1)), requires_grad=True)  if self.l_max_3b > 0 else None

            # self.c_param_2 = torch.nn.Parameter(torch.ones([self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1)]), requires_grad=False)
            # self.c_param_3 = torch.nn.Parameter(torch.ones([self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1)]), requires_grad=False)

            # self.c_param_2 = torch.nn.Parameter(torch.normal(mean=0, std=0.5, size = (self.ntypes, self.ntypes, (self.n_max_radial+1), (self.n_base_radial+1))), requires_grad=True)
            # self.c_param_3 = torch.nn.Parameter(torch.normal(mean=0, std=0.5, size = (self.ntypes, self.ntypes, (self.n_max_angular+1), (self.n_base_angular+1))), requires_grad=True)
            
            # self.common_bias = torch.nn.Parameter(torch.tensor(energy_shift), requires_grad=True)
            # self.common_bias = None  # for nep common bias test

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

    def get_fitnet_index(self, atom_type: torch.Tensor) -> List[int]:
        fitnet_index: List[int] = []
        for i, atom in enumerate(atom_type):
            if atom == 0: # for hybrid training, 0 means no atom
                continue
            index = self.get_index(self.atom_type, atom)
            fitnet_index.append(index)
        return fitnet_index
   
    def forward(self, 
                NN_radial: torch.Tensor,
                NL_radial: torch.Tensor,
                Ri_radial: torch.Tensor, 
                NN_angular: torch.Tensor, 
                NL_angular: torch.Tensor, 
                Ri_angular: torch.Tensor,
                num_atom: torch.Tensor, 
                atom_type_map: torch.Tensor, 
                Egroup_weight: Optional[torch.Tensor] = None, 
                divider: Optional[torch.Tensor] = None, 
                is_calc_f: Optional[bool] = True,
                charge_label: Optional[torch.Tensor] = None,
                position: Optional[torch.Tensor] = None,
                box_original: Optional[torch.Tensor] = None,
                volume: Optional[torch.Tensor] = None,
                need_force: Optional[bool] = True,
                need_bec: Optional[bool] = True,
                need_charge_virial: Optional[bool] = True,
                need_charge_energy: Optional[bool] = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass of the model.

        Args:
            list_neigh (torch.Tensor): Tensor representing the neighbor list. Shape: (batch_size, natoms_sum, max_neighbor * ntypes).
            Imagetype_map (torch.Tensor): The tensor mapping atom types to image types.. Shape: (natoms_sum).
            atom_type (torch.Tensor): Tensor representing the image's atom types. Shape: (ntypes).
            ImageDR (torch.Tensor): Tensor representing the image DRneigh. Shape: (batch_size, natoms_sum, max_neighbor * ntypes, 4).
            nghost (int): Number of ghost atoms.
            Egroup_weight (Optional[torch.Tensor], optional): Tensor representing the Egroup weight. Defaults to None.
            divider (Optional[torch.Tensor], optional): Tensor representing the divider. Defaults to None.
            is_calc_f (Optional[bool], optional): Flag indicating whether to calculate forces and virial. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: Tuple containing the total energy (Etot), atomic energies (Ei), forces (Force), energy group (Egroup), and virial (Virial).
        """
        # check_cuda_memory(-1, -1, "=====FORWAR START=====") #self.fitting_net[0].layers[0].weight
        # t0 = time.time()
        device = Ri_radial.device
        dtype = Ri_radial.dtype
        natoms_sum = NL_radial.shape[0]#no use
        # fitnet_index = self.get_fitnet_index()
        Ri, Ri_d, Ri_angular, Ri_d_angular = self.calculate_Ri(Ri_radial, Ri_angular, device, dtype)
        Ri = Ri_radial
        Ri.requires_grad_()

        Ri_angular = Ri_angular
        Ri_angular.requires_grad_()
        radial_NL = NL_radial
        radial_Ri = Ri
        radial_Ri_d = Ri_d
        use_analytical_nep_grad = bool(
            self.use_analytical_nep_grad and
            device.type != "cpu" and
            is_calc_f is not False and
            need_force is not False)
        radial_context = None
        angular_context = None

        if device.type == "cpu":
            NL_radial_type = radial_NL.new_full(radial_NL.shape, -1).requires_grad_(False)
            mask = radial_NL != -1
            NL_radial_type[mask] = atom_type_map[radial_NL[mask]]

            NL_angular_type = NL_angular.new_full(NL_angular.shape, -1).requires_grad_(False)
            mask = NL_angular != -1
            NL_angular_type[mask] = atom_type_map[NL_angular[mask]]

            feats = self.calculate_qn(atom_type_map, NL_radial_type, radial_Ri, NL_angular_type, Ri_angular, device, dtype)
        else:# cuda ops
            if self.train_2b:
                feat_2b = torch.zeros(natoms_sum, self.two_feat_num, dtype=dtype, device=device, requires_grad=True)
                radial_outputs = CalcOps.calculateNepFeatWithGradContext(
                                                self.c_param_2.contiguous(),
                                                Ri,
                                                NL_radial,
                                                atom_type_map,
                                                feat_2b,
                                                self.cutoff_radial,
                                                self.multi_feat_num,
                                                int(self.input_param.nep_param.fix_cij)
                                                ) if use_analytical_nep_grad else CalcOps.calculateNepFeat(self.c_param_2.contiguous(),
                                                Ri, 
                                                NL_radial, 
                                                atom_type_map,
                                                feat_2b, 
                                                self.cutoff_radial,
                                                self.multi_feat_num,
                                                int(self.input_param.nep_param.fix_cij)
                                                )
                feat_2b = radial_outputs[0]
                if use_analytical_nep_grad:
                    radial_context = radial_outputs[1:]
            if self.l_max_3b > 0:
                feat_3b = torch.zeros(natoms_sum, self.multi_feat_num, dtype=dtype, device=device, requires_grad=True)
                angular_outputs = CalcOps.calculateNepMbFeatWithGradContext(
                                                        self.c_param_3.contiguous(),
                                                        Ri_angular,
                                                        NL_angular,
                                                        atom_type_map,
                                                        feat_3b,
                                                        self.two_feat_num,
                                                        self.l_max_3b,
                                                        self.l_max_4b,
                                                        self.l_max_5b,
                                                        self.cutoff_angular,
                                                        int(self.input_param.nep_param.fix_cij)
                                                        ) if use_analytical_nep_grad else CalcOps.calculateNepMbFeat(self.c_param_3.contiguous(),
                                                        Ri_angular, 
                                                        NL_angular, 
                                                        atom_type_map, 
                                                        feat_3b, 
                                                        self.two_feat_num,
                                                        self.l_max_3b, 
                                                        self.l_max_4b, 
                                                        self.l_max_5b, 
                                                        self.cutoff_angular,
                                                        int(self.input_param.nep_param.fix_cij)
                                                        )
                feat_3b = angular_outputs[0]
                if use_analytical_nep_grad:
                    angular_context = angular_outputs[1:]

                if self.train_2b:
                    feats = torch.concat([feat_2b, feat_3b], dim=-1)
                else:
                    feats = feat_3b
            else:
                feats = feat_2b
        feats_in = self.q_scaler * feats
        # feats_in = (feats-self.q_min)/(self.q_max-self.q_min)
        if use_analytical_nep_grad:
            Ei, charge, grad_feat_E_scaled, grad_feat_Q_scaled = self.calculate_Ei_with_grad(atom_type_map, feats_in, device)
        else:
            Ei, charge = self.calculate_Ei(atom_type_map, feats_in, device)
            grad_feat_E_scaled, grad_feat_Q_scaled = None, None
        assert Ei is not None
        charge_predict = None
        self.charge_predict = None
        self.atomic_charge = None
        self.atomic_charge_shifted = None
        self.atomic_bec = None
        if self.charge_mode:
            self.atomic_charge = charge
            charge_predict, self.atomic_charge_shifted = self.shift_total_charge(charge, num_atom, charge_label)
            self.charge_predict = charge_predict
            if need_bec:
                self.atomic_bec = self.calculate_bec(
                    charge,
                    self.atomic_charge_shifted,
                    radial_Ri,
                    radial_Ri_d,
                    radial_NL,
                    Ri_angular,
                    Ri_d_angular,
                    NL_angular,
                    device,
                    dtype)

        charge_energy = None
        charge_virial = None
        charge_position = None
        charge_box_original = None
        charge_volume = None
        if (self.charge_mode and self.atomic_charge_shifted is not None and
                position is not None and box_original is not None and
                (need_charge_energy or need_charge_virial)):
            charge_box_original = box_original.to(dtype=dtype, device=device)
            charge_volume = volume.to(dtype=dtype, device=device) if volume is not None else None
            if need_charge_energy:
                charge_position = position.detach().clone().to(dtype=dtype, device=device).requires_grad_(True)
                if need_charge_virial:
                    charge_energy, charge_virial = self.calculate_charge_energy_virial(
                        charge_position,
                        charge_box_original,
                        charge_volume,
                        num_atom,
                        self.atomic_charge_shifted,
                        dtype,
                        device)
                else:
                    charge_energy = self.calculate_charge_energy(
                        charge_position,
                        charge_box_original,
                        charge_volume,
                        num_atom,
                        self.atomic_charge_shifted,
                        dtype,
                        device)
            elif need_charge_virial:
                charge_virial = self.calculate_charge_virial(
                    position.to(dtype=dtype, device=device),
                    charge_box_original,
                    charge_volume,
                    num_atom,
                    self.atomic_charge_shifted,
                    dtype,
                    device)
        
        Egroup = self.get_egroup(Ei, Egroup_weight, divider) if Egroup_weight is not None else None
        # Ei = torch.squeeze(Ei, 1)


        # t1 = time.time()
        # check_cuda_memory(-1, -1, "FORWAR Ei")
        exist_rij = False
        if self.zbl is not None:
            condition = (Ri_angular[:, :, 0] > 0) & (Ri_angular[:, :, 0] < self.zbl)
            exist_rij = condition.any().item()
            if exist_rij:
                Ei_zbl, ri_zbl, ri_d_zbl, neigh_zbl = self.calculate_zbl(Ri_angular, Ri_d_angular, NL_angular, atom_type_map)
                Ei = Ei + Ei_zbl
            else:
                ri_zbl, ri_d_zbl, neigh_zbl = None, None, None
        else:
            ri_zbl, ri_d_zbl, neigh_zbl = None, None, None
        # t2 = time.time()
        # check_cuda_memory(-1, -1, "FORWAR E_zbl")

        split_sizes = num_atom.reshape(-1).tolist()
        energy_per_image = Ei.split(split_sizes)
        nep_Etot = torch.stack([x.sum() for x in energy_per_image]).unsqueeze(-1)
        Etot_for_energy = nep_Etot
        if charge_energy is not None:
            Etot_for_energy = Etot_for_energy + charge_energy.reshape(-1, 1)
        Etot_for_force = Etot_for_energy
        Etot = Etot_for_energy
        # Etot = torch.sum(Ei, 1).unsqueeze(1)

        if is_calc_f is False or need_force is False: #False: # is_calc_f is False:   ##is_calc_f is False
            Force, Virial = None, None
            # print("==single time: tall {} ei {} zbl ei {}".format(t2-t0, t1-t0, t2-t1))
        else:
            # t4 = time.time()
            if use_analytical_nep_grad:
                grad_feat_E_raw = grad_feat_E_scaled * self.q_scaler
                dE_radial = None
                dE_angular = None
                dE_zbl = None
                if self.train_2b:
                    radial_seed = grad_feat_E_raw[:, :self.two_feat_num]
                    dfeat_c2, dfeat_2b, dfeat_2b_noc = radial_context
                    dE_radial = CalcOps.calculateNepFeatInputGrad(
                        radial_seed,
                        self.c_param_2.contiguous(),
                        radial_Ri,
                        radial_NL,
                        dfeat_c2,
                        dfeat_2b,
                        dfeat_2b_noc,
                        atom_type_map,
                        self.multi_feat_num,
                        int(self.input_param.nep_param.fix_cij))
                if self.l_max_3b > 0:
                    angular_seed = grad_feat_E_raw[:, self.two_feat_num:]
                    dfeat_c3, dfeat_3b, dfeat_3b_noc, sum_fxyz = angular_context
                    dE_angular = CalcOps.calculateNepMbFeatInputGrad(
                        angular_seed,
                        self.c_param_3.contiguous(),
                        Ri_angular,
                        NL_angular,
                        dfeat_c3,
                        dfeat_3b,
                        dfeat_3b_noc,
                        sum_fxyz,
                        atom_type_map,
                        self.two_feat_num,
                        self.l_max_3b,
                        self.l_max_4b,
                        self.l_max_5b,
                        self.cutoff_angular,
                        int(self.input_param.nep_param.fix_cij))
                if ri_zbl is not None:
                    dE_zbl = torch.autograd.grad(
                        Etot_for_force,
                        ri_zbl,
                        grad_outputs=torch.ones_like(Etot_for_force),
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True)[0]
                    if dE_zbl is None:
                        dE_zbl = torch.zeros_like(ri_zbl)
                Force, Virial = self.calculate_force_virial_from_descriptor_grad(
                    dE_radial,
                    dE_angular,
                    dE_zbl,
                    radial_Ri,
                    radial_Ri_d,
                    Ri_angular,
                    Ri_d_angular,
                    ri_zbl,
                    ri_d_zbl,
                    radial_NL,
                    NL_angular,
                    neigh_zbl,
                    num_atom,
                    device,
                    dtype)
            else:
                Force, Virial = self.calculate_force_virial(radial_Ri, radial_Ri_d, 
                                                        Ri_angular, Ri_d_angular, 
                                                        ri_zbl, ri_d_zbl,
                                                        Etot_for_force, natoms_sum,
                                                        radial_NL, 
                                                        NL_angular, 
                                                        neigh_zbl,
                                                        num_atom,
                                                        device, dtype)
            if charge_energy is not None and charge_position is not None:
                charge_force = -torch.autograd.grad(
                    charge_energy.sum(),
                    charge_position,
                    retain_graph=True,
                    create_graph=True)[0]
                Force = Force + charge_force
                if charge_virial is not None:
                    Virial = Virial + charge_virial
            
            # t3 = time.time()
            # print("==single time: tall {} ei {} zbl ei {} force {}".format(t3-t0, t1-t0, t2-t1, t3-t2))
            # ==single time: t1 0.0015997886657714844 t2 0.0016467571258544922 t3 0.03717923164367676 t4 2.8371810913085938e-05 t5 0.0011038780212402344 t6 4.267692565917969e-05 t7 0.08994221687316895
            # print("==single time: t1 {} t2 {} t3 {} t4 {} t5 {} t6 {} t7 {}".format(t1-t0, t2-t1, t3-t2, t4-t3, t5-t4, t6-t5, t7-t6))
            # check_cuda_memory(-1, -1, "FORWAR calculate_force")
        return Etot, Ei, Force, Egroup, Virial, charge_predict, self.atomic_bec

    def calculate_bec(
        self,
        charge: torch.Tensor,
        charge_shifted: torch.Tensor,
        Ri: torch.Tensor,
        Ri_d: torch.Tensor,
        NL_radial: torch.Tensor,
        Ri_angular: torch.Tensor,
        Ri_d_angular: torch.Tensor,
        NL_angular: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype) -> torch.Tensor:
        atom_num = charge.shape[0]
        identity = torch.eye(3, dtype=dtype, device=device)
        bec = charge_shifted.reshape(-1, 1, 1) * identity.reshape(1, 3, 3)

        sources = []
        radial_source_index = None
        angular_source_index = None
        if self.train_2b:
            radial_source_index = len(sources)
            sources.append(Ri)
        if self.l_max_3b > 0:
            angular_source_index = len(sources)
            sources.append(Ri_angular)
        if len(sources) == 0:
            return (bec * self.sqrt_epsilon_inf.to(dtype=dtype, device=device)).reshape(atom_num, 9)

        def add_descriptor_bec(
            bec_value: torch.Tensor,
            grad_value: Optional[torch.Tensor],
            descriptor: torch.Tensor,
            descriptor_d: torch.Tensor,
            neigh: torch.Tensor,
            center: int) -> torch.Tensor:
            if grad_value is None:
                return bec_value
            valid = neigh[center] >= 0
            if not valid.any():
                return bec_value
            neigh_index = neigh[center][valid].to(torch.int64)
            r12 = descriptor[center, valid, 1:4]
            f12 = torch.mul(grad_value[center, valid].unsqueeze(-1), descriptor_d[center, valid]).sum(dim=-2)
            contribution = 0.5 * r12.unsqueeze(-1) * f12.unsqueeze(-2)
            center_update = torch.zeros_like(bec_value)
            neighbor_update = torch.zeros_like(bec_value)
            center_update[center] = contribution.sum(dim=0)
            neighbor_update.index_add_(0, neigh_index, -contribution)
            return bec_value + center_update + neighbor_update

        grads = torch.autograd.grad(
            charge.sum(),
            sources,
            retain_graph=True,
            create_graph=True,
            allow_unused=True)
        for center in range(atom_num):
            if radial_source_index is not None:
                bec = add_descriptor_bec(
                    bec,
                    grads[radial_source_index],
                    Ri,
                    Ri_d,
                    NL_radial,
                    center)
            if angular_source_index is not None:
                bec = add_descriptor_bec(
                    bec,
                    grads[angular_source_index],
                    Ri_angular,
                    Ri_d_angular,
                    NL_angular,
                    center)

        return (bec * self.sqrt_epsilon_inf.to(dtype=dtype, device=device)).reshape(atom_num, 9)

    def calculate_charge_energy(
        self,
        position: torch.Tensor,
        box_original: torch.Tensor,
        volume: Optional[torch.Tensor],
        num_atom: torch.Tensor,
        charge: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
        split_sizes = num_atom.reshape(-1).tolist()
        atom_starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=num_atom.dtype, device=device), num_atom.reshape(-1)[:-1]]),
            dim=0)
        energies = []
        alpha = torch.as_tensor(self.Pi / self.cutoff_radial, dtype=dtype, device=device)
        alpha_factor = 0.25 / (alpha * alpha)
        for image_idx, (start_tensor, atom_num) in enumerate(zip(atom_starts, split_sizes)):
            start = int(start_tensor.item())
            end = start + atom_num
            image_charge = charge[start:end]
            image_position = position[start:end]
            image_energy = self.calculate_charge_reciprocal_energy(
                image_position,
                image_charge,
                box_original[image_idx],
                volume[image_idx] if volume is not None else None,
                alpha,
                alpha_factor,
                dtype,
                device)
            energies.append(image_energy)
        return torch.stack(energies)

    def calculate_charge_energy_virial(
        self,
        position: torch.Tensor,
        box_original: torch.Tensor,
        volume: Optional[torch.Tensor],
        num_atom: torch.Tensor,
        charge: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        split_sizes = num_atom.reshape(-1).tolist()
        atom_starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=num_atom.dtype, device=device), num_atom.reshape(-1)[:-1]]),
            dim=0)
        energies = []
        virials = []
        alpha = torch.as_tensor(self.Pi / self.cutoff_radial, dtype=dtype, device=device)
        alpha_factor = 0.25 / (alpha * alpha)
        identity = torch.eye(3, dtype=dtype, device=device)
        for image_idx, (start_tensor, atom_num) in enumerate(zip(atom_starts, split_sizes)):
            start = int(start_tensor.item())
            end = start + atom_num
            image_energy, image_virial = self.calculate_charge_reciprocal_energy_virial(
                position[start:end],
                charge[start:end],
                box_original[image_idx],
                volume[image_idx] if volume is not None else None,
                alpha,
                alpha_factor,
                identity,
                dtype,
                device)
            energies.append(image_energy)
            virials.append(image_virial.reshape(9))
        return torch.stack(energies), torch.stack(virials)

    def calculate_charge_virial(
        self,
        position: torch.Tensor,
        box_original: torch.Tensor,
        volume: Optional[torch.Tensor],
        num_atom: torch.Tensor,
        charge: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
        split_sizes = num_atom.reshape(-1).tolist()
        atom_starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=num_atom.dtype, device=device), num_atom.reshape(-1)[:-1]]),
            dim=0)
        virials = []
        alpha = torch.as_tensor(self.Pi / self.cutoff_radial, dtype=dtype, device=device)
        alpha_factor = 0.25 / (alpha * alpha)
        identity = torch.eye(3, dtype=dtype, device=device)
        for image_idx, (start_tensor, atom_num) in enumerate(zip(atom_starts, split_sizes)):
            start = int(start_tensor.item())
            end = start + atom_num
            strain = torch.zeros((3, 3), dtype=dtype, device=device, requires_grad=True)
            deformation = identity + strain
            image_position = position[start:end].matmul(deformation.T)
            image_box = deformation.matmul(box_original[image_idx].reshape(3, 3)).reshape(-1)
            image_energy = self.calculate_charge_reciprocal_energy(
                image_position,
                charge[start:end],
                image_box,
                None,
                alpha,
                alpha_factor,
                dtype,
                device)
            image_virial = torch.autograd.grad(
                image_energy,
                strain,
                retain_graph=True,
                create_graph=True)[0]
            virials.append(image_virial.reshape(9))
        return -torch.stack(virials)

    def get_charge_kvecs(
        self,
        reciprocal: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        b3: torch.Tensor,
        abs_det: torch.Tensor,
        alpha: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        volume_k = (2.0 * self.Pi) ** 3 / abs_det
        n1_max = int(torch.floor(alpha * 2.0 * self.Pi * torch.linalg.cross(b2, b3).norm() / volume_k).item())
        n2_max = int(torch.floor(alpha * 2.0 * self.Pi * torch.linalg.cross(b3, b1).norm() / volume_k).item())
        n3_max = int(torch.floor(alpha * 2.0 * self.Pi * torch.linalg.cross(b1, b2).norm() / volume_k).item())
        n1_values = torch.arange(0, n1_max + 1, dtype=torch.int64, device=device)
        n2_values = torch.arange(-n2_max, n2_max + 1, dtype=torch.int64, device=device)
        n3_values = torch.arange(-n3_max, n3_max + 1, dtype=torch.int64, device=device)
        n1_grid, n2_grid, n3_grid = torch.meshgrid(n1_values, n2_values, n3_values, indexing="ij")
        nonzero = (n1_grid * n1_grid + n2_grid * n2_grid + n3_grid * n3_grid) != 0
        half_space = ~((n1_grid == 0) & ((n2_grid < 0) | ((n2_grid == 0) & (n3_grid < 0))))
        k_indices = torch.stack(
            [n1_grid[nonzero & half_space], n2_grid[nonzero & half_space], n3_grid[nonzero & half_space]],
            dim=-1,
        ).to(dtype=dtype)
        kvecs = k_indices.matmul(reciprocal.T)
        ksq = torch.sum(kvecs * kvecs, dim=-1)
        valid = ksq < (2.0 * self.Pi) ** 2 * alpha * alpha
        return kvecs[valid], ksq[valid]

    def calculate_charge_reciprocal_energy_virial(
        self,
        position: torch.Tensor,
        charge: torch.Tensor,
        box: torch.Tensor,
        volume: Optional[torch.Tensor],
        alpha: torch.Tensor,
        alpha_factor: torch.Tensor,
        identity: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        lattice = box.reshape(3, 3)
        det = torch.det(lattice)
        abs_det = torch.abs(det)
        if volume is not None:
            abs_det = torch.abs(volume.reshape(-1)[0])
        reciprocal = 2.0 * self.Pi * torch.linalg.inv(lattice).T
        b1, b2, b3 = reciprocal[:, 0], reciprocal[:, 1], reciprocal[:, 2]
        kvecs, ksq = self.get_charge_kvecs(reciprocal, b1, b2, b3, abs_det, alpha, dtype, device)
        prefactor = 2.0 * torch.abs((2.0 * self.Pi) / det)
        kr = position.matmul(kvecs.T)
        s_real = torch.sum(charge.reshape(-1, 1) * torch.cos(kr), dim=0)
        s_imag = -torch.sum(charge.reshape(-1, 1) * torch.sin(kr), dim=0)
        g = prefactor * torch.exp(-ksq * alpha_factor) / ksq
        k_energy = g * (s_real * s_real + s_imag * s_imag)
        energy = torch.sum(k_energy)
        k_outer = kvecs.reshape(-1, 3, 1) * kvecs.reshape(-1, 1, 3)
        virial_term = identity.reshape(1, 3, 3) - 2.0 * (alpha_factor + 1.0 / ksq).reshape(-1, 1, 1) * k_outer
        virial = torch.sum(k_energy.reshape(-1, 1, 1) * virial_term, dim=0)
        return self.K_C_SP * energy, self.K_C_SP * virial

    def calculate_charge_reciprocal_energy(
        self,
        position: torch.Tensor,
        charge: torch.Tensor,
        box: torch.Tensor,
        volume: Optional[torch.Tensor],
        alpha: torch.Tensor,
        alpha_factor: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device) -> torch.Tensor:
        lattice = box.reshape(3, 3)
        det = torch.det(lattice)
        abs_det = torch.abs(det)
        if volume is not None:
            abs_det = torch.abs(volume.reshape(-1)[0])
        reciprocal = 2.0 * self.Pi * torch.linalg.inv(lattice).T
        b1, b2, b3 = reciprocal[:, 0], reciprocal[:, 1], reciprocal[:, 2]
        kvecs, ksq = self.get_charge_kvecs(reciprocal, b1, b2, b3, abs_det, alpha, dtype, device)
        prefactor = 2.0 * torch.abs((2.0 * self.Pi) / det)
        kr = position.matmul(kvecs.T)
        s_real = torch.sum(charge.reshape(-1, 1) * torch.cos(kr), dim=0)
        s_imag = -torch.sum(charge.reshape(-1, 1) * torch.sin(kr), dim=0)
        g = prefactor * torch.exp(-ksq * alpha_factor) / ksq
        energy = torch.sum(g * (s_real * s_real + s_imag * s_imag))
        return self.K_C_SP * energy

    def shift_total_charge(
        self,
        charge: torch.Tensor,
        num_atom: torch.Tensor,
        charge_label: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        split_sizes = num_atom.reshape(-1).tolist()
        charge_per_image = charge.split(split_sizes)
        charge_sum = torch.stack([x.sum() for x in charge_per_image]).reshape(-1, 1)
        if charge_label is None:
            charge_label = torch.zeros_like(charge_sum)
        else:
            charge_label = charge_label.reshape(-1, 1).to(dtype=charge.dtype, device=charge.device)
        correction = (charge_label - charge_sum) / num_atom.reshape(-1, 1).to(dtype=charge.dtype)
        shifted = []
        for image_charge, image_correction in zip(charge_per_image, correction):
            shifted.append(image_charge + image_correction)
        return charge_sum, torch.cat(shifted, dim=0)

    def calculate_Ri(self,
                     ImagedR: torch.Tensor, 
                     ImagedR_angular: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = ImagedR[:, :, 0].abs() > 1e-5
        Ri_d = torch.zeros(ImagedR.shape[0], ImagedR.shape[1], 4, 3, dtype=dtype, device=device)
        Ri_d[:, :, 0, 0][mask] = ImagedR[:, :, 1][mask] / ImagedR[:, :, 0][mask]
        Ri_d[:, :, 1, 0][mask] = 1
        # dy
        Ri_d[:, :, 0, 1][mask] = ImagedR[:, :, 2][mask] / ImagedR[:, :, 0][mask]
        Ri_d[:, :, 2, 1][mask] = 1
        # dz
        Ri_d[:, :, 0, 2][mask] = ImagedR[:, :, 3][mask] / ImagedR[:, :, 0][mask]
        Ri_d[:, :, 3, 2][mask] = 1 


        mask = ImagedR_angular[:, :, 0].abs() > 1e-5
        Ri_d_angular = torch.zeros(ImagedR_angular.shape[0], ImagedR_angular.shape[1], 4, 3, dtype=dtype, device=device)
        Ri_d_angular[:, :, 0, 0][mask] = ImagedR_angular[:, :, 1][mask] / ImagedR_angular[:, :, 0][mask]
        Ri_d_angular[:, :, 1, 0][mask] = 1
        # dy
        Ri_d_angular[:, :, 0, 1][mask] = ImagedR_angular[:, :, 2][mask] / ImagedR_angular[:, :, 0][mask]
        Ri_d_angular[:, :, 2, 1][mask] = 1
        # dz
        Ri_d_angular[:, :, 0, 2][mask] = ImagedR_angular[:, :, 3][mask] / ImagedR_angular[:, :, 0][mask]
        Ri_d_angular[:, :, 3, 2][mask] = 1 

        return ImagedR, Ri_d, ImagedR_angular, Ri_d_angular

    def calculate_Ei(self, 
                     Imagetype_map: torch.Tensor,
                     feats: torch.Tensor,
                     device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
        Ei = torch.zeros(Imagetype_map.shape[0], dtype=self.dtype, device=device)
        charge = torch.zeros(Imagetype_map.shape[0], dtype=self.dtype, device=device) if self.charge_mode else None
        # fit_net_dict = {idx: fit_net for idx, fit_net in enumerate(self.fitting_net)}
        for idx, fit_net in enumerate(self.fitting_net):
            # fit_net = fit_net_dict.get(nn_i)
            # S_Rij = Ri[:, indices, ntype_1 * self.maxNeighborNum:(ntype_1+1) * self.maxNeighborNum, 0].unsqueeze(-1)
            mask = (Imagetype_map == idx)
            if not mask.any():
                continue
            indices = torch.arange(len(Imagetype_map.flatten()),device=device)[mask]  
            feat = feats[indices, :]
            output_ntype = fit_net.forward(feat)
            if self.charge_mode:
                energy_ntype, charge_ntype = output_ntype
                Ei[mask] = energy_ntype.reshape(-1)
                charge[mask] = charge_ntype.reshape(-1)
            else:
                Ei[mask] = output_ntype.reshape(-1)
        if self.charge_mode and self.gpumd_nep4:
            Ei = Ei + self.common_bias
        return Ei, charge

    def calculate_Ei_with_grad(
            self,
            Imagetype_map: torch.Tensor,
            feats_scaled: torch.Tensor,
            device: torch.device
            ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        Ei = torch.zeros(Imagetype_map.shape[0], dtype=self.dtype, device=device)
        grad_feat_E_scaled = torch.zeros_like(feats_scaled)
        if self.charge_mode:
            charge = torch.zeros(Imagetype_map.shape[0], dtype=self.dtype, device=device)
            grad_feat_Q_scaled = torch.zeros_like(feats_scaled)
        else:
            charge = None
            grad_feat_Q_scaled = None

        for idx, fit_net in enumerate(self.fitting_net):
            mask = (Imagetype_map == idx)
            if not mask.any():
                continue
            indices = torch.arange(len(Imagetype_map.flatten()), device=device)[mask]
            feat = feats_scaled[indices, :]
            if self.charge_mode:
                energy_ntype, charge_ntype, grad_e_ntype, grad_q_ntype = fit_net.forward_with_input_grad(feat)
                Ei[mask] = energy_ntype.reshape(-1)
                charge[mask] = charge_ntype.reshape(-1)
                grad_feat_E_scaled[indices, :] = grad_e_ntype
                grad_feat_Q_scaled[indices, :] = grad_q_ntype
            else:
                energy_ntype, grad_e_ntype = fit_net.forward_with_input_grad(feat)
                Ei[mask] = energy_ntype.reshape(-1)
                grad_feat_E_scaled[indices, :] = grad_e_ntype

        if self.charge_mode and self.gpumd_nep4:
            Ei = Ei + self.common_bias
        return Ei, charge, grad_feat_E_scaled, grad_feat_Q_scaled

    def calculate_force_virial_from_descriptor_grad(
            self,
            dE: Optional[torch.Tensor],
            dE_angular: Optional[torch.Tensor],
            dE_zbl: Optional[torch.Tensor],
            Ri: Optional[torch.Tensor],
            Ri_d: Optional[torch.Tensor],
            Ri_angular: Optional[torch.Tensor],
            Ri_d_angular: Optional[torch.Tensor],
            Ri_zbl: Optional[torch.Tensor],
            Ri_d_zbl: Optional[torch.Tensor],
            list_neigh: Optional[torch.Tensor],
            list_neigh_angular: Optional[torch.Tensor],
            list_neigh_zbl: Optional[torch.Tensor],
            num_atom: torch.Tensor,
            device: torch.device,
            dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        num_atom_flat = num_atom.reshape(-1)
        natoms_sum = int(num_atom_flat.sum().item())
        batch_size = num_atom_flat.shape[0]

        def aggregate_cpu(branch_dE, branch_Ri, branch_Ri_d, branch_list_neigh):
            branch_dE = torch.unsqueeze(branch_dE, dim=-1)
            dE_Rid = torch.mul(branch_dE, branch_Ri_d).sum(dim=-2)
            force = torch.zeros((natoms_sum + 1, 3), device=device, dtype=dtype)
            force[1:natoms_sum + 1, :] = -1 * dE_Rid.sum(dim=-2)
            indice = (branch_list_neigh + 1).flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64)
            values = dE_Rid.view(-1, 3)
            force.scatter_add_(0, indice, values).view(natoms_sum + 1, 3)
            force = force[1:, :]

            image_atom_index = torch.cumsum(num_atom_flat, dim=0)
            image_atom_index = torch.cat((torch.tensor([0], device=device), image_atom_index), dim=0)
            virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
            for i in range(0, batch_size):
                start = image_atom_index[i]
                end = image_atom_index[i + 1]
                virial[i, 0] = (branch_Ri[start:end, :, 1] * dE_Rid[start:end, :, 0]).flatten().sum(dim=0)
                virial[i, 1] = (branch_Ri[start:end, :, 1] * dE_Rid[start:end, :, 1]).flatten().sum(dim=0)
                virial[i, 2] = (branch_Ri[start:end, :, 1] * dE_Rid[start:end, :, 2]).flatten().sum(dim=0)
                virial[i, 4] = (branch_Ri[start:end, :, 2] * dE_Rid[start:end, :, 1]).flatten().sum(dim=0)
                virial[i, 5] = (branch_Ri[start:end, :, 2] * dE_Rid[start:end, :, 2]).flatten().sum(dim=0)
                virial[i, 8] = (branch_Ri[start:end, :, 3] * dE_Rid[start:end, :, 2]).flatten().sum(dim=0)
                virial[i, 3] = virial[i, 1]
                virial[i, 6] = virial[i, 2]
                virial[i, 7] = virial[i, 5]
            return force, virial

        def aggregate_gpu(branch_dE, branch_Ri, branch_Ri_d, branch_list_neigh):
            branch_Ri_d = branch_Ri_d.view(natoms_sum, -1, 3)
            dE_tmp = branch_dE.view(natoms_sum, 1, -1)
            force = -1 * torch.matmul(dE_tmp, branch_Ri_d).squeeze(-2)
            image_dr = branch_Ri[:, :, 1:].clone()
            force = CalcOps.calculateNepForce(branch_list_neigh, branch_dE, branch_Ri_d, force)[0]
            virial = CalcOps.calculateNepVirial(branch_list_neigh, branch_dE, image_dr, branch_Ri_d, num_atom)[0]
            return force, virial

        aggregate = aggregate_cpu if device.type == "cpu" else aggregate_gpu
        force_total = torch.zeros((natoms_sum, 3), device=device, dtype=dtype)
        virial_total = torch.zeros((batch_size, 9), device=device, dtype=dtype)

        if self.train_2b and dE is not None:
            force, virial = aggregate(dE, Ri, Ri_d, list_neigh)
            force_total = force_total + force
            virial_total = virial_total + virial
        if self.l_max_3b > 0 and dE_angular is not None:
            force, virial = aggregate(dE_angular, Ri_angular, Ri_d_angular, list_neigh_angular)
            force_total = force_total + force
            virial_total = virial_total + virial
        if Ri_zbl is not None and dE_zbl is not None:
            force, virial = aggregate(dE_zbl, Ri_zbl, Ri_d_zbl, list_neigh_zbl)
            force_total = force_total + force
            virial_total = virial_total + virial

        return -force_total, -virial_total
     
    def calculate_force_virial(self, 
                                Ri: torch.Tensor,
                                Ri_d: torch.Tensor,
                                Ri_angular: torch.Tensor,
                                Ri_d_angular: torch.Tensor,
                                Ri_zbl: torch.Tensor,
                                Ri_d_zbl: torch.Tensor,
                                Etot: torch.Tensor,
                                natoms_sum: int,
                                list_neigh: torch.Tensor,
                                list_neigh_angular: torch.Tensor,
                                list_neigh_zbl: torch.Tensor,
                                num_atom: torch.Tensor,
                                device: torch.device,
                                dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # t7 = time.time()
        grad_inputs = []
        grad_names = []
        if self.train_2b:
            grad_inputs.append(Ri)
            grad_names.append("radial")
        if self.l_max_3b > 0:
            grad_inputs.append(Ri_angular)
            grad_names.append("angular")
        if Ri_zbl is not None:
            grad_inputs.append(Ri_zbl)
            grad_names.append("zbl")

        grads = torch.autograd.grad(
            Etot,
            grad_inputs,
            grad_outputs=torch.ones_like(Etot),
            retain_graph=True,
            create_graph=True,
            allow_unused=True)
        grad_map = dict(zip(grad_names, grads))
        if self.train_2b:
            dE = grad_map["radial"]
            if dE is None:
                dE = torch.zeros_like(Ri)
        if self.l_max_3b > 0:
            dE_angular = grad_map["angular"]
            if dE_angular is None:
                dE_angular = torch.zeros_like(Ri_angular)
        if Ri_zbl is not None:
            dE_zbl = grad_map["zbl"]
            if dE_zbl is None:
                dE_zbl = torch.zeros_like(Ri_zbl)
        return self.calculate_force_virial_from_descriptor_grad(
            dE if self.train_2b else None,
            dE_angular if self.l_max_3b > 0 else None,
            dE_zbl if Ri_zbl is not None else None,
            Ri,
            Ri_d,
            Ri_angular,
            Ri_d_angular,
            Ri_zbl,
            Ri_d_zbl,
            list_neigh,
            list_neigh_angular,
            list_neigh_zbl,
            num_atom,
            device,
            dtype)
        # t8 = time.time()
        '''
        # this result is same as the above code
        mask: List[Optional[torch.Tensor]] = [torch.ones_like(Ei)]
        dE = torch.autograd.grad([Ei], [Ri], grad_outputs=mask, retain_graph=True, create_graph=True)[0]
        '''
        if device.type == "cpu": #True: 
            batch_size = num_atom.shape[0]
            image_atom_index = torch.cumsum(num_atom, dim=0).squeeze(-1)
            image_atom_index = torch.cat((torch.tensor([0], device=device), image_atom_index), dim=0)
            if self.train_2b:
                dE = torch.unsqueeze(dE, dim=-1)
                dE_Rid = torch.mul(dE, Ri_d).sum(dim=-2)
                Force = torch.zeros((natoms_sum + 1, 3), device=device, dtype=dtype)
                Force[1:natoms_sum + 1, :] = -1 * dE_Rid.sum(dim=-2)
                Virial = torch.zeros((batch_size, 9), device=device, dtype=dtype)
                indice = (list_neigh+1).flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64) # list_neigh's index start from 1, so the Force's dimension should be natoms_sum + 1
                values = dE_Rid.view(-1, 3)
                Force.scatter_add_(0, indice, values).view(natoms_sum + 1, 3)
                
                for i in range(0, batch_size):
                    Virial[i, 0] = (Ri[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid[image_atom_index[i]:image_atom_index[i+1], :, 0]).flatten().sum(dim=0) # xx
                    Virial[i, 1] = (Ri[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid[image_atom_index[i]:image_atom_index[i+1], :, 1]).flatten().sum(dim=0) # xy
                    Virial[i, 2] = (Ri[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # xz
                    Virial[i, 4] = (Ri[image_atom_index[i]:image_atom_index[i+1], :, 2] * dE_Rid[image_atom_index[i]:image_atom_index[i+1], :, 1]).flatten().sum(dim=0) # yy
                    Virial[i, 5] = (Ri[image_atom_index[i]:image_atom_index[i+1], :, 2] * dE_Rid[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # yz
                    Virial[i, 8] = (Ri[image_atom_index[i]:image_atom_index[i+1], :, 3] * dE_Rid[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # zz
                    Virial[i, 3] = Virial[i, 1]
                    Virial[i, 6] = Virial[i, 2]
                    Virial[i, 7] = Virial[i, 5]
                Force = Force[1:, :]
            if self.l_max_3b > 0:
                dE_angular = torch.unsqueeze(dE_angular, dim=-1)
                dE_Rid_angular = torch.mul(dE_angular, Ri_d_angular).sum(dim=-2)
                Force_angular = torch.zeros((natoms_sum + 1, 3), device=device, dtype=dtype)
                Force_angular[1:natoms_sum + 1, :] = -1 * dE_Rid_angular.sum(dim=-2)
                Virial_angular = torch.zeros((batch_size, 9), device=device, dtype=dtype)
                indice = (list_neigh_angular+1).flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64) # list_neigh's index start from 1, so the Force's dimension should be natoms_sum + 1
                values = dE_Rid_angular.view(-1, 3)
                Force_angular.scatter_add_(0, indice, values).view(natoms_sum + 1, 3)

                for i in range(0, batch_size):
                    Virial_angular[i, 0] = (Ri_angular[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid_angular[image_atom_index[i]:image_atom_index[i+1], :, 0]).flatten().sum(dim=0) # xx
                    Virial_angular[i, 1] = (Ri_angular[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid_angular[image_atom_index[i]:image_atom_index[i+1], :, 1]).flatten().sum(dim=0) # xy
                    Virial_angular[i, 2] = (Ri_angular[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid_angular[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # xz
                    Virial_angular[i, 4] = (Ri_angular[image_atom_index[i]:image_atom_index[i+1], :, 2] * dE_Rid_angular[image_atom_index[i]:image_atom_index[i+1], :, 1]).flatten().sum(dim=0) # yy
                    Virial_angular[i, 5] = (Ri_angular[image_atom_index[i]:image_atom_index[i+1], :, 2] * dE_Rid_angular[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # yz
                    Virial_angular[i, 8] = (Ri_angular[image_atom_index[i]:image_atom_index[i+1], :, 3] * dE_Rid_angular[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # zz
                    Virial_angular[i, 3] = Virial_angular[i, 1]
                    Virial_angular[i, 6] = Virial_angular[i, 2]
                    Virial_angular[i, 7] = Virial_angular[i, 5]
                Force_angular = Force_angular[1:, :]

            if Ri_zbl is not None:
                dE_zbl = torch.unsqueeze(dE_zbl, dim=-1)
                dE_Rid_zbl = torch.mul(dE_zbl, Ri_d_zbl).sum(dim=-2)
                Force_zbl = torch.zeros((natoms_sum + 1, 3), device=device, dtype=dtype)
                Force_zbl[1:natoms_sum + 1, :] = -1 * dE_Rid_zbl.sum(dim=-2)
                Virial_zbl = torch.zeros((batch_size, 9), device=device, dtype=dtype)
                indice = (list_neigh_zbl+1).flatten().unsqueeze(-1).expand(-1, 3).to(torch.int64)
                values = dE_Rid_zbl.view(-1, 3)
                Force_zbl.scatter_add_(0, indice, values).view(natoms_sum + 1, 3)

                for i in range(0, batch_size):
                    Virial_zbl[i, 0] = (Ri_zbl[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid_zbl[image_atom_index[i]:image_atom_index[i+1], :, 0]).flatten().sum(dim=0) # xx
                    Virial_zbl[i, 1] = (Ri_zbl[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid_zbl[image_atom_index[i]:image_atom_index[i+1], :, 1]).flatten().sum(dim=0) # xy
                    Virial_zbl[i, 2] = (Ri_zbl[image_atom_index[i]:image_atom_index[i+1], :, 1] * dE_Rid_zbl[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # xz
                    Virial_zbl[i, 4] = (Ri_zbl[image_atom_index[i]:image_atom_index[i+1], :, 2] * dE_Rid_zbl[image_atom_index[i]:image_atom_index[i+1], :, 1]).flatten().sum(dim=0) # yy
                    Virial_zbl[i, 5] = (Ri_zbl[image_atom_index[i]:image_atom_index[i+1], :, 2] * dE_Rid_zbl[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # yz
                    Virial_zbl[i, 8] = (Ri_zbl[image_atom_index[i]:image_atom_index[i+1], :, 3] * dE_Rid_zbl[image_atom_index[i]:image_atom_index[i+1], :, 2]).flatten().sum(dim=0) # zz
                    Virial_zbl[i, 3] = Virial_zbl[i, 1]
                    Virial_zbl[i, 6] = Virial_zbl[i, 2]
                    Virial_zbl[i, 7] = Virial_zbl[i, 5]
                Force_zbl = Force_zbl[1:, :]
        else: # gpu code
            if self.train_2b:
                Ri_d = Ri_d.view(natoms_sum, -1, 3)
                dE_tmp = dE.view(natoms_sum, 1, -1)
                Force = -1 * torch.matmul(dE_tmp, Ri_d).squeeze(-2)
                ImageDR = Ri[:,:,1:].clone()
                # tmp_list_neigh = torch.unsqueeze(list_neigh,2)
                # tmp_list_neigh = (tmp_list_neigh - 1).type(torch.int)
                Force = CalcOps.calculateNepForce(list_neigh, dE, Ri_d, Force)[0] # the save order in memory of dE and dE_tmp are in the same
                Virial,atom_virial = CalcOps.calculateNepVirial(list_neigh, dE, ImageDR, Ri_d, num_atom)
            if self.l_max_3b > 0:
                Ri_d_angular = Ri_d_angular.view(natoms_sum, -1, 3)
                dE_angular_tmp = dE_angular.view(natoms_sum, 1, -1)
                Force_angular = -1 * torch.matmul(dE_angular_tmp, Ri_d_angular).squeeze(-2)
                ImageDR_angular = Ri_angular[:,:,1:].clone()
                # tmp_list_neigh_angular = torch.unsqueeze(list_neigh_angular,2)
                # tmp_list_neigh_angular = (tmp_list_neigh_angular - 1).type(torch.int)
                Force_angular = CalcOps.calculateNepForce(list_neigh_angular, dE_angular, Ri_d_angular, Force_angular)[0]
                Virial_angular = CalcOps.calculateNepVirial(list_neigh_angular, dE_angular, ImageDR_angular, Ri_d_angular, num_atom)[0]
            if Ri_zbl is not None:
                Ri_d_zbl = Ri_d_zbl.view(natoms_sum, -1, 3)
                dE_zbl_tmp = dE_zbl.view(natoms_sum, 1, -1)
                Force_zbl = -1 * torch.matmul(dE_zbl_tmp, Ri_d_zbl).squeeze(-2)
                ImageDR_zbl = Ri_zbl[:,:,1:].clone()
                # list_neigh_zbl = torch.unsqueeze(list_neigh_zbl,2)
                # list_neigh_zbl = (list_neigh_zbl - 1).type(torch.int)
                Force_zbl = CalcOps.calculateNepForce(list_neigh_zbl, dE_zbl, Ri_d_zbl, Force_zbl)[0]
                Virial_zbl = CalcOps.calculateNepVirial(list_neigh_zbl, dE_zbl, ImageDR_zbl, Ri_d_zbl, num_atom)[0]                
        # t9 = time.time()
        # print("t8 {} t9 {}".format(t8-t7, t9-t8))
        # del dE ???
        # print(-Force)
        if Ri_zbl is not None:
            if self.train_2b and self.l_max_3b > 0:
                return -(Force + Force_angular + Force_zbl), -(Virial + Virial_angular + Virial_zbl)
            elif self.l_max_3b > 0:
                return -(Force_angular + Force_zbl), -(Virial_angular + Virial_zbl)
            else:
                return -(Force + Force_zbl), -(Virial + Virial_zbl)
        else:
            if self.train_2b and self.l_max_3b > 0:
                return -(Force + Force_angular), -(Virial + Virial_angular)
            elif self.l_max_3b > 0:
                return -Force_angular, -Virial_angular
            return -Force, -Virial


    def calculate_qn(self,
                     Imagetype_map: torch.Tensor,
                     j_type_map: torch.Tensor,
                     Ri: torch.Tensor, 
                     j_type_map_angular: torch.Tensor,
                     Ri_angular: torch.Tensor, 
                     device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn start")
        if self.train_2b:
            c2 = self.get_c(self.c_param_2, self.n_max_radial,  self.n_base_radial,  Imagetype_map, j_type_map)
            feat_2b = self.cal_feat_2body(Ri[:, :, 0], Imagetype_map, 
                                        c2,
                                        self.n_max_radial, self.n_base_radial, self.cutoff_radial, self.rcinv_radial)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b end")
        # R = Ri_angular[:, :, :, 0]
        # xyz = Ri_angular[:, :, :, 1:]

        if self.l_max_3b > 0:
            c3 = self.get_c(self.c_param_3, self.n_max_angular, self.n_base_angular, Imagetype_map, j_type_map_angular)  if self.l_max_3b > 0 else None
            multi_feat = self.cal_feat_multi_body(Ri_angular[:, :, 0], Ri_angular[:, :, 1:], Imagetype_map, 
                                            c3,
                                            self.n_max_angular, self.n_base_angular, self.cutoff_angular, self.rcinv_angular, self.l_max_3b)            
            return torch.concat([feat_2b, multi_feat], dim=-1)
        else:
            return feat_2b

    def get_c(self,
            c_2b : torch.Tensor,
            n_max_r : int,
            n_base_r:int,
            Imagetype_map : torch.Tensor,
            j_type_map : torch.Tensor) -> torch.Tensor: #get c params from c[n_type,n_type, n_max, n_base] 
        atom_nums = j_type_map.shape[0]
        j_list_nums = j_type_map.shape[1]

        # j_type_map = j_type_map.clone()
        mask = j_type_map > -1
        j_type_map[mask] = (Imagetype_map*self.ntypes).unsqueeze(-1).repeat(1, j_type_map.shape[1])[mask]+j_type_map[mask]
        j_type_map3 = j_type_map.flatten()
        mask2 = j_type_map3 > -1
        
        c_list = torch.zeros([j_type_map3.shape[0], n_max_r+1, n_base_r+1], dtype=c_2b.dtype, device=c_2b.device)
        c2 = c_2b.reshape(self.ntypes_sq, c_2b.shape[-2],c_2b.shape[-1])
        c_list[mask2] = c2[j_type_map3[mask2]]
        # c1 = c[Imagetype_map, :, :, :] # search by i
        c2 = c_list.view(atom_nums, j_list_nums, n_max_r+1, n_base_r+1)
        return c2.transpose(2, 1)

    def cal_fk(self,
                rij: torch.Tensor,
                n_base: int,
                rcut: float,
                rcinv: float) -> torch.Tensor:
        mask = (rij.abs() > 1e-5) & (rij <= rcut) # 超过截断半径的rij, fk(rij) 为0，那么 c*t*fc = 0,导数也为0，因为fk=0, dfk=0
        fc = torch.zeros_like(rij)
        fc[mask]  = 0.5 + 0.5 * torch.cos(self.Pi * rij[mask] * rcinv)

        tk  = torch.zeros([rij.shape[0], rij.shape[1], n_base+1], dtype=rij.dtype).to(rij.device)# [b,i,j,M]
        fk  = torch.zeros([rij.shape[0], rij.shape[1], n_base+1], dtype=rij.dtype).to(rij.device)# [b,i,j,M]
        
        x = torch.zeros_like(rij)
        x[mask] = 2 * (rij[mask] * rcinv - 1)**2 - 1

        # 先不要考虑n_max_r计算完之后做扩展,再与c做乘法, fc也要最后再乘
        tk[:, :, 0][mask]   = 1.0 # t0
        tk[:, :, 1][mask]   = x[mask] # t1

        fk[:, :, 0][mask]   = fc[mask]   # 0.5 *( t0(x) + 1) * fc(rij), t0(x) = 1
        fk[:, :, 1][mask]   = 0.5 * (x[mask] + 1) * fc[mask]   # 0.5 *( t1(x) + 1 ) * fc(rij), t1(x) = x
        # fk[:,:,:,1] = torch.tensor(x.data).unsqueeze(2).repeat(1,1,fk.shape[2],1)
        for n in range(2, n_base + 1):## 参考nep-cpu
            tk[:,:,n][mask]      = 2 * x[mask] * tk[:,:,n - 1][mask] - tk[:,:,n - 2][mask]
            fk[:,:,n][mask]      = 0.5 * (tk[:,:,n][mask] +1) * fc[mask]                  # [b,i,N,j,M]
            
        return fk

    def cal_feat_2body(self,
                        rij: torch.Tensor,
                        Imagetype_map: torch.Tensor,
                        # j_type_map: torch.Tensor,
                        c2:torch.Tensor,
                        n_max: int,
                        n_base: int,
                        rcut: float,
                        rcinv: float) -> torch.Tensor:
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b start")
        # c2 = self.get_c(self.c_param_2, n_max, n_base, Imagetype_map, j_type_map)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b c2")
        fk = self.cal_fk(rij, n_base, rcut, rcinv)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b fk")
        fk_res = fk.unsqueeze(1).repeat(1, n_max+1, 1, 1)    # n_max_r+1 个feature区别是在c系数上，fk是一样的 c2 [4, 96, 5, 200, 13]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b fk_res")
        feat_2b = (c2 * fk_res).sum(-1).sum(-1) # sum n_base_r and sum j
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 2b feat_2b")
        # type 1 [0,0,:25]  type 2 list [0,0,25:78]
        # mask_q0000: List[Optional[torch.Tensor]] = [torch.ones_like(feat_2b[0,0,0])]
        # dfeat_c2 = torch.autograd.grad([feat_2b[0,0,0]], [self.c_param_2], grad_outputs=mask_q0000, retain_graph=True, create_graph=True)[0]

        return feat_2b 

    '''
    description: 
        for nep_cpu, the qn of 3b, 4b, 5b orders are :
                n=0       n=1       n=2       n=3       n=4 (n to max_angular+1)
        L=1 q_3b_01   q_3b_11   q_3b_21   q_3b_31   q_3b_41
        L=2 q_3b_02   q_3b_12   q_3b_22   q_3b_32   q_3b_42
        L=3 q_3b_03   q_3b_13   q_3b_23   q_3b_33   q_3b_43
        L=4 q_3b_04   q_3b_14   q_3b_24   q_3b_34   q_3b_44    
        L=4 q_4b_022  q_4b_122  q_4b_222  q_4b_322  q_4b_422
        L=4 q_5b_0111 q_5b_1111 q_5b_2111 q_5b_3111 q_5b_4111
    return {*}
    author: wuxingxing
    '''    
    def cal_feat_multi_body(
                    self,
                    rij: torch.Tensor,
                    xyz: torch.Tensor,
                    Imagetype_map: torch.Tensor,
                    # j_type_map: torch.Tensor,
                    c3:torch.Tensor,
                    n_max: int,
                    n_base: int,
                    rcut: float,
                    rcinv: float,
                    l_max_3b: int) -> torch.Tensor:
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 3b start")
        # c3 = self.get_c(self.c_param_3, rij, n_max, n_base, Imagetype_map)
        # c3 = self.get_c(self.c_param_3, n_max, n_base, Imagetype_map, j_type_map)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn ck start")
        fk = self.cal_fk(rij, n_base, rcut, rcinv)
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn fk start")
        # c * tk # now fk 是对的 c3 不对，rij 对，
        gn1 = (c3 * (fk.unsqueeze(1).repeat(1, n_max+1, 1, 1))).sum(-1)   # n_max_r 个feature区别是在c系数上，fk是一样的 # sum n_base c3 [4, 96, 5, 200, 13]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn gn1 start")
        gn2 = gn1.unsqueeze(-1).repeat(1, 1, 1, 24) # lmax_3body = 4 [1, 96, 5, 200, 24]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn gn2 start")
        blm = self.cal_blm_ij(rij, xyz, rcut) #[1, 96, 200, 24]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn blm start")
        blm2 = blm.unsqueeze(1).repeat(1, n_max + 1, 1, 1)# [1, 96, 5, 200, 24] 这里blm = blm(xij,yij,zij) / rij^l
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn blm2 start")
        snlm = (gn2 * blm2).sum(2) #gn * blm, then sum j : [1, 96, 5, 200, 24] -> [1, 96, 5, 24]
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn snlm start")
        snlm_sq = snlm**2
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn snlm_sq start")
        # 常系数C
        c_lm = self.C3B.unsqueeze(0).unsqueeze(0).repeat(snlm_sq.shape[0],snlm_sq.shape[1],1)
        qnlm = c_lm * snlm_sq
        qnl = torch.zeros([snlm_sq.shape[0], snlm_sq.shape[1], l_max_3b], dtype=qnlm.dtype, device=qnlm.device)
        qnl[:, :, 0] = qnlm[:, :, 0:3].sum(-1)
        qnl[:, :, 1] = qnlm[:, :, 3:8].sum(-1)
        qnl[:, :, 2] = qnlm[:, :, 8:15].sum(-1)
        qnl[:, :, 3] = qnlm[:, :, 15:24].sum(-1)
        # feat_3b = qnl.view(qnl.shape[0], qnl.shape[1], -1) # 3体feature
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 3b end")

        # feature 4
        # feat_4b = None
        if self.l_max_4b != 0:
            sn20_sq = snlm_sq[:, :, 3]
            sn21_sq = snlm_sq[:, :, 4]
            sn22_sq = snlm_sq[:, :, 5]
            sn23_sq = snlm_sq[:, :, 6]
            sn24_sq = snlm_sq[:, :, 7]
            feat_4b = self.C4B[0] * snlm[:, :, 3] * sn20_sq + \
                    self.C4B[1] * snlm[:, :, 3] * (sn21_sq + sn22_sq) +\
                    self.C4B[2] * snlm[:, :, 3] * (sn23_sq + sn24_sq) + \
                    self.C4B[3] * snlm[:, :, 6] * (sn22_sq - sn21_sq) +\
                    self.C4B[4] * snlm[:, :, 4] * snlm[:, :, 5] * snlm[:, :, 7]
        else:
            feat_4b = None
        # feature 5
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 4b end")
        # feat_5b = None
        if self.l_max_5b != 0:
            sn10_sq = snlm_sq[:, :, 0]
            sn11_sq = snlm_sq[:, :, 1]
            sn12_sq = snlm_sq[:, :, 2]
            feat_5b = self.C5B[0] * sn10_sq * sn10_sq + self.C5B[1] * sn10_sq * (sn11_sq + sn12_sq) + self.C5B[2] * (sn11_sq + sn12_sq)**2
        else:
            feat_5b = None
        # check_cuda_memory(-1, -1, "FORWAR calculate_qn 5b end")

        if feat_5b is not None:
            feat_5b = feat_5b.unsqueeze(-2)
        if feat_4b is not None:
            feat_4b = feat_4b.unsqueeze(-2)

        if feat_5b is not None and feat_4b is not None:
            return torch.concat([qnl.transpose(2,1), feat_4b, feat_5b], dim=-2).view(qnl.shape[0], -1)
        elif feat_5b is not None:
            return torch.concat([qnl.transpose(2,1), feat_5b], dim=-2).view(qnl.shape[0],  -1)
        elif feat_4b is not None:
            return torch.concat([qnl.transpose(2,1), feat_4b], dim=-2).view(qnl.shape[0], -1)
        else:
            return qnl.transpose(2,1).reshape(qnl.shape[0], -1)
            
    def cal_blm_ij(self,
            rij: torch.Tensor,
            xyz: torch.Tensor,
            rcut: float,
            ) -> torch.Tensor:
        mask = (rij.abs() > 1e-5) & (rij <= rcut)
        d12inv = torch.zeros_like(rij)
        d12inv[mask] = 1/rij[mask]
        x12 = d12inv[mask] * xyz[:, :, 0][mask]
        y12 = d12inv[mask] * xyz[:, :, 1][mask]
        z12 = d12inv[mask] * xyz[:, :, 2][mask]
        
        x12sq = x12 ** 2
        y12sq = y12 ** 2
        z12sq = z12 ** 2
        x12sq_minus_y12sq = x12sq - y12sq

        blm = torch.zeros([xyz.shape[0], xyz.shape[1], 24], dtype=xyz.dtype, device=xyz.device)
        blm[:, :, 0][mask] = z12                                                            # Y10       b10 / r^1 
        blm[:, :, 1][mask] = x12                                                            # Y11_real  b11 / r^1
        blm[:, :, 2][mask] = y12                                                            # Y11_imag  b12 / r^1
        blm[:, :, 3][mask] = (3.0 * z12sq - 1.0)                                            # Y20       b20 / r^2
        blm[:, :, 4][mask] = x12 * z12                                                      # Y21_real  b21 / r^2
        blm[:, :, 5][mask] = y12 * z12                                                      # Y21_imag  b22 / r^2
        blm[:, :, 6][mask] = x12sq_minus_y12sq                                              # Y22_real  b23 / r^2
        blm[:, :, 7][mask] = 2.0 * x12 * y12                                                # Y22_imag  b24 / r^2
        blm[:, :, 8][mask] = (5.0 * z12sq - 3.0) * z12                                      # Y30       b30 / r^3       
        blm[:, :, 9][mask] = (5.0 * z12sq - 1.0) * x12                                      # Y31_real  b31 / r^3
        blm[:, :, 10][mask] = (5.0 * z12sq - 1.0) * y12                                      # Y31_imag  b32 / r^3
        blm[:, :, 11][mask] = x12sq_minus_y12sq * z12                                        # Y32_real  b33 / r^3
        blm[:, :, 12][mask] = 2.0 * x12 * y12 * z12                                          # Y32_imag  b34 / r^3
        blm[:, :, 13][mask] = (x12 * x12 - 3.0 * y12 * y12) * x12                            # Y33_real  b35 / r^3
        blm[:, :, 14][mask] = (3.0 * x12 * x12 - y12 * y12) * y12                            # Y33_imag  b36 / r^3
        blm[:, :, 15][mask] = ((35.0 * z12sq - 30.0) * z12sq + 3.0)                          # Y40       b40 / r^4
        blm[:, :, 16][mask] = (7.0 * z12sq - 3.0) * x12 * z12                                # Y41_real  b41 / r^4
        blm[:, :, 17][mask] = (7.0 * z12sq - 3.0) * y12 * z12                                # Y41_iamg  b42 / r^4
        blm[:, :, 18][mask] = (7.0 * z12sq - 1.0) * x12sq_minus_y12sq                        # Y42_real  b43 / r^4
        blm[:, :, 19][mask] = (7.0 * z12sq - 1.0) * x12 * y12 * 2.0                          # Y42_imag  b44 / r^4
        blm[:, :, 20][mask] = (x12sq - 3.0 * y12sq) * x12 * z12                              # Y43_real  b45 / r^4
        blm[:, :, 21][mask] = (3.0 * x12sq - y12sq) * y12 * z12                              # Y43_imag  b46 / r^4
        blm[:, :, 22][mask] = (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq)  # Y44_real  b47 / r^4
        blm[:, :, 23][mask] = (4.0 * x12 * y12 * x12sq_minus_y12sq)                          # Y44_imag  b48 / r^4

        return blm

    def calculate_zbl(self,
        Ri_angular :torch.Tensor, 
        Ri_d_angular :torch.Tensor, 
        list_neigh_angular :torch.Tensor, 
        type_map :torch.Tensor):
        # 获取真实原子序数 Z
        Z_i = self.atom_type_device[type_map].unsqueeze(1)           # [n_atoms, 1]
        # 近邻类型索引 → Z_j
        type_j_idx = torch.full_like(list_neigh_angular, -1, dtype=torch.long)
        valid_neigh = list_neigh_angular != -1
        type_j_idx[valid_neigh] = type_map[list_neigh_angular[valid_neigh]]
        
        Z_j = torch.full_like(list_neigh_angular, -1, dtype=torch.long)
        valid_z = type_j_idx != -1
        Z_j[valid_z] = self.atom_type_device[type_j_idx[valid_z]]

        if self.zbl_factor is not None:
            # 计算每对 (i,j) 的 new_rcut = min(self.zbl, (cov_i + cov_j) * factor)
            cov_i = self.COVALENT_RADIUS[Z_i]                        # [n_atoms, 1]
            cov_j = self.COVALENT_RADIUS[Z_j]                        # [n_atoms, n_neigh]
            cov_sum = (cov_i + cov_j) * self.zbl_factor
            rcut_per_pair = torch.minimum(
                torch.full_like(cov_sum, self.zbl, dtype=cov_sum.dtype, device=cov_sum.device),
                cov_sum
            )
        else:
            # 固定 self.zbl 为 outer rcut
            rcut_per_pair = torch.full_like(Ri_angular[:, :, 0], self.zbl, 
                                           dtype=Ri_angular.dtype, device=Ri_angular.device)

        rij = Ri_angular[:, :, 0]
        # 1. ri_zbl：rij > new_rcut 的位置置 0
        mask_zero = (rij > rcut_per_pair)
        ri_zbl = Ri_angular.clone().detach()
        ri_zbl[mask_zero] = 0
        ri_zbl.requires_grad_()

        # 2. ri_d_zbl
        ri_d_zbl = Ri_d_angular.clone().detach()
        ri_d_zbl[mask_zero] = 0
        
        # 3. neigh_zbl
        neigh_zbl = list_neigh_angular.clone().detach()
        neigh_zbl[mask_zero] = -1
        
        # 4. type_zbl（保存 type index，用于后续取 Z）
        type_zbl = torch.full_like(list_neigh_angular, -1, dtype=torch.long)
        type_zbl[valid_neigh] = type_j_idx[valid_neigh]
        type_zbl[mask_zero] = -1
        
        # 计算 ZBL 能量
        Ei_zbl = self.cal_zbl(ri_zbl, type_zbl, type_map, rcut_per_pair)
        return Ei_zbl, ri_zbl, ri_d_zbl, neigh_zbl

    def cal_zbl(self,
                ri_zbl: torch.Tensor,
                type_zbl: torch.Tensor,      # type index (0/1 等)
                type_map:torch.Tensor,
                rcut_per_pair: torch.Tensor) -> torch.Tensor:
        rij = ri_zbl[:, :, 0]
        safe_rij = torch.where(rij.abs() > 1e-8, rij, torch.tensor(1e-8, dtype=rij.dtype, device=rij.device))
        
        fc = torch.zeros_like(rij, dtype=rij.dtype)
        if self.zbl_factor is not None:
            # fc = 0.5 + 0.5 * cos(π * r / rcut_per_pair)   for r < rcut_per_pair
            # fc = 0                                        for r >= rcut_per_pair
            # inner = 0, outer = rcut_per_pair
            mask_inner = (rij.abs() < rcut_per_pair) & (rij.abs() > 1e-8)
            if mask_inner.any():
                r_val = rij[mask_inner]
                rcut_val = rcut_per_pair[mask_inner]
                fc[mask_inner] = 0.5 + 0.5 * torch.cos(self.Pi * r_val / rcut_val)
        else:
            # inner = rcut/2, outer = rcut (rcut = self.zbl)
            rcut = rcut_per_pair
            mask_inner = (rij.abs() < rcut * 0.5)
            fc[mask_inner] = 1.0
            
            mask_switch = (rij.abs() >= rcut * 0.5) & (rij.abs() <= rcut)
            if mask_switch.any():
                r_val = rij[mask_switch]
                rcut_val = rcut[mask_switch]
                fc[mask_switch] = 0.5 + 0.5 * torch.cos(
                    (self.Pi / (rcut_val * 0.5)) * (r_val - rcut_val * 0.5)
                )
        # ==================== 真实原子序数 Z ====================
        Z_i = self.atom_type_device[type_map].unsqueeze(1).expand_as(type_zbl)
        Z_j = torch.full_like(type_zbl, -1, dtype=torch.long)
        valid = type_zbl != -1
        Z_j[valid] = self.atom_type_device[type_zbl[valid]]
        
        # ==================== 计算 phi 和 ei_zbl ====================
        mask_compute = valid & (rij.abs() > 1e-8)
        
        x = torch.zeros_like(rij, dtype=rij.dtype)
        x[mask_compute] = rij[mask_compute] * (
            Z_i[mask_compute]**0.23 + Z_j[mask_compute]**0.23
        ) * 2.134563
        
        phi = torch.zeros_like(rij, dtype=rij.dtype)
        phi[mask_compute] = (
            self.zbl_para[0] * torch.exp(-self.zbl_para[1] * x[mask_compute]) +
            self.zbl_para[2] * torch.exp(-self.zbl_para[3] * x[mask_compute]) +
            self.zbl_para[4] * torch.exp(-self.zbl_para[5] * x[mask_compute]) +
            self.zbl_para[6] * torch.exp(-self.zbl_para[7] * x[mask_compute])
        )
        
        ei_zbl = torch.zeros_like(rij, dtype=rij.dtype)
        ei_zbl[mask_compute] = (
            self.K_C_SP *
            Z_i[mask_compute] *
            Z_j[mask_compute] *
            phi[mask_compute] *
            fc[mask_compute] /
            safe_rij[mask_compute]
        )
        
        return 0.5 * ei_zbl.sum(dim=-1)

    # def cal_zbl_fc(self,
    #             rij: torch.Tensor,
    #             rcut: float) -> torch.Tensor:
    #     mask = (rij.abs() >= rcut) & (rij <= 2 * rcut) # 超过截断半径的rij, fk(rij) 为0，那么 c*t*fc = 0,导数也为0，因为fk=0, dfk=0
    #     fc = torch.zeros_like(rij)
    #     fc[mask]  = 0.5 + 0.5 * torch.cos((self.Pi / rcut) * (rij[mask] * rcut))
    #     mask = (rij.abs() < rcut)
    #     fc[mask] = 1
    #     return fc
    
    # def cal_zbl_phi(self,
    #     rij: torch.Tensor,
    #     type_zbl:torch.Tensor,
    #     type_map:torch.Tensor,
    #     atom_type: torch.Tensor
    #     ):
    #     zj = torch.zeros_like(type_zbl)
    #     mask = type_zbl != -1
    #     zj[mask] = atom_type[type_zbl[mask]]
    #     zi = atom_type[type_map]
    #     alpha = ((zi.view(1, zi.shape[0], 1))**0.23 + zj**0.23) * 2.134563
    #     x = rij[mask] * alpha[mask]
    #     phi = self.zbl_para[0] * torch.exp(-self.zbl_para[1]* x) + \
    #             self.zbl_para[2] * torch.exp(-self.zbl_para[3]* x) + \
    #                 self.zbl_para[4] * torch.exp(-self.zbl_para[5]* x) + \
    #                     self.zbl_para[6] * torch.exp(-self.zbl_para[7]* x)
    #     ZiZj = zi.view(1, zi.shape[0], 1) * zj

    #     Ei_zbl = self.K_C_SP * ZiZj * phi / rij
