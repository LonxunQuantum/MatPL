import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
sys.path.append(os.getcwd())
#import parameters as pm    
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+"/pre_data")

import use_para as pm    
# import prepare as pp
# pp.readFeatnum()
from src.model.dmirror import dmirror_FC
from src.model.FC import f_FC
# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15

# setup logging module
logger = logging.getLogger('train.MLFF')

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

# MLFF contains two parts:
# MLFF_dmirror implementation and MLFF_autograd implementation
#
def d_sigmoid(x):
    return torch.sigmoid(x) * (1.0 - torch.sigmoid(x))
def d_tanh(x):
    return 1-torch.tanh(x)**2
def d_relu(x):
    res = torch.zeros_like(x)
    mask = x > 0
    res[mask] = 1
    return res
LeakyReLU = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
def d_LeakyReLU(x):
    res = torch.ones_like(x)
    mask = x < 0
    res[mask] = 0.01
    return res

class MLFF(nn.Module):
    def __init__(self, net_cfg, activation_type, device, magic=False, autograd=True):
        super(MLFF, self).__init__()
        # config parameters
        self.atomType = pm.atomType
        self.device = device
        self.dim_feat = pm.nFeatures
        self.nneighbors = pm.maxNeighborNum
        # network
        if(autograd == False):
            if (net_cfg == 'default'):
                self.net_cfg = pm.MLFF_dmirror_cfg
                info("MLFF_dmirror: using default net_cfg: pm.MLFF_dmirror_cfg")
                info(self.net_cfg)
            else:
                net_cfg = 'pm.' + net_cfg
                self.net_cfg = eval(net_cfg)
                info("MLFF_dmirror: using specified net_cfg: %s" %net_cfg)
                info(self.net_cfg)
            
            if (activation_type == 'sigmoid'):
                self.activation_type = 'sigmoid'
                info("MLFF_dmirror: using sigmoid activation")
                self.net = dmirror_FC(self.net_cfg, torch.sigmoid, d_sigmoid, magic)
            elif (activation_type == 'tanh'):
                self.activation_type = 'tanh'
                info("MLFF_dmirror: using tanh activation")
                self.net = dmirror_FC(self.net_cfg, torch.tanh, d_tanh, magic)
            elif (activation_type == 'relu'):
                self.activation_type = 'relu'
                info("MLFF_dmirror: using relu activation")
                self.net = dmirror_FC(self.net_cfg, torch.relu, d_relu, magic)
            elif (activation_type == 'softplus'):
                self.activation_type = 'softplus'
                info("MLFF_dmirror: using softplus activation")
                self.net = dmirror_FC(self.net_cfg, F.softplus, F.sigmoid, magic)
            else:
                error("MLFF_dmirror: unsupported activation_type: %s" %activation_type)
                raise RuntimeError("MLFF_dmirror: unsupported activation_type: %s" %activation_type)
        else:
            if (net_cfg == 'default'):
                self.net_cfg = pm.MLFF_autograd_cfg
                info("MLFF_autograd: using default net_cfg: pm.MLFF_autograd_cfg")
                info(self.net_cfg)
            else:
                net_cfg = 'pm.' + net_cfg
                self.net_cfg = eval(net_cfg)
                info("MLFF_autograd: using specified net_cfg: %s" %net_cfg)
                info(self.net_cfg)
            
            if (activation_type == 'sigmoid'):
                self.activation_type = 'sigmoid'
                info("MLFF_autograd: using sigmoid activation")
                self.net = f_FC(self.net_cfg, torch.sigmoid, magic)
            elif (activation_type == 'softplus'):
                self.activation_type = 'softplus'
                info("MLFF_autograd: using softplus activation")
                self.net = f_FC(self.net_cfg, F.softplus, magic)
            elif (activation_type == 'tanh'):
                self.activation_type = 'tanh'
                info("MLFF_autograd: using tanh activation")
                self.net = f_FC(self.net_cfg, torch.tanh, magic)
            elif (activation_type == 'relu'):
                self.activation_type = 'relu'
                info("MLFF_autograd: using relu activation")
                self.net = f_FC(self.net_cfg, torch.relu, magic)
            else:
                error("MLFF_autograd: unsupported activation_type: %s" %activation_type)
                raise RuntimeError("MLFF_autograd: unsupported activation_type: %s" %activation_type)
    
    def forward(self, image, dfeat, neighbor, Egroup_weight, divider):
        batch_size = image.shape[0]
        natoms_index = [0]
        temp = 0
        for i in pm.natoms:
            temp += i
            natoms_index.append(temp)    #[0,32,64]

        result_dEi_dFeat_fortran = torch.zeros((batch_size, natoms_index[-1] + 1, self.dim_feat), dtype=torch.double).to(self.device)
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            temp_image = image[:, natoms_index[i]:natoms_index[i+1]]
            temp_result_Ei = torch.zeros((batch_size, natoms_index[i+1])).to(self.device)
            temp_result_dEi_dFeat = torch.zeros((batch_size, natoms_index[i+1], self.dim_feat)).to(self.device)
            temp_result_Ei, temp_result_dEi_dFeat = self.net(temp_image)
            if(i==0):
                Ei= temp_result_Ei 
                dE = temp_result_dEi_dFeat
            else:
                Ei = torch.cat((Ei, temp_result_Ei), dim=1)    #[64,1]
                dE = torch.cat((dE, temp_result_dEi_dFeat), dim=1)
        result_dEi_dFeat_fortran[:, 1:, :] = dE
        
        Etot = torch.sum(Ei,1)
        # Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        Force = torch.zeros((batch_size, pm.natoms[-1], 3)).to(self.device)

        # first version of calculate force
        '''
        for batch_idx in range(batch_size):
            for i in range(natoms_index[-1]):
                # get atom_idx & neighbor_idx
                my_neighbor = neighbor[batch_idx, i]
                neighbor_idx = my_neighbor.nonzero().squeeze().type(torch.int64)
                atom_idx = my_neighbor[neighbor_idx].type(torch.int64) - 1
                # calculate Force
                #   a.shape = [neighbor_num, 1, self.dim_feat]
                #   b.shape = [neighbor_num, self.dim_feat, 3]
                #   Force.shape = [batch_size, self.natoms, 3]
                a = dE[batch_idx, atom_idx].unsqueeze(1)
                b = dfeat[batch_idx, i, neighbor_idx]
                Force[batch_idx, i, :] = torch.matmul(a, b).sum([0, 1])
        '''

        # dEi_neighbors.shape = [batch_size, self.natoms, self.nneighbors, 1, self.dim_feat]
        # dfeat.shape         = [batch_size, self.natoms, self.nneighbors, self.dim_feat, 3]
        #
        # n_a_idx_fortran: neighbor's atom index in fortran style, starts
        # from 1, and 0 means empty neighbor slot. result_dEi_dFeat_fortran
        # matches this in it's atom dimension
        # 
        # n_a_ofs_fortran_b: offset of neighbor's atom index in the batched
        # atom list, the offset step value for each image is (natoms + 1)
        # to match our fortran style atom index accommodation
        #
        # n_a_idx_fortran_b: neighbor's atom index in the batched atom list
        #
        n_a_idx_fortran = neighbor.reshape(batch_size * natoms_index[-1] * self.nneighbors)
        n_a_ofs_fortran_b = torch.arange(0, batch_size * (natoms_index[-1] + 1), natoms_index[-1] + 1)\
                            .repeat_interleave(natoms_index[-1] * self.nneighbors).to(self.device)
        n_a_idx_fortran_b = n_a_idx_fortran.type(torch.int64) + n_a_ofs_fortran_b
        dEi_neighbors = result_dEi_dFeat_fortran.reshape(batch_size * (natoms_index[-1] + 1), self.dim_feat)[n_a_idx_fortran_b,]\
                        .reshape(batch_size, natoms_index[-1], self.nneighbors, 1, self.dim_feat)
        Force = torch.matmul(dEi_neighbors, dfeat).sum([2, 3])

        return Etot, Ei, Force

    def get_egroup(self, Ei, Egroup_weight, divider):
        batch_size = Ei.shape[0]
        Egroup = torch.zeros_like(Ei)
        for i in range(batch_size):
            Etot1 = Ei[i]
            weight_inner = Egroup_weight[i]
            E_inner = torch.matmul(weight_inner, Etot1)
            Egroup[i] = E_inner
        Egroup_out = torch.divide(Egroup, divider)
        return Egroup_out
