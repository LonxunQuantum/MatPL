#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
sys.path.append(os.getcwd())
#import parameters as pm    
import use_para as pm    
# import prepare as pp
# pp.readFeatnum()


################################################################
# fully connection nn
# Ei Neural Network
################################################################

# ACTIVE = torch.tanh
# ACTIVE = F.softplus 
# ACTIVE = torch.relu  
def dsigmoid(x):
    return torch.sigmoid(x) * (1 - torch.sigmoid(x))
ACTIVE = torch.sigmoid
 
dACTIVE = dsigmoid

B_INIT= -0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LNNet(nn.Module):
    def __init__(self, itype = 0):  #atomtypes=len(pm.atomType)
        super(LNNet,self).__init__()
        self.itype = itype  # itype =0,1 if CuO
        self.atomType = pm.atomType
        self.natoms = pm.natoms   #[32,32]
        self.biases = []
        self.weights = nn.ParameterList()
        for i in range(pm.nLayers-1):
            in_putsize = pm.nFeatures if i==0 else pm.nNodes[i-1,itype] #0为type1,1为type2
            w = nn.Parameter(torch.randn(pm.nNodes[i, itype], in_putsize))
            # self.b = nn.Parameter(torch.randn(pm.nNodes[i, itype], 1))
            self.weights.append(w)
            # self.biases.append(self.b)
        if (pm.nLayers == 1):
            self.output = nn.Parameter(torch.randn(1, input_size))
        else:
            self.output = nn.Parameter(torch.randn(1, pm.nNodes[pm.nLayers-2,itype]))  #最后一层
        # self.register_parameter("output", self.output)
    #     self.__set__init(self.output)

    # def __set__init(self,layer):
    #     init.normal_(layer.weight, mean = 0, std = 1)       
    #     init.constant_(layer.bias,val=B_INIT)

    def forward(self,  image, dfeat, neighbor, Egroup_weight, divider):
        natoms_index = [0]
        temp = 0
        for i in self.natoms:
            temp += i
            natoms_index.append(temp)    #[0,32,64]
        input_data = image
        
        batch_size = image.shape[0]
        # input_grad_allatoms = torch.zeros([batch_size, natoms_index[-1], pm.nFeatures])
        for batch_index in range(batch_size):
            atom_index_temp = 0
            for idx, natom in enumerate(self.natoms):  #[32,32]  [108] 
                for i in range(natom):
                    x = input_data[batch_index, atom_index_temp+i, :]  #[42]
                    L = []
                    dL = []
                    x = nn.functional.linear(x, self.weights[0])
                    L.append(ACTIVE(x))
                    dL.append(dACTIVE(x))
                    for layer in range(1, pm.nLayers-1):
                        L.append(ACTIVE(nn.functional.linear(L[layer-1], self.weights[layer])))
                        dL.append(dACTIVE(nn.functional.linear(L[layer-1], self.weights[layer])))
                    predict = nn.functional.linear(L[pm.nLayers-2], self.output)  #网络的最后一层
                    if(i==0):
                        Ei = predict #[1]
                    else:
                        Ei = torch.cat((Ei, predict), dim=0)    #[108]
                    res = self.output.t()  #(30,1)
                    while layer >= 0:
                        res = dL[layer].squeeze() * res.squeeze()  #[30,1]*[1,30]-->[30,30]
                        res = res * self.weights[layer].t() #[30]*[60,30]-->[60,30]
                        res = res.sum(axis=-1)  #[60,30]-->[60]
                        layer -= 1
                    # input_grad_allatoms.index_put_(2, res)
                    if(i==0):
                        one_sample_input_grad_allatoms = res.unsqueeze(0)  #[1, 42]
                    else:
                        one_sample_input_grad_allatoms = torch.cat((one_sample_input_grad_allatoms, res.unsqueeze(0)), dim=0)  #[108,42]
                one_sample_Etot = Ei.sum(dim=0)  #[108]-->[1]
            if(batch_index==0):
                batches_Ei = Ei.unsqueeze(0) #[108]-->[1,108]
                Etot = one_sample_Etot.unsqueeze(0) #[1] --> [1,1]
                input_grad_allatoms = one_sample_input_grad_allatoms.unsqueeze(0) #[108,42]-->[1,108,42]
            else:
                batches_Ei = torch.cat((batches_Ei, Ei.unsqueeze(0)), dim=0)  #[1,108] --> [2,108]
                Etot = torch.cat((Etot, one_sample_Etot.unsqueeze(0)), dim=0) #[1,1]-->[2,1]   -->2
                input_grad_allatoms = torch.cat((input_grad_allatoms, one_sample_input_grad_allatoms.unsqueeze(0)), dim=0) #[2,108,42]

        Force = torch.zeros((batch_size, natoms_index[-1], 3)).to(device)
        for batch_index in range(batch_size):
            atom_index_temp = 0
            for idx, natom in enumerate(self.natoms):  #[32,32]    
                for i in range(natom):
                    neighbori = neighbor[batch_index, atom_index_temp + i]  # neighbor [40, 64, 100] neighbori [1, 100]
                    neighbor_number = neighbori.shape[-1]
                    atom_force = torch.zeros((1, 3)).to(device)
                    for nei in range(neighbor_number):
                        nei_index = neighbori[nei] - 1 #第几个neighbor
                        if(nei_index == -1):
                            break 
                        atom_force += torch.matmul(input_grad_allatoms[batch_index, nei_index, :], dfeat[batch_index, atom_index_temp + i, nei, :, :])
                        # print("The dEtot/dfeature for batch_index %d, neighbor_inde %d" %(batch_index, nei_index))
                        # print(input_grad_allatoms[batch_index, nei_index, :])
                    Force[batch_index, atom_index_temp+i] = atom_force * 1e10

        # Egroup = self.get_egroup(Ei, Egroup_weight, divider)

        return Etot, Force

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
