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
import time
# import prepare as pp
# pp.readFeatnum()


################################################################
# fully connection nn
# Ei Neural Network
################################################################

# ACTIVE = torch.tanh
# ACTIVE = F.softplus 
# ACTIVE = torch.relu  

# def dsigmoid(x):
#     return torch.sigmoid(x) * (1 - torch.sigmoid(x))
# ACTIVE = torch.sigmoid
 
# dACTIVE = dsigmoid

ACTIVE = F.softplus
dACTIVE = torch.sigmoid

# ACTIVE = torch.tanh
# def dtanh(x):
#     return 1-torch.tanh(x)**2
# dACTIVE = dtanh

# ACTIVE = torch.relu
# def drelu(x):
#     res = torch.zeros_like(x)
#     mask = x > 0
#     res[mask] = 1
#     return res
# dACTIVE = drelu

# def no_act(x):
#     return x
# def no_dact(x):
#     return torch.ones_like(x)
# ACTIVE = no_act
# dACTIVE = no_dact

# ACTIVE = torch.nn.LeakyReLU(negative_slope=0.01, inplace=False)
# def dLeakyReLU(x):
#     res = torch.ones_like(x)
#     mask = x < 0
#     res[mask] = 0.01
#     return res
# dACTIVE = dLeakyReLU


B_INIT= -0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FCNet(nn.Module):
    def __init__(self, BN = False, Dropout = False, itype = 0):  #atomtypes=len(pm.atomType)
        super(FCNet,self).__init__()
        self.dobn = BN
        self.dodrop = Dropout                                  
        self.fcs=[]
        self.bns=[]
        self.drops=[]
        self.itype= itype  # itype =0,1 if CuO         
        self.weights = nn.ParameterList()
        self.bias = nn.ParameterList()

        for i in range(pm.nLayers-1):
            in_putsize = pm.nFeatures if i==0 else pm.nNodes[i-1,itype] #0为type1,1为type2
            w = nn.Parameter(torch.randn(pm.nNodes[i, itype], in_putsize))
            b = nn.Parameter(torch.randn(pm.nNodes[i, itype]))
            self.weights.append(w)
            self.bias.append(b)
        w = nn.Parameter(torch.randn(1, pm.nNodes[pm.nLayers-2,itype]))  #最后一层  #ones
        b = nn.Parameter(torch.randn(1))  #最后一层    #zeros
        self.weights.append(w)
        self.bias.append(b)

    def forward(self, x):
        L = []
        dL = []
        # print("L[0]=F.linear(x, self.weights[0]")
        # print(x)
        # print('~'*10)
        # print(self.weights[0])
        # print('~'*10)
        # dL.append(dACTIVE(F.linear(x, self.weights[0], bias=self.bias[0])))
        # L.append(ACTIVE(F.linear(x, self.weights[0], bias=self.bias[0])))
        # Fout_0=F.linear(x, self.weights[0], bias=self.bias[0])
        Fout_0 = torch.matmul(x, self.weights[0].t()) + self.bias[0]
        L.append(ACTIVE(Fout_0))
        dL.append(dACTIVE(Fout_0))
        for ilayer in range(1, pm.nLayers-1):
            # Fout_temp = F.linear(L[ilayer-1], self.weights[ilayer], bias=self.bias[ilayer])
            # L.append(ACTIVE(Fout_temp))
            # dL.append(dACTIVE(Fout_temp))  
            Fout_temp = torch.matmul(L[ilayer-1], self.weights[ilayer].t()) + self.bias[ilayer]
            L.append(ACTIVE(Fout_temp))
            dL.append(dACTIVE(Fout_temp)) 
        # print('L[1]='+str(L[1]))
        # print('dL[1]='+str(dL[1]))
        # predict = F.linear(L[pm.nLayers-2], self.weights[-1], bias=self.bias[-1])  #网络的最后一层
        predict = torch.matmul(L[pm.nLayers-2], self.weights[-1].t()) + self.bias[-1]
        ilayer += 1
        grad = self.weights[ilayer]
        ilayer -= 1
        while ilayer >= 0:
            grad = dL[ilayer] * grad   #(2,108,30)*(1,30)-->(2,108,30)
            grad = grad.unsqueeze(2) * self.weights[ilayer].t()  #(2,108,1,30)*(60,30)-->(2,108,60,30)
            grad = grad.sum(axis=-1)  #(2,108,60,30)-->(2,108,60)
            ilayer -= 1
        
        return predict, grad


class preMLFFNet(nn.Module):
    def __init__(self, atomType = pm.atomType, natoms = pm.natoms):  # atomType=[8,32]
        super(preMLFFNet,self).__init__()
        self.atomType = atomType
        self.natoms = pm.natoms   #[32,32]
        self.models = nn.ModuleList()
        for i in range(len(self.atomType)):  #i=[0,1]
            self.models.append(FCNet(itype = i, Dropout=True))   # Dropout=True


    def forward(self, image, dfeat, neighbor):
        natoms_index = [0]
        temp = 0
        for i in self.natoms:
            temp += i
            natoms_index.append(temp)    #[0,32,64]
        input_data = image
        
        for i in range(len(natoms_index)-1):
            x = input_data[:, natoms_index[i]:natoms_index[i+1]]
            _, predict = self.models[i](x)
            if(i==0):
                Ei = predict #[32, 1]
            else:
                Ei = torch.cat((Ei, predict), dim=1)    #[64,1]
        Etot = Ei.sum(dim=1)
        return Etot, Ei


class MLFFNet(nn.Module):
    def __init__(self, scalers, atomType = pm.atomType, natoms = pm.natoms):  #atomType=[8,32]
        super(MLFFNet,self).__init__()
        self.atomType = atomType
        self.natoms = pm.natoms   #[32,32]
        self.models = nn.ModuleList()
        self.scalers = scalers
        for i in range(len(self.atomType)):  #i=[0,1]
            self.models.append(FCNet(itype = i, Dropout=True))   # Dropout=True


    def forward(self, image, dfeat, neighbor, Egroup_weight, divider):
        start = time.time()
        natoms_index = [0]
        temp = 0
        for i in self.natoms:
            temp += i
            natoms_index.append(temp)    #[0,32,64]
        
        # for i in range(len(natoms_index)-1):
        for i in range(pm.ntypes):
            itype = pm.atomType[i]
            x = image[:, natoms_index[i]:natoms_index[i+1]]
            predict, grad = self.models[i](x)
            
            # scale_feat_a = torch.tensor(self.scalers.feat_as[itype], device=device, dtype=torch.float)
            # grad = grad * scale_feat_a
            if(i==0):
                Ei = predict #[32, 1]
                dE = grad
            else:
                Ei = torch.cat((Ei, predict), dim=1)    #[64,1]
                dE = torch.cat((dE, grad), dim=1)
        # de = self.get_de(image, dfeat, neighbor)  #函数的方法计算de
        # input_grad_allatoms = dE     #手动计算的dE
        
        cal_ei_de = time.time()
        Etot = Ei.sum(dim=1)
        
        # test = Ei.sum()
        # test.backward(retain_graph=True)
        # dE = image.grad

        Ei.unsqueeze(2)
        dEtest = torch.autograd.grad(Ei, x, grad_outputs=torch.ones_like(Ei), create_graph=True, retain_graph=True)
        dEtest = torch.stack(list(dEtest), dim=0).squeeze(0)
        dE = dEtest
        # import ipdb;ipdb.set_trace()
        # dfeat = dfeat.transpose(3, 4) # 40 108 100 3 42
        # dE = dE.unsqueeze(2).unsqueeze(2)
        # force_all = dfeat * dE
        # force_all = force_all.sum(-1) # 40 108 100 3
        flag = neighbor > 0 # 40 108 100
        batch_size = image.shape[0]
        natom = image.shape[1]
        neighbor -= 1
        force = torch.zeros(image.shape[0], image.shape[1], 3).to(device)
        for batch_id in range(batch_size):
            for atom_id in range(natom):
                neighbor_list = neighbor[batch_id, atom_id, flag[batch_id, atom_id]].type(torch.int64)
                tmp_de = dE[batch_id, neighbor_list.tolist()].unsqueeze(1)
                tmp_dfeat = dfeat[batch_id, atom_id, :len(neighbor_list)]
                tmp_force = torch.matmul(tmp_de, tmp_dfeat).sum([0, 1])
                force[batch_id, atom_id] = tmp_force
        # import ipdb;ipdb.set_trace()
                # tmp = force_all[batch_id, atom_id, neighbor[batch_id, atom_id, flag[batch_id, atom_id]]]
        # batch_size = image.shape[0]
        # import ipdb; ipdb.set_trace()
        # Force = torch.zeros((batch_size, natoms_index[-1], 3)).to(device)
        # for batch_index in range(batch_size):
        #     atom_index_temp = 0
        #     for idx, natom in enumerate(self.natoms):  #[32,32]    
        #         for i in range(natom):
        #             neighbori = neighbor[batch_index, atom_index_temp + i]  # neighbor [40, 64, 100] neighbori [1, 100]
        #             neighbor_number = neighbori.shape[-1]
        #             atom_force = torch.zeros((1, 3)).to(device)
        #             for nei in range(neighbor_number):
        #                 nei_index = neighbori[nei] - 1 #第几个neighbor
        #                 if(nei_index == -1):
        #                     break 
        #                 atom_force += torch.matmul(input_grad_allatoms[batch_index, nei_index, :], dfeat[batch_index, atom_index_temp + i, nei, :, :])
        #                 # print("The dEtot/dfeature for batch_index %d, neighbor_inde %d" %(batch_index, nei_index))
        #                 # print(input_grad_allatoms[batch_index, nei_index, :])
        #             Force[batch_index, atom_index_temp+i] = atom_force
        Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        # return Force, Etot, Ei, Egroup
        end = time.time()
        # print("cal ei de time:", cal_ei_de - start, 's')
        # print("cal force time:", end - cal_ei_de, 's')
        return Etot,  Ei, force

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

    def get_de(self, image, dfeat, neighbor):
        batch_size = image.shape[0]
        atom_type_num = len(self.atomType)
        for i in range(atom_type_num):
            model_weight = self.models[i].state_dict()
            layer_name = []
            for weight_name in model_weight.keys():
                layer_name.append(weight_name)
            layers = int(len(layer_name)/2)
            W = []
            B = []
            L = []
            dL = []
            # W.append(model_weight['fc0.weight'].transpose(0, 1))            
            # B.append(model_weight['fc0.bias'])   
            W.append(model_weight[layer_name[0]])            
            B.append(model_weight[layer_name[layers]])       
            dL.append(dACTIVE(torch.matmul(image, W[0].T) + B[0]))
            L.append(ACTIVE(torch.matmul(image, W[0].T) + B[0]))
            for ilayer in range(1, pm.nLayers-1):
                # W.append(model_weight['fc' + str(ilayer) + '.weight'].transpose(0, 1))            
                # B.append(model_weight['fc' + str(ilayer) + '.bias'])   
                W.append(model_weight[layer_name[ilayer]])            
                B.append(model_weight[layer_name[layers + ilayer]])         
                dL.append(dACTIVE(torch.matmul(L[ilayer-1], W[ilayer].T) + B[ilayer]))
                L.append(ACTIVE(torch.matmul(L[ilayer-1], W[ilayer].T) + B[ilayer]))
            ilayer += 1
            # W.append(model_weight['output.weight'].transpose(0, 1))            
            # B.append(model_weight['output.bias']) 
            W.append(model_weight[layer_name[ilayer]])            
            B.append(model_weight[layer_name[layers + ilayer]])           
            # res = W[ilayer].transpose(0, 1)  
            res = W[ilayer]
            ilayer -= 1
            while ilayer >= 0:
                res = dL[ilayer] * res   #(2,108,30)*(1,30)-->(2,108,30)
                res = res.unsqueeze(2) * W[ilayer].T  #(2,108,1,30)*(60,30)-->(2,108,60,30)
                res = res.sum(axis=-1)  #(2,108,60,30)-->(2,108,60)
                ilayer -= 1
            return res
