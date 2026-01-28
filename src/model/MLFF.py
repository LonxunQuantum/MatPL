#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import init
from torch.autograd import Variable
import sys, os

"""
sys.path.append(os.getcwd())
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../pre_data') 
""" 
#import parameters as pm    
import default_para as pm    
import time

"""
if pm.torch_dtype == 'float32':
    torch_dtype = torch.float32
    #print('info: torch.dtype = torch.float32 in Pytorch training.')
else:
    torch_dtype = torch.float64
    torch.set_default_dtype(torch.float64)
    #print('info: torch.dtype = torch.float64 in Pytorch training. (it may be slower)')
"""

torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)

################################################################
# fully connection nn
# Ei Neural Network
################################################################

def dtanh(x):
    return 1.0-torch.tanh(x)**2

ACTIVE = torch.tanh
dACTIVE = dtanh

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


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
            
            w = nn.Parameter(nn.init.xavier_uniform_(torch.randn(pm.nNodes[i, itype], in_putsize)))
            
            b = torch.empty(pm.nNodes[i,itype])
            b = nn.Parameter(torch.full_like(b,0.0))

            self.weights.append(w)
            self.bias.append(b)

        # Final layer
        w = nn.Parameter(nn.init.xavier_uniform_(torch.randn(1,pm.nNodes[pm.nLayers-2,itype])))
        
        b = nn.Parameter(torch.tensor([pm.itype_Ei_mean[itype]]), requires_grad=True)
        
        #print (itype,pm.itype_Ei_mean[itype])

        self.weights.append(w)
        self.bias.append(b)


    def forward(self, x):
        L = []
        dL = []

        #print ("index of FCN:",self.itype)

        """
            First layer 
        """ 

        #print ("x.shape:",x.shape)
        #print ("self.weights[0].t().shape",self.weights[0].t().shape)

        Fout_0 = torch.matmul(x, self.weights[0].t()) + self.bias[0]

        #print ("weight dim:", self.weights[0].shape)
        #print ("input dim:", x.shape)
        #print ("output dim:",Fout_0.shape)
        
        # this layer is wrong. 
        #F_Wx = torch.matmul(x, self.weights[0].t()) 

        """
        print("output of the layer 1 without bias, atom type ", self.itype)
        print("output dim:", F_Wx.shape)
        print(F_Wx)
        print("*********************************************************************\n")

        print("output of the layer 1 before activation, atom type ", self.itype)
        print("output dim:", Fout_0.shape)
        print(Fout_0)
        print("*********************************************************************\n")        
        """

        L.append(ACTIVE(Fout_0))
        dL.append(dACTIVE(Fout_0))

        """
        print("output of the layer 1 after activation, atom type ", self.itype)
        print("output dim:", L[0].shape)
        print(L[0])
        print("*********************************************************************\n")
        """

        """
            Hidde layers
        """
        for ilayer in range(1, pm.nLayers-1):

            Fout_temp = torch.matmul(L[ilayer-1], self.weights[ilayer].t()) + self.bias[ilayer]

            L.append(ACTIVE(Fout_temp))
            dL.append(dACTIVE(Fout_temp)) 

        """
            Final Layer 
        """ 
        
        predict = torch.matmul(L[pm.nLayers-2], self.weights[-1].t()) + self.bias[-1]

        '''warning!!! if loss this dL, will make bias.2 to be None grad'''

        dL.append(1.0*predict)
        ilayer += 1
        grad = dL[-1]*self.weights[ilayer]
        ilayer -= 1

        while ilayer >= 0:
            grad = dL[ilayer] * grad   #(2,108,30)*(1,30)-->(2,108,30)
            grad = grad.unsqueeze(2) * self.weights[ilayer].t()  #(2,108,1,30)*(60,30)-->(2,108,60,30)
            grad = grad.sum(axis=-1)  #(2,108,60,30)-->(2,108,60)
            ilayer -= 1


        return predict, grad

class MLFFNet(nn.Module):

    def __init__(self, device, atomType = pm.atomType, Dropout = False):  #atomType=[8,32]

        super(MLFFNet,self).__init__()
        self.atomType = pm.atomType
        self.models = nn.ModuleList()
        self.device = device

        """ 
            atom type #1, atom type #2, ... 
        """
        for i in range(len(self.atomType)):  #i=[0,1]
            self.models.append(FCNet(itype = i, Dropout=Dropout))   # Dropout=True

        #print (self.atomType)
        #print (len(self.models))

    def forward(self, image, dfeat, neighbor, natoms_img, atom_type, Egroup_weight = None, divider = None, is_calc_f = True):
        """
            single image at a time 
            add label to avoid force calculation 
        """
        start = time.time()
        
        """
            image: 
            natoms_img: array of the form [total_atom_num, num_element_1, num_element_2, ... ] 
        """
        #image.requires_grad_(True)
        
        natoms_index = [0]
        temp = 0

        for i in natoms_img[0, 1:]:
            q = temp + i 
            temp += i 
            natoms_index.append(q)    #[0,32,64] 
            
        """
            For each type of atomic network , feed in the corresponding input. 
            pm.atomType must have the same order as in MOVEMENT!
            How to make it more flexible?
        """ 
        """
            use an extra array atomIndex
            atomIndex = [atomIdx in each segment of image]
            only carry out forward when pm.atomType[i] == atomIdx
            
        """ 
        atom_type = list(np.array(atom_type[0].cpu()))
        for i in range(len(atom_type)):            
            itype = pm.atomType[i]
            # for atom in atom_type:
            #     i = pm.atomType.index(atom)
            # get the segment corresponding to the current atom type 
            x = image[:, natoms_index[i]:natoms_index[i+1]]
            fit_model_index = pm.atomType.index(atom_type[i])
            predict, grad = self.models[fit_model_index](x)
            
            #print ("Ei prediction value:\n", predict)
            if(i==0):
                Ei = predict #[32, 1]
                dE = grad
            else:
                Ei = torch.cat((Ei, predict), dim=1)    #[64,1]
                dE = torch.cat((dE, grad), dim=1)

        # de = self.get_de(image, dfeat, neighbor)
        input_grad_allatoms = dE
        cal_ei_de = time.time()
        Etot = Ei.sum(dim=1)

        batch_size = image.shape[0]
        natom = image.shape[1]
        F = torch.zeros((batch_size, natom, 3), device=self.device)
        Virial = None, #torch.zeros((batch_size, 9), device=self.device) unrealized
        Egroup = None
        if Egroup_weight is not None:
            Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        if is_calc_f == False:
            return Etot, Ei, F, Egroup, Virial
        
        test = Ei.sum()
        mask = torch.ones_like(test)
        test_grad = torch.autograd.grad(test,image,grad_outputs=mask, create_graph=True,retain_graph=True)
        test_grad = test_grad[0]
           
        neighbor_num=dfeat.shape[2]

        dim_feat=pm.nFeatures

        result_dEi_dFeat_fortran = torch.zeros((batch_size, natom + 1, dim_feat)).to(self.device)
        result_dEi_dFeat_fortran[:, 1:, :]=test_grad

        n_a_idx_fortran = neighbor.reshape(batch_size * natom * neighbor_num)
        n_a_ofs_fortran_b = torch.arange(0, batch_size * (natom+1), (natom+1))\
                            .repeat_interleave(natom * neighbor_num)
        n_a_idx_fortran_b = n_a_idx_fortran.type(torch.int64) + n_a_ofs_fortran_b.to(self.device)

        dEi_neighbors = result_dEi_dFeat_fortran\
                        .reshape(batch_size * (natom+1), dim_feat)[n_a_idx_fortran_b,]\
                        .reshape(batch_size, natom, neighbor_num, 1, dim_feat)
        Force = torch.matmul(dEi_neighbors.to(self.device), dfeat).sum([2, 3])

        self.Ei = Ei
        self.Etot = Etot
        self.Force = Force  

        return Etot, Ei, Force, Egroup, Virial #Virial not used
        # if Egroup is not None:
        #     return Etot, Ei, Force, Egroup, Virial #Virial not used
        # else:
        #     return Etot, Ei, Force, Virial

    def get_egroup(self, Ei, Egroup_weight, divider):
        batch_size = Ei.shape[0]
        Egroup = torch.zeros_like(Ei)

        for i in range(batch_size):
            Etot1 = Ei[i]
            weight_inner = Egroup_weight[i]
            E_inner = torch.matmul(weight_inner, Etot1)
            Egroup[i] = E_inner
        Egroup_out = torch.divide(Egroup, divider) # Egroup.squeeze(-1) not need
        return Egroup_out
    
    # def get_egroup(self,Ei_input,Egroup_weight, divider):
    #     #batch_size = self.Ei.shape[0]
    #     batch_size  = Ei_input.shape[0]
    #     # egroup is an array with the same size as self.Ei 
    #     Egroup = torch.zeros_like(Ei_input)
    #     for i in range(batch_size):
    #         """
    #             2nd dimension of weight is the max number in all systems 
    #             In this case it is 144 
    #         """
    #         Etot1 = Ei_input[i]
    #         numAtoms = Egroup_weight[i].shape[0]
    #         weight_inner = Egroup_weight[i]
    #         """
    #             only take the first natom rows in weight matrix! 
    #         """
    #         E_inner = torch.matmul(weight_inner[:,:numAtoms], Etot1)
    #         #E_inner = torch.matmul(torch.t(Etot1),weight_inner)
    #         Egroup[i] = E_inner
    #     Egroup_out = torch.divide(Egroup, divider)
    #     return Egroup_out

    def get_de(self, image, dfeat, neighbor):
        
        batch_size = image.shape[0]
        atom_type_num = len(pm.atomType)

        for i in range(atom_type_num):
            model_weight = self.models[i].state_dict()
            W = []
            B = []
            L = []
            dL = []

            W.append(model_weight['fc0.weight'].transpose(0, 1))            
            B.append(model_weight['fc0.bias'])            
            dL.append(dACTIVE(torch.matmul(image, W[0]) + B[0]))
            L.append(ACTIVE(torch.matmul(image, W[0]) + B[0]))
            
            for ilayer in range(1, pm.nLayers-1):
                W.append(model_weight['fc' + str(ilayer) + '.weight'].transpose(0, 1))            
                B.append(model_weight['fc' + str(ilayer) + '.bias'])            
                dL.append(dACTIVE(torch.matmul(L[ilayer-1], W[ilayer]) + B[ilayer]))
                L.append(ACTIVE(torch.matmul(L[ilayer-1], W[ilayer]) + B[ilayer]))
            
            ilayer += 1
            W.append(model_weight['output.weight'].transpose(0, 1))            
            B.append(model_weight['output.bias'])            
            res = W[ilayer].transpose(0, 1)
            ilayer -= 1
            
            while ilayer >= 0:
                res = dL[ilayer] * res   #(2,108,30)*(1,30)-->(2,108,30)
                res = res.unsqueeze(2) * W[ilayer]  #(2,108,1,30)*(60,30)-->(2,108,60,30)
                res = res.sum(axis=-1)  #(2,108,60,30)-->(2,108,60)
                ilayer -= 1
            return res


