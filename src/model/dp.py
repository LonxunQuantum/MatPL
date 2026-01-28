from sys import path
from builtins import print
from re import S
from tkinter import N
import numpy as np
import torch
from torch import embedding
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
import sys, os
import datetime
import time
import math

#git test

import default_para as pm    
# import prepare as pp
# pp.readFeatnum()
from src.model.embedding import EmbedingNet, FittingNet
from src.model.calculate_force import CalculateForce
# logging and our extension
import logging
logging_level_DUMP = 5
logging_level_SUMMARY = 15


class DP(nn.Module):
    def __init__(self, net_cfg, activation_type, device, stat, magic, is_reconnect):
        super(DP, self).__init__()
        # config parameters
        self.is_reconnect = is_reconnect
        self.ntypes = pm.ntypes
        self.device = device
        self.stat = stat
        
        """
        if pm.training_dtype == "float64":
            self.dtype = torch.double
        elif pm.training_dtype == "float32":
            self.dtype = torch.float32
        else:
            raise RuntimeError("train(): unsupported training_dtype %s" % pm.training_dtype)
        """
        # network   
        """
        if (net_cfg == 'default'):
            self.net_cfg = pm.MLFF_autograd_cfg
            print("DP: using default net_cfg: pm.DP_cfg")
            print(self.net_cfg)
        else:
            net_cfg = 'pm.' + net_cfg
            self.net_cfg = eval(net_cfg)
            print("DP: using specified net_cfg: %s" %net_cfg)
            print(self.net_cfg)
        """
        # set network config
        self.net_cfg = net_cfg
        
        self.embeding_net = nn.ModuleList()
        self.fitting_net = nn.ModuleList()

        self.M2 = pm.dp_M2 
        # wlj debubg 
        #  
        # self.embeding_net = EmbedingNet(self.net_cfg['embeding_net'], magic)
        for i in range(self.ntypes):
            for j in range(self.ntypes):
                #self.embeding_net.append(EmbedingNet(self.net_cfg['embeding_net'], magic, True))
                self.embeding_net.append(EmbedingNet(self.net_cfg['embeding_net'], magic, self.is_reconnect))
            fitting_net_input_dim = self.net_cfg['embeding_net']['network_size'][-1]
            #self.fitting_net.append(FittingNet(self.net_cfg['fitting_net'], self.M2 * fitting_net_input_dim, self.stat[2][i], magic, self.is_reconnect))
            self.fitting_net.append(FittingNet(self.net_cfg['fitting_net'], self.M2 * fitting_net_input_dim, self.stat[2][i], magic, True))


    def get_egroup(self, Egroup_weight, divider):
        batch_size = self.Ei.shape[0]
        Egroup = torch.zeros_like(self.Ei)
        for i in range(batch_size):
            Etot1 = self.Ei[i]
            weight_inner = Egroup_weight[i]
            E_inner = torch.matmul(weight_inner, Etot1)
            Egroup[i] = E_inner
        Egroup_out = torch.divide(Egroup, divider)

        return Egroup_out
            
    def forward(self, Ri, dfeat, list_neigh, natoms_img, Egroup_weight, divider, is_calc_f=None):

        torch.autograd.set_detect_anomaly(True)
        
        Ri_d = dfeat
        natoms = natoms_img[0, 1:]
        natoms_sum = Ri.shape[1]

        # when batch_size > 1. don't support multi movement file

        if pm.batch_size > 1 and torch.unique(natoms_img[:, 0]).shape[0] > 1:
            raise ValueError("batch size must be 1")
        atom_sum = 0

        for ntype in range(self.ntypes):
            for ntype_1 in range(self.ntypes):

                S_Rij = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1*pm.maxNeighborNum:(ntype_1+1)*pm.maxNeighborNum, 0].unsqueeze(-1)
                embedding_index = ntype * self.ntypes + ntype_1

                G = self.embeding_net[embedding_index](S_Rij)
                
                # dbg starts 
                #if ntype == 0 and ntype_1 == 0:
                #    !print ("printing dgb info")
                #    print (G[0].shape)
                #    print (G[0,0,0:4,:])
                # dbg ends  
                tmp_a = Ri[:, atom_sum:atom_sum+natoms[ntype], ntype_1*pm.maxNeighborNum:(ntype_1+1)*pm.maxNeighborNum].transpose(-2, -1)
                tmp_b = torch.matmul(tmp_a, G)
                
                if ntype_1 == 0:
                    xyz_scater_a = tmp_b

                else:
                    xyz_scater_a = xyz_scater_a + tmp_b

            xyz_scater_a = xyz_scater_a * 4.0 / (pm.maxNeighborNum * self.ntypes * 4)
            xyz_scater_b = xyz_scater_a[:, :, :, :self.M2]
            
            DR_ntype = torch.matmul(xyz_scater_a.transpose(-2, -1), xyz_scater_b)
            DR_ntype = DR_ntype.reshape(pm.batch_size, natoms[ntype], -1)

            if ntype == 0:
                DR = DR_ntype
            else:
                DR = torch.concat((DR, DR_ntype), dim=1)

            Ei_ntype = self.fitting_net[ntype](DR_ntype)
            
            if ntype == 0:
                Ei = Ei_ntype
            else:
                Ei = torch.concat((Ei, Ei_ntype), dim=1)
            atom_sum = atom_sum + natoms[ntype]

        self.Ei = Ei
        Etot = torch.sum(self.Ei, 1)
        
        # Egroup = self.get_egroup(Ei, Egroup_weight, divider)
        F = torch.zeros((pm.batch_size, atom_sum, 3), device=self.device)

        if is_calc_f == False:
            return Etot, Ei, F
        # start_autograd = time.time()
        # print("fitting time:", start_autograd - start_fitting, 's')

        mask = torch.ones_like(Ei)
        # get direvatives to calculate force
        dE = torch.autograd.grad(Ei, Ri, grad_outputs=mask, retain_graph=True, create_graph=True)
        dE = torch.stack(list(dE), dim=0).squeeze(0)  #[:,:,:,:-1] #[2,108,100,4]-->[2,108,100,3]

        Ri_d = Ri_d.reshape(pm.batch_size, natoms_sum, -1, 3)
        dE = dE.reshape(pm.batch_size, natoms_sum, 1, -1)
        
        # start_force = time.time()
        # print("autograd time:", start_force - start_autograd, 's')
        F = torch.matmul(dE, Ri_d).squeeze(-2) # batch natom 3
        F = F * (-1)
        
        #print (list_neigh)
        list_neigh = (list_neigh - 1).type(torch.int)
        F = CalculateForce.apply(list_neigh, dE, Ri_d, F)
        
        return Etot, Ei, F