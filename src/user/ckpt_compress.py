import os
import sys
import numpy as np
import torch
from src.user.input_param import InputParam
from src.PWMLFF.dp_network import dp_network
from src.model.dp_dp import DP
from src.model.dp_dp_typ_emb import TypeDP
from utils.atom_type_emb_dict import get_normalized_data_list
'''
description: 
    issue: Should we use the normalized multiples or adjust the previous multiples for the upper bound setting of sij?
    here we use the previous

param {*} ckpt_file
param {*} cmd_type
return {*}
author: wuxingxing
'''
def compress_force_field(ckpt_file):
    #json_file
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dx', help='specify Sij grid partition interval', type=float, default=0.01)
    parser.add_argument('-s', '--savename', help='specify the compressed model prefix name', type=str, default='cmp_dp_model')
    parser.add_argument('-o', '--order', help='specify the compressed order', type=int, default=3)
    parser.add_argument('-w', '--work_dir', help='Specify the compressed model storage directory', type=str, default=os.getcwd())
    args = parser.parse_args(sys.argv[3:])
    os.chdir(args.work_dir)
    order = args.order
    dx = args.dx
    model_compress_name = args.savename.replace('.ckpt', '') + ".ckpt"

    model_checkpoint = torch.load(ckpt_file,map_location=torch.device("cpu"), weights_only=False)
    atom_type_order = model_checkpoint['atom_type_order']
    if atom_type_order.size == 1:   #
        atom_type_order = [atom_type_order.tolist()]
    davg = model_checkpoint['davg']
    dstd = model_checkpoint['dstd']
    energy_shift = model_checkpoint['energy_shift']
    sij_max = float(model_checkpoint["sij_max"]) # sij before davg&dstd scaled

    json_dict = model_checkpoint["json_file"]
    json_dict["work_dir"] = os.getcwd()
    json_dict["model_load_file"] = ckpt_file
    dp_param = InputParam(json_dict, "train".upper())
    dp_trainer = dp_network(dp_param)
    model = dp_trainer.load_model_with_ckpt(davg, dstd, energy_shift)
 
    sij, scal_min, scal_max, sij_out_max, sij_len, sij_out_len = set_sij_range(davg, dstd, atom_type_order, \
                        dp_trainer.training_type, dp_trainer.device, sij_max, dx)

    type_vector = None
    if dp_param.type_embedding is False:
        if order == 2:
            cmp_tab = dp_model_compress_2order(sij, model)
        elif order == 3:
            cmp_tab = dp_model_compress_3order(sij, model, dx, sij_len, sij_out_len)
        elif order == 5:
            cmp_tab = dp_model_compress_5order(sij, model, dx, sij_len, sij_out_len)
        else:
            raise Exception("Error ! The input compress order {} not realized yet. Order options: 2, 3, or 5".format(order))
    else:
        physical_property = model.config["net_cfg"]["type_embedding_net"]["physical_property"]
        type_vector = get_normalized_data_list(atom_type_order, physical_property)
        if order == 3:
            cmp_tab = dp_type_model_compress_3order(sij, model, dx, atom_type_order, type_vector, sij_len, sij_out_len, dp_trainer.device)
        elif order == 5:
            cmp_tab = dp_type_model_compress_5order(sij, model, dx, atom_type_order, type_vector, sij_len, sij_out_len, dp_trainer.device)
        else:
            raise Exception("Error ! The input compress order {} not realized yet. Order options: 2, 3, or 5".format(order))
        
    # cmp_tab = model_compress_sintest(sij, model, dx)
    compress = {}
    compress["table"] = cmp_tab
    compress["dx"] = dx
    compress["order"] = order
    compress["davg"] = davg[:,0]
    compress["dstd"] = dstd[:,0]
    compress["sij_min"] = scal_min
    compress["sij_max"] = scal_max
    compress["sij_out_max"] = sij_out_max
    compress["sij_len"] = sij_len
    compress["sij_out_len"] = sij_out_len
    compress["type_vector"] = type_vector
    model_checkpoint["compress"] = compress

    torch.save(model_checkpoint, os.path.join(os.getcwd(), model_compress_name))
    print("dp model compress success!")

def dp_model_compress_2order(sij:torch.Tensor, model:DP):
    ntypes = model.ntypes
    sij_tab = []
    for type_0 in range(0, ntypes):
        for type_1 in range(0, ntypes):
            y_d = None
            embedding_index = type_0 * model.ntypes + type_1
            G = model.embedding_net[embedding_index](sij)
            for m in range(G.shape[-1]):
                ym = G[:,m].unsqueeze(-1)
                mask = torch.ones_like(ym)
                ym_1d = torch.stack(list(torch.autograd.grad(ym, sij, grad_outputs=mask, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                y_d = ym_1d if y_d is None else torch.concat((y_d, ym_1d), dim=1)
            # save to table
            print()
            sij_tab.append(np.array(torch.concat((G,y_d), dim=1).data.cpu()))
    sij_tab = np.array(sij_tab)
    return sij_tab

def dp_model_compress_3order(sij:torch.Tensor, model:DP, dx:float, len_sij: int, len_out_max: int):
    ntypes = model.ntypes
    sij_tab = []
    num_sij = sij.shape[0]
    for type_0 in range(0, ntypes):
        for type_1 in range(0, ntypes):
            embedding_index = type_0 * model.ntypes + type_1
            coef_L = []
            for index in range(len([len_sij, len_out_max])):
                if index == 0:
                    S_Rij = sij[0:len_sij+1]
                else:
                    S_Rij = sij[len_sij:]
                    dx = 10*dx
                y = model.embedding_net[embedding_index](S_Rij) #S_Rij shape is [1000, 1], out y shape is [1000, 25]
                am, bm, cm, dm= None, None, None, None
                # m_coef = None
                for m in range(y.shape[-1]):
                    ym = y[:,m].unsqueeze(-1)
                    mask = torch.ones_like(ym)
                    ym_1d = torch.stack(list(torch.autograd.grad(ym, S_Rij, grad_outputs=mask, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                    
                    ym_2 = ym[1:, :] #attention to the order of assignment
                    ym = ym[:-1, :]

                    ym_2_1d = ym_1d[1:, :]
                    ym_1d = ym_1d[:-1, :]
                    _am = 1/(dx**3) * ((ym_2_1d+ym_1d)*dx-2*(ym_2-ym))
                    _bm = 1/(dx**2) * (-(ym_2_1d+2*ym_1d)*dx+3*(ym_2-ym))
                    _cm = ym_1d
                    _dm = ym

                    am = _am if am is None else torch.concat((am, _am), dim=1) #[L+1, m]
                    bm = _bm if bm is None else torch.concat((bm, _bm), dim=1)
                    cm = _cm if cm is None else torch.concat((cm, _cm), dim=1)
                    dm = _dm if dm is None else torch.concat((dm, _dm), dim=1)
                for L in range(S_Rij.shape[0]-1):
                    coef_sij = []
                    for m in range(0, y.shape[-1]):
                        _coef_sij = [float(am[L][m]), float(bm[L][m]), float(cm[L][m]), float(dm[L][m])]
                        coef_sij.append(_coef_sij)
                    coef_L.append(coef_sij)
                # save to table
            sij_tab.append(coef_L)
    sij_tab = np.array(sij_tab)
    return sij_tab

def dp_model_compress_5order(sij:torch.Tensor, model:DP, dx:float, len_sij: int, len_out_max: int):
    ntypes = model.ntypes
    sij_tab = []
    num_sij = sij.shape[0]
    for type_0 in range(0, ntypes):
        for type_1 in range(0, ntypes):
            # print(type_0, "\t\t", ntype_1)
            # dim of Ri: batch size, natom_sum, ntype*max_neigh_num, local environment matrix , ([10,9,300,4])
            # S_Rij = sij[:,type_0].unsqueeze(-1)
            embedding_index = type_0 * model.ntypes + type_1
            coef_L = []
            for index in range(len([len_sij, len_out_max])):
                if index == 0:
                    S_Rij = sij[0:len_sij+1]
                else:
                    S_Rij = sij[len_sij:]
                    dx = 10*dx
                # determines which embedding net
                # itermediate output of embedding net 
                # dim of G: batch size, natom of ntype, max_neigh_num, final layer dim
                y = model.embedding_net[embedding_index](S_Rij) #S_Rij shape is [1000, 1], out y shape is [1000, 25]
                am, bm, cm, dm, em, fm = None, None, None, None, None, None
                # m_coef = None
                for m in range(y.shape[-1]):
                    ym = y[:,m].unsqueeze(-1)
                    mask = torch.ones_like(ym)
                    ym_1d = torch.stack(list(torch.autograd.grad(ym, S_Rij, grad_outputs=mask, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                    mask2 = torch.ones_like(ym_1d)
                    ym_2d = torch.stack(list(torch.autograd.grad(ym_1d, S_Rij, grad_outputs=mask2, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                    
                    ym_2 = ym[1:, :] #attention to the order of assignment
                    ym = ym[:-1, :]

                    ym_2_1d = ym_1d[1:, :]
                    ym_1d = ym_1d[:-1, :]
                    
                    ym_2_2d = ym_2d[1:, :]
                    ym_2d = ym_2d[:-1, :]

                    _am = 1/(2*dx**5) * (12*(ym_2-ym)-6*(ym_2_1d+ym_1d)*dx+(ym_2_2d-ym_2d)*dx**2)
                    _bm = 1/(2*dx**4) * (-30*(ym_2-ym)+(14*ym_2_1d+16*ym_1d)*dx+(3*ym_2d-2*ym_2_2d)*dx**2)
                    _cm = 1/(2*dx**3) * (20*(ym_2-ym)-(8*ym_2_1d+12*ym_1d)*dx-(3*ym_2d-ym_2_2d)*dx**2)
                    _dm = 1/2*(ym_2d)
                    _em = ym_1d
                    _fm = ym

                    # m_coef = torch.concat((_am, _bm, _cm, _dm, _em, _fm), dim=1)
                    am = _am if am is None else torch.concat((am, _am), dim=1) #[L+1, m]
                    bm = _bm if bm is None else torch.concat((bm, _bm), dim=1)
                    cm = _cm if cm is None else torch.concat((cm, _cm), dim=1)
                    dm = _dm if dm is None else torch.concat((dm, _dm), dim=1)
                    em = _em if em is None else torch.concat((em, _em), dim=1)
                    fm = _fm if fm is None else torch.concat((fm, _fm), dim=1)
                # contruct coefficient matrix [L+1, m, 6]
                # for m in range(0, y.shape[-1]):
                #     coef_sij = torch.concat((am[:,m].unsqueeze(-1), bm[:,m].unsqueeze(-1), \
                #                              cm[:,m].unsqueeze(-1), dm[:,m].unsqueeze(-1), \
                #                                 em[:,m].unsqueeze(-1), fm[:,m].unsqueeze(-1)),\
                #                                     dim=1)
                #     coef_L = coef_sij if coef_L is None else torch.concat((coef_L, coef_sij), dim=1)
                # print(coef_L.shape)
                for L in range(S_Rij.shape[0]-1):
                    coef_sij = []
                    for m in range(0, y.shape[-1]):
                        _coef_sij = [float(am[L][m]), float(bm[L][m]), float(cm[L][m]), float(dm[L][m]), float(em[L][m]), float(fm[L][m])]
                        coef_sij.append(_coef_sij)
                    coef_L.append(coef_sij)
                # save to table
            sij_tab.append(coef_L)
    sij_tab = np.array(sij_tab)
    return sij_tab

def dp_type_model_compress_3order(sij:torch.Tensor, model:TypeDP, dx:float, atom_type_order:list, type_vector:dict, len_sij: int, len_out_max: int, device:torch.device):
    sij_tab = []
    for atom_type in atom_type_order:
        coef_L = []
        for index in range(len([len_sij, len_out_max])):
            if index == 0:
                S_Rij = sij[0:len_sij+1]
            else:
                S_Rij = sij[len_sij:]
                dx = 10*dx
            t_vector = torch.tensor(type_vector[atom_type], dtype=model.dtype, device=device).repeat(S_Rij.shape[0],1)
            S_Rij_input = torch.concat((S_Rij, t_vector), dim=1)
            y = model.embedding_net[-1](S_Rij_input) #S_Rij shape is [1000, 1], out y shape is [1000, 25]
            am, bm, cm, dm= None, None, None, None
            # m_coef = None
            for m in range(y.shape[-1]):
                ym = y[:,m].unsqueeze(-1)
                mask = torch.ones_like(ym)
                ym_1d = torch.stack(list(torch.autograd.grad(ym, S_Rij, grad_outputs=mask, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                
                ym_2 = ym[1:, :] #attention to the order of assignment
                ym = ym[:-1, :]

                ym_2_1d = ym_1d[1:, :]
                ym_1d = ym_1d[:-1, :]

                _am = 1/(dx**3) * ((ym_2_1d+ym_1d)*dx-2*(ym_2-ym))
                _bm = 1/(dx**2) * (-(ym_2_1d+2*ym_1d)*dx+3*(ym_2-ym))
                _cm = ym_1d
                _dm = ym

                am = _am if am is None else torch.concat((am, _am), dim=1) #[L+1, m]
                bm = _bm if bm is None else torch.concat((bm, _bm), dim=1)
                cm = _cm if cm is None else torch.concat((cm, _cm), dim=1)
                dm = _dm if dm is None else torch.concat((dm, _dm), dim=1)

            for L in range(S_Rij.shape[0]-1):
                coef_sij = []
                for m in range(0, y.shape[-1]):
                    _coef_sij = [float(am[L][m]), float(bm[L][m]), float(cm[L][m]), float(dm[L][m])]
                    coef_sij.append(_coef_sij)
                coef_L.append(coef_sij)
            # save to table
        sij_tab.append(coef_L)
    sij_tab = np.array(sij_tab)
    return sij_tab

def dp_type_model_compress_5order(sij:torch.Tensor, model:TypeDP, dx:float, atom_type_order:list, type_vector:dict, len_sij: int, len_out_max: int, device:torch.device):
    sij_tab = []
    num_sij = sij.shape[0]
    for atom_type in atom_type_order:
        coef_L = []
        for index in range(len([len_sij, len_out_max])):
            if index == 0:
                S_Rij = sij[0:len_sij+1]
            else:
                S_Rij = sij[len_sij:]
                dx = 10*dx
            t_vector = torch.tensor(type_vector[atom_type], dtype=model.dtype, device=device).repeat(S_Rij.shape[0],1)
            S_Rij_input = torch.concat((S_Rij, t_vector), dim=1)
            y = model.embedding_net[-1](S_Rij_input) #S_Rij shape is [1000, 1], out y shape is [1000, 25]
            am, bm, cm, dm, em, fm= None, None, None, None, None, None
            # m_coef = None
            for m in range(y.shape[-1]):
                ym = y[:,m].unsqueeze(-1)
                mask = torch.ones_like(ym)
                ym_1d = torch.stack(list(torch.autograd.grad(ym, S_Rij, grad_outputs=mask, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                mask2 = torch.ones_like(ym_1d)
                ym_2d = torch.stack(list(torch.autograd.grad(ym_1d, S_Rij, grad_outputs=mask2, retain_graph=True, create_graph=True)), dim=0).squeeze(0)
                  
                ym_2 = ym[1:, :] #attention to the order of assignment
                ym = ym[:-1, :]

                ym_2_1d = ym_1d[1:, :]
                ym_1d = ym_1d[:-1, :]
                
                ym_2_2d = ym_2d[1:, :]
                ym_2d = ym_2d[:-1, :]
                
                _am = 1/(2*dx**5) * (12*(ym_2-ym)-6*(ym_2_1d+ym_1d)*dx+(ym_2_2d-ym_2d)*dx**2)
                _bm = 1/(2*dx**4) * (-30*(ym_2-ym)+(14*ym_2_1d+16*ym_1d)*dx+(3*ym_2d-2*ym_2_2d)*dx**2)
                _cm = 1/(2*dx**3) * (20*(ym_2-ym)-(8*ym_2_1d+12*ym_1d)*dx-(3*ym_2d-ym_2_2d)*dx**2)
                _dm = 1/2*(ym_2d)
                _em = ym_1d
                _fm = ym

                # m_coef = torch.concat((_am, _bm, _cm, _dm, _em, _fm), dim=1)
                am = _am if am is None else torch.concat((am, _am), dim=1) #[L+1, m]
                bm = _bm if bm is None else torch.concat((bm, _bm), dim=1)
                cm = _cm if cm is None else torch.concat((cm, _cm), dim=1)
                dm = _dm if dm is None else torch.concat((dm, _dm), dim=1)
                em = _em if em is None else torch.concat((em, _em), dim=1)
                fm = _fm if fm is None else torch.concat((fm, _fm), dim=1)
                
            for L in range(S_Rij.shape[0]-1):
                coef_sij = []
                for m in range(0, y.shape[-1]):
                    _coef_sij = [float(am[L][m]), float(bm[L][m]), float(cm[L][m]), float(dm[L][m]), float(em[L][m]), float(fm[L][m])]
                    coef_sij.append(_coef_sij)
                coef_L.append(coef_sij)
                # save to table
        sij_tab.append(coef_L)
    sij_tab = np.array(sij_tab)
    return sij_tab

'''
description: 
different type atom has different davg and dstd
param {list} davg
param {list} dstd
param {list} atom_type_order
param {str} dtype
param {str} device
param {float} sij_max
param {float} sij_min
param {*} dx
return {*}  
author: wuxingxing
'''
def set_sij_range(davg:list, dstd:list, atom_type_order:list, dtype:str, device:str, sij_max:float, dx:float):
    scal_max = 0
    scal_min = 0
    for i in range(len(atom_type_order)):
        # get max sacled sij, this data can get from training scaning
        scal_sij_max = (sij_max-davg[i,0])/dstd[i,0]
        scal_max = scal_sij_max if scal_sij_max > scal_max else scal_max
        # get min sacled sij, when rij > rc, sij = 0
        scal_sij_min = (0-davg[i,0])/dstd[i,0]
        scal_min = scal_sij_min if scal_sij_min < scal_min else scal_min

    sij_range = []
    sij_ = scal_min
    while sij_ < scal_max:
        sij_range.append(sij_)
        sij_ += dx
    len_sij = len(sij_range)
    sij_out_max = scal_max*10
    len_out_max = 0
    while sij_ < sij_out_max:
        sij_range.append(sij_)
        sij_ += dx*10
        len_out_max += 1

    Sij = torch.tensor(sij_range, dtype=dtype, device=device, requires_grad=True).unsqueeze(-1)
    # davg = torch.tensor(davg[:,0], dtype=dtype, device=device)
    # dstd = torch.tensor(dstd[:,0], dtype=dtype, device=device)
    # Sij = (Sij-davg)/dstd
    # Sij = Sij.squeeze(-1)
    return Sij, scal_min, scal_max, sij_out_max, len_sij, len_out_max
