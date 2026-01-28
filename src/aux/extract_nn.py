#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# run this code in working dir, it will read the file ./fread_dfeat/NN_output/NNFi/Wij.npy and output txt to fread_dfeat/Wij.txt

import os
import numpy as np
import torch
'''
codepath=os.path.abspath(sys.path[0])
sys.path.append(codepath+'/../src/lib')
sys.path.append(os.getcwd())
import use_para as pm
import parse_input
parse_input.parse_input()
# every py file in git_version/bin/ should be invoke directly from shell, and must parse parameters.
'''

import os

#f_npfile='data_scaler.npy'

# nfeature = 42, and net info [60, 30, 1]
# 42 -> 60 -> 30 -> 1
# torch wij shape [60,42], [30,60], [1,30]
# torch bj  shape [60], [30], [1]
# Wij.txt structure: w0, b0, w1, b1, w2, b2
#   [42,60], [60], [60,30], [30], [30,1], [1]

def read_wij(model_path, ntype):
    #first argument as input file path
    model_checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    nn_model = model_checkpoint['state_dict']
    nlayers = len(nn_model) // ntype // 2
    #print('nlayers %d' % (nlayers))

    info_net = []
    for i in range(nlayers):
        info_net.append(np.array(nn_model[r'models.0.weights.'+str(i)].shape))

    wij_all = [ [ np.zeros((info_net[ilayer]),dtype=float) for ilayer in range(nlayers) ] for itype in range(ntype)]
    bij_all = [ [ np.zeros((info_net[ilayer]),dtype=float) for ilayer in range(nlayers) ] for itype in range(ntype)]
    
    with open(os.path.join('fread_dfeat/Wij.txt'),'w') as f:
        f.write('test ' + str(model_path) + '  \n')
        f.write('shape '+str(nlayers*2*ntype)+'\n')
        f.write('dim 1'+'\n')
        f.write('size '+str(nlayers*2*ntype)+'\n')

        count = 0

        for itype in range(ntype):
            for ilayer in range(nlayers):
                wij_all[itype][ilayer] = nn_model[r'models.'+str(itype)+r'.weights.'+str(ilayer)]
                bij_all[itype][ilayer] = nn_model[r'models.'+str(itype)+r'.bias.'+str(ilayer)]
                wij = wij_all[itype][ilayer]
                bij = bij_all[itype][ilayer]
                m1 = info_net[ilayer][1]
                m2 = info_net[ilayer][0]
                #print('wij_shape:')
                #print(wij.shape)
                #print('m1 m2: %d %d' % (m1, m2))
                # write wij
                f.write('m12= '+str(m1) + ',' +str(m2)+', 1\n')
                for i in range(0,m1):
                    for j in range(0,m2):
                        f.write(str(i)+'  '+str(j)+'  '+str(float(wij[j][i]))+'\n')
                # write bj
                f.write('m12= 1,' + str(m2)+', 1\n')
                for j in range(0,m2):
                    f.write(str(j)+',  0  '+str(float(bij[j]))+'\n')


def read_scaler(src_name, ntype):
    model_checkpoint = torch.load(src_name, map_location=torch.device("cpu"), weights_only=False)
    scaler = model_checkpoint['scaler']

    fout = open('fread_dfeat/data_scaler.txt', 'w')
    fout.write('test\n')
    fout.write('shape, ignored\n')
    fout.write('dim, %d  ignored\n' % (ntype))
    fout.write('size, %d ignored\n' % (ntype*2))
    for i in range(ntype):
        fout.write('m12= %d, 0, 1\n' % (scaler.scale_.shape[0]))
        for j in range(scaler.scale_.shape[0]):
            fout.write('%5d  %5d   %.12f\n' % (j, 0, scaler.scale_[j]))

            #print(type(scaler.scale_[j]))
            #print(str(scaler.scale_[j]))
        
        fout.write('m12= %d, 0, 1\n' % (scaler.scale_.shape[0]))
        for j in range(scaler.scale_.shape[0]):
            fout.write('%5d  %5d   %.12f\n' % (j, 0, scaler.min_[j])) # x_std = x * scale_ + min_
    fout.close()
    # liuliping data_scaler.txt end
    
if __name__ =="__main__":
        
    read_wij()
    
    print("network parameters extracted")
    
    read_scaler()

    print("scaler extracted")
