import os
import shutil
from src.user.input_param import InputParam
import torch
import numpy as np
import src.aux.extract_ff as extract_ff
from utils.atom_type_emb_dict import get_normalized_data_list

def extract_force_field(dp_params:InputParam):
    config = dp_params.get_dp_net_dict()
    forcefield_dir = dp_params.file_paths.forcefield_dir
    if os.path.exists(forcefield_dir):
        shutil.rmtree(forcefield_dir)
    os.makedirs(forcefield_dir)
    
    cwd = os.getcwd()
    os.chdir(forcefield_dir)
    extract_model_para(config, dp_params)
    print("extract_model_para function need redo")

    mk = config["net_cfg"]["fitting_net"]["resnet_dt"]

    extract_type_physical_property(dp_params)

    extract_ff.extract_ff(ff_name = dp_params.file_paths.forcefield_name, 
                          model_type = 5, 
                          atom_type = dp_params.atom_type, 
                          max_neigh_num = dp_params.max_neigh_num, 
                          is_fitting_recon = mk, 
                          is_type_embedding = dp_params.descriptor.type_embedding)
    os.chdir(cwd)
    
def extract_model_para(config:dict, dp_params:InputParam):
    """ 
        extract the model parameters of DP network
        NEED TO ADD SESSION DIR NAME 
    """ 
    extract_model_name = dp_params.file_paths.model_save_path
    
    print ("extracting network parameters from:",extract_model_name )
    
    netConfig = config["net_cfg"]

    isEmbedingNetResNet = netConfig["embedding_net"]["resnet_dt"]
    isFittingNetResNet  = netConfig["fitting_net"]["resnet_dt"]

    embedingNetSizes = netConfig['embedding_net']['network_size']
    nLayerEmbedingNet = len(embedingNetSizes)   

    print("layer number of embeding net:"+str(nLayerEmbedingNet))
    print("size of each layer:"+ str(embedingNetSizes) + '\n')

    fittingNetSizes = netConfig['fitting_net']['network_size']
    nLayerFittingNet = len(fittingNetSizes)

    print("layer number of fitting net:"+str(nLayerFittingNet))
    print("size of each layer:"+ str(fittingNetSizes) + '\n')

    embedingNet_output = 'embeding.net' 
    fittingNet_output = 'fitting.net'
    
    model_checkpoint = torch.load(extract_model_name,map_location=torch.device("cpu"), weights_only=False)
    raw = model_checkpoint['state_dict']
    
    has_module = "module." if "module" in list(raw.keys())[0] else ""
    module_sign = True if "module" in list(raw.keys())[0] else False

    #determining # of networks 

    # nEmbedingNet = len(config["atomType"])**2  
    # nFittingNet = len(config["atomType"])
    nEmbedingNet, nFittingNet = count_net_nums(list(raw.keys()))
    
    print("number of embedding network:",nEmbedingNet)
    print("\n")
    print("number of fitting network:",nFittingNet)
    
    # first dim of w matrix 
    first_w_matrix = raw[list(raw.keys())[0]]
    first_dim = len(first_w_matrix)
    # write embedding network
    f = open(embedingNet_output, 'w')
    # total number of embeding network
    f.write(str(nEmbedingNet)+'\n') 
    #layer of embeding network
    f.write(str(nLayerEmbedingNet) + '\n')
    #size of each layer
    f.write(str(first_dim) + " ")
    for i in embedingNetSizes:
        f.write(str(i)+' ')
    f.write('\n')

    #f.writelines([str(i) for i in embedingNetSizes])
    print("******** converting embedding network starts ********")
    for idxNet in range(nEmbedingNet):
        print ("converting embedding network No."+str(idxNet))
        for idxLayer in range(nLayerEmbedingNet):
            print ("converting layer "+str(idxLayer) )	
            #write wij
            label_W = catNameEmbedingW(idxNet,idxLayer,has_module)
            for item in raw[label_W]:
                dump(item,f)

            print("w matrix dim:" +str(len(raw[label_W])) +str('*') +str(len(raw[label_W][0])))

            #write bi
            label_B = catNameEmbedingB(idxNet,idxLayer,has_module)
            dump(raw[label_B][0],f)
            print ("b dim:" + str(len(raw[label_B][0])))

        print ('\n')

    f.close()

    print("******** converting embedding network ends  *********")
    print("\n")
    """
        write fitting network
    """

    f = open(fittingNet_output, 'w')

    # total number of embeding network
    f.write(str(nFittingNet)+'\n')

    #layer of embeding network
    f.write(str(nLayerFittingNet) + '\n')

    #size of each layer

    f.write(str(len(raw[catNameFittingW(0,0,has_module)]))+' ')

    for i in fittingNetSizes:
        f.write(str(i)+' ')

    f.write('\n')

    print("******** converting fitting network starts ********")
    for idxNet in range(nFittingNet):
        print ("converting fitting network No."+str(idxNet))
        for idxLayer in range(nLayerFittingNet):
            print ("converting layer "+str(idxLayer) )  

            #write wij
            label_W = catNameFittingW(idxNet,idxLayer,has_module)
            for item in raw[label_W]:
                dump(item,f)

            print("w matrix dim:" +str(len(raw[label_W])) +str('*') +str(len(raw[label_W][0])))

            #write bi
            label_B = catNameFittingB(idxNet,idxLayer,has_module)
            dump(raw[label_B][0],f)
            print ("b dim:" + str(len(raw[label_B][0])))

        print ('\n')
            #break
    f.close()

    print("******** converting fitting network ends  *********")

    print("\n")
    """
        writing ResNets
    """
    print("******** converting resnet starts  *********")

    # only extract fitting net now 
    if config["net_cfg"]["fitting_net"]["resnet_dt"]:
        numResNet = 0

        """
            The format below fits Dr.Wang's Fortran routine
        """

        for keys in list(raw.keys()):
            tmp = keys.split('.')
            if module_sign:
                if tmp[1] == "fitting_net" and tmp[2] == '0' and tmp[4] == 'resnet_dt':
                    numResNet +=1
            else:
                if tmp[0] == "fitting_net" and tmp[1] == '0' and tmp[4] == 'resnet_dt':
                    numResNet +=1 

        print ("number of resnet: " + str(numResNet))

        filename  = "fittingNet.resnet"

        f= open(filename, "w")
        # itype: number of fitting network 
        f.write(str(nFittingNet)+'\n') 

        #nlayer: 
        f.write(str(nLayerFittingNet) + '\n')

        #dim of each layer 
        f.write(str(len(raw[catNameFittingW(0,0)]))+' ')

        for i in fittingNetSizes:
            f.write(str(i)+' ')	
        f.write("\n")

        for i in range(0,len(fittingNetSizes)+1):
            if (i > 1) and (i < len(fittingNetSizes)):
                f.write("1 ")
            else:
                f.write("0 ")

        f.write("\n")
        #f.write(str(numResNet)+"\n")

        for fittingNetIdx in range(nFittingNet):
            for resNetIdx in range(1,numResNet+1):
                f.write(str(fittingNetSizes[resNetIdx])+"\n")
                label_resNet = catNameFittingRes(fittingNetIdx,resNetIdx)   
                dump(raw[label_resNet][0],f)

        f.close()
    else:
        print ("FITTING IS NOT RECONNECTED.OMIT")

    print("******** converting resnet done *********\n")

    print("******** generating gen_dp.in  *********\n")
    
    orderedAtomList = [str(atom) for atom in dp_params.atom_type]
    
    # from where 
    davg = model_checkpoint['davg']
    dstd = model_checkpoint['dstd']

    davg_size = len(davg)
    dstd_size = len(dstd)

    assert(davg_size == dstd_size)
    assert(davg_size == len(orderedAtomList))
    
    f_out = open("gen_dp.in","w")
    
    # in default_para.py, Rc is the max cut, beyond which S(r) = 0 
    # Rm is the min cut, below which S(r) = 1

    f_out.write(str(config["Rc_M"]) + '\n') 
    # f_out.write(str(config["maxNeighborNum"])+"\n")
    f_out.write(str(config["M2"])+"\n")
    f_out.write(str(dstd_size)+"\n")
    
    for i,atom in enumerate(orderedAtomList):
        f_out.write(atom+"\n")
        f_out.write(str(config["atomType"][0]["Rc"])+' '+str(config["atomType"][0]["Rm"])+'\n')

        for idx in range(4):
            f_out.write(str(davg[i][idx])+" ")
        
        f_out.write("\n")

        for idx in range(4):
            f_out.write(str(dstd[i][idx])+" ")
        f_out.write("\n")
    
    f_out.close() 

    print("******** gen_dp.in generation done *********")

def extract_type_physical_property(dp_params:InputParam):
    """
        extract the type embedding parameters of DP network
    """
    type_vector = get_normalized_data_list(dp_params.atom_type, dp_params.descriptor.type_physical_property)
    print("******** writting type embedding starts ********")
    f = open("type_physical_properties.in","w")
    if dp_params.descriptor.type_embedding:
        f.write("1 " + "\n")         # if 1, then type embedding is used
        for i, atom in enumerate(type_vector):
            f.write(str(atom)+" \n")
            f.write(" ".join(map(str, type_vector[atom])) + " \n")
    else:
        f.write("0 " + "\n")
    f.close()
    print("******** writting type embedding ends ********")
"""
    parameter extraction related functions
""" 
def catNameEmbedingW(idxNet, idxLayer, has_module=""):
    return "{}embedding_net.".format(has_module)+str(idxNet)+".layers."+str(idxLayer)+".weight"

def catNameEmbedingB(idxNet, idxLayer, has_module=""):
    return "{}embedding_net.".format(has_module)+str(idxNet)+".layers."+str(idxLayer)+".bias"

def catNameFittingW(idxNet, idxLayer, has_module=""):
    return "{}fitting_net.".format(has_module)+str(idxNet)+".layers."+str(idxLayer)+".weight"

def catNameFittingB(idxNet, idxLayer, has_module=""):
    return "{}fitting_net.".format(has_module)+str(idxNet)+".layers."+str(idxLayer)+".bias"

def catNameFittingRes(idxNet, idxResNet, has_module=""):
    return "{}fitting_net.".format(has_module)+str(idxNet)+".layers."+str(idxResNet)+".resnet_dt"

def count_net_nums(net_list:list):
    net = {}
    for i in net_list:
        net_type, index = i.split('.')[:2]
        key = "{}_{}".format(net_type, index)
        if net_type not in net.keys():
            net[net_type] = [key]
        else:
            if key not in net[net_type]:
                net[net_type].append(key)
    return [len(_[1]) for _ in net.items()]

def dump(item, f):
    raw_str = ''
    for num in item:
        raw_str += (str(float(num))+' ')
    f.write(raw_str)
    f.write('\n')

def load_davg_dstd_from_checkpoint(model_path):
    model_checkpoint = torch.load(model_path,map_location=torch.device("cpu"), weights_only=False)
    davg = model_checkpoint['davg']
    dstd = model_checkpoint['dstd']
    atom_type_order = model_checkpoint['atom_type_order']
    energy_shift = model_checkpoint['energy_shift']
    if atom_type_order.size == 1:   #
        atom_type_order = [atom_type_order.tolist()]
    return davg, dstd, atom_type_order, energy_shift

def load_atomtype_energyshift_from_checkpoint(model_path):
    model_checkpoint = torch.load(model_path,map_location=torch.device("cpu"), weights_only=False)
    atom_type_order = model_checkpoint['atom_type_order']
    energy_shift = model_checkpoint['energy_shift']
    
    if isinstance(atom_type_order, float):
        atom_type_order  = [atom_type_order]
    if isinstance(atom_type_order, np.ndarray):
        if atom_type_order.size == 1:   #
            atom_type_order = [atom_type_order.tolist()]
        else:
            atom_type_order = atom_type_order.tolist()
    return atom_type_order, energy_shift

'''
description: load davg from feature paths, \
    at least one of the feature paths should contail all atom types, when doing hybrid training.
param {list} feature_paths
return {*}
author: wuxingxing
'''
def load_davg_dstd_from_feature_path(feature_paths:list):
    atom_type_order = []
    num_atom_type = []
    for feature_path in feature_paths:
        atom_map = np.loadtxt(os.path.join(feature_path, "train", "atom_map.raw"), dtype=int)
        atom_type_order.append(atom_map)
        num_atom_type.append(atom_map.size)
    feature_path = feature_paths[num_atom_type.index(max(num_atom_type))]
    davg = np.load(os.path.join(feature_path, "train", "davg.npy"))
    dstd = np.load(os.path.join(feature_path, "train", "dstd.npy"))
    energy_shift = np.loadtxt(os.path.join(feature_path, "train", "energy_shift.raw"))
    if energy_shift.size == 1:
        energy_shift = [energy_shift.tolist()]
    return davg, dstd, atom_type_order[num_atom_type.index(max(num_atom_type))].tolist(), energy_shift

