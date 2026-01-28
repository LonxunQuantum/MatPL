import torch
import os 
import sys

element_table = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
                 'Ar',
                 'K', 'Ca',
                 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                 'Rb', 'Sr', 'Y',
                 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs',
                 'Ba', 'La',
                 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                 'W', 'Re',
                 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                 'U', 'Np',
                 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']

element_table_2 = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109}

def get_atomic_number_from_name(atomic_names:list[str]):
    res = []
    for name in atomic_names:
        res.append(element_table_2[name])
    return res

def get_atomic_name_from_number(atomic_number:list[int]):
    res = []
    for number in atomic_number:
        res.append(element_table[int(number)])
    return res

def get_atomic_name_from_str(atom_strs):
    try:
        return [int(_) for _ in atom_strs]
    except ValueError:
        return get_atomic_number_from_name(atom_strs)

def check_atom_type_name(atom_types:list[str]):
    return all([_ in element_table_2.keys() for _ in atom_types])

'''
description: 

the example of hfo2
m.keys()
dict_keys(['json_file', 'epoch', 'state_dict', 'energy_shift', 'q_scaler', 'atom_type_order'])

m['state_dict'].keys()
odict_keys(['c_param_2', 'c_param_3', \
    'fitting_net.0.layers.0.weight', 'fitting_net.0.layers.0.bias', 'fitting_net.0.layers.1.weight', 'fitting_net.0.layers.1.bias', \
    'fitting_net.1.layers.0.weight', 'fitting_net.1.layers.0.bias', 'fitting_net.1.layers.1.weight', 'fitting_net.1.layers.1.bias'])

m['json_file']['model']['descriptor']
{'Rmax': 6.0, 'Rmin': 0.5, 'cutoff': [6.0, 6.0], 'n_max': [4, 4], 'basis_size': [12, 12], 'l_max': [4, 2, 1], 'type_weight': [1.0, 1.0]}

    param {str} nep_path
return {*}
author: wuxingxing
'''
def extract_model(nep_path:str):
    model = torch.load(nep_path, map_location=torch.device('cpu'), weights_only=False)
    model_type = model['json_file']['model_type']
    if model_type.upper() != "NEP":
        raise Exception("Error! the input model is not NEP model, please check the model!")
    model_atom_type = model['json_file']['atom_type']

    if "max_neighbor" in model.keys():
        max_NN_radial, max_NN_angular = model['max_neighbor']
    else:
        max_NN_radial = 500
        max_NN_angular = 300
    # the nep.txt head content
    cutoff = model['json_file']['model']['descriptor']['cutoff']
    n_max  = model['json_file']['model']['descriptor']['n_max']
    basis_size = model['json_file']['model']['descriptor']['basis_size']
    l_max  = model['json_file']['model']['descriptor']['l_max']
    if isinstance(model['json_file']['model']['fitting_net'], int):
        ann = model['json_file']['model']['fitting_net']
    elif isinstance(model['json_file']['model']['fitting_net'], list):
        ann = model['json_file']['model']['fitting_net'][0]
    elif isinstance(model['json_file']['model']['fitting_net']['network_size'], int):
        ann = model['json_file']['model']['fitting_net']['network_size']
    else :
        ann = model['json_file']['model']['fitting_net']['network_size'][0]
    zbl = model['json_file']['model']['descriptor']['zbl'] if 'zbl' in model['json_file']['model']['descriptor'].keys() else None
    atom_names = get_atomic_name_from_number(model_atom_type)
    if zbl is None:
        head_content =  "nep5   {} {}\n".format(len(atom_names), " ".join(map(str, atom_names)))
    else:
        head_content =  "nep5_zbl   {} {}\n".format(len(atom_names), " ".join(map(str, atom_names)))
        head_content +=  "zbl   {} {}\n".format(zbl/2, zbl)
    
    head_content += "cutoff {} {} {} {}\n".format(cutoff[0], cutoff[1], max_NN_radial, max_NN_angular)            #cutoff 6.0 6.0
    head_content += "n_max  {}\n".format(" ".join(map(str, n_max)))             #n_max  4 4
    head_content += "basis_size {}\n".format(" ".join(map(str, basis_size)))    #basis_size 12 12
    head_content += "l_max  {}\n".format(" ".join(map(str, l_max)))             #l_max  4 2 1
    head_content += "ANN    {} {}\n".format(ann, 0)                             #ANN    100 0

    #param lists
    nn_list = []
    c_list = []
    q_list = []
    for i in range(0, len(model_atom_type)):
        nn_list.extend(list(model['state_dict']['fitting_net.{}.layers.0.weight'.format(i)].transpose(1, 0).flatten().cpu().detach().numpy()))
        nn_list.extend((-model['state_dict']['fitting_net.{}.layers.0.bias'.format(i)]).flatten().cpu().detach().numpy())
        nn_list.extend(model['state_dict']['fitting_net.{}.layers.1.weight'.format(i)].flatten().cpu().detach().numpy())
        _last_bias = float(-model['state_dict']['fitting_net.{}.layers.1.bias'.format(i)])
        nn_list.append(_last_bias)

    nn_list.append(0.0) #last common bais
    c_list.extend(list(model['state_dict']['c_param_2'].permute(2, 3, 0, 1).flatten().cpu().detach().numpy()))
    if l_max[0] > 0:
        c_list.extend(list(model['state_dict']['c_param_3'].permute(2, 3, 0, 1).flatten().cpu().detach().numpy()))
    q_list.extend(list(model['q_scaler']))

    # check param nums
    # feature nums
    two_feat_num   = n_max[0] + 1
    three_feat_num = (n_max[1] + 1) * l_max[0]
    four_feat_num  = (n_max[1] + 1) if l_max[1] > 0 else 0
    five_feat_num  = (n_max[1] + 1) if l_max[2] > 0 else 0
    feature_nums   = two_feat_num + three_feat_num + four_feat_num + five_feat_num
    assert len(q_list) == feature_nums
    # c param nums, the 4-body and 5-body use the same c param of 3-body, their N_base_a the same
    ntypes_sq   = len(model_atom_type)*len(model_atom_type)
    two_c_num   = ntypes_sq * (n_max[0]+1)  * (basis_size[0]+1)
    three_c_num = ntypes_sq * (n_max[1]+1) * (basis_size[1]+1)
    if l_max[0] > 0:
        assert len(c_list) == two_c_num + three_c_num
    else:
        assert len(c_list) == two_c_num
    nn_params = len(model_atom_type) * (feature_nums * ann + ann + ann)
    assert len(nn_list) == nn_params+1+len(model_atom_type)
    head_content += "\n".join(map(str, nn_list))
    head_content += "\n"
    head_content += "\n".join(map(str, c_list))
    head_content += "\n"
    head_content += "\n".join(map(str, q_list))

    return head_content, model_atom_type, atom_names

def nep_ckpt_to_gpumd(cmd_list):
    infos = "\n\nThis cmd is used to convert the nep_model.ckpt trained by MatPL to nep5.txt for Lammps or GPUMD !\n\n"
    # infos += "This cmd requires installation of pytorch in your Python environment, and there is no mandatory version requirement.\n"
    infos += "The command example 'MatPL totxt nep_model.ckpt'.\n\n"
    print(infos)

    nep_model_path = cmd_list[0]
    if len(cmd_list) > 1:
        save_name = cmd_list[1]
    else:
        save_name = "nep5.txt"
    # os.chdir("/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/nep_lkf/model_record")
    # nep_model_path = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/nep_lkf/model_record/nep_model.ckpt"
    # atom_types = [8, 72]
    # atom_type_nums = [10, 10]
    # save_name = "nep_to_gpumd.txt"

    nep_content, model_atom_type, atom_names = extract_model(nep_model_path)
    with open(save_name, 'w') as wf:
        wf.writelines(nep_content)
    
    print("Successfully converted from MatPL nep.model.ckpt to GPUMD nep.txt format!")
    print("The result file is {}.".format(save_name))
    

def calculate_common_bias(model_bias:dict, input_type:list[int], atom_nums:list[int]):
    common_bias = 0
    all_num = 0
    for idx,atom in enumerate(input_type):
        common_bias += model_bias[atom] * atom_nums[idx]
        all_num += atom_nums[idx]
    return common_bias/all_num

if __name__ == "__main__":
    nep_ckpt_to_gpumd(sys.argv[1:])