import os
import numpy as np
import torch
from utils.atom_type_emb_dict import element_table
from utils.json_operation import get_parameter
from utils.nep_to_gpumd import get_atomic_name_from_str

'''
description: 
 this NEP params default value is the same as 
    https://gpumd.org/nep/input_files/nep_in.html#nep-in

return {*}
author: wuxingxing
'''
class NepParam(object):
    '''
    description: 
        if nep.in file provided by user, prioritize using it
        elif nep.in file not exisited, use params set in user input json file
        else use default params
    param {*} self
    param {*} nep_dict
    param {str} nep_in_file
    param {list} type_list
    param {list} type_list_weight
    param {str} nep_save_file
    return {*}
    author: wuxingxing
    '''    
    def __init__(self) -> None:
        self.nep_in_file = None
        self.nep_txt_file = None
        self.train_2b = True
        self.max_NN_radial = None
        self.max_NN_angular = None
        self.max_nn_from_txt = False
        self.fix_cij = False
        self.fix_hiddenlayer=False
        self.fix_outlayer=False
    '''
    description: 
        extract nep params from nep.in file
    param {*} self
    param {*} nep_in_file
    param {list} type_list
    return {*}
    author: wuxingxing
    '''
    def set_nep_param_from_nep_in(self, nep_in_file, type_list:list[int]=[]) -> None:
        nep_file_dict = {}
        self.nep_in_file = nep_in_file
        if nep_in_file is not None:
            if not os.path.exists(nep_in_file):
                raise Exception("ERROR! the nep.in file is not exist, please check the file: {}".format(nep_in_file))
            nep_file_dict = self.read_nep_param_from_nep_file(nep_in_file)
            
        self.version = get_parameter("version", nep_file_dict, 4) # select between NEP2, NEP3, and NEP4
        if self.version != 4:
            raise Exception("ERROR! Only support NEP4!")
        type_str = self.set_atom_type(type_list)
        self.type = get_parameter("type", nep_file_dict, type_str) # number of atom types and list of chemical species
        self.type_num = len(self.type.split()[1:])
        type_list_weight_default = [1.0 for _ in range(0, self.type_num)]
        self.type_weight = get_parameter("type_weight", nep_file_dict, type_list_weight_default) # force weights for different atom types
        self.model_type = get_parameter("model_type", nep_file_dict, 0) # select to train potential 0, dipole 1, or polarizability 2
        self.zbl = get_parameter("zbl", nep_file_dict, None) # outer cutoff for the universal ZBL potential [Ziegler1985]
        
        self.cutoff = get_parameter("cutoff", nep_file_dict, [8.0, 4.0], out_format=4) # radial () and angular () cutoffs # use dp rcut, default to 6
        self.n_max = get_parameter("n_max", nep_file_dict, [4, 4], out_format=2) # size of radial () and angular () basis
        if len(self.n_max) != 2:
            raise Exception("Error ! The input 'n_max' should has 2 values, default is [4, 4]")
        if self.n_max[0] <=0 or self.n_max[0] >= 20 or self.n_max[1] <=0 or self.n_max[1] >= 20:
            raise Exception("Error ! The input 'n_max' value should: 0 < n_max < 20")
        
        self.basis_size = get_parameter("basis_size", nep_file_dict, [8, 8], out_format=2) # number of radial () and angular () basis functions
        if len(self.basis_size) != 2:
            raise Exception("the input 'basis_size' should has 2 values, such as [8, 8]")
        if self.n_max[0] <=0 or self.n_max[0] >= 20 or self.n_max[1] <=0 or self.n_max[1] >= 20:
            raise Exception("Error ! The input 'basis_size' value should: 0 < basis_size < 20")
        
        self.l_max = get_parameter("l_max", nep_file_dict, [4, 2, 1], out_format=2) # expansion order for angular terms
        if len(self.l_max) != 3 or (self.l_max[0] != 4) or (self.l_max[1] != 2) or (self.l_max[2] > 1) :
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features.\n"
            raise Exception(error_log)
        if self.l_max[2] != 0 and self.l_max[2] != 1:
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features, '0' means not use 5 body features\n"
            raise Exception(error_log)
        self.neuron = get_parameter("neuron", nep_file_dict, 100, out_format=1) # number of neurons in the hidden layer
        self.neuron = [self.neuron, 1]
        self.set_feature_params()

    '''
    description: 
    recover model from nep.txt file
    for nep.txt in gpumd, the bias is '-' op,  but in pwmlff, the bias is '+' op
    param {str} nep_txt_file
    return {*}
    author: wuxingxing
    '''    
    def set_nep_nn_c_param_from_nep_txt(self, nep_txt_file:str, atom_type_train:list[int]=None):
        self.nep_txt_file = nep_txt_file
        with open(nep_txt_file, "r") as rf:
            lines =rf.readlines()

        line_num = len(lines)

        line_1 = lines[0].split()
        version, type_num, type_list = line_1[0], int(line_1[1]), line_1[2:]
        type_list = get_atomic_name_from_str(type_list)

        set1, set2 = set(atom_type_train), set(type_list)
        only_in_arr1 = set1 - set2
        only_in_arr2 = set2 - set1
        if len(only_in_arr1) > 0:
            raise Exception("ERROR! The element type specified by the 'atom_type' parameter is not contained in the nep.txt file! ")
        if len(only_in_arr2) > 0:
            print("Extracting the training parameters corresponding to the specified element type from nep.txt:")
            print(f"the atom type in nep.txt:\n {type_list}")
            print(f"the target atom type:\n {atom_type_train}")
        self.type_num = type_num
        self.zbl = None
        use_zbl = False
        self.use_fixed_zbl = False
        if "zbl" in lines[1]:
            zbl_line = lines[1].split()
            if float(zbl_line[1]) < 1e-04 and float(zbl_line[2]) < 1e-04:
                self.use_fixed_zbl = True
                raise Exception("ERROR! The fixed zbl is not supported!")
            else:
                self.zbl = float(zbl_line[2])
                # print("use zbl: {}".format(self.zbl))
                # if self.zbl[0] > self.zbl[1] or self.zbl[0] > 2.5 :
                #     raise Exception("ERROR! the inner should smaller than outer of zbl in nep.txt! And both of them should be smaller than 2.5!\n The inner = 0.5*outer should be a reasonable choice. ")
                # elif self.zbl[1] > 2.5:
                #     print("Warning! the input zbl outer param should smaller than 2.5! Automatically adjust outer to 2.5!")
                #     self.zbl[1] = 2.5
                lines[1:-1] = lines[2:]
                
            use_zbl = True
            
        cutoffs = lines[1].split()
        self.cutoff = [float(cutoffs[1]), float(cutoffs[2])]
        if len(cutoffs) > 3:
            self.max_NN_radial, self.max_NN_angular = [int(cutoffs[3]), int(cutoffs[4])]
            self.max_nn_from_txt = True
            
        n_maxs =  lines[2].split() 
        self.n_max = [int(n_maxs[1]), int(n_maxs[2])]

        basis_size =  lines[3].split() 
        self.basis_size = [int(basis_size[1]),int(basis_size[2])]

        l_maxs =  lines[4].split() 
        self.l_max = [int(l_maxs[1]), int(l_maxs[2]), int(l_maxs[3])]

        anns = lines[5].split()
        ann_num = int(anns[1])
        self.neuron = [ann_num, 1]
        self.set_feature_params()

        #num_w0 
        self.model_wb = []
        start_index = 6
        w0_num = self.feature_nums*ann_num
        b0_num = ann_num
        if "5" in version:
            ann_nums= (w0_num + b0_num*2) * self.type_num + self.type_num + 1
        else:
            ann_nums= (w0_num + b0_num*2) * self.type_num + self.type_num
        need_line = 6 + ann_nums + self.c_num + self.feature_nums
        if use_zbl:
            if self.use_fixed_zbl:
                need_line += 1 + (self.type_num+1)*self.type_num/2
            else:
                need_line += 1
        if "4" in version:
            is_gpumd_nep = False if need_line == line_num else True
        else:
            is_gpumd_nep = False
        
        nep5_bias = []
        for i in range(0, self.type_num):
            w0 = np.array([float(_) for _ in lines[start_index        : start_index+w0_num]]).reshape(ann_num, self.feature_nums).transpose(1,0)
            start_index = start_index + w0_num
            self.model_wb.append(w0)
            b0 = np.array([-float(_) for _ in lines[start_index : start_index + b0_num]]).reshape(1,b0_num)
            start_index = start_index + b0_num
            self.model_wb.append(b0)
            w1 = np.array([float(_) for _ in lines[start_index : start_index + b0_num]]).reshape(b0_num, 1)
            start_index = start_index + b0_num
            self.model_wb.append(w1)
            if "5" in version:
                nep5_bias.append(-float(lines[start_index]))
                start_index += 1

        if "4" in version and is_gpumd_nep:
            print("The nep.txt file is from GPUMD")
            common_bias = -float(lines[start_index])
            self.bias_lastlayer = np.array([common_bias for _ in range(0, self.type_num)])
            start_index = start_index + 1
        if "4" in version and is_gpumd_nep is False:
            self.bias_lastlayer = np.array([-float(_) for _ in lines[start_index : start_index + self.type_num]])
            start_index = start_index + self.type_num
        if "5" in version:
            start_index = start_index + 1 # the 0 of comm bias 
            self.bias_lastlayer = np.array(nep5_bias)
        # attention: the value order in c++ duda is same as in cpu memory of the inintial numpy array
        _c2_param = np.array([float(_) for _ in lines[start_index:start_index + self.two_c_num]]).reshape(self.n_max[0]+1, self.basis_size[0]+1, self.type_num, self.type_num).transpose(2, 3, 0, 1)
        self.c2_param = []
        for _ in _c2_param:
            self.c2_param.append(_)
        self.c2_param = np.array(self.c2_param).reshape(self.type_num, self.type_num, self.n_max[0]+1, self.basis_size[0]+1)
        start_index = start_index + self.two_c_num
        if self.l_max[0] > 0:
            _c3_param = np.array([float(_) for _ in lines[start_index:start_index + self.three_c_num]]).reshape(self.n_max[1]+1, self.basis_size[1]+1, self.type_num, self.type_num).transpose(2, 3, 0, 1)
            self.c3_param = []
            for _ in _c3_param:
                self.c3_param.append(_)
            self.c3_param = np.array(self.c3_param).reshape(self.type_num, self.type_num, self.n_max[1]+1, self.basis_size[1]+1)
            start_index = start_index + self.three_c_num
        else:
            self.c3_param = None

        self.q_scaler = np.array([float(_) for _ in lines[start_index:start_index + self.feature_nums]])
        start_index += self.feature_nums
        if use_zbl:
            if start_index + 1 != line_num:
                raise Exception("extract the nep.txt {} error! the params need {} but has {}\n".format(nep_txt_file, start_index + 1, len(lines)))
        else:
            if start_index != line_num:
                raise Exception("extract the nep.txt {} error! the params need {} but has {}\n".format(nep_txt_file, start_index, len(lines)))

        if atom_type_train is not None:
            # 根据输入，重新调整元素类型对应的参数顺序
            _nn_param = []
            index = []
            
            for atom_in in atom_type_train:
                _index = type_list.index(atom_in)
                index.append(_index)
                _nn_param.append(self.model_wb[_index*3])
                _nn_param.append(self.model_wb[_index*3+1])
                _nn_param.append(self.model_wb[_index*3+2])
            self.model_wb = _nn_param
            self.c2_param = self.c2_param[index, :, :, :][:, index, :, :]
            self.c3_param = self.c3_param[index, :, :, :][:, index, :, :]
            self.type_num = len(atom_type_train)
            self.bias_lastlayer = self.bias_lastlayer[index]

    '''
    description: 
    extract nep params from input json file
    param {*} self
    param {dict} json_dict
    param {list} type_list
    return {*}
    author: wuxingxing
    '''
    def set_nep_param_from_json(self, json_dict:dict, type_list:list[int]=[]):
        model_dict = get_parameter("model", json_dict, {})
        descriptor_dict = get_parameter("descriptor", model_dict, {})
        # optimizer_dict = get_parameter("optimizer", json_dict, {})

        self.version = 4 # select between NEP2, NEP3, and NEP4
        type_str = self.set_atom_type(type_list)
        self.type = type_str # number of atom types and list of chemical species
        self.type_num = len(type_list)
        type_list_weight_default = [1.0 for _ in range(0, self.type_num)]
        self.type_weight = get_parameter("type_weight", descriptor_dict, type_list_weight_default) # force weights for different atom types
        self.model_type = 0 # select to train potential 0, dipole 1, or polarizability 2
        self.prediction = 0 # select between training and prediction (inference)
        self.zbl = get_parameter("zbl", descriptor_dict, None)
        self.train_2b = get_parameter("train_2b", descriptor_dict, True)
        if self.zbl is not None:
            if self.zbl > 5 or self.zbl < 1.0:
                raise Exception("ERROR! the 'zbl' in json file should be between 1.0 and 2.5")
        self.use_fixed_zbl = False

        self.cutoff = get_parameter("cutoff", descriptor_dict, [6.0, 6.0]) # radial () and angular () cutoffs # use dp rcut, default to 6
        self.n_max = get_parameter("n_max", descriptor_dict, [4, 4]) # size of radial () and angular () basis
        if len(self.n_max) != 2:
            raise Exception("the input 'n_max' should has 2 values, such as [4, 4]")
        self.basis_size = get_parameter("basis_size", descriptor_dict, [12, 12]) # number of radial () and angular () basis functions
        if len(self.basis_size) != 2:
            raise Exception("the input 'basis_size' should has 2 values, such as [12, 12]")
        self.l_max = get_parameter("l_max", descriptor_dict, [4, 2, 1]) # expansion order for angular terms
        if len(self.l_max) != 3 :
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 0, 0] (only use three body features), [4, 2, 0] (use 3 and 4 body features) or [4, 2, 1] (use 3,4,5 body features).\n"
            raise Exception(error_log)
        
        if (self.l_max[2] in [0, 1]) is False or (self.l_max[1] in [0, 2]) is False:
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 0, 0] (only use three body features), [4, 2, 0] (use 3 and 4 body features) or [4, 2, 1] (use 3,4,5 body features).\n"
            raise Exception(error_log)
        if "fitting_net" in model_dict.keys():
            self.neuron = get_parameter("network_size", model_dict["fitting_net"], [40]) # number of neurons in the hidden layer
            if not isinstance(self.neuron, list):
                self.neuron = [self.neuron]
            self.fix_cij = get_parameter("fix_cij", model_dict["fitting_net"], False)
            self.fix_hiddenlayer =get_parameter("fix_hiddenlayer", model_dict["fitting_net"], False)
            self.fix_outlayer =get_parameter("fix_outlayer", model_dict["fitting_net"], False)        
        else:
            self.neuron = [40]
        if self.neuron[-1] != 1:
            self.neuron.append(1) # output layer of fitting net
        self.set_feature_params()

    def set_fixed_params(self, json_dict):
        model_dict = get_parameter("model", json_dict, {})
        if "fitting_net" in model_dict.keys():
            self.fix_cij = get_parameter("fix_cij", model_dict["fitting_net"], False)
            self.fix_hiddenlayer =get_parameter("fix_hiddenlayer", model_dict["fitting_net"], False)
            self.fix_outlayer =get_parameter("fix_outlayer", model_dict["fitting_net"], False)

    def set_feature_params(self):
        # features
        self.two_feat_num = self.n_max[0]+1
        self.three_feat_num = (self.n_max[1]+1)*self.l_max[0]
        self.four_feat_num = (self.n_max[1]+1) if self.l_max[1] != 0 else 0
        self.five_feat_num = (self.n_max[1]+1) if self.l_max[2] != 0 else 0
        self.feature_nums = self.two_feat_num + self.three_feat_num + self.four_feat_num + self.five_feat_num
        # c params, the 4-body and 5-body use the same c param of 3-body, their N_base_a the same
        self.two_c_num = self.type_num*self.type_num*(self.n_max[0]+1)*(self.basis_size[0]+1)
        self.three_c_num = self.type_num*self.type_num*(self.n_max[1]+1)*(self.basis_size[1]+1)
        self.c_num = self.two_c_num + self.three_c_num if self.l_max[0] > 0 else self.two_c_num

        self.c2_param = None
        self.c3_param = None
        self.q_scaler = None
        self.model_wb = None

    '''
    read params from nep.txt
    description: 
    param {*} self
    param {*} file_path
    return {*}
    author: wuxingxing
    '''    
    def set_params_from_neptxt(self, file_path="nep.txt"):
        # self.c2_param = None
        # self.c3_param = None
        # self.q_scaler = None
        # self.model_wb = None
        pass
    
    def read_nep_param_from_nep_file(self, nep_in_file:str):
        with open(nep_in_file, 'r') as rf:
            lines = rf.readlines()
        nep_dict = {}
        for line in lines:
            substring = line.split("#")[0].strip()
            if len(substring.split()) > 1:
                key = substring.split()[0]
                value_str = substring.replace(key, "")
                nep_dict[key.lower()] = value_str.strip()
        return nep_dict

    def to_nep_in_txt(self):
        content = ""
        content += "version     {}\n".format(self.version)
        content += "type        {}\n".format(self.type)
        # if self.type_weight is not None:
        #     content += "type_weight {}\n".format(self.type_weight)
        content += "model_type  {}\n".format(self.model_type)
        content += "prediction  {}\n".format(self.prediction)
        if self.zbl is not None: #' '.join(map(str, int_list))
            content += "zbl         {}\n".format(self.zbl)
        content += "cutoff      {}\n".format(" ".join(map(str, self.cutoff)))
        content += "n_max       {}\n".format(" ".join(map(str, self.n_max)))
        content += "basis_size  {}\n".format(" ".join(map(str, self.basis_size)))
        content += "l_max       {}\n".format(" ".join(map(str, self.l_max)))
        content += "neuron      {}\n".format(self.neuron[0]) # filter the output layer
        return content
    
    def to_nep_txt(self, max_NN_radial=None, max_NN_angular=None):
        content = ""
        if self.zbl is not None:
            content += "nep4_zbl   {}\n".format(self.type)    #line1
            content += "zbl   {} {}\n".format(self.zbl/2, self.zbl)    #line zbl
        else:
            content += "nep4   {}\n".format(self.type)    #line1
        if max_NN_radial is not None:
            cutoff_line = "{} {} {} {}".format(self.cutoff[0], self.cutoff[1], max_NN_radial, max_NN_angular)
        else:
            cutoff_line = "{} {}".format(self.cutoff[0], self.cutoff[1])
        content += "cutoff {}\n".format(cutoff_line)    #line2
        content += "n_max  {}\n".format(" ".join(map(str, self.n_max)))    #line3
        content += "basis_size {}\n".format(" ".join(map(str, self.basis_size)))    #line4
        content += "l_max  {}\n".format(" ".join(map(str, self.l_max)))    #line5
        content += "ANN    {} {}\n".format(self.neuron[0], 0)    #line6
        return content

    '''
    description: 
        the format of type in nep.in file is as:
            "type 2 Te Pb # this is a mandatory keyword"
    param {*} self
    param {*} atom_type_list
    return {*}
    author: wuxingxing
    '''    
    def set_atom_type(self, atom_type_list):
        atom_names = []
        for atom in atom_type_list:
            atom_names.append(element_table[atom])
        return str(len(atom_type_list)) + " " + " ".join(atom_names)