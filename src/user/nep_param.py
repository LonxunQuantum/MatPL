import os
import numpy as np
from src.utils.atom_type_emb_dict import element_table
from src.utils.json_operation import get_parameter
from src.utils.nep_to_gpumd import get_atomic_name_from_str

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
        self.charge_mode = 0
        self.charge_output_num = 1
        self.sqrt_epsilon_inf = None

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
        self.charge_mode = 0
        self.sqrt_epsilon_inf = None
        if "charge" in version:
            charge_token = version.split("charge")[-1]
            self.charge_mode = int(charge_token)
            if self.charge_mode not in [1, 2, 3]:
                raise Exception("ERROR! charge_mode in nep.txt should be 1, 2, or 3. \n 1 for NEP-charge include both real-space and k-space; 2 for NEP-Charge include k-space only; 3 for NEP-Charge-VdW and include k-space only.")
        self.charge_output_num = 3 if self.charge_mode >= 3 else (2 if self.charge_mode else 1)
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
        self.use_typewise_cutoff_zbl = None
        self.use_fixed_zbl = False
        if "zbl" in lines[1]:
            zbl_line = lines[1].split()
            if float(zbl_line[1]) < 1e-04 and float(zbl_line[2]) < 1e-04:
                self.use_fixed_zbl = True
                raise Exception("ERROR! The fixed zbl is not supported!")
            else:
                self.zbl = float(zbl_line[2])
                if len(zbl_line) > 3:
                    self.use_typewise_cutoff_zbl = float(zbl_line[3])
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
        w1_num = b0_num * self.charge_output_num
        if "5" in version:
            ann_nums= (w0_num + b0_num + w1_num) * self.type_num + self.type_num + 1 # bais-N + common_bias 0
        else:
            ann_nums= (w0_num + b0_num + w1_num) * self.type_num + self.type_num # 不是错误，用于过滤元素类型只有1个
            if self.charge_mode:
                ann_nums += 2
        need_line = 6 + ann_nums + self.c_num + self.feature_nums
        if use_zbl:
            if self.use_fixed_zbl:
                need_line += 1 + (self.type_num+1)*self.type_num/2
            else:
                need_line += 1
        if "4" in version:
            is_gpumd_nep = False if need_line == line_num else True  #元素类型只有1个，可以认为是nep5
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
            w1_values = np.array([float(_) for _ in lines[start_index : start_index + w1_num]])
            if self.charge_output_num == 1:
                w1 = w1_values.reshape(b0_num, 1)
            else:
                w1 = w1_values.reshape(self.charge_output_num, b0_num).transpose(1, 0)
            start_index = start_index + w1_num
            self.model_wb.append(w1)
            if "5" in version:
                nep5_bias.append(-float(lines[start_index]))
                start_index += 1

        if "4" in version and is_gpumd_nep:
            print("The nep.txt file is from GPUMD")
            if self.charge_mode:
                self.sqrt_epsilon_inf = float(lines[start_index])
                common_bias = -float(lines[start_index + 1])
                start_index = start_index + 2
            else:
                common_bias = -float(lines[start_index])
                start_index = start_index + 1
            if self.charge_output_num == 1:
                self.bias_lastlayer = np.array([common_bias for _ in range(0, self.type_num)])
            else:
                self.bias_lastlayer = np.zeros((self.type_num, self.charge_output_num))
                self.bias_lastlayer[:, 0] = common_bias
        if "4" in version and is_gpumd_nep is False:
            bias_values = np.array([-float(_) for _ in lines[start_index : start_index + self.type_num]])
            if self.charge_output_num == 1:
                self.bias_lastlayer = bias_values
            else:
                self.bias_lastlayer = np.zeros((self.type_num, self.charge_output_num))
                self.bias_lastlayer[:, 0] = bias_values
            start_index = start_index + self.type_num
            if self.charge_mode:
                self.sqrt_epsilon_inf = float(lines[start_index])
                start_index = start_index + 2
        if "5" in version:
            start_index = start_index + 1 # the 0 of comm bias 
            if self.charge_output_num == 1:
                self.bias_lastlayer = np.array(nep5_bias)
            else:
                self.bias_lastlayer = np.zeros((self.type_num, self.charge_output_num))
                self.bias_lastlayer[:, 0] = np.array(nep5_bias)
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

        self.version = 5
        type_str = self.set_atom_type(type_list)
        self.type = type_str # number of atom types and list of chemical species
        self.type_num = len(type_list)
        type_list_weight_default = [1.0 for _ in range(0, self.type_num)]
        self.type_weight = get_parameter("type_weight", descriptor_dict, type_list_weight_default) # force weights for different atom types
        self.model_type = 0 # select to train potential 0, dipole 1, or polarizability 2
        self.prediction = 0 # select between training and prediction (inference)
        self.charge_mode = get_parameter("charge_mode", descriptor_dict, None)
        if  self.charge_mode is not None and self.charge_mode not in [1, 2, 3]:
            raise Exception("ERROR! charge_mode should be 1, 2, or 3. \n 1 for NEP-charge include both real-space and k-space; 2 for NEP-Charge include k-space only; 3 for NEP-Charge-VdW and include k-space only.")
        self.charge_output_num = 3 if self.charge_mode >= 3 else (2 if self.charge_mode else 1)
        self.zbl = get_parameter("zbl", descriptor_dict, None)
        self.use_typewise_cutoff_zbl = get_parameter("use_typewise_cutoff_zbl", descriptor_dict, None)
        if self.zbl is None and self.use_typewise_cutoff_zbl is not None:
            raise Exception("ERROR! The use_typewise_cutoff_zbl parameter must be used in conjunction with the zbl parameter.")
        self.train_2b = get_parameter("train_2b", descriptor_dict, True)
        # if self.zbl is not None:
        #     if self.zbl > 5 or self.zbl < 1.0:
        #         raise Exception("ERROR! the 'zbl' in json file should be between 1.0 and 2.5")
        self.use_fixed_zbl = False

        self.cutoff = get_parameter("cutoff", descriptor_dict, [8.0, 4.0]) # radial () and angular () cutoffs # use dp rcut, default to 6
        self.n_max = get_parameter("n_max", descriptor_dict, [4, 4]) # size of radial () and angular () basis
        if len(self.n_max) != 2:
            raise Exception("the input 'n_max' should has 2 values, such as [4, 4]")
        if self.n_max[0] <=0 or self.n_max[0] >= 20 or self.n_max[1] <=0 or self.n_max[1] >= 20:
            raise Exception("Error ! The input 'n_max' value should: 0 < n_max < 20")
                
        self.basis_size = get_parameter("basis_size", descriptor_dict, [12, 12]) # number of radial () and angular () basis functions
        if len(self.basis_size) != 2:
            raise Exception("the input 'basis_size' should has 2 values, such as [12, 12]")
        if self.basis_size[0] <=0 or self.basis_size[0] >= 20 or self.basis_size[1] <=0 or self.basis_size[1] >= 20:
            raise Exception("Error ! The input 'basis_size' value should: 0 < basis_size < 20")
        
        self.l_max = get_parameter("l_max", descriptor_dict, [4, 2, 1]) # expansion order for angular terms
        if len(self.l_max) != 3 or (self.l_max[0] != 4) or (self.l_max[1] != 2) or (self.l_max[2] > 1) :
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features.\n"
            raise Exception(error_log)
        if self.l_max[2] != 0 and self.l_max[2] != 1:
            error_log = "the input 'l_max' should has 3 values. The values should be [4, 2, 0] or [4, 2, 1]. The last num '1', means use 5 body features, '0' means not use 5 body features\n"
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


    # def to_nep_in_txt(self):
    #     content = ""
    #     content += "version     {}\n".format(self.version)
    #     content += "type        {}\n".format(self.type)
    #     # if self.type_weight is not None:
    #     #     content += "type_weight {}\n".format(self.type_weight)
    #     content += "model_type  {}\n".format(self.model_type)
    #     content += "prediction  {}\n".format(self.prediction)
    #     if self.zbl is not None: #' '.join(map(str, int_list))
    #         content += "zbl         {}\n".format(self.zbl)
    #         if self.use_typewise_cutoff_zbl is not None:
    #             content += "use_typewise_cutoff_zbl         {}\n".format(self.use_typewise_cutoff_zbl)
    #     content += "cutoff      {}\n".format(" ".join(map(str, self.cutoff)))
    #     content += "n_max       {}\n".format(" ".join(map(str, self.n_max)))
    #     content += "basis_size  {}\n".format(" ".join(map(str, self.basis_size)))
    #     content += "l_max       {}\n".format(" ".join(map(str, self.l_max)))
    #     content += "neuron      {}\n".format(self.neuron[0]) # filter the output layer
    #     return content
    
    # def to_nep_txt(self, max_NN_radial=None, max_NN_angular=None):
    #     content = ""
    #     if self.zbl is not None:
    #         content += "nep4_zbl   {}\n".format(self.type)    #line1
    #         content += "zbl   {} {}\n".format(self.zbl/2, self.zbl)    #line zbl
    #     else:
    #         content += "nep4   {}\n".format(self.type)    #line1
    #     if max_NN_radial is not None:
    #         cutoff_line = "{} {} {} {}".format(self.cutoff[0], self.cutoff[1], max_NN_radial, max_NN_angular)
    #     else:
    #         cutoff_line = "{} {}".format(self.cutoff[0], self.cutoff[1])
    #     content += "cutoff {}\n".format(cutoff_line)    #line2
    #     content += "n_max  {}\n".format(" ".join(map(str, self.n_max)))    #line3
    #     content += "basis_size {}\n".format(" ".join(map(str, self.basis_size)))    #line4
    #     content += "l_max  {}\n".format(" ".join(map(str, self.l_max)))    #line5
    #     content += "ANN    {} {}\n".format(self.neuron[0], 0)    #line6
    #     return content

    # '''
    # read params from nep.txt
    # description: 
    # param {*} self
    # param {*} file_path
    # return {*}
    # author: wuxingxing
    # '''    
    # def set_params_from_neptxt(self, file_path="nep.txt"):
    #     # self.c2_param = None
    #     # self.c3_param = None
    #     # self.q_scaler = None
    #     # self.model_wb = None
    #     pass
    
    # def read_nep_param_from_nep_file(self, nep_in_file:str):
    #     with open(nep_in_file, 'r') as rf:
    #         lines = rf.readlines()
    #     nep_dict = {}
    #     for line in lines:
    #         substring = line.split("#")[0].strip()
    #         if len(substring.split()) > 1:
    #             key = substring.split()[0]
    #             value_str = substring.replace(key, "")
    #             nep_dict[key.lower()] = value_str.strip()
    #     return nep_dict
