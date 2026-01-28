import numpy as np
from utils.json_operation import get_parameter, get_required_parameter

class Descriptor(object):
    def __init__(self, json_input:dict, model_type:str, cmd:str, feature_type:list=None, type_embedding:bool=False) -> None:
        self.model_type = model_type
        self.cmd = cmd
        self.feature_type = feature_type
        self.supported_feature_group = [[1, 2], [3, 4], [5], [6], [7], [8]]
        self.Rmax = get_parameter("Rmax", json_input, 6.0)
        self.Rmin = get_parameter("Rmin", json_input, 0.5)
        self.M2 = get_parameter("M2", json_input, 16)
        self.E_tolerance = get_parameter("E_tolerance", json_input, 9999999.0)
        self.cheby_order = get_parameter("cheby_order", json_input, 10)
        self.radial_num1 = get_parameter("radial_num1", json_input, 25)
        self.radial_num2 = get_parameter("radial_num2", json_input, 4)
        if self.feature_type is not None:   #if feature type specified at first layer of json 
            if "feature_type" in json_input.keys(): # if feature type sepcified at descriptor layer of json
                if self.feature_type != json_input["feature_type"]: # if this two values are different, show WARNING info.
                    print("\n\nWARNNING: feature_type is specified twice with inconsistent values. \nThe program will prioritize the value specified within the Descriptor.\n\n")
                    self.feature_type = json_input["feature_type"]
        else:   # the first layer not specified in both first layer and descriptor layer, use default value
            self.feature_type = [int(_) for _ in get_parameter("feature_type", json_input, [3,4])]

        # dp params These parameters are only used for outputting to the standard output json file
        self.network_size = get_parameter("network_size", json_input, [25, 25, 25])
        self.bias = get_parameter("bias", json_input, True)
        self.resnet_dt = get_parameter("resnet_dt", json_input, False) # resnet in embedding net is False.
        self.activation = get_parameter("activation", json_input, "tanh")

        # type embedding params These parameters are only used for outputting to the standard output json file
        type_embedding_dict = get_parameter("type_embedding", json_input, {})
        self.type_embedding = False if (not type_embedding_dict.items()) and type_embedding is False else True
        # set dp embedding net params
        self.type_physical_property = get_parameter("physical_property", type_embedding_dict, ["atomic_number", "atom_radius", "atom_mass", "electron_affin", "pauling"])
        self.type_network_size = get_parameter("network_size", type_embedding_dict, None)
        if self.type_network_size is not None:
            self.type_network_size = [len(self.type_physical_property)] + self.type_network_size

        self.type_bias = get_parameter("bias", type_embedding_dict, True)
        self.type_resnet_dt = get_parameter("resnet_dt", type_embedding_dict, False) # resnet in embedding net is False.
        self.type_activation = get_parameter("activation", type_embedding_dict, "tanh")

        if self.feature_type not in self.supported_feature_group:
            raise Exception("ERROR: The input feature type group {} is not support, \
                            we currently support the following combinations:\n {}".format(self.feature_type, self.supported_feature_group))
        self.feature_dict = {}  # for feature generating
        self.feature_dict_out = {}  # for std output json
        for ftype in self.feature_type:
            ftype = "{}".format(ftype)
            ftype_dict = get_parameter("{}".format(ftype), json_input, {})
            # self.feature_dict[ftype] = getattr(self, "set_ftype{}_para".format(ftype), ftype_dict)
            if '1' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype1_para(ftype_dict)
            elif '2' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype2_para(ftype_dict)
            elif '3' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype3_para(ftype_dict)
            elif '4' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype4_para(ftype_dict)
            elif '5' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype5_para(ftype_dict)
            elif '6' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype6_para(ftype_dict)
            elif '7' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype7_para(ftype_dict)
            elif '8' == ftype:
                self.feature_dict[ftype], self.feature_dict_out[ftype] = self.set_ftype8_para(ftype_dict)
        # for nep param
        self.cutoff = None
        self.n_max = None
        self.basis_size = None
        self.l_max = None
        self.type_weight = None
    '''
    description: 
    the return dict is used to assignment for value Ftype1_para in dfault_param.py
        the 'n2b' from json file is 'numOf2bfeat' value in Ftype1_para
    param {*} self
    param {dict} ftype_dict
    return {*}
    author: wuxingxing
    '''
    def set_ftype1_para(self, ftype_dict:dict):
        numOf2bfeat = get_parameter("n2b", ftype_dict, 24)
        iflag_grid = get_parameter("iflag_grid", ftype_dict, 3)
        fact_base = get_parameter("fact_base", ftype_dict, 0.2)
        dR1 = get_parameter("dR1", ftype_dict, 0.5)
        iflag_ftype = get_parameter("iflag_ftype", ftype_dict, 3)
        return {
            "numOf2bfeat":[numOf2bfeat for tmp in range(10)],
            "Rc": [self.Rmax for tmp in range(10)],
            "Rm": [self.Rmin for tmp in range(10)],
            "iflag_grid": [iflag_grid for tmp in range(10)],   # 1 or 2 or 3
            "fact_base": [fact_base for tmp in range(10)],
            "dR1": [dR1 for tmp in range(10)],
            "iflag_ftype": iflag_ftype  
        },  \
        {
            "n2b":numOf2bfeat,
            "iflag_grid": iflag_grid,   # 1 or 2 or 3
            "fact_base": fact_base,
            "dR1": dR1,
            "iflag_ftype": iflag_ftype  
        } 


    '''
    description: 
    the return dict is used to assignment for value Ftype2_para in dfault_param.py
        the 'n3b1' from json file is 'numOf3bfeat1' value in Ftype2_para
        the 'n3b2' from json file is 'numOf3bfeat2' value in Ftype2_para
    param {*} self
    param {dict} ftype_dict
    return {*}
    author: wuxingxing
    '''

    def set_ftype2_para(self, ftype_dict:dict):
        numOf3bfeat1 = get_parameter("n3b1", ftype_dict, 3)
        numOf3bfeat2 = get_parameter("n3b2", ftype_dict, 3)
        iflag_grid = get_parameter("iflag_grid", ftype_dict, 3)
        fact_base = get_parameter("fact_base", ftype_dict, 0.2)
        dR1 = get_parameter("dR1", ftype_dict, 0.5)
        dR2 = get_parameter("dR2", ftype_dict, 0.5)
        iflag_ftype = get_parameter("iflag_ftype", ftype_dict, 3)

        return {
            "numOf3bfeat1":[numOf3bfeat1 for tmp in range(10)],
            "numOf3bfeat2":[numOf3bfeat2 for tmp in range(10)],
            "Rc":[self.Rmax for tmp in range(10)],
            "Rc2":[self.Rmax for tmp in range(10)],
            "Rm":[self.Rmin for tmp in range(10)],
            "iflag_grid":[iflag_grid for tmp in range(10)],    # 1 or 2 or 3
            "fact_base":[fact_base for tmp in range(10)],
            "dR1":[dR1 for tmp in range(10)],
            "dR2":[dR2 for tmp in range(10)],
            "iflag_ftype":iflag_ftype   # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
        }, \
         {
            "n3b1":numOf3bfeat1,
            "n3b2":numOf3bfeat1,
            "dR1":dR1,
            "dR2":dR2,
            "iflag_grid":iflag_grid,    # 1 or 2 or 3
            "fact_base":fact_base,
            "iflag_ftype":iflag_ftype   # same value for different types, iflag_ftype:1,2,3 when 3, iflag_grid must be 3
        }

    def set_ftype3_para(self, ftype_dict:dict):
        n2b = get_parameter("n2b", ftype_dict, 6)
        w = get_parameter("w", ftype_dict, [1.0, 1.5, 2.0])

        return {
            "Rc":[self.Rmax for tmp in range(10)],     # 5.4 number of elements in Rc = num atom type
            "n2b":[n2b for tmp in range(10)],       # 6 number of elements in n2b = num atom type
            "w": w #[1.0, 1.5, 2.0],
        }, \
        {
            "n2b":n2b,
            "w": w
        }
        # 1/w^2 is the \eta in formula, and w is the width of gaussian fuction

    def set_ftype4_para(self, ftype_dict:dict):
        n3b = get_parameter("n3b", ftype_dict, 20)
        zeta = get_parameter("zeta", ftype_dict, 2.0)
        w = get_parameter("w", ftype_dict, [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
        return {
            "Rc":[self.Rmax for tmp in range(10)],     # 5.4 number of elements in Rc = num atom type
            "n3b":[n3b for tmp in range(10)],   # 20
            # n3b must be less than n_zeta * n_w * n_lambda, by default, n3b < 7*10*2=126
            # it is better if n3b is divisible by (n_w*2)
            "zeta": [ (zeta ** np.array(range(20))).tolist() for tmp in range(10)],  #2.0 feature changed
            "w":    [ w for tmp in range(10)],  # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] feature changed
            # 1/w^2 is the \eta in formula, and w is the width of gaussian fuction
            # 'lambda':[ [1.0,-1.0] for tmp in range(10)], # lambda === [1.0, -1.0]
        },\
        {
            "n3b":n3b, 
            "zeta": zeta, 
            "w": w 
        }
        # 1/w^2 is the \eta in formula, and w is the width of gaussian fuction

    def set_ftype5_para(self, ftype_dict:dict):
        n_MTP_line = get_parameter("n_MTP_line", ftype_dict, 5)
        return {
            "Rc":[self.Rmax for tmp in range(10)],     # number of elements in Rc = num atom type
            "Rm":[self.Rmin  for tmp in range(10)],     # number of elements in Rc = num atom type
            "n_MTP_line": [n_MTP_line for tmp in range(10)], # 5~14
            
            "tensors":[
                        [
                        '1, 4, 0, ( )                              ',
                        '2, 3,3, 0,0, ( ), ( )                     ',
                        '2, 3,3, 1,1, ( 21 ), ( 11 )               ',
                        '2, 3,3, 2,2, ( 21, 22 ), ( 11, 12 )       ',
                        '3, 2,2,2, 2,1,1 ( 21, 31 ), ( 11 ), ( 12 )',
                        '3, 3,3,3, 2,1,1 ( 21, 31 ), ( 11 ), ( 12 )',
                        '3, 2,2,2, 3,2,1 ( 21, 22, 31 ), ( 11, 12 ), ( 13 )',
                        '3, 3,3,3, 3,2,1 ( 21, 22, 31 ), ( 11, 12 ), ( 13 )',
                        '3, 2,2,2, 4,2,2 ( 21, 22, 31, 32 ), ( 11, 12 ), ( 13, 14 )',
                        '3, 3,3,3, 4,2,2 ( 21, 22, 31, 32 ), ( 11, 12 ), ( 13, 14 )',
                        '4, 2,2,2,2 3,1,1,1 ( 21, 31, 41 ), ( 11 ), ( 12 ), ( 13 )',
                        '4, 3,3,3,3 3,1,1,1 ( 21, 31, 41 ), ( 11 ), ( 12 ), ( 13 )',
                        '4, 2,2,2,2 4,2,1,1 ( 21, 22, 31, 41 ), ( 11, 12 ), ( 13 ), ( 14 )',
                        '4, 3,3,3,3 4,2,1,1 ( 21, 22, 31, 41 ), ( 11, 12 ), ( 13 ), ( 14 )',
                        ] for tmp in range(10)
                    ],
            }, \
            {
            "n_MTP_line": n_MTP_line # 5~14
            }

    def set_ftype6_para(self, ftype_dict:dict):
        J = get_parameter("J", ftype_dict, 3.0)
        n_w_line = get_parameter("n_w_line", ftype_dict, 2)
        w1 = get_parameter("w1", ftype_dict, [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4])
        w2 = get_parameter("w2", ftype_dict, [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.3, 0.6])
        
        return {
            'Rc':[self.Rmax for tmp in range(10)],     # number of elements in Rc = num atom type
            'J' :[J for tmp in range(10)],  #3.0
            'n_w_line': [n_w_line for tmp in range(10)],#2
            'w1':[ w1 for tmp in range(10)],  # shape(w1) = (ntype, n_w_line) [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4]
            'w2':[ w2 for tmp in range(10) ], # [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.3, 0.6] 
        }, \
            {
            'J' :J,
            'n_w_line': n_w_line,
            'w1':w1,
            'w2':w2
        }

    def set_ftype7_para(self, ftype_dict:dict):
        M = get_parameter("M", ftype_dict, 25)
        M2 = get_parameter("M2", ftype_dict, 4)
        weight_r = get_parameter("weight_r", ftype_dict, 1.0)
        
        return {
            'Rc':[self.Rmax  for tmp in range(10)],     # number of elements in Rc = num atom type
            'Rc2':[self.Rmin  for tmp in range(10)],
            'Rm':[1.0 for tmp in range(10)],    #Chebyshev param , not rmin
            'M': [M for tmp in range(10)],#25
            'M2': [M2  for tmp in range(10)],#4
            'weight_r': [weight_r  for tmp in range(10)], #1.0
        }, \
         {
            'M': M,#25
            'M2': M2,#4
            'weight_r': weight_r #1.0
        }

    def set_ftype8_para(self, ftype_dict:dict):
        M = get_parameter("M", ftype_dict, 8)
        weight_r = get_parameter("weight_r", ftype_dict, 1.0)
        w = get_parameter("w", ftype_dict, [1.0, 1.5, 2.0, 2.5 ] )
        return {
            'Rc':[self.Rmax for tmp in range(10)],     # number of elements in Rc = num atom type
            'M':[M for tmp in range(10)], # 8
            'weight_r':[weight_r for tmp in range(10)], #1.0
            'w':w
        }, \
        {
            'M':M, # 8
            'weight_r':weight_r, #1.0
            'w':w
        }
    
    def to_dict(self):
        dicts = {}

        # dicts["E_tolerance"] = self.E_tolerance
        if self.model_type == "DP".upper():
            dicts["Rmax"] = self.Rmax
            dicts["Rmin"] = self.Rmin
            dicts["M2"] = self.M2
            dicts["network_size"] = self.network_size
            dicts["M2"] = self.M2
            # dicts["bias"] = self.bias
            # dicts["resnet_dt"] = self. resnet_dt 
            # dicts["activation"] = self.activation
        elif self.model_type == "NN".upper() or self.model_type == "Linear".upper():
            dicts["Rmax"] = self.Rmax
            dicts["Rmin"] = self.Rmin
            dicts["feature_type"] = self.feature_type
            for feature in self.feature_type:
                feature = "{}".format(feature)
                dicts[feature] = self.feature_dict_out[feature]
        elif self.model_type == "CHEBY".upper():
            dicts["cheby_order"] = self.cheby_order
            dicts["radial_num1"] = self.radial_num1
            dicts["radial_num2"] = self.radial_num2
        elif self.model_type.upper() == "NEP":
            dicts["cutoff"] = self.cutoff
            if self.zbl is not None:
                dicts["zbl"] = self.zbl
            dicts["n_max"] = self.n_max
            dicts["basis_size"] = self.basis_size
            dicts["l_max"] = self.l_max
            # dicts["type_weight"] = self.type_weight
        else:
            raise Exception("descriptor to dict: the model type not realized:{}".format(self.model_type))
        
        if self.type_embedding is True:
            dicts["type_embedding"] = {}
            dicts["type_embedding"]["physical_property"] = self.type_physical_property
            if self.type_network_size is not None:
               dicts["type_embedding"]["network_size"] = self.type_network_size[1:]
        return dicts