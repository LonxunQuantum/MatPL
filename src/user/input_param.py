import os
import json
from utils.json_operation import get_parameter, get_required_parameter
from utils.nep_to_gpumd import get_atomic_name_from_str
from src.user.model_param import ModelParam
from src.user.optimizer_param import OptimizerParam
from src.user.work_file_param import WorkFileStructure
from src.user.nn_feature_type import Descriptor

from src.user.nep_param import NepParam
'''
description: 
    covert the input json file to class
return {*}
author: wuxingxing
'''
class InputParam(object):
    '''
    description: 
        covert the input json file to class
    param {*} self
    param {dict} json_input: the input json file from user
    return {*}
    author: wuxingxing
    '''
    def __init__(self, json_input:dict, cmd) -> None:
        self.nep_param = None

        self.cmd = cmd
        self.inference = True if self.cmd == "test".upper() else False
        self.model_type = get_required_parameter("model_type", json_input).upper()
        # self.atom_type = get_required_parameter("atom_type", json_input)
        self.atom_type = get_atomic_name_from_str(get_required_parameter("atom_type", json_input))
        self.model_num = get_parameter("model_num", json_input, 1)
        self.recover_train = get_parameter("recover_train", json_input, True)
        self.max_neigh_num = get_parameter("max_neigh_num", json_input, 100)
        self.save_step = get_parameter("save_step", json_input, None)
        self.max_save_num = get_parameter("max_save_num", json_input, 10)
        self.profiling = get_parameter("profiling", json_input, False)#not realized

        self.set_feature_params(json_input)
        self.set_workdir_structures(json_input)
        if self.inference and self.model_type in ["NN", "LINEAR"]:
            self.file_paths.nn_work = os.path.join(self.file_paths.json_dir, "work_test_dir")

        if self.model_type in ["DP", "NN", "NEP", "LINEAR", "CHEBY"]:
            self.set_model_init_params(json_input)
            self.set_default_multi_gpu_info(json_input)
            # set optimizer
            self.set_optimizer(json_input)
            if self.model_type in ["NN", "LINEAR"]:
                self.optimizer_param.batch_size = 1
                print("Warning! NN and Linear models only support single batch, automatically adjust batch_size=1.")
        # elif self.model_type in ["NEP"]:
        #     self.set_nep_in_params(json_input)
            
        elif self.model_type in ["GNN"]:
            raise Exception("GNN is not realized yet!")
        else:
            raise Exception("ERROR! {} model type not supported!".format(self.model_type))

        # DP, NN, and NEP common file paths
        self.file_paths.set_model_file_paths(json_input)
        self.file_paths.set_train_valid_file(json_input)

    '''
    description: 
        extract params from nep.in file or 'model' dict, it both exsits, prioritize taking from nep.in file
        对于NEP 参数，存在nep.in文件，则从nep.in提取，否则从json字典提取，先提取未nepparam对象，然后从中取出值，复制给模型和优化器
        需要补充 从 nep.txt 中取参数，支持对nep.txt 继续训练
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''    
    def set_nep_params(self, json_input:dict):
        self.model_param = ModelParam()
        nep_dict = get_parameter("model", json_input, {})
        self.model_param.set_type_embedding_net(
                                    network_size=self.descriptor.type_network_size, 
                                    bias=self.descriptor.type_bias, 
                                    resnet_dt=self.descriptor.type_resnet_dt, 
                                    activation=self.descriptor.type_activation,
                                    physical_property=self.descriptor.type_physical_property)
        self.model_param.set_nn_fitting_net(get_parameter("fitting_net",nep_dict, {}))
        self.is_dfeat_sparse = get_parameter("is_dfeat_sparse", json_input, False)  #for nn, 'true' not realized

        # nep_in_file = get_parameter("nep_in_file", json_input, None)
        nep_param = NepParam()
        # if nep_in_file is not None:
        #     nep_param.set_nep_param_from_nep_in(nep_in_file, self.atom_type)
        nep_txt_file = get_parameter("nep_txt_file", json_input, None)
        if nep_txt_file is not None:
            nep_param.set_nep_nn_c_param_from_nep_txt(nep_txt_file, self.atom_type)
        else:
            nep_param.set_nep_param_from_json(json_input, self.atom_type)
        # if fixed
        nep_param.set_fixed_params(json_input)
        self.nep_param = nep_param
        self.model_param.fitting_net.network_size = nep_param.neuron
        self.descriptor.cutoff = nep_param.cutoff
        self.descriptor.n_max = nep_param.n_max
        self.descriptor.basis_size = nep_param.basis_size
        self.descriptor.l_max = nep_param.l_max
        self.descriptor.zbl = nep_param.zbl
        # self.descriptor.type_weight = nep_param.type_weight


    '''
    description: 
        params for dp and nn model training/inference work
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''
    def set_model_init_params(self, json_input:dict):
        # if feature_type specified at first layer of JSON, use this value as feature type in descriptor
        # nn feature types default is [3,4]
        self.feature_type = get_parameter("feature_type", json_input, None) 

        self.type_embedding = get_parameter("type_embedding",json_input, False)
        model_dict = get_parameter("model", json_input, {})
        self.descriptor = Descriptor(get_parameter("descriptor", model_dict, {}), self.model_type, self.cmd, self.feature_type, self.type_embedding)
        
        self.atom_type_dict = []
        for idx in self.atom_type:
            #the last 4 param used in function gen_config_inputfile()
            single_dict = {'type': idx, 'Rc': self.descriptor.Rmax, 'Rm': self.descriptor.Rmin, 'iflag_grid': 3, 'fact_base': 0.2, 'dR1': 0.5, "b_init":166.0}    
            self.atom_type_dict.append(single_dict)

        self.type_embedding = self.descriptor.type_embedding
        if self.model_type == "DP":
            self.model_param = ModelParam()
            self.model_param.set_type_embedding_net(
                                                    network_size=self.descriptor.type_network_size, 
                                                    bias=self.descriptor.type_bias, 
                                                    resnet_dt=self.descriptor.type_resnet_dt, 
                                                    activation=self.descriptor.type_activation,
                                                    physical_property=self.descriptor.type_physical_property)
            self.model_param.set_embedding_net(network_size=self.descriptor.network_size, 
                                               bias=self.descriptor.bias, 
                                               resnet_dt=self.descriptor.resnet_dt, 
                                               activation=self.descriptor.activation)
            self.model_param.set_dp_fitting_net(get_parameter("fitting_net",model_dict, {}))
        elif self.model_type == "NN":
            self.model_param = ModelParam()
            self.model_param.set_type_embedding_net(
                                        network_size=self.descriptor.type_network_size, 
                                        bias=self.descriptor.type_bias, 
                                        resnet_dt=self.descriptor.type_resnet_dt, 
                                        activation=self.descriptor.type_activation,
                                        physical_property=self.descriptor.type_physical_property)
            self.model_param.set_nn_fitting_net(get_parameter("fitting_net",model_dict, {}))
            self.is_dfeat_sparse = get_parameter("is_dfeat_sparse", json_input, False)  #'true' not realized
        elif self.model_type == "Linear".upper():
            pass
        elif self.model_type == "NEP".upper():
            self.set_nep_params(json_input) #  nep.in 输入的适配可能并不需要
            self.file_paths.set_nep_native_file_paths()#  nep.in 输入的适配可能并不需要
        elif self.model_type == "CHEBY".upper():
            self.model_param = ModelParam()
            self.model_param.set_nn_fitting_net(get_parameter("fitting_net",model_dict, {}))
        else: # linear
            raise Exception("model_type {} not realized yet".format(self.model_type))
    
    def set_optimizer(self, json_input:dict):
        self.optimizer_param = OptimizerParam()
        self.optimizer_param.set_optimizer( json_input ,self.nep_param)
        #

    '''
    description: 
        Reserved param interface for multi-GPU, multi-node parallel training of DP/NN models
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''    
    def set_default_multi_gpu_info(self, json_input:dict):
        # multi GPU train params  these params not used
        # "number of data loading workers (default: 4)
        self.workers = get_parameter("workers", json_input, 0)
        # dist training by horovod, when multi GPU train, need True,
        self.hvd = get_parameter("hvd", json_input, False)
        self.world_size = get_parameter("world_size", json_input, -1)
        self.rank = get_parameter("rank", json_input, -1)
        self.dist_url = get_parameter("dist_url", json_input, "tcp://localhost:23456")  
        self.dist_backend = get_parameter("dist_backend", json_input, "nccl")
        self.distributed = get_parameter("distributed", json_input, False)
        self.gpu = get_parameter("gpu", json_input, None)

    '''
    description: 
        set feature common params (dp/nn/nep/gnn)
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''    
    def set_feature_params(self, json_input:dict):
        # set feature related params
        self.valid_shuffle = get_parameter("valid_shuffle", json_input, False)
        self.data_shuffle = get_parameter("data_shuffle", json_input, True)
        self.seed = get_parameter("seed", json_input, 2023)
        self.precision = get_parameter("precision", json_input, "float64")

    '''
    description: 
    set common workdir
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''
    def set_workdir_structures(self, json_input:dict):
        # set file structures
        self.file_paths = WorkFileStructure(json_dir=os.getcwd(), 
                            reserve_work_dir=get_parameter("reserve_work_dir", json_input, False), 
                            reserve_feature = get_parameter("reserve_feature", json_input, False), 
                            model_type=self.model_type)

    '''
    description: 
        set dp and NN inference params
    param {*} self
    param {dict} json_input
            is_nep_txt for nep.txt file load
    return {*}
    author: wuxingxing
    '''    
    def set_test_relative_params(self, json_input:dict, is_nep_txt:bool=False):
        self.inference = True
        self.recover_train = True
        self.optimizer_param.batch_size = 1     # set batch size to 1, so that each image inference info will be saved
        self.data_shuffle = False
        self.valid_shuffle = False
        self.file_paths.set_inference_paths(json_input,is_nep_txt = is_nep_txt)
    
    '''
    description: 
        this dict is used for dp model initialization
    param {*} self
    return {*}
    author: wuxingxing
    '''
    def get_dp_net_dict(self):
        net_dict = {}
        net_dict["model_type"] = self.model_type
        net_dict["net_cfg"] = {}
        net_dict["net_cfg"][self.model_param.embedding_net.net_type] = self.model_param.embedding_net.to_dict()
        net_dict["net_cfg"][self.model_param.fitting_net.net_type] = self.model_param.fitting_net.to_dict()
        net_dict["net_cfg"][self.model_param.type_embedding_net.net_type] = self.model_param.type_embedding_net.to_dict()
        net_dict["M2"] = self.descriptor.M2
        net_dict["maxNeighborNum"] = self.max_neigh_num
        net_dict["atomType"]=self.atom_type_dict
        net_dict["training_type"] = self.precision
        net_dict["Rc_M"] = self.descriptor.Rmax
        return net_dict
    
    '''
    description: 
        this dict is used for feature generation of NN models
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_data_file_dict(self):
        data_file_dict = {}
        # data_file_dict = self.file_paths.get_data_file_structure()
        data_file_dict["M2"] = self.descriptor.M2
        data_file_dict["maxNeighborNum"] = self.max_neigh_num
        data_file_dict["atomType"]=self.atom_type_dict
        data_file_dict["Rc_M"] = self.descriptor.Rmax
        data_file_dict["E_tolerance"] = self.descriptor.E_tolerance
        data_file_dict["train_egroup"] = self.optimizer_param.train_egroup

        return data_file_dict
    
    def to_dict(self):
        params_dict = {}
        params_dict["model_type"] = self.model_type
        params_dict["atom_type"] = self.atom_type
        if self.model_type !="NEP":
            params_dict["max_neigh_num"] = self.max_neigh_num
        if self.seed is not None:
            params_dict["seed"] = self.seed
        if self.model_num > 1 :
            params_dict["model_num"] = self.model_num
        # params_dict["E_tolerance"] = self.descriptor.E_tolerance
        # params_dict["Rmax"] = self.descriptor.Rmax
        # params_dict["Rmin"] = self.Rmin
        # params_dict["M2"] = self.descriptor.M2
        # params_dict["data_shuffle"] = self.data_shuffle
        # if self.cmd == "train".upper():
        #     params_dict["train_valid_ratio"] = self.train_valid_ratio
        
        # params_dict["precision"] = self.precision

        # params_dict["profiling"] = self.profiling
        # params_dict["workers"] = self.workers
        # params_dict["hvd"] = self.hvd
        # params_dict["world_size"] = self.world_size
        # params_dict["rank"] = self.rank
        # params_dict["dist_url"] = self.dist_url
        # params_dict["dist_backend"] = self.dist_backend
        # params_dict["distributed"] = self.distributed

        # params_dict["inference"] = self.inference
        
        if self.cmd == "train".upper() and self.model_type != "Linear".upper():
            params_dict["recover_train"] = self.recover_train
        
        params_dict["model"] = {}
        if self.model_type in ["DP", "NN", "LINEAR", "CHEBY", "NEP"]:
            params_dict["model"]["descriptor"] = self.descriptor.to_dict()

            if self.model_type == "Linear".upper():
                params_dict["optimizer"] = self.optimizer_param.to_linear_dict()
            else:
                params_dict["model"]["fitting_net"] = self.model_param.fitting_net.to_dict_std()
                params_dict["optimizer"] = self.optimizer_param.to_dict()
        # elif self.model_type in ["NEP"]:
        #     nep_in_content = self.model_param.nep_param.to_txt()
        elif self.model_type in ["GNN"]:
            raise Exception("The model GNN not realized yet!")

        file_dir_dict = self.file_paths.to_dict()
        for key in file_dir_dict:
            params_dict[key] = file_dir_dict[key]
        return params_dict
    
    def print_input_params(self, json_file_save_name="json_all.json"):
        params_dict = self.to_dict()
        json.dump(params_dict, open(os.path.join(self.file_paths.json_dir, json_file_save_name), "w"), indent=4)
        print(params_dict)
        
def help_info():
    print("\n" + "=" * 50) 
    print("         MatPL Basic Information")
    print("=" * 50) 
    print("Version: 2025.03")
    print("Compatible pwdata: >= 0.5.0")
    print("Compatible pwact: >= 0.3.3")
    print("Last Commit: 2025.05.07")
    print("Contact: support@pwmat.com")
    print("Citation: https://github.com/LonxunQuantum/PWMLFF")
    print("Manual online: http://doc.lonxun.com/PWMLFF/")
    print("=" * 50)  
    print("\n\n")
