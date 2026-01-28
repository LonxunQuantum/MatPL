import os
from utils.json_operation import get_parameter, get_required_parameter
from src.user.work_file_param import WorkFileStructure

'''
description: 
    covert the input json file to class
return {*}
author: wuxingxing
'''
class Extract_Param(object):
    '''
    description: 
        covert the input json file to class
    param {*} self
    param {dict} json_input: the input json file from user
    return {*}
    author: wuxingxing
    '''
    def __init__(self, json_input:dict) -> None:
        self.set_params(json_input)
        self.set_workdir_structures(json_input)
        self.file_paths._set_NN_PWdata_dirs(json_input)
        self.file_paths.set_train_valid_file(json_input)
        
    '''
    description: 
        set feature common params (dp/nn/nep/gnn)
    param {*} self
    param {dict} json_input
    return {*}
    author: wuxingxing
    '''    
    def set_params(self, json_input:dict):
        # set feature related params
        self.valid_shuffle = get_parameter("valid_shuffle", json_input, False)
        self.train_valid_ratio = get_parameter("train_valid_ratio", json_input, 0.8)
        self.seed = get_parameter("seed", json_input, 2024)
        self.format = get_parameter("format", json_input, "movement")
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
                            model_type=get_parameter("reserve_feature", json_input, ""))

    '''
    description: 
        this dict is used for feature generation of NN models
    param {*} self
    return {*}
    author: wuxingxing
    '''    
    def get_data_file_dict(self):
        data_file_dict = self.file_paths.get_data_file_structure()
        return data_file_dict
        
def help_info():
    print("train: do model training")
    print("test: do dp model inference")
    