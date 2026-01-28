import os
import json
import torch
from src.user.input_param import InputParam
from src.PWMLFF.dp_param_extract import extract_force_field
from src.PWMLFF.dp_network import dp_network
from utils.file_operation import delete_tree, copy_tree, copy_file
from utils.json_operation import get_parameter, get_required_parameter
from src.pre_data.find_maxneighbor import get_max_neighbor

'''
description: do dp training
    step1. generate feature from MOVEMENTs
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def dp_train(input_json: json, cmd:str):
    dp_param = InputParam(input_json, cmd) 

    # max_neighbor, _, _, _ , dataset = get_max_neighbor(
    #         data_paths=dp_param.file_paths.train_data_path,
    #         format=dp_param.file_paths.format,
    #         atom_types=dp_param.atom_type,
    #         cutoff_radial=dp_param.descriptor.Rmax,
    #         with_type=True
    # )
    # dp_param.max_neigh_num = max(max_neighbor, dp_param.max_neigh_num)
    # dp_param.print_input_params(json_file_save_name="std_input.json")
    dp_trainer = dp_network(dp_param)
    dp_trainer.train()
    if os.path.exists(dp_param.file_paths.model_save_path) is False:
        if os.path.exists(dp_param.file_paths.model_load_path):
            dp_param.file_paths.model_save_path = dp_param.file_paths.model_load_path
    # extract_force_field(dp_param) # not not generate the forcefield dir

'''
description: 
    do dp inference:
    setp0. read params from mode.cpkt file, and set model related params to test
    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def dp_test(input_json: json, cmd:str):
    model_load_path = get_required_parameter("model_load_file", input_json)
    model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
    json_dict_train = model_checkpoint["json_file"]
    model_checkpoint["json_file"]["datasets_path"] = []
    json_dict_train["train_data"] = []
    json_dict_train["valid_data"] = []
    json_dict_train["test_data"] = input_json["test_data"]
    json_dict_train["format"] = get_parameter("format", input_json, "pwmat/movement")
    dp_param = InputParam(json_dict_train, "test".upper())
    # set inference param
    dp_param.set_test_relative_params(input_json)
    dp_trainer = dp_network(dp_param)
    dp_trainer.inference()

'''
description: 
    copy model dir under workdir to target_dir
    copy forcefild dir under workdir to target_dir
    delete feature files under workdir
param {*} target_dir
param {*} model_store_dir
param {*} forcefield_dir
param {*} train_dir
return {*}
author: wuxingxing
'''
def copy_train_result(target_dir, model_store_dir, forcefield_dir):
    # copy model
    target_model_path = os.path.join(target_dir, os.path.basename(model_store_dir))
    copy_tree(model_store_dir, target_model_path)
    # copy forcefild
    target_forcefield_dir = os.path.join(target_dir, os.path.basename(forcefield_dir))
    copy_tree(forcefield_dir, target_forcefield_dir)


'''
description: 
    copy inference result under work dir to target_dir

param {*} test_dir
param {*} json_dir
return {*}
author: wuxingxing
'''
def copy_test_result(target_dir, test_dir):
    # copy inference result
    target_test_dir = os.path.join(target_dir, os.path.basename(test_dir))
    for file in os.listdir(test_dir):
        if file.endswith(".txt") or file.endswith('.csv'):
            copy_file(os.path.join(test_dir, file), os.path.join(target_test_dir, os.path.basename(file)))
    
