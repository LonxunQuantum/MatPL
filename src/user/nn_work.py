import os
import json
import torch
from src.user.input_param import InputParam
from src.PWMLFF.nn_param_extract import extract_force_field
from src.PWMLFF.nn_network import nn_network
from utils.file_operation import delete_tree, copy_tree, copy_file
from utils.json_operation import get_parameter, get_required_parameter
from src.pre_data.find_maxneighbor import get_max_neighbor
'''
description: do nn training
    step1. generate feature from MOVEMENTs
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def nn_train(input_json: json, cmd:str):
    nn_param = InputParam(input_json, cmd) 
    # set max neighbor
    max_neighbor, _, _, _ , dataset = get_max_neighbor(
            data_paths=nn_param.file_paths.train_data_path,
            format="pwmat/movement",
            atom_types=nn_param.atom_type,
            cutoff_radial=nn_param.descriptor.Rmax * 1.25,
            with_type=False
    )
    nn_param.max_neigh_num = max(max_neighbor, nn_param.max_neigh_num)
    nn_param.print_input_params(json_file_save_name="std_input.json")
    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.train_data_path) > 0:
        chunk_size = min(10, len(dataset))
        feature_path = nn_trainer.generate_data(
                                chunk_size = chunk_size,
                                shuffle=False, 
                                movement_path = nn_param.file_paths.train_data_path,
                                feature_type="train_feature")
        nn_param.file_paths.set_train_feature_path([feature_path])

    if len(nn_param.file_paths.valid_data_path) > 0:
        feature_path = nn_trainer.generate_data(shuffle=False, 
                                movement_path = nn_param.file_paths.valid_data_path,
                                feature_type="valid_feature")
        nn_param.file_paths.set_valid_feature_path([feature_path])
    nn_trainer.train()
    # if the input epochs to the end, model will not be trained and will not be saved at work_dir
    if os.path.exists(nn_param.file_paths.model_save_path) is False:
        if os.path.exists(nn_param.file_paths.model_load_path):
            nn_param.file_paths.model_save_path = nn_param.file_paths.model_load_path
    extract_force_field(nn_param)
    
    if os.path.realpath(nn_param.file_paths.json_dir) != os.path.realpath(nn_param.file_paths.nn_work) :
        copy_train_result(nn_param.file_paths.json_dir, \
                    nn_param.file_paths.model_store_dir, nn_param.file_paths.forcefield_dir)
        # delete train_feature and valid_feature files
        if nn_param.file_paths.reserve_feature is False:
            if len(nn_param.file_paths.train_feature_path) > 0:
                for feature_path in nn_param.file_paths.train_feature_path:
                    delete_tree(feature_path)
            if len(nn_param.file_paths.valid_feature_path) > 0:
                for feature_path in nn_param.file_paths.valid_feature_path:   
                    delete_tree(feature_path)
        # delete workdir
        if nn_param.file_paths.reserve_work_dir is False:
            delete_tree(nn_param.file_paths.nn_work)

def gen_nn_feature(input_json: json, cmd:str):
    nn_param = InputParam(input_json, cmd) 
    nn_param.print_input_params(json_file_save_name="std_input.json")
    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.train_movement_path) > 0:
        feature_path = nn_trainer.generate_data()
    print("feature generated done, the dir path is: \n{}".format(feature_path))
    return feature_path

'''
description: 
    do nn inference:
    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def nn_test(input_json: json, cmd:str):
    model_load_path = get_required_parameter("model_load_file", input_json)
    model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
    json_dict_train = model_checkpoint["json_file"]
    json_dict_train["train_data"] = []
    json_dict_train["valid_data"] = []
    json_dict_train["test_data"] = input_json["test_data"]
    json_dict_train["format"] = get_parameter("format", input_json, "pwmat/movement")

    max_neighbor, _, _, _ , dataset = get_max_neighbor(
            data_paths=json_dict_train["test_data"],
            format="pwmat/movement",
            atom_types=json_dict_train["atom_type"],
            cutoff_radial=json_dict_train['model']['descriptor']['Rmax'] * 1.25,
            with_type=False
    )
    
    json_dict_train["max_neigh_num"] = max(max_neighbor, get_parameter("max_neigh_num", json_dict_train, 100))

    nn_param = InputParam(json_dict_train, "test".upper())
    # set inference param
    nn_param.set_test_relative_params(input_json)
    nn_param.print_input_params(json_file_save_name="std_input.json")

    nn_trainer = nn_network(nn_param)
    if len(nn_param.file_paths.test_data_path) > 0:
        feature_path = nn_trainer.generate_data(shuffle=False, 
            movement_path = nn_param.file_paths.test_data_path,
            feature_type="test_feature")
        nn_param.file_paths.set_test_feature_path([feature_path])

    nn_trainer.inference()

    if os.path.realpath(nn_param.file_paths.json_dir) != os.path.realpath(nn_param.file_paths.nn_work) :
        copy_test_result(nn_param.file_paths.json_dir, nn_param.file_paths.test_dir)
        if nn_param.file_paths.reserve_work_dir is False:
            delete_tree(nn_param.file_paths.nn_work)

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
    copy inference dir under work dir to target_dir
param {*} test_dir
param {*} json_dir
return {*}
author: wuxingxing
'''
def copy_test_result(target_dir, test_dir):
    # copy inference result
    target_test_dir = os.path.join(target_dir, os.path.basename(test_dir))
    for file in os.listdir(test_dir):
        if os.path.isfile(os.path.join(test_dir, file)):
        # if file.endswith(".txt") or file.endswith('.csv'):
            copy_file(os.path.join(test_dir, file), os.path.join(target_test_dir, os.path.basename(file)))