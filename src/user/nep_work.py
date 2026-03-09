import os
import torch
import json
import socket
from contextlib import closing
import argparse

import torch
import torch.multiprocessing as mp

from src.user.input_param import InputParam
from src.PWMLFF.nep_network import nep_network, save_checkpoint
from src.utils.file_operation import delete_tree, copy_tree, copy_file
from src.utils.atom_type_emb_dict import element_table
from src.utils.file_operation import delete_tree, copy_tree, copy_file
from src.utils.json_operation import get_parameter, get_required_parameter
from src.utils.atom_type_emb_dict import get_atomic_number_from_name
from src.utils.nep_to_gpumd import extract_model

def find_free_port():
    """查找一个空闲的 TCP 端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # 绑定到任意地址，端口 0 让系统分配空闲端口
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])  # 返回分配的端口号

def main_worker(rank, world_size, nep_param):
    try:
        if nep_param.multi_nodes is False and nep_param.multi_gpus: # single node and mulit GPUs
            nep_param.rank = rank
            nep_param.local_rank = rank
        nep_net = nep_network(nep_param)
        nep_net.train()
    except Exception as e:
        print(f"Rank {nep_param.rank}, LocalRank {nep_param.local_rank}: Error occurred: {e}")
        raise

'''
description: do nep training
    step1. generate feature to xyz format files
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def nep_train(input_json: json, cmd:str):
    nep_param = InputParam(input_json, cmd)
    
    nep_param.multi_gpus = False
    nep_param.multi_nodes = False
    nep_param.world_size = 1
    nep_param.rank = 0
    nep_param.local_rank = 0
    main_worker(0, 1, nep_param)

# '''
# description: 
#     do dp inference:
#     setp0. read params from mode.cpkt file, and set model related params to test
#         the params need to be set by nep.txt or nep.in file:

#     step1. generate feature, the movement from json file 'test_movement_path'
#     step2. load model and do inference
#     step3. copy inference result files to the same level directory of jsonfile
# param {json} input_json
# param {str} cmd
# return {*}
# author: wuxingxing
# '''
def nep_test(input_json: json, cmd:str):
    model_load_path = get_parameter("model_load_file", input_json, None)
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        nep_content, model_atom_type, atom_names = extract_model(model_load_path)
        model_load_path = os.path.join(os.path.dirname(os.path.abspath(model_load_path)), "tmp_pwmlff_nep_test.txt")
        with open(model_load_path, 'w') as wf:
            wf.writelines(nep_content)
    except Exception as e:
        with open(model_load_path, 'r') as rf:
            line = rf.readline()
        if "nep" in line:
            print("the input model file is txt format")
        else:
            raise Exception("ERROR! The input model file cannot be parsed!")

    if model_load_path is None or not os.path.exists(model_load_path):
        raise Exception("Error! NEP test should have nep.txt files or nep_model.ckpt file!")
    input_dict = {}
    input_dict["model_type"] = "NEP"
    # get atom_type from nep.txt file
    with open(model_load_path, 'r') as rf:
        atom_type_str = rf.readline().split()[2:]
    atom_type_list = get_atomic_number_from_name(atom_type_str)
    input_dict["atom_type"] = atom_type_list
    input_dict["nep_txt_file"] = model_load_path
    input_dict["test_data"] = get_parameter("test_data", input_json, [])
    input_dict["datasets_path"] = get_parameter("datasets_path", input_json, [])
    input_dict["raw_files"] = get_parameter("raw_files", input_json, [])
    input_dict["format"] = get_parameter("format", input_json, "pwmat/config")
    input_dict["optimizer"] = {}
    input_dict["optimizer"]["optimizer"] = "ADAM"        
    nep_param = InputParam(input_dict, "test".upper())
    # set inference param
    nep_param.set_test_relative_params(input_json, is_nep_txt=True)
    nep_trainer = nep_network(nep_param)
    # nep_trainer.inference()
    # nep_trainer.gpu_nep_inference(model_load_path)
    nep_trainer.multi_cpus_nep_inference(model_load_path) # the speed is 1cpu > 1gpu
    # if nep_trainer.device.type == 'cuda':
    #     nep_trainer.inference()
        
    #     # nep_trainer.gpu_nep_inference(model_load_path)
    # else: #cpu
    #     nep_trainer.multi_cpus_nep_inference(model_load_path)
    if "tmp_pwmlff_nep_test" in model_load_path:
        os.remove(model_load_path)

    """
    else: ckpt file test
        # load from nep.ckpt file
        model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        json_dict_train = model_checkpoint["json_file"]
        model_checkpoint["json_file"]["datasets_path"] = []
        json_dict_train["optimizer"] = {}
        json_dict_train["optimizer"]["optimizer"] = "ADAM"
        nep_param = InputParam(json_dict_train, "test".upper())
        # set inference param
        nep_param.set_test_relative_params(input_json)
        nep_trainer = nep_network(nep_param)

    if len(nep_param.file_paths.raw_path) > 0:
        data_paths = nep_trainer.generate_data()
        nep_param.file_paths.set_datasets_path(data_paths)
    nep_trainer.inference()
    """

def nep_test_ckpt(input_json: json, cmd:str):

    model_load_path = get_required_parameter("model_load_file", input_json)
    model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
    json_dict_train = model_checkpoint["json_file"]
    model_checkpoint["json_file"]["datasets_path"] = []
    json_dict_train["train_data"] = []
    json_dict_train["valid_data"] = []
    json_dict_train["test_data"] = input_json["test_data"]
    json_dict_train["format"] = get_parameter("format", input_json, "pwmat/movement")
    nep_param = InputParam(json_dict_train, "test".upper())
    # set inference param
    nep_param.set_test_relative_params(input_json)
    dp_trainer = nep_network(nep_param)
    dp_trainer.inference()
    
# def toneplmps(cmd_list:list[str]):
# this function is move to togpumd -> totxt for nep5
#     infos = "\n\nThis cmd is used to convert the nep_model.ckpt trained by MatPL to nep.txt for lammps!\n"
#     infos += "After the command execution is completed, you will receive a file named 'nep_to_lmps.txt'\n\n"
#     # infos += "This cmd requires installation of pytorch in your Python environment, and there is no mandatory version requirement.\n"
#     infos += "The command example: \n"
#     infos += "    'MatPL toneplmps nep_model.ckpt'\n"
#     # print(infos)

#     if "-h" in cmd_list or "-help" in cmd_list or "--help" in cmd_list:
#         print(infos)
#     else:
#         ckpt_file = cmd_list[0]
#         head_content, model_atom_type, atom_names = extract_model(ckpt_file, togpumd=False)
#         with open(os.path.join(os.path.dirname(os.path.abspath(ckpt_file)), "nep_to_lmps.txt"), 'w') as wf:
#             wf.write(head_content)
#         print("Successfully convert to nep.in and nep.txt file.") 

def togpumd(cmd_list:list[str]):
    from src.utils.nep_to_gpumd import nep_ckpt_to_gpumd
    nep_ckpt_to_gpumd(cmd_list)

# def tonepckpt(cmd_list:list[str], save_ckpt:bool=True):
#     infos = "\n\nThis cmd is used to convert the nep.txt from GPUMD to PWMLFF '.ckpt' format!\n"
#     infos += "After the command execution is completed, you will receive a file named 'nep_from_gpumd.ckpt'\n\n"
#     # infos += "This cmd requires installation of pytorch in your Python environment, and there is no mandatory version requirement.\n"
#     infos += "The command example: \n"
#     infos += "    'PWMLFF topwmlff nep.txt'\n"
#     # print(infos)

#     if "-h" in cmd_list or "-help" in cmd_list or "--help" in cmd_list:
#         print(infos)
#     else:
#         nep_txt_path = cmd_list[0]
#         # nep_in_path = cmd_list[1]
#         input_dict = {}
#         input_dict["model_type"] = "NEP"
#         # get atom_type from nep.txt file
#         with open(nep_txt_path, 'r') as rf:
#             line_one = rf.readline()
#             if len(line_one.split()) < 3:
#                 raise Exception("ERROR! The nep.txt file is not correct! The first line should have at least three elements, such as 'nep4 1 Li'! Please check the path {}".format(nep_txt_path))
#             atom_type_str = line_one.split()[2:]
#         # with open(nep_in_path, 'r') as rf:
#         #     line_one = rf.readline()
#         #     if len(line_one.split()) != 2:
#         #         raise Exception("ERROR! The nep.in file is not correct! The first line should have 2 elements,such as 'version 4'! Please check the path {}".format(nep_in_path))
#         atom_type_list = get_atomic_number_from_name(atom_type_str)
#         input_dict["atom_type"] = atom_type_list
#         # input_dict["nep_in_file"] = nep_in_path
#         input_dict["nep_txt_file"] = nep_txt_path
#         input_dict["datasets_path"] = []
#         input_dict["raw_files"] = []
#         input_dict["format"] = "pwmat/config"
#         input_dict["optimizer"] = {}
#         input_dict["optimizer"]["optimizer"] = "ADAM"        
#         nep_param = InputParam(input_dict, "test".upper())
#         # set inference param
#         nep_param.set_test_relative_params(input_dict, is_nep_txt=True)

#         nep_trainer = nep_network(nep_param)
#         energy_shift = [1.0 for _ in atom_type_list]
#         model, optimizer = nep_trainer.load_model_optimizer(energy_shift)

#         if save_ckpt:
#             save_checkpoint(
#                 {
#                 "json_file":nep_param.to_dict(),
#                 "epoch": 1,
#                 "state_dict": model.state_dict(),
#                 "energy_shift":energy_shift,
#                 "atom_type_order": atom_type_list,    #atom type order of davg/dstd/energy_shift, the user input order
#                 # "sij_max":Sij_max,
#                 "q_scaler": model.get_q_scaler(),
#                 "optimizer":optimizer.state_dict()
#                 },
#                 "nep_from_txt.ckpt",
#                 os.getcwd()
#             )
#         print("Convert successfully!")
#         return model, "NEP", nep_param