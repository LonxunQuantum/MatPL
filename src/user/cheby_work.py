import os
import json
import torch
from src.user.input_param import InputParam
from src.PWMLFF.cheby_network import cheby_network
from utils.json_operation import get_required_parameter

def cheby_train(input_json: json, cmd:str):
    cheby_param = InputParam(input_json, cmd) 
    cheby_param.print_input_params(json_file_save_name="std_input.json")
    cheby_trainer = cheby_network(cheby_param)
    if len(cheby_param.file_paths.raw_path) > 0:
        datasets_path = cheby_trainer.generate_data()
        cheby_param.file_paths.set_datasets_path(datasets_path)
    scaler, energy_shift, max_atom_nums = cheby_trainer._get_stat()
    cheby_trainer.train(scaler, energy_shift, max_atom_nums)
    if os.path.exists(cheby_param.file_paths.model_save_path) is False:
        if os.path.exists(cheby_param.file_paths.model_load_path):
            cheby_param.file_paths.model_save_path = cheby_param.file_paths.model_load_path

def cheby_test(input_json: json, cmd:str):
    model_load_path = get_required_parameter("model_load_file", input_json)
    model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
    json_dict_train = model_checkpoint["json_file"]
    model_checkpoint["json_file"]["datasets_path"] = []
    cheby_param = InputParam(json_dict_train, "test".upper())
    # set inference param
    cheby_param.set_test_relative_params(input_json)
    cheby_param.print_input_params(json_file_save_name="std_input.json")
    cheby_trainer = cheby_network(cheby_param)
    if len(cheby_param.file_paths.raw_path) > 0:
        datasets_path = cheby_trainer.generate_data()
        cheby_param.file_paths.set_datasets_path(datasets_path)
    scaler, energy_shift, max_atom_nums = cheby_trainer._get_stat()
    cheby_trainer.inference(scaler, energy_shift, max_atom_nums)
