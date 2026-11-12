import torch
from src.utils.nep_model_type import is_nep_model_type, nep_model_type_from_txt_header
# import os
# import numpy as np
# from src.user.input_param import InputParam
# from src.PWMLFF.dp_network import dp_network
# from src.PWMLFF.dp_mods.dp_trainer import train_KF, save_checkpoint

# def conver_to_dp_torch2_version(model_file, atom_type, data, format, savename, save_dir, davg_dir, data_dict=None):
#     ckpt = torch.load(model_file, map_location="cpu", weights_only=False)
#     state_dict_list = list(ckpt['state_dict'].keys())
#     if "embedding_net.0.layers.0.weight" in state_dict_list[0]:
#         print("The model input is dp torch2 version, does not need convert!")
#         return
#     if 'embedding_net.0.weights.weight0' in state_dict_list[0]:
#         model_dict, emb_net_size, fit_net_size, atom_num, M2 = read_structure_dp(ckpt)
#     else:
#         raise Exception("The model constructure can not be recognized!")

#     assert len(atom_type) == atom_num, \
#         "The input element list {} does not match the model, as the model contains {} elements"\
#             .format(atom_type, atom_num)
#     input_json = make_json_input(model_dict, emb_net_size, fit_net_size, atom_type, M2, data, format)
#     data_dict = convert_to_dp(model_dict, input_json, savename, save_dir, ckpt['epoch'], davg_dir,  data_dict)
#     return data_dict

# def convert_to_dp(model_dict:dict, input_json:dict, savename:str, save_dir:str, epoch:int, davg_dir:str, data_dict=None):
#     dp_param = InputParam(input_json, "train") 
#     # dp_param.print_input_params(json_file_save_name="std_input.json")
#     dp_trainer = dp_network(dp_param)
#     if len(dp_param.file_paths.raw_path) > 0:
#         datasets_path = dp_trainer.generate_data()
#         dp_param.file_paths.set_datasets_path(datasets_path)
#     davg, dstd, energy_shift, max_atom_nums = dp_trainer.load_davg_from_ckpt()#跟energy_shift没关系
#     davg = np.load(os.path.join(davg_dir, "davg.npy"))
#     dstd = np.load(os.path.join(davg_dir, "dstd.npy"))
#     davg, dstd, energy_shift, atom_map, train_loader, val_loader = dp_trainer.load_data(davg, dstd, energy_shift, max_atom_nums) #davg, dstd, energy_shift, atom_map
#     model, optimizer = dp_trainer.load_model_optimizer(davg, dstd, energy_shift)
#     if data_dict is None:
#         loss, loss_Etot, loss_Etot_per_atom, loss_Force, loss_Ei, loss_egroup, loss_virial, loss_virial_per_atom, Sij_max = train_KF(
#                         train_loader, model, dp_trainer.criterion, optimizer, 1, dp_trainer.device, dp_trainer.dp_params
#                     ) # epoch = 1
#     else:
#         Sij_max = data_dict["Sij_max"]
#     # copy net param
#     target_state_dict = copy_net_param(model_dict, model)
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     save_checkpoint(
#         {
#         "json_file":dp_trainer.dp_params.to_dict(),
#         "epoch": epoch,
#         "state_dict": target_state_dict,
#         "davg":davg,
#         "dstd":dstd,
#         "energy_shift":energy_shift,
#         "atom_type_order": atom_map,    #atom type order of davg/dstd/energy_shift
#         "sij_max":Sij_max
#         },
#         savename,
#         save_dir
#     )
#     data_dict = {}
#     data_dict["Sij_max"] = Sij_max
#     return data_dict

# def copy_net_param(input_model, target_model):
#     target_state_dict = target_model.state_dict()
#     assert len(target_state_dict) == len(input_model)
#     for layer in target_state_dict:
#         target_state_dict[layer] = input_model[layer].cpu().to(target_model.state_dict()[layer].device)
#     return target_state_dict

# def make_json_input(model_dict, emb_net_size:list, fit_net_size:list, atom_list:int, M2:int, data:list[str], format:str):
#     has_resnet = False
#     for key in model_dict.keys():
#         if "resnet" in key:
#             has_resnet = True
#             break

#     input_dict = {}
#     input_dict["atom_type"] = atom_list
#     input_dict["model_type"] = "DP"
#     # input_dict["chunk_size"] = 1
#     descriptor = {}
#     fitting_net = {}
#     optimizer = {}
#     optimizer["optimizer"] = "ADAM"
#     optimizer["batch_size"] = 32
#     descriptor["M2"] = M2
#     descriptor["network_size"] = emb_net_size
#     fitting_net["network_size"] = fit_net_size
#     fitting_net["resnet_dt"] = has_resnet

#     input_dict["model"] = {}
#     input_dict["model"]["descriptor"] = descriptor
#     input_dict["model"]["fitting_net"] = fitting_net
#     input_dict["optimizer"] = optimizer

#     if format is not None:
#         input_dict["raw_data"] = data
#         input_dict["format"] = format
#     else:
#         if isinstance(data, str):
#             data = [data]
#         input_dict["datasets_path"] = data
#     return input_dict


# '''
# description: 
# param {str} ckpt_file
# return {*}
#     model_structure, embeddingnet shape, fitting net shape, atom type nums
# author: wuxingxing
# '''
# def read_structure_dp(ckpt):
#     model_dict = {}
#     emb_net_size = []
#     fit_net_size = []
#     fit_net_nums = []
#     state_dict_list = list(ckpt['state_dict'].keys())
#     if 'embedding_net.0.weights.weight0' in state_dict_list[0]:# 
#         for layer in state_dict_list:
#             # 'embedding_net.0.weights.weight0' 'embedding_net.0.layers.1.weight'
#             #
#             # 'embedding_net.0.layers.0.weight', 'embedding_net.0.layers.0.bias', 
#             # 'embedding_net.0.layers.1.weight', 'embedding_net.0.layers.1.bias'
#             net_name, net_id, layer_name, _layer_id = layer.split('.')
#             if "weights" in layer_name:
#                 layer_name = "weight"
#             if 'weight' in _layer_id:
#                 layer_id = _layer_id.replace('weight', '')
#             elif 'bias' in _layer_id:
#                 layer_id = _layer_id.replace('bias', '')
#             elif 'resnet_dt' in _layer_id:
#                 layer_id = _layer_id.replace('resnet_dt', '')
#             if "fitting_net." in layer and "fitting_net.{}".format(net_id) not in fit_net_nums:
#                 fit_net_nums.append("fitting_net.{}".format(net_id))
#             if "embedding_net.0" in layer:
#                 if ".weights." in layer:
#                     emb_net_size.append(ckpt['state_dict'][layer].shape[1])
#             if "fitting_net.0" in layer:
#                 if ".weights." in layer:
#                     fit_net_size.append(ckpt['state_dict'][layer].shape[1])
#             M2 = int(ckpt['state_dict']["fitting_net.0.weights.weight0"].shape[0]/emb_net_size[-1])
#             new_layer_name = "{}.{}.{}.{}.{}".format(net_name, net_id, 'layers', layer_id, layer_name)
#             model_dict[new_layer_name] = ckpt['state_dict'][layer]

#         return model_dict, emb_net_size, fit_net_size, len(fit_net_nums), M2
#     else:
#         return None, None, None, None, None

def get_model_type(model_load_path):
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        model_type = _model_checkpoint['json_file']['model_type']
        if model_type == "DP":
            return "DP"
        elif model_type == "NN":
            return "NN"
        elif is_nep_model_type(model_type):
            return model_type.upper()
        return None
    except Exception as e:
        with open(model_load_path, 'r') as rf:
            line = rf.readline()
        return nep_model_type_from_txt_header(line)

def is_nep_ckpt(model_load_path):
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        model_type = _model_checkpoint['json_file']['model_type']
        if is_nep_model_type(model_type):
            return True
    except Exception as e:
        with open(model_load_path, 'r') as rf:
            line = rf.readline()
        if "nep" in line:
            return False
    return False

def is_nep_txt(model_load_path):
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        return False
    except Exception as e:
        with open(model_load_path, 'r') as rf:
            line = rf.readline()
        if "nep" in line:
            return True
    return False
