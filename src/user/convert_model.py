import torch

def copy_net_param(input_model, target_model):
    target_state_dict = target_model.state_dict()
    assert len(target_state_dict) == len(input_model)
    for layer in target_state_dict:
        target_state_dict[layer] = input_model[layer].cpu().to(target_model.state_dict()[layer].device)
    return target_state_dict

def make_json_input(model_dict, emb_net_size:list, fit_net_size:list, atom_list:int, M2:int, data:list[str], format:str):
    has_resnet = False
    for key in model_dict.keys():
        if "resnet" in key:
            has_resnet = True
            break

    input_dict = {}
    input_dict["atom_type"] = atom_list
    input_dict["model_type"] = "DP"
    # input_dict["chunk_size"] = 1
    descriptor = {}
    fitting_net = {}
    optimizer = {}
    optimizer["optimizer"] = "ADAM"
    optimizer["batch_size"] = 32
    descriptor["M2"] = M2
    descriptor["network_size"] = emb_net_size
    fitting_net["network_size"] = fit_net_size
    fitting_net["resnet_dt"] = has_resnet

    input_dict["model"] = {}
    input_dict["model"]["descriptor"] = descriptor
    input_dict["model"]["fitting_net"] = fitting_net
    input_dict["optimizer"] = optimizer

    if format is not None:
        input_dict["raw_data"] = data
        input_dict["format"] = format
    else:
        if isinstance(data, str):
            data = [data]
        input_dict["datasets_path"] = data
    return input_dict

def get_model_type(model_load_path):
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        model_type = _model_checkpoint['json_file']['model_type']
        if model_type == "NEP":
            return "NEP"
        return None
    except Exception as e:
        with open(model_load_path, 'r') as rf:
            line = rf.readline()
        if "nep" in line:
            return "NEP"
        else:
            raise Exception("ERROR! The input model file cannot be parsed!")

def is_nep_ckpt(model_load_path):
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        model_type = _model_checkpoint['json_file']['model_type']
        if model_type == "NEP":
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