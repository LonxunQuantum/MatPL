import torch
from src.mods.infer import Inference
import os 
import glob
import numpy as np
from pwdata import Config
from src.utils.nep_to_gpumd import get_atomic_name_from_number, check_atom_type_name

def _extract_infer_options(args: list[str]):
    valid_methods = ["ewald", "pppm"]
    filtered_args = []
    kspace_method = "ewald"
    total_charge = None
    i = 0
    while i < len(args):
        key = args[i].lower()
        if key == "kspace":
            if i + 1 < len(args):
                method = args[i + 1].lower()
                if method in valid_methods:
                    kspace_method = method
                else:
                    print(
                        "WARNING! Detected kspace option but method '{}' is invalid. "
                        "Valid methods are 'ewald' and 'pppm'. Use default 'ewald'.".format(args[i + 1])
                    )
                    kspace_method = "ewald"
                i += 2
            else:
                print(
                    "WARNING! Detected kspace option without method. "
                    "Valid methods are 'ewald' and 'pppm'. Use default 'ewald'."
                )
                i += 1
            continue
        if key == "total_charge":
            if i + 1 < len(args):
                try:
                    total_charge = float(args[i + 1])
                except ValueError:
                    print(
                        "WARNING! Detected total_charge option but value '{}' is invalid. "
                        "Use default total_charge=0.0.".format(args[i + 1])
                    )
                    total_charge = None
                i += 2
            else:
                print("WARNING! Detected total_charge option without value. Use default total_charge=0.0.")
                i += 1
            continue
        filtered_args.append(args[i])
        i += 1
    return filtered_args, kspace_method, total_charge

def infer_main(sys_cmd:list[str]):
    ckpt_file = sys_cmd[0]
    sys_index = 0
    structures_file = sys_cmd[1]
    format = sys_cmd[2] if len(sys_cmd) > 2 else "pwmat/config"
    sys_index = 2
    use_nep_txt = False
    device = None
    
    atom_names = sys_cmd[sys_index+1:]
    atom_names, kspace_method, total_charge = _extract_infer_options(atom_names)
    if isinstance(atom_names, list) is False:
        atom_names = [atom_names]
    if format.lower() == "lammps/dump" or format.lower() == "lammps/lmp":
        if atom_names is None:
            raise Exception("Error! For lammps/dump or lammps/lmp file, the atom type list of config should be set!")
        try:
            atom_types = get_atomic_name_from_number(atom_names)
        except Exception as e:
            if check_atom_type_name(atom_names):
                atom_types = atom_names
            else:
                raise Exception("Error! The input atom_type {} is not valid, please check!".format(" ".join(atom_names)))
    else:
        atom_types = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: {}".format(device.type))
    infer = Inference(ckpt_file, device, use_nep_txt)
    if os.path.isdir(structures_file):
        traj_list = glob.glob(os.path.join(structures_file, "*"))
        try:
            trajs = sorted(traj_list, key=lambda x:int(os.path.basename(x).split('.')[0]))
        except Exception:
            trajs = sorted(traj_list)
    else:
        trajs = [structures_file]
    for ti, traj in enumerate(trajs):
        image_read = Config(data_path=traj, format=format, atom_names=atom_types).images
        if infer.model_type == "DP":
            infer.inference(image_read)
        elif infer.model_type == "NEP":
            infer.inference_nep_txt(image_read, kspace_method=kspace_method, total_charge=total_charge)

def model_devi(ckpt_file_list, structure_dir, format, save_path, atom_names:list[str]=None):
    # set atom_types in trajs
    if format.lower() == "lammps/dump" or format.lower() == "lammps/lmp":
        if atom_names is None:
            raise Exception("Error! For lammps/dump or lammps/lmp file, the atom type list of config should be set by '-t'!")
        try:
            atom_types = get_atomic_name_from_number(atom_names)
        except Exception as e:
            if check_atom_type_name(atom_names):
                atom_types = atom_names
            else:
                raise Exception("The input '-t' or '--atom_type': '{}' is not valid, please check the input".format(" ".join(atom_names)))
    else:
        atom_types = None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device: {}".format(device.type))
    if os.path.isdir(structure_dir):
        traj_list = glob.glob(os.path.join(structure_dir, "*"))
        try:
            trajs = sorted(traj_list, key=lambda x:int(os.path.basename(x).split('.')[0]))
        except Exception:
            trajs = sorted(traj_list)
    else:
        trajs = [structure_dir]
    headline = "#   avg_devi_f       min_devi_f       max_devi_f       avg_devi_e       min_devi_e       max_devi_e\n"

    with open(save_path, 'w') as wf:
        wf.write(headline)
    model_lists = []
    for mi, model in enumerate(ckpt_file_list):
        model_lists.append(Inference(model, device))

    for ti, traj in enumerate(trajs):
        force_i = {}
        ei_i = {}
        Etot_i = {}
        image_read = Config(data_path=traj, format=format, atom_names=atom_types).images
        for id, model in enumerate(model_lists):
            force_i[id] = []
            ei_i[id] = []
            Etot_i[id] = []
            if model.model_type == "DP":
                _etot_list, _ei_list, _force_list, _virial_list = model.inference(image_read, do_deviation=True)
            elif model.model_type == "NEP":
                _etot_list, _ei_list, _force_list, _virial_list, _, _ = model.inference_nep_txt(image_read, do_deviation=True)
            
            for idj in range(0, len(_etot_list)):
                force_i[id].append(_force_list[idj])
                ei_i[id].append(_ei_list[idj])
                Etot_i[id].append(_etot_list[idj])
        
        for idj in range(0, len(Etot_i[0])):
            # calculate model deviation
            ei = np.squeeze(np.array([ei_i[_][idj] for _ in range(0, len(model_lists))]))
            force = np.squeeze(np.array([force_i[_][idj] for _ in range(0, len(model_lists))]))
            etot = np.squeeze(np.array([Etot_i[_][idj] for _ in range(0, len(model_lists))]))

            avg_force = np.mean(force, axis=0)
            _f_error = np.transpose((force - avg_force[np.newaxis, :, :])**2, (1, 0, 2))
            sqrt_f_error = np.sqrt(np.mean(np.sum(_f_error, axis=-1), axis=-1))
            res_devi_foce = np.max(sqrt_f_error)
            res_devi_min_force = np.min(sqrt_f_error)
            res_devi_mean_force= np.mean(sqrt_f_error)

            avg_ei = np.mean(ei, axis=0)
            sqrt_ei_error = np.sqrt(np.mean((ei - avg_ei[np.newaxis, :])**2, axis=1))
            res_devi_ei = np.max(sqrt_ei_error)
            res_devi_min_ei  = np.min(sqrt_ei_error)
            res_devi_mean_ei = np.max(sqrt_ei_error)

            with open(save_path, 'a') as wf:
                line = "    {:<17.6f}{:<17.6f}{:<17.6f}{:<17.6f}{:<17.6f}{:<17.6f}\n".format(
                    res_devi_mean_force, res_devi_min_force, res_devi_foce,
                    res_devi_mean_ei, res_devi_min_ei, res_devi_ei
                )
                wf.write(line)

