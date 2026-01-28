import torch
import numpy as np
import os
import glob
from src.PWMLFF.dp_network import dp_network
from src.user.input_param import InputParam
from pwdata import Save_Data
from src.pre_data.dp_data_loader import type_map, find_neighbore
from src.optimizer.KFWrapper import KFOptimizerWrapper

class KPU_CALCULATE(object):
    def __init__(self, 
                 ckpt_file: str) -> None:
        self.ckpt_file = ckpt_file
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.optimizer = self.load_model(ckpt_file)
        self.KFOptWrapper = KFOptimizerWrapper(self.model, self.optimizer, 24, 6)

    def load_model(self, ckpt_file: str):
        model_checkpoint = torch.load(ckpt_file, map_location = torch.device("cpu"), weights_only=False)
        stat = [model_checkpoint["davg"], model_checkpoint["dstd"], model_checkpoint["energy_shift"]]
        model_checkpoint["json_file"]["model_load_file"] = ckpt_file
        model_checkpoint["json_file"]["datasets_path"] = []
        dp_param = InputParam(model_checkpoint["json_file"], "train".upper())
        dp_param.inference = True 
        dp_trainer = dp_network(dp_param)
        model, optimizier = dp_trainer.load_model_optimizer(davg=stat[0], dstd=stat[1], energy_shift=stat[2])
        
        # model.load_state_dict(model_checkpoint["state_dict"])
        if "compress" in model_checkpoint.keys():
            model.set_comp_tab(model_checkpoint["compress"])
        
        model.to(self.device)
        model.eval()
        return model, optimizier

    def kpu_dp(self, structure_dir:str, format:str, atom_names:list[str], savepath:str, \
        is_etot_kpu:bool=True, is_force_kpu:bool=True, force_kpu_detail:bool=False):
        traj_list = glob.glob(os.path.join(structure_dir, "*"))
        trajs = sorted(traj_list, key = lambda x : int(os.path.basename(x).split(".")[0]))
        model_config = self.model.config
        # read atom names from atom type file
        if os.path.isfile(atom_names[0]):
            with open(atom_names[0], 'r') as rf:
                line = rf.readline()
            atom_names = line.split()
        Egroup = 0
        nghost = 0
        force_kpu_list = []
        model_devi_list = []
        for ti, traj in enumerate(trajs):
            traj_index = int(os.path.basename(traj).split(".")[0])
            Ei = np.zeros(1)
            list_neigh, type_maps, atom_types, ImageDR = self.processed_data(traj, model_config, Ei, Egroup, format, atom_names)
            # Etot, Ei, Force, Egroup, Virial = self.model(list_neigh, type_maps, atom_types, ImageDR, nghost)
            if is_etot_kpu:
                etot_kpu, Etot = self.KFOptWrapper.cal_kpu_etot(list_neigh, type_maps, atom_types, ImageDR, nghost)
            if is_force_kpu:
                force_kpu, Force = self.KFOptWrapper.cal_kpu_force(list_neigh, type_maps, atom_types, ImageDR, nghost)

            model_devi_list.append([traj_index, float(etot_kpu), np.mean(force_kpu), np.max(force_kpu), np.min(force_kpu)])

            if force_kpu_detail:
                force_kpu_list.append(force_kpu)

        if force_kpu_detail and len(force_kpu_list) > 0:
            np.save("force_kpu_detail.npy", np.array(force_kpu_list))
        
        if len(model_devi_list) > 0: # maybe the traj is none, this could be error, should be fixed at active learning code
            header = "    step        etot_kpu      f_kpu_mean       f_kpu_max       f_kpu_min"
            np.savetxt(savepath, np.array(model_devi_list), fmt=["%10d", "%15.2f", "%15.2f", "%15.2f", "%15.2f"], delimiter=' ', header=header)

    def processed_data(self, structrue_file, model_config, Ei, Egroup, format, atom_names=None):
        infer = Save_Data(data_path=structrue_file, format=format, atom_names=atom_names)
        struc_num = 1
        if infer.image_nums != struc_num:
            raise Exception("Error! the image num in structrue file is not 1!")
        atom_types_struc = infer.atom_types_image
        atom_types = infer.atom_type[0]
        ntypes = len(atom_types)
        position = infer.position.reshape(struc_num, -1, 3)
        natoms = position.shape[1]
        lattice = infer.lattice.reshape(struc_num, 3, 3)
        input_atom_types = np.array(self.model.atom_type)
        img_max_types = self.model.ntypes
        if ntypes > img_max_types:
            raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
        m_neigh = self.model.maxNeighborNum
        Rc_M = self.model.Rmax
        Rc_type = np.array([(_['Rc']) for _ in model_config["atomType"]])
        Rm_type = np.array([(_['Rm']) for _ in model_config["atomType"]])
        type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
        list_neigh, dR_neigh, _, _, _, _ = find_neighbore(type_maps, position, lattice, natoms, Ei, 
                                                          img_max_types, Rc_type, Rm_type, m_neigh, Rc_M, Egroup)   
        
        list_neigh = self.to_tensor(list_neigh).unsqueeze(0)
        type_maps = self.to_tensor(type_maps).squeeze(0)
        atom_types = self.to_tensor(atom_types)
        ImageDR = self.to_tensor(dR_neigh).unsqueeze(0)
        return list_neigh, type_maps, atom_types, ImageDR

    def to_tensor(self, data):
        data = torch.from_numpy(data).to(self.device)
        return data