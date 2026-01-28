import torch
from torch.autograd import Variable
import numpy as np
import os

from src.user.input_param import InputParam
from pwdata import Save_Data
from pwdata import Config
from utils.nep_to_gpumd import extract_model, get_atomic_number_from_name
from src.user.convert_model import get_model_type, is_nep_txt, is_nep_ckpt
from utils.atom_type_emb_dict import type_map
class Inference(object):
    def __init__(self, 
                 ckpt_file: str, 
                 device: torch.device = None,
                 nep_txt:bool = False) -> None:
        self.ckpt_file = ckpt_file
        self.device = device
        self.model_atom_type = None
        self.model_type = get_model_type(ckpt_file)
        
        if self.model_type == "NEP":
            if is_nep_ckpt(ckpt_file):
                nep_content, self.model_atom_type, atom_names = extract_model(ckpt_file)
                self.model_atom_type = get_atomic_number_from_name(atom_names)
                self.ckpt_file = os.path.join(os.path.dirname(os.path.abspath(ckpt_file)), "tmp_matpl_nep.txt")
                with open(self.ckpt_file, 'w') as wf:
                    wf.write(nep_content)
            else:
                self.ckpt_file = ckpt_file
                with open(ckpt_file, 'r') as rf:
                    line = rf.readline()
                    atom_names = line.split()[2:]
                    self.model_atom_type = get_atomic_number_from_name(atom_names)
            
            if self.device.type == "cpu":
                from src.feature.nep_find_neigh.findneigh import FindNeigh
                self.calc = FindNeigh()
                self.calc.init_model(self.ckpt_file)
            else:
                from src.feature.NEP_GPU.build.nep_gpu import NEP3
                self.calc = NEP3()
                self.calc.init_from_file(self.ckpt_file, 1, 0)
            
            if "tmp_matpl_nep" in self.ckpt_file:
                os.remove(self.ckpt_file)
    
    def inference_nep_txt(self, structrue_file, format="pwmat/config", atom_names=None, do_deviation=False):
        # infer = Save_Data(data_path=structrue_file, format=format)
        image_read = Config(data_path=structrue_file, format=format, atom_names=atom_names).images
        if not isinstance(image_read, list): # for lammps/dumps or movement .images will be list
            image_read = [image_read]
        input_atom_types = np.array(self.model_atom_type)
        img_max_types = len(self.model_atom_type)
        
        etot_list = []
        ei_list = []
        force_list = []
        virial_list = []

        for idx, image in enumerate(image_read):
            atom_types_struc = image.atom_types_image
            atom_types = image.atom_type
            ntypes = len(atom_types)
            # print("=========position==========\n")
            # print(image.position)
            # cart_postion = image.position
            # if image.cartesian is True:
            #     image._set_fractional()
            if image.cartesian is False:
                image._set_cartesian()
            atom_nums = image.atom_nums

            if ntypes > img_max_types:
                raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
            type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)

            ei_predict, force_predict, virial_predict = self.calc.inference(
                    list(type_maps[0]), 
                    list(np.array(image.lattice).transpose(1, 0).reshape(-1)), 
                    np.array(image.position).transpose(1, 0).reshape(-1)
            )

            ei_predict   = np.array(ei_predict).reshape(atom_nums)
            force_predict = np.array(force_predict).reshape(3, atom_nums).transpose(1, 0)
            virial_predict = np.array(virial_predict)
            etot_predict = np.sum(ei_predict)

            etot_list.append(etot_predict)
            ei_list.append(ei_predict)
            force_list.append(force_predict)
            virial_list.append(virial_predict)
            
            if not do_deviation:
                with np.printoptions(threshold=np.inf):
                    print("----------image   {}  -------".format(idx))
                    print("----------Total Energy-------\n", etot_predict)
                    print("----------Atomic Energy------\n", ei_predict)
                    print("----------Force--------------\n", force_predict)
                    print("----------Virial-------------\n", virial_predict)
                    print("\n")
                
        return etot_list, ei_list, force_list, virial_list

    def ase_nep_infer(self, lattice, cart_postions, symbols):
        # infer = Save_Data(data_path=structrue_file, format=format)
        input_atom_types = np.array(self.model_atom_type)
        atom_nums = cart_postions.shape[0]
        atom_type_list = get_atomic_number_from_name(symbols) # the atom type lists of per atom in config
        type_maps = np.array(type_map(atom_type_list, input_atom_types)).reshape(1, -1)
        ei_predict, force_predict, virial_predict = self.calc.inference(
                    list(type_maps[0]), 
                    list(np.array(lattice).transpose(1, 0).reshape(-1)), 
                    np.array(cart_postions).transpose(1, 0).reshape(-1)
            )

        ei_predict   = np.array(ei_predict).reshape(atom_nums)
        force_predict = np.array(force_predict).reshape(3, atom_nums).transpose(1, 0)
        virial_predict = np.array(virial_predict)
        etot_predict = np.sum(ei_predict)

        return etot_predict, ei_predict, force_predict, virial_predict
