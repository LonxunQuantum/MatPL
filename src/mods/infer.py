import torch
from torch.autograd import Variable
import numpy as np
import os

from src.PWMLFF.dp_network import dp_network
from src.user.input_param import InputParam
from pwdata import Save_Data
from src.pre_data.dp_data_loader import type_map
# , find_neighbore
from src.pre_data.dpuni_data_loader import find_neighbore
from pwdata import Config
from src.utils.nep_to_gpumd import extract_model, get_atomic_number_from_name
from src.user.convert_model import get_model_type, is_nep_txt, is_nep_ckpt

class Inference(object):
    def __init__(self, 
                 ckpt_file: str, 
                 device: torch.device = None,
                 nep_txt:bool = False) -> None:
        self.ckpt_file = ckpt_file
        self.device = device
        self.model_atom_type = None
        self.model_type = get_model_type(ckpt_file)
        
        if self.model_type == "DP":
            self.model, self.input_param = self.load_dp_model(ckpt_file)
            self.model_atom_type = self.model.atom_type
        elif self.model_type == "NEP":
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
                from src.feature.NEP_GPU.build.nep_gpu import NEP
                self.calc = NEP()
                self.calc.init_from_file(self.ckpt_file, 1, 0)
            
            if "tmp_matpl_nep" in self.ckpt_file:
                os.remove(self.ckpt_file)

    '''
    description: 
        load dp_model.ckpt
        the nep_model.ckpt does not use this function
    param {*} self
    param {str} ckpt_file
    return {*}
    author: wuxingxing
    '''
    def load_dp_model(self, ckpt_file: str):
        model_checkpoint = torch.load(ckpt_file, map_location = torch.device("cpu"), weights_only=False)
        json_dict_train = model_checkpoint["json_file"]
        json_dict_train["model_load_file"] = ckpt_file
        json_dict_train["datasets_path"] = []
        json_dict_train["train_data"] = []
        json_dict_train["valid_data"] = []
        json_dict_train["test_data"] = []

        dp_param = InputParam(json_dict_train, "test".upper())
        
        dp_param.set_test_relative_params(json_dict_train)
        dp_trainer = dp_network(dp_param)
        davg, dstd, atom_map, energy_shift = dp_trainer.load_davg_from_ckpt()
        model, optimizer, _ = dp_trainer.load_model_optimizer(davg, dstd, energy_shift)
        # model = dp_trainer.load_model_with_ckpt(davg=stat[0], dstd=stat[1], energy_shift=stat[2])
        # model.load_state_dict(model_checkpoint["state_dict"])
        # if "compress" in model_checkpoint.keys():
        #     model.set_comp_tab(model_checkpoint["compress"])

        model.to(self.device)
        model.eval()
        return model, dp_param

    def inference(self, image_read, do_deviation=False): # for dp infer 
        model_config = self.model.config
        Ei = np.zeros(1)
        Egroup = 0
        nghost = 0
        # list_neigh, type_maps, atom_types, ImageDR = self.processed_data(structrue_file, model_config, Ei, Egroup, format, atom_names)
        # infer = Save_Data(data_path=structrue_file, format=format, atom_names=atom_names, train_ratio=1)
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
            if not hasattr(image, 'atom_type_map'):
                image.atom_type_map = np.array([self.model_atom_type.index(_) for _ in image.atom_types_image])

            image2 = image
            # if image.cartesian is True:
            #     image._set_fractional()
            if image.cartesian is False: # DP 统一修改为笛卡尔坐标集
                image._set_cartesian()
            # atom_nums = image.atom_nums
            if ntypes > img_max_types:
                raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
            type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)


            atom_types_struc = image.atom_types_image
            atom_types = image.atom_type
            ntypes = len(atom_types)
            position = np.array(image.position).reshape(1, -1, 3)
            natoms = position.shape[1]
            lattice = np.array(image.lattice).reshape(1, 3, 3)
            input_atom_types = np.array(self.model.atom_type)
            img_max_types = self.model.ntypes
            if ntypes > img_max_types:
                raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
            m_neigh = self.model.maxNeighborNum
            Rc_M = self.model.Rmax
            Rc_type = np.array([(_['Rc']) for _ in model_config["atomType"]])
            Rm_type = np.array([(_['Rm']) for _ in model_config["atomType"]])
            type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
            list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = \
                find_neighbore(image.atom_type_map, 
                            np.array(image.lattice).transpose(1, 0).reshape(-1),
                            np.array(image.position).transpose(1, 0).reshape(-1),
                            Rc_type, 
                            Rm_type, 
                            Rc_M,
                            m_neigh
                )
                # find_neighbore(image.atom_type_map, 
                #                 image.position, 
                #                 image.lattice, 
                #                 image.position.shape[0], 
                #                 image.atomic_energy, 
                #                 img_max_types, 
                #                 Rc_type, 
                #                 Rm_type, 
                #                 m_neigh, 
                #                 Rc_M, 
                #                 Egroup,
                #                 image2
                # )

            list_neigh = Variable(torch.tensor(np.expand_dims(list_neigh, axis=0)).int().to(self.device))
            type_maps = Variable(torch.tensor(type_maps, dtype=torch.int32).to(self.device))
            atom_types = Variable(torch.tensor(np.array(atom_types), dtype=torch.int32).to(self.device))
            ImageDR = Variable(torch.tensor(np.expand_dims(dR_neigh, axis=0)).to(self.device))

            Etot, Ei, Force, Egroup, Virial = self.model(list_neigh, type_maps[0], atom_types, ImageDR, 0, None, None)
            Etot = Etot.squeeze(0).cpu().detach().numpy()
            Ei = Ei.squeeze(0).cpu().detach().numpy()
            Force = Force.squeeze(0).cpu().detach().numpy()
            Virial = Virial.squeeze(0).cpu().detach().numpy()

            etot_list.append(Etot)
            ei_list.append(Ei)
            force_list.append(Force)
            virial_list.append(Virial)
            
            if not do_deviation:
                with np.printoptions(threshold=np.inf):
                    print("----------image   {}  -------".format(idx))
                    print("----------Total Energy-------\n", Etot)
                    print("----------Atomic Energy------\n", Ei)
                    print("----------Force--------------\n", Force)
                    print("----------Virial-------------\n", Virial)
                    print("\n")
            return etot_list, ei_list, force_list, virial_list

    def ase_dp_infer(self, lattice, cart_postions, symbols): # for dp infer 
        Egroup = 0
        model_config = self.model.config
        m_neigh = self.model.maxNeighborNum
        Rc_M = self.model.Rmax
        Rc_type = np.array([(_['Rc']) for _ in model_config["atomType"]])
        Rm_type = np.array([(_['Rm']) for _ in model_config["atomType"]])
        input_atom_types = np.array(self.model.atom_type)
        atom_type_list = get_atomic_number_from_name(symbols) # the atom type lists of per atom in config
        atom_type = np.unique(atom_type_list) # the atom type of config
        type_maps = np.array(type_map(atom_type_list, input_atom_types)).reshape(1, -1)

        list_neigh, dR_neigh, max_ri, Egroup_weight, Divider, Egroup = \
            find_neighbore(
                type_maps[0], 
                np.array(lattice).transpose(1, 0).reshape(-1),
                np.array(cart_postions).transpose(1, 0).reshape(-1),
                Rc_type, 
                Rm_type, 
                Rc_M, 
                m_neigh 
                )
                  
            # find_neighbore(type_maps, 
            #                 frac_postions, 
            #                 lattice, 
            #                 frac_postions.shape[0], 
            #                 None, 
            #                 self.model.ntypes, 
            #                 Rc_type, 
            #                 Rm_type, 
            #                 m_neigh, 
            #                 Rc_M, 
            #                 Egroup
            # )

        list_neigh = Variable(torch.tensor(np.expand_dims(list_neigh, axis=0)).int().to(self.device))
        type_maps = Variable(torch.tensor(type_maps, dtype=torch.int32).to(self.device))
        atom_types = Variable(torch.tensor(np.array(atom_type), dtype=torch.int32).to(self.device))
        ImageDR = Variable(torch.tensor(np.expand_dims(dR_neigh, axis=0)).to(self.device))

        Etot, Ei, Force, Egroup, Virial = self.model(list_neigh, type_maps[0], atom_types, ImageDR, 0, None, None)
        Etot = Etot.squeeze(0).cpu().detach().numpy()
        Ei = Ei.squeeze(0).cpu().detach().numpy()
        Force = Force.squeeze(0).cpu().detach().numpy()
        Virial = Virial.squeeze(0).cpu().detach().numpy()

        return Etot, Ei, Force, Virial

    '''
    description: 
    not used, this function is replaced by Config when doing infer work
    param {*} self
    param {*} structrue_file
    param {*} model_config
    param {*} Ei
    param {*} Egroup
    param {*} format
    param {*} atom_names
    return {*}
    author: wuxingxing
    '''                                                
    def processed_data(self, structrue_file, model_config, Ei, Egroup, format, atom_names=None):
        infer = Save_Data(data_path=structrue_file, format=format, atom_names=atom_names)
        struc_num = 1
        if infer.image_nums != struc_num:
            raise Exception("Error! the image num in structrue file is not 1!")
        atom_types_struc = infer.atom_types_image
        atom_types = infer.atom_type[0]
        ntypes = len(atom_types)
        position = infer.position.reshape(-1, 3)
        natoms = position.shape[1]
        lattice = infer.lattice.reshape(3, 3)
        input_atom_types = np.array(self.model.atom_type)
        img_max_types = self.model.ntypes
        if ntypes > img_max_types:
            raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
        m_neigh = self.model.maxNeighborNum
        Rc_M = self.model.Rmax
        Rc_type = np.array([(_['Rc']) for _ in model_config["atomType"]])
        Rm_type = np.array([(_['Rm']) for _ in model_config["atomType"]])
        type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
        list_neigh, dR_neigh, _, _, _, _ = \
            find_neighbore(
                type_maps, 
                np.array(infer.lattice).transpose(1, 0).reshape(-1),
                np.array(infer.position).transpose(1, 0).reshape(-1),
                Rc_type, 
                Rm_type,
                Rc_M, 
                m_neigh
            )
            # find_neighbore(type_maps, position, lattice, natoms, Ei, 
            #                                               img_max_types, Rc_type, Rm_type, m_neigh, Rc_M, Egroup)   


        list_neigh = self.to_tensor(list_neigh).unsqueeze(0)
        type_maps = self.to_tensor(type_maps).squeeze(0)
        atom_types = self.to_tensor(atom_types)
        ImageDR = self.to_tensor(dR_neigh).unsqueeze(0)
        return list_neigh, type_maps, atom_types, ImageDR
    
    def to_tensor(self, data):
        data = torch.from_numpy(data).to(self.device)
        return data

    def inference_nep_txt(self, image_read, do_deviation=False, return_charge_bec=False):
        # infer = Save_Data(data_path=structrue_file, format=format)
        # structrue_file, format="pwmat/config", atom_names=None, 
        # image_read = Config(data_path=structrue_file, format=format, atom_names=atom_names).images
        if not isinstance(image_read, list): # for lammps/dumps or movement .images will be list
            image_read = [image_read]
        input_atom_types = np.array(self.model_atom_type)
        img_max_types = len(self.model_atom_type)
        
        etot_list = []
        ei_list = []
        force_list = []
        virial_list = []
        charge_list = []
        bec_list = []

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

            calc_args = (
                    list(type_maps[0]), 
                    list(np.array(image.lattice).transpose(1, 0).reshape(-1)), 
                    np.array(image.position).transpose(1, 0).reshape(-1)
            )
            if return_charge_bec:
                ei_predict, force_predict, virial_predict, charge_predict, bec_predict = self.calc.inference_with_charge(*calc_args)
            else:
                ei_predict, force_predict, virial_predict = self.calc.inference(*calc_args)

            ei_predict   = np.array(ei_predict).reshape(atom_nums)
            force_predict = np.array(force_predict).reshape(3, atom_nums).transpose(1, 0)
            virial_predict = np.array(virial_predict)
            etot_predict = np.sum(ei_predict)
            if return_charge_bec:
                charge_predict = np.array(charge_predict).reshape(atom_nums)
                bec_predict = np.array(bec_predict).reshape(9, atom_nums).transpose(1, 0).reshape(atom_nums, 3, 3)

            etot_list.append(etot_predict)
            ei_list.append(ei_predict)
            force_list.append(force_predict)
            virial_list.append(virial_predict)
            if return_charge_bec:
                charge_list.append(charge_predict)
                bec_list.append(bec_predict)
            
            if not do_deviation:
                with np.printoptions(threshold=np.inf):
                    print("----------image   {}  -------".format(idx))
                    print("----------Total Energy-------\n", etot_predict)
                    print("----------Atomic Energy------\n", ei_predict)
                    print("----------Force--------------\n", force_predict)
                    print("----------Virial-------------\n", virial_predict)
                    if return_charge_bec:
                        print("----------Charge-------------\n", charge_predict)
                        print("----------BEC----------------\n", bec_predict)
                    print("\n")
                
        if return_charge_bec:
            return etot_list, ei_list, force_list, virial_list, charge_list, bec_list
        return etot_list, ei_list, force_list, virial_list

    def ase_nep_infer(self, lattice, cart_postions, symbols, return_charge_bec=False):
        # infer = Save_Data(data_path=structrue_file, format=format)
        input_atom_types = np.array(self.model_atom_type)
        atom_nums = cart_postions.shape[0]
        atom_type_list = get_atomic_number_from_name(symbols) # the atom type lists of per atom in config
        type_maps = np.array(type_map(atom_type_list, input_atom_types)).reshape(1, -1)
        calc_args = (
                    list(type_maps[0]), 
                    list(np.array(lattice).transpose(1, 0).reshape(-1)), 
                    np.array(cart_postions).transpose(1, 0).reshape(-1)
            )
        if return_charge_bec:
            ei_predict, force_predict, virial_predict, charge_predict, bec_predict = self.calc.inference_with_charge(*calc_args)
        else:
            ei_predict, force_predict, virial_predict = self.calc.inference(*calc_args)

        ei_predict   = np.array(ei_predict).reshape(atom_nums)
        force_predict = np.array(force_predict).reshape(3, atom_nums).transpose(1, 0)
        virial_predict = np.array(virial_predict)
        etot_predict = np.sum(ei_predict)

        if return_charge_bec:
            charge_predict = np.array(charge_predict).reshape(atom_nums)
            bec_predict = np.array(bec_predict).reshape(9, atom_nums).transpose(1, 0).reshape(atom_nums, 3, 3)
            return etot_predict, ei_predict, force_predict, virial_predict, charge_predict, bec_predict
        return etot_predict, ei_predict, force_predict, virial_predict

    # def inference_nep(self, structrue_file, format="pwmat/config", atom_names=None):
    #     Ei = np.zeros(1)
    #     Egroup = 0
    #     nghost = 0
    #     from src.feature.nep_find_neigh.findneigh import FindNeigh

    #     calc = FindNeigh()

    #     # infer = Save_Data(data_path=structrue_file, format=format)
    #     image_read = Config(data_path=structrue_file, format=format, atom_names=atom_names).images
    #     if isinstance(image_read, list): # for lammps/dumps or movement .images will be list
    #         image = image_read[0]
    #     else:
    #         image = image_read
    #     struc_num = 1
    #     atom_types_struc = image.atom_types_image
    #     atom_types = image.atom_type
    #     ntypes = len(atom_types)
    #     # print("=========position==========\n")
    #     # print(image.position)
    #     cart_postion = image.position
    #     if image.cartesian is True:
    #         image._set_fractional()
    #     atom_nums = image.atom_nums
    #     input_atom_types = np.array(self.model.atom_type)
    #     img_max_types = self.model.ntypes
    #     if ntypes > img_max_types:
    #         raise Exception("Error! the atom types in structrue file is larger than the max atom types in model!")
    #     type_maps = np.array(type_map(atom_types_struc, input_atom_types)).reshape(1, -1)
    #     # type_maps list 1 dim
    #     # Lattice [10.104840279, 0.0, -1.7274452448, 0.0, 10.28069973, 0.0, -5.1064545896e-16, 0.0, 10.275204659] 做转置后拉成一列
    #     # Position [96,3] 转置后拉成一列
    #     # 34622.19498329725 d12_radial
    #     d12_radial, d12_agular, NL_radial, NL_angular, NLT_radial, NLT_angular = calc.getNeigh(
    #                        self.input_param.descriptor.cutoff[0],self.input_param.descriptor.cutoff[1], 
    #                         len(self.input_param.atom_type)*self.input_param.max_neigh_num, list(type_maps[0]), list(np.array(image.lattice).transpose(1, 0).reshape(-1)), np.array(image.position).transpose(1, 0).reshape(-1))

    #     neigh_radial_rij   = np.array(d12_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
    #     neigh_angular_rij  = np.array(d12_agular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num, 4)
    #     neigh_radial_list  = np.array(NL_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
    #     neigh_angular_list = np.array(NL_angular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
    #     neigh_radial_type_list  =  np.array(NLT_radial).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)
    #     neigh_angular_type_list = np.array(NLT_angular).reshape(atom_nums, len(self.input_param.atom_type)*self.input_param.max_neigh_num)

    #     neigh_radial_rij = self.to_tensor(neigh_radial_rij).unsqueeze(0)
    #     neigh_angular_rij = self.to_tensor(neigh_angular_rij).unsqueeze(0)

    #     neigh_radial_list = self.to_tensor(neigh_radial_list).unsqueeze(0)
    #     neigh_angular_list = self.to_tensor(neigh_angular_list).unsqueeze(0)

    #     neigh_radial_type_list = self.to_tensor(neigh_radial_type_list).unsqueeze(0)
    #     neigh_angular_type_list = self.to_tensor(neigh_angular_type_list).unsqueeze(0)
        
    #     type_maps = self.to_tensor(type_maps).squeeze(0)
    #     atom_types = self.to_tensor(np.array(atom_types))

    #     # Etot, Ei, Force, Egroup, Virial = self.model(neigh_radial_list, type_maps, atom_types, neigh_radial_rij, neigh_radial_type_list, nghost)

    #     Etot, Ei, Force, Egroup, Virial = self.model(
    #         neigh_radial_list, 
    #         neigh_radial_rij,
    #         neigh_radial_type_list,
    #         neigh_angular_list,
    #         neigh_angular_rij,
    #         neigh_angular_type_list,
    #         type_maps, 
    #         atom_types, 
    #         0
    #         )

    #     ### debug start
    #     # dR_neigh_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/dR_neigh.txt"
    #     # dR_neigh = np.loadtxt(dR_neigh_txt).reshape(atom_nums, len(self.input_param.atom_type) * self.input_param.max_neigh_num, 4)
    #     # imagetype_map_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/imagetype_map.txt"
    #     # imagetype_map = np.loadtxt(imagetype_map_txt, dtype=int).reshape(atom_nums)
    #     # neighbor_list_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/neighbor_list.txt"
    #     # neighbor_list = np.loadtxt(neighbor_list_txt, dtype=int).reshape(atom_nums, len(self.input_param.atom_type) * self.input_param.max_neigh_num)
    #     # neighbor_type_list_txt = "/data/home/wuxingxing/datas/lammps_test/nep_hfo2_lmps/lmp_dug/neighbor_type_list.txt"
    #     # neighbor_type_list = np.loadtxt(neighbor_type_list_txt, dtype=int).reshape(atom_nums, len(self.input_param.atom_type) * self.input_param.max_neigh_num)

    #     # neigh_radial_rij2 = self.to_tensor(dR_neigh).unsqueeze(0)
    #     # neigh_radial_list2 = self.to_tensor(neighbor_list).unsqueeze(0)
    #     # neigh_radial_type_list2 = self.to_tensor(neighbor_type_list).unsqueeze(0)
    #     # type_maps2 = self.to_tensor(imagetype_map).squeeze(0)
    #     # # atom_types = self.to_tensor(np.array(atom_types))

    #     # Etot2, Ei2, Force2, Egroup2, Virial2 = self.model(neigh_radial_list2, type_maps2, atom_types, neigh_radial_rij2, neigh_radial_type_list2, 1214)
        
    #     ### debug end
    #     Etot = Etot.cpu().detach().numpy()
    #     Ei = Ei.cpu().detach().numpy()
    #     Force = Force.cpu().detach().numpy()
    #     Virial = Virial.cpu().detach().numpy()
    #     try:
    #         Egroup = Egroup.cpu().detach().numpy()
    #     except:
    #         Egroup = None
    #     print("----------Total Energy-------\n", Etot)
    #     print("----------Atomic Energy------\n", Ei)
    #     print("----------Force--------------\n", Force)
    #     print("----------Virial-------------\n", Virial)
    #     return Etot, Ei, Force, Egroup, Virial    

