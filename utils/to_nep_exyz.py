import os
import random
import numpy as np
import math

from pwdata.movement import MOVEMENT
# from pwdata.extendedxyz import save_to_extxyz
from pwdata.main import Config

from pwdata.calculators.const import elements

def random_index(image_nums:int, ratio:float, is_random:bool=False, seed:int=None):
    arr = np.arange(image_nums)
    if seed:
        np.random.seed(seed)
    if is_random is True:
        np.random.shuffle(arr)
    split_idx = math.ceil(image_nums*ratio)
    train_data = arr[:split_idx]
    test_data = arr[split_idx:]
    return sorted(train_data), sorted(test_data)

def convert_to_xyz(input_list:list, 
                    input_format:str, 
                    atom_types:str,
                    save_dir:str,
                    train_save_name:str, 
                    valid_save_name:str, 
                    valid_shuffle:bool=False, 
                    ratio:float=0.2, 
                    is_valid:bool=False,
                    seed:int=None
                    # trainDataPath:str="train", 
                    # validDataPath:str="valid"
                    ):
    # if the save_file exists before, delete it
    if os.path.exists(os.path.join(save_dir, train_save_name)):
        os.remove(os.path.join(save_dir, train_save_name))
    if os.path.exists(os.path.join(save_dir, valid_save_name)):
        os.remove(os.path.join(save_dir, valid_save_name))
    
    for pwdata_dir in input_list:
        # if input_format != "pwmlff/npy":# for raw datas
        image_data = Config(data_path=pwdata_dir, format=input_format, atom_names=atom_types)
        image_list = image_data.images
        image_nums = len(image_list)

        if image_nums == 0:
            raise Exception("ERROR! The input dir is empty file, please check {}".format(pwdata_dir))
        if is_valid:
            train_indexs = list(range(0, image_nums))
            valid_indexs = []
        else:
            train_indexs, valid_indexs = random_index(image_nums, ratio, valid_shuffle, seed)
        train_images = []
        valid_images = []
        for index in train_indexs:
            train_images.append(image_list[index])
        for index in valid_indexs:
            valid_images.append(image_list[index])
    
        assert(len(train_images) + len(valid_images)) == len(image_list)
        
        if is_valid:# the inference input file of nep is train.xyz and prediction = 0
            save_to_extxyz(train_images, save_dir, train_save_name, write_patthen='a')
        else:
            save_to_extxyz(train_images, save_dir, train_save_name, write_patthen='a')
            save_to_extxyz(valid_images, save_dir, valid_save_name, write_patthen='a')

def save_to_extxyz(image_data_all: list, output_path: str, data_name: str, write_patthen='w'):
    data_name = open(os.path.join(output_path, data_name), write_patthen)
    for i in range(len(image_data_all)):
        image_data = image_data_all[i]
        if not image_data.cartesian:
            image_data._set_cartesian()
        data_name.write("%d\n" % image_data.atom_nums)
        # data_name.write("Iteration: %s\n" % image_data.iteration)
        output_head = 'Lattice="%f %f %f %f %f %f %f %f %f" Properties=species:S:1:pos:R:3:force:R:3 pbc="T T T" energy={}\n'.format(image_data.Ep)
        output_extended = (image_data.lattice[0][0], image_data.lattice[0][1], image_data.lattice[0][2], 
                                image_data.lattice[1][0], image_data.lattice[1][1], image_data.lattice[1][2], 
                                image_data.lattice[2][0], image_data.lattice[2][1], image_data.lattice[2][2])
        data_name.write(output_head % output_extended)
        for j in range(image_data.atom_nums):
            properties_format = "%s %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f\n"
            properties = (elements[image_data.atom_types_image[j]], image_data.position[j][0], image_data.position[j][1], image_data.position[j][2], 
                            image_data.force[j][0], image_data.force[j][1], image_data.force[j][2])
            data_name.write(properties_format % properties)
    data_name.close()
    print("Convert to %s successfully!" % data_name)

if __name__ == "__main__":
    datasets_path = ["/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/lmps_test/pwmlff_setatom/96atoms/traj/0.lammpstrj"]
    save_dir = "/data/home/wuxingxing/datas/pwmat_mlff_workdir/hfo2/lmps_test/pwmlff_setatom/96k/gpumd"
    input_format  = "lammps/dump" # 支持格式："pwmlff/npy","pwmat/movement","vasp/outcar",dpdata/npy","dpdata/raw"
    valid_shuffle = False             #分割训练集验证集时，是否随机分割
    train_valid_ratio = 1          #分割训练集、测试集比例
    seed = 2024                      #随机分割时的random seed
    if "lammps" in input_format:
        atom_types = ["Hf", "O"]
    else:
        atom_types = None
    convert_to_xyz( input_list      =datasets_path,
                    input_format    =input_format,
                    atom_types      =atom_types,
                    save_dir        =save_dir,
                    train_save_name ="train.xyz",
                    valid_save_name ="test.xyz",
                    valid_shuffle   =valid_shuffle,
                    ratio           =train_valid_ratio, 
                    is_valid        =True,
                    seed            =seed
    )
