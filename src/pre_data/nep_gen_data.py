from utils.random_utils import random_index
import os

from pwdata.movement import MOVEMENT
from pwdata.extendedxyz import save_to_extxyz
from pwdata.main import Config
'''
description: 
    convert movements to train.xyz and valid.xyz 
param {list} mvm_file_list
param {str} save_dir 
param {str} train_save_path "train.xyz"
param {str} valid_save_path "valid.xyz"
param {bool} valid_shuffle
param {float} ratio
param {int} seed
return {*}
author: wuxingxing
'''
def convert_mvmfiles_to_xyz(mvm_file_list:list, save_dir:str, train_save_path:str, valid_save_path:str, valid_shuffle:bool=False, ratio:float=0.2, seed:int=None):
    # if the save_file exists before, delete it
    if os.path.exists(os.path.join(save_dir, train_save_path)):
        os.remove(os.path.join(save_dir, train_save_path))
    if os.path.exists(os.path.join(save_dir, valid_save_path)):
        os.remove(os.path.join(save_dir, valid_save_path))
        
    for mvm in mvm_file_list:
        image_data = MOVEMENT(mvm)
        image_nums = len(image_data.get())
        if image_nums == 0:
            raise Exception("ERROR! The input movement file is empty file, please check {}".format(mvm))
        train_indexs, valid_indexs = random_index(image_nums, ratio, valid_shuffle, seed)
        train_images = []
        valid_images = []
        for index in train_indexs:
            train_images.append(image_data.image_list[index])
        for index in valid_indexs:
            valid_images.append(image_data.image_list[index])
        
        assert(len(train_images) + len(valid_images)) == len(image_data.get())
        
        save_to_extxyz(train_images, save_dir, train_save_path, write_patthen='a')
        save_to_extxyz(valid_images, save_dir, valid_save_path, write_patthen='a')

def convert_to_xyz(input_list:list, 
                    input_format:str, 
                    save_dir:str,
                    train_save_path:str, 
                    valid_save_path:str, 
                    is_valid:bool=False,
                    is_append:bool=False,
                    valid_shuffle:bool=False, 
                    ratio:float=0.2, 
                    seed:int=None
                    # trainDataPath:str="train", 
                    # validDataPath:str="valid"
                    ):
    # if the save_file exists before, delete it
    if not is_append:
        if os.path.exists(os.path.join(save_dir, train_save_path)):
            os.remove(os.path.join(save_dir, train_save_path))
        if os.path.exists(os.path.join(save_dir, valid_save_path)):
            os.remove(os.path.join(save_dir, valid_save_path))
    
    for pwdata_dir in input_list:
        # if input_format != "pwmlff/npy":# for raw datas
        image_data = Config(data_path=pwdata_dir, format=input_format)
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
            save_to_extxyz(train_images, save_dir, train_save_path, write_patthen='a')
        else:
            save_to_extxyz(train_images, save_dir, train_save_path, write_patthen='a')
            save_to_extxyz(valid_images, save_dir, valid_save_path, write_patthen='a')

