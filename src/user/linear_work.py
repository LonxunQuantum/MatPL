import os
import json
from src.user.input_param import InputParam
from src.PWMLFF.linear_regressor import linear_regressor
from utils.file_operation import delete_tree, copy_tree, copy_file
from src.pre_data.find_maxneighbor import get_max_neighbor

'''
description: do linear training
    step1. generate feature from MOVEMENTs
    step2. load features and do training
    step3. extract forcefield files
    step4. copy features, trained model files to the same level directory of jsonfile
param {json} input_json
return {*}
author: wuxingxing
'''
def linear_train(input_json: json, cmd:str):
    linear_param = InputParam(input_json, cmd) 
    max_neighbor, _, _, _ , dataset = get_max_neighbor(
            data_paths=linear_param.file_paths.train_data_path,
            format="pwmat/movement",
            atom_types=linear_param.atom_type,
            cutoff_radial=linear_param.descriptor.Rmax * 1.25,
            with_type=False
    )
    linear_param.max_neigh_num = max(max_neighbor, linear_param.max_neigh_num)

    linear_param.print_input_params("std_input.json")
    linear_trainer = linear_regressor(linear_param)
    if len(linear_param.file_paths.train_data_path) > 0:
        feature_path = linear_trainer.generate_data(
            movement_path = linear_param.file_paths.train_data_path,
            feature_type="train_feature")
        linear_param.file_paths.set_train_feature_path([feature_path])

    if len(linear_param.file_paths.valid_data_path) > 0:
        print("Warning! Linear model does not support valid test during training!")

    # linear_param.file_paths.set_train_feature_path([feature_path])
    work_path = os.path.join(linear_param.file_paths.nn_work, "train_feature")
    linear_trainer.train(work_path)
    linear_trainer.extract_force_field(linear_param.file_paths.forcefield_name)
    # copy fread_dfeat, input, output dir to model_record
    source_fread_dir = os.path.join(work_path, "fread_dfeat")
    target_fread_dir = os.path.join(linear_param.file_paths.json_dir, 
                                    os.path.basename(linear_param.file_paths.forcefield_dir), "fread_dfeat")
    copy_tree(source_fread_dir, target_fread_dir)
    copy_tree(os.path.join(os.path.dirname(source_fread_dir), "input"),
              os.path.join(os.path.dirname(target_fread_dir), "input"))
    copy_tree(os.path.join(os.path.dirname(source_fread_dir), "output"),
              os.path.join(os.path.dirname(target_fread_dir), "output"))
    # copy forcefield files to forcefield dir
    copy_file(os.path.join(os.path.dirname(source_fread_dir), linear_param.file_paths.forcefield_name),
              os.path.join(os.path.dirname(target_fread_dir), linear_param.file_paths.forcefield_name))

    if linear_param.recover_train is False:
        if os.path.realpath(linear_param.file_paths.json_dir) != os.path.realpath(linear_param.file_paths.nn_work) :
            if linear_param.file_paths.reserve_feature is False:
                delete_tree(work_path)
            if linear_param.file_paths.reserve_work_dir is False:
                delete_tree(linear_param.file_paths.nn_work)
'''
description: 
    do linear inference:
    step1. generate feature, the movement from json file 'test_movement_path'
    step2. load model and do inference
    step3. copy inference result files to the same level directory of jsonfile
param {json} input_json
param {str} cmd
return {*}
author: wuxingxing
'''
def linear_test(input_json: json, cmd:str):
    linear_param = InputParam(input_json, cmd)
    linear_param.set_test_relative_params(input_json)
    linear_param.print_input_params("std_input.json")
    linear_trainer = linear_regressor(linear_param)
    work_path = os.path.join(linear_param.file_paths.nn_work, "test_feature")
    if len(linear_param.file_paths.test_data_path) > 0:
        linear_trainer.evaluate_prepare_data(linear_param.file_paths.test_data_path, work_path)
    linear_trainer.evaluate(work_path, num_thread = 1, plot_elem = True, save_data = True)

    # copy plot dir 
    source_dir = os.path.join(work_path, "plot_data")
    target_dir = os.path.join(linear_param.file_paths.json_dir, os.path.basename(linear_param.file_paths.test_dir))
    copy_tree(source_dir, target_dir)

    if linear_param.recover_train is False:
        if os.path.realpath(linear_param.file_paths.json_dir) != os.path.realpath(linear_param.file_paths.work_dir) :
            if linear_param.file_paths.reserve_feature is False:
                delete_tree(work_path)
            if linear_param.file_paths.reserve_work_dir is False:
                delete_tree(linear_param.file_paths.work_dir)
            