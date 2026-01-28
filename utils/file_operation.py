import os, subprocess
import shutil
import filecmp

import numpy as np
import torch

def check_model_type(model_load_path):
    try:
        _model_checkpoint = torch.load(model_load_path, map_location=torch.device("cpu"), weights_only=False)
        model_type = _model_checkpoint['json_file']['model_type']
    except Exception as e:
        with open(model_load_path, 'r') as rf:
            line = rf.readline()
        if "nep" not in line:
            raise Exception("ERROR! The input model file cannot be parsed!")
        else:
            model_type  = "NEP"
    return model_type

def write_line_to_file(file_path, line, write_patthen="w"):
    with open(file_path, write_patthen) as wf:
        wf.write(line)
        wf.write('\n')
    
def write_arrays_to_file(file_path, arrays, write_patthen="w", head_line=None):
    with open(file_path, write_patthen) as wf:
        if head_line is not None:
            wf.write(head_line)
            wf.write('\n')
        for data in arrays:
            if isinstance(data, np.ndarray):
                if data.ndim == 1:
                    line = ' '.join(data.astype('str'))
                    wf.write(line)
                    wf.write('\n')
                else:
                    for _data in data:
                        line = ' '.join(_data.astype('str'))
                        wf.write(line)
                        wf.write('\n')
            elif isinstance(data, list):
                line = ' '.join(np.array(data).astype('str'))
                wf.write(line)
                wf.write('\n')
            else:
                line = "{}".format(data)
                wf.write(line)
                wf.write('\n')

def write_force_ei(file_path, ei_list, f_list, write_patthen="w"):
    with open(file_path, write_patthen) as wf:
        for image_idx, eis in enumerate(ei_list):
            for ei_idx, ei in enumerate(eis):
                out_str = "{} {} {} {}".format(ei, f_list[image_idx][ei_idx*3], f_list[image_idx][ei_idx*3+1], f_list[image_idx][ei_idx*3+2])
                wf.write(out_str)
                wf.write('\n')
                
'''
description: copy file from souce to target dir:
    if target dir does not exist, create it
    if target_file is exist, replace it by source file
param {*} source_file_path
param {*} target_file_path
return {*}
author: wuxingxing
'''
def copy_file(source_file_path, target_file_path):
    if os.path.exists(os.path.dirname(target_file_path)) is False:
        os.makedirs(os.path.dirname(target_file_path))
    if os.path.exists(target_file_path):
        os.remove(target_file_path)
    shutil.copy(source_file_path, target_file_path)

'''
description: copy souce dir to target dir, if target dir exists, replace it.
param {*} souce_dir
param {*} target_dir
return {*}
author: wuxingxing
'''
def copy_tree(souce_dir, target_dir):
    if os.path.exists(target_dir):
        if os.path.islink(target_dir):
            os.remove(target_dir)
        else:
            shutil.rmtree(target_dir)
    shutil.copytree(souce_dir, target_dir)

def delete_tree(dir):
    if os.path.exists(dir):
        if os.path.islink(dir):
            os.remove(dir)
        else:
            shutil.rmtree(dir)

'''
description: 
    copy source_movement_list to target_dir, each movement file at target_dir has a sinlge dir. for example:
        souce_movement_paths=[".../MOVMENET1", ".../MOVEMENT2"], after copy, at the target_dir:
        target_dir/MOVEMENT1_0/MOVEMENT1, target_dir/MOVEMENT2_1/MOVEMENT2
param {*} source_movement_paths
param {*} target_dir
param {*} trainSetDir
param {*} movement_name
return data_work_dir
author: wuxingxing
'''
def copy_movements_to_work_dir(source_movement_paths, target_dir, trainSetDir, movement_name):
    if os.path.exists(target_dir) is True:
        shutil.rmtree(target_dir)
    data_work_dir = os.path.join(target_dir, trainSetDir)
    for i, mvm in enumerate(source_movement_paths):
        mvm_dir = os.path.join(data_work_dir, "{}_{}".format(os.path.basename(mvm), i))
        target_mvm = os.path.join(mvm_dir, movement_name)
        copy_file(mvm, target_mvm)
    return data_work_dir

'''
description: delete diractory
param {*} dir
return {*}
author: wuxingxing
'''
def delete_tree(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir)

'''
description: 
    reset the file's path in pm.py, the work_dir is root dir
    this is used in NN hrybrid training
param {*} pm
param {*} work_dir
return {*}
author: wuxingxing
'''
def reset_pm_params(pm, work_dir):
    pm.sourceFileList = []
    # pm.atomType = atom_type
    # pm.atomTypeNum = len(pm.atomType)       #this step is important. Unexpected error
    # pm.ntypes = len(pm.atomType)
    # reset dirs
    pm.train_data_path = os.path.join(work_dir, r'train_data/final_train')
    pm.test_data_path =os.path.join(work_dir,  r'train_data/final_test')
    pm.prefix = work_dir
    pm.trainSetDir = os.path.join(work_dir, r'PWdata')
    # fortranFitSourceDir=r'/home/liuliping/program/nnff/git_version/src/fit'
    pm.fitModelDir = os.path.join(work_dir, r'fread_dfeat')
    pm.dRneigh_path = pm.trainSetDir + r'dRneigh.dat'

    pm.trainSetDir=os.path.abspath(pm.trainSetDir)
    #genFeatDir=os.path.abspath(genFeatDir)
    pm.fbinListPath=os.path.join(pm.trainSetDir,'location')
    pm.InputPath=os.path.abspath(os.path.join(work_dir, 'input'))
    pm.OutputPath=os.path.abspath(os.path.join(work_dir, 'output'))
    pm.Ftype1InputPath=os.path.join('./input/',pm.Ftype_name[1]+'.in')
    pm.Ftype2InputPath=os.path.join('./input/',pm.Ftype_name[2]+'.in')

    pm.featCollectInPath=os.path.join(pm.fitModelDir,'feat_collect.in')
    pm.fitInputPath_lin=os.path.join(pm.fitModelDir,'fit_linearMM.input')
    pm.fitInputPath2_lin=os.path.join(pm.InputPath,'fit_linearMM.input')
    pm.featCollectInPath2=os.path.join(pm.InputPath,'feat_collect.in')

    pm.linModelCalcInfoPath=os.path.join(pm.fitModelDir,'linear_feat_calc_info.txt')
    pm.linFitInputBakPath=os.path.join(pm.fitModelDir,'linear_fit_input.txt')

    pm.dir_work = os.path.join(pm.fitModelDir,'NN_output/')

    pm.f_train_feat = os.path.join(pm.dir_work,'feat_train.csv')
    pm.f_test_feat = os.path.join(pm.dir_work,'feat_test.csv')

    pm.f_train_natoms = os.path.join(pm.dir_work,'natoms_train.csv')
    pm.f_test_natoms = os.path.join(pm.dir_work,'natoms_test.csv')        

    pm.f_train_dfeat = os.path.join(pm.dir_work,'dfeatname_train.csv')
    pm.f_test_dfeat  = os.path.join(pm.dir_work,'dfeatname_test.csv')

    pm.f_train_dR_neigh = os.path.join(pm.dir_work,'dR_neigh_train.csv')
    pm.f_test_dR_neigh  = os.path.join(pm.dir_work,'dR_neigh_test.csv')

    pm.f_train_force = os.path.join(pm.dir_work,'force_train.csv')
    pm.f_test_force  = os.path.join(pm.dir_work,'force_test.csv')

    pm.f_train_egroup = os.path.join(pm.dir_work,'egroup_train.csv')
    pm.f_test_egroup  = os.path.join(pm.dir_work,'egroup_test.csv')

    pm.f_train_ep = os.path.join(pm.dir_work,'ep_train.csv')
    pm.f_test_ep  = os.path.join(pm.dir_work,'ep_test.csv')

    pm.d_nnEi  = os.path.join(pm.dir_work,'NNEi/')
    pm.d_nnFi  = os.path.join(pm.dir_work,'NNFi/')

    pm.f_Einn_model   = pm.d_nnEi+'allEi_final.ckpt'
    pm.f_Finn_model   = pm.d_nnFi+'Fi_final.ckpt'

    pm.f_data_scaler = pm.d_nnFi+'data_scaler.npy'
    pm.f_Wij_np  = pm.d_nnFi+'Wij.npy'

'''
description: 
    write multi movement files to one file
param {*} sourceFileList
param {*} save_path
return {*}
author: wuxingxing
'''
def combine_movement(sourceFileList, save_path):
    #with open(os.path.join(os.path.abspath(pm.trainSetDir),'MOVEMENTall'), 'w') as outfile:     
    with open(os.path.join(save_path), 'w') as outfile:     
        # Iterate through list 
        for names in sourceFileList:
            # Open each file in read mode 
            with open(names) as infile:     
                # read the data from file1 and 
                # file2 and write it in file3 
                outfile.write(infile.read()) 
            # Add '\n' to enter data of file2 
            # from next line 
            outfile.write("\n")

def is_alive_atomic_energy(movement_list:list):
    if len(movement_list) < 1:
        return False
    # Declare is_real_Ep as a global variable
    command = 'grep Atomic-Energy ' + movement_list[0] + ' | head -n 1'
    print('running-shell-command: ' + command)
    result = subprocess.run(command, stdout=subprocess.PIPE, encoding='utf-8', shell=True)
    if 'Atomic-Energy' in result.stdout:
        alive_atomic_energy = True
    else:
        alive_atomic_energy = False
    return alive_atomic_energy
        
'''
description: 
    smlink souce file to target file
param {*} source_file
param {*} target_fie
return {*}
author: wuxingxing
'''
def smlink_file(source_file, target_fie):
    if os.path.exists(target_fie) is False:
        os.symlink(source_file, target_fie)


'''
description: 
    compare dir1 and dir2 are same dir
param {str} dir1
param {str} dir2
return {*}
author: wuxingxing
'''
def same_dir(dir1:str, dir2:str):
    comp = filecmp.dircmp(dir1, dir2)
    if comp.same_files and not comp.diff_files and not comp.funny_files:
        return True
    else:
        return False

'''
description: 
compare file1 and file2 are same file
param {str} file1
param {str} file2
return {*}
author: wuxingxing
'''
def same_file(file1:str, file2:str):
    if os.path.samefile(file1, file2):
        return True
    else:
        return False