#!/usr/bin/env python
import json
import os, random, shutil
from utils.file_operation import delete_tree
from src.slurm.slurm import SlurmJob, Mission
from src.user.input_param import InputParam
"""
This script generates multiple training jobs for PWmatMLFF. It takes a json file and a slurm file as input, and generates multiple training jobs based on the number of models specified in the json file. The script first generates features using the PWMLFF gen_feat command, and then trains each model using the PWMLFF train command. The script also modifies the json file for each model to specify the working directory, model store directory, and forcefield directory. The script uses subprocess to submit jobs to the slurm scheduler.
"""

def multi_train(json_path:str, cmd_type, slurm_file):
    # generate feature
    do_gen_feature_job(json_path, slurm_file)
    
    # train multi model
    # use std json file from gen feature step
    std_json = os.path.join(os.path.dirname(os.path.abspath(json_path)), "std_input.json")
    do_train_job(std_json, cmd_type, slurm_file)

def do_gen_feature_job(json_path, slurm_file):
    command_gen_feat = "PWMLFF gen_feat " + json_path
    command_submit_gen_feat = "sbatch gen_slurm.sh"
    # make gen_feature slurm script
    shutil.copy(slurm_file, "gen_slurm.sh")
    tag = "gen_feat_success.tag"
    with open("gen_slurm.sh", 'r+') as f:
        lines = f.readlines()
        if command_gen_feat + '\n' not in lines:
            f.write('\n' + command_gen_feat + '\n')
            f.write('\n'+ "echo 0 > {}\n".format(tag))

    # run gen_feature slurm script
    mission = Mission()
    slurm_job = SlurmJob()
    slurm_job.set_tag(tag)
    slurm_job.set_cmd(command_submit_gen_feat)
    mission.add_job(slurm_job)
    mission.commit_jobs()   # commit job
    mission.check_running_job() # check job is done
    print("\ngen feautre done!\n")

def do_train_job(json_path, cmd, slurm_template):
    json_dir = os.path.dirname(os.path.abspath(json_path))
    json_file = json.load(open(json_path))
    dp_param = InputParam(json_file, cmd) 
    
    json_file = json.load(open(json_path))
    model_num = json_file['model_num']
    mission = Mission()

    command_train = "PWMLFF train {}"
    command_slrum = "sbatch {}"
    for i in range(0, model_num):
        # make new json file which data source from feature path
        json_file_name = os.path.join(json_dir, "std_input_{}.json".format(i))
        json_file_i = set_json_file(json_file, dp_param, i)
        json.dump(json_file_i, open(json_file_name, "w"), indent=4)
        #make training slurm script
        slurm_file = "train_slurm_{}.sh".format(i)
        tag = "success_train_model_{}.tag".format(i)
        shutil.copy(slurm_template, slurm_file)
        with open(slurm_file, 'a') as wf:
            wf.write('\n' + command_train.format(json_file_name) + '\n')
            wf.write('\n' + "echo 0 > {}\n\n".format(tag) + '\n')
        slurm_cmd = command_slrum.format(slurm_file)
        
        slurm_job = SlurmJob()
        slurm_job.set_tag(tag)
        slurm_job.set_cmd(slurm_cmd)
        mission.add_job(slurm_job)
    
    # run training mission
    if len(mission.job_list) > 0:
        mission.commit_jobs()
        mission.check_running_job()
        assert(mission.all_job_finished())

    print("train model done")
    # post process of multi train
    post_process_multi_train(dp_param)

def set_json_file(json_file:json, dp_param:InputParam, index):
    # set json file dir name
    json_file["model_num"] = 1
    json_file["model_store_dir"] = "{}_{}".format(os.path.basename(dp_param.file_paths.model_store_dir), index)
    json_file["forcefield_dir"] = "{}_{}".format(os.path.basename(dp_param.file_paths.forcefield_dir), index)
    json_file["train_movement_file"] = []
    json_file["train_feature_path"] = [dp_param.file_paths.train_dir]
    json_file["reserve_work_dir"] = True
    json_file["reserve_feature"] = True
    json_file["seed"] = random.randint(1,1000000)
    return json_file

def post_process_multi_train(dp_param:InputParam):
    # delete slurm file
    file_list = os.listdir(dp_param.file_paths.json_dir)
    for file in file_list:
        if ".out" in file:  #delete slurm log
            os.remove(file)
    # delete feature file
    if dp_param.file_paths.reserve_feature is False:
        delete_tree(dp_param.file_paths.train_dir)
    if dp_param.file_paths.reserve_work_dir is False:
        delete_tree(dp_param.file_paths.work_dir)
    