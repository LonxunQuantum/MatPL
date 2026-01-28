import json
import os

from src.mods.adaptive_trainer import adaptive_trainer
from src.user.lmps_param import LmpParam
from src.user.gpumd_param import GPUmdParam
from src.user.gpumd_work import GPUMD

def ff2lmps_explore(input_json: json):
    lmps_param = LmpParam(input_json, os.getcwd())
    adpt_trainer = adaptive_trainer(lmps_param)
    #adpt_trainer.initialize() 
    adpt_trainer.explore()
    #adpt_trainer.dbg_explore()  
    #adpt_trainer.run_scf()
    #adpt_trainer.train() 
    #adpt_trainer.lmp_get_err()

'''
description: 
This method has been abandoned
param {json} input_json
return {*}
author: wuxingxing
'''
def run_gpumd(input_json:json):
    gpumd_param = GPUmdParam(input_json)
    gpumd = GPUMD(gpumd_param)
    gpumd.run_md()
