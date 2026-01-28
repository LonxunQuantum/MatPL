import json
import os

from src.mods.adaptive_trainer import adaptive_trainer
from src.user.lmps_param import LmpParam

def ff2lmps_explore(input_json: json):
    lmps_param = LmpParam(input_json, os.getcwd())
    adpt_trainer = adaptive_trainer(lmps_param)
    #adpt_trainer.initialize() 
    adpt_trainer.explore()
    #adpt_trainer.dbg_explore()  
    #adpt_trainer.run_scf()
    #adpt_trainer.train() 
    #adpt_trainer.lmp_get_err()



    