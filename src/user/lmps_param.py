import json
import os 

from utils.json_operation import get_parameter, get_required_parameter
from utils.nep_to_gpumd import get_atomic_name_from_str
class LmpParam(object):
    def __init__(self, lmp_json:json, working_dir:str) -> None:
        self.json_dir = os.getcwd()
        self.working_dir = get_parameter("working_dir", lmp_json,  working_dir)
        self.model_num = get_parameter("model_num", lmp_json,  4)
        self.model_type = get_required_parameter("model_type", lmp_json).upper()
        self.num_cand_per_cfg = get_parameter("num_cand_per_cfg", lmp_json, 10)
        self.process_num = get_parameter("process_num", lmp_json,  1)
        self.iter_num = get_parameter("iter_num", lmp_json,  1)
        self.temp = get_parameter("temp", lmp_json,  [])
        self.pressure = get_parameter("pressure", lmp_json,  [1])
        self.traj_step = get_parameter("traj_step", lmp_json,  500)
        self.md_dt = get_parameter("md_dt", lmp_json,  0.001)
        self.ensemble = get_parameter("ensemble", lmp_json,  "nvt")
        self.lmp_iso = get_parameter("lmp_iso", lmp_json,  "tri")
        self.kspacing = get_parameter("kspacing", lmp_json,  0.16)
        self.silent_mode = get_parameter("silent_mode", lmp_json,  True)
        self.num_select_per_group = get_parameter("num_select_per_group", lmp_json,  20)
        self.psp_dir = get_parameter("psp_dir", lmp_json,  "/share/psp/NCPP-SG15-PBE")
        self.etot_file = get_parameter("etot_file", lmp_json,  './')
        self.struct_dir = get_parameter("struct_dir", lmp_json,  None)
        self.ff_file = get_parameter("ff_file", lmp_json,  [])
        self.node_num = get_parameter("node_num", lmp_json,  1)
        self.atom_type = get_atomic_name_from_str(get_parameter("atom_type", lmp_json,  []))
        self.success_bar = get_parameter("success_bar", lmp_json,  0.15)
        self.candidate_bar = get_parameter("candidate_bar", lmp_json,  0.35)
        self.lmp_damp = get_parameter("lmp_damp", lmp_json,  25)
        self.lmp_nprocs = get_parameter("lmp_nprocs", lmp_json,  1)
        self.is_single_node = get_parameter("is_single_node", lmp_json,  True,      )
        self.lmp_partition_name = get_parameter("lmp_partition_name", lmp_json,  None)
        self.lmp_ntask_per_node = get_parameter("lmp_ntask_per_node", lmp_json,  None,)
        self.lmp_wall_time = get_parameter("lmp_wall_time", lmp_json,  7200)
        self.lmp_custom_lines = get_parameter("lmp_custom_lines", lmp_json, [])

    def to_dict(self):
        dicts = {}
        dicts["working_dir"] = self.working_dir
        dicts["model_num"] = self.model_num
        dicts["num_cand_per_cfg"] = self.num_cand_per_cfg
        dicts["process_num"] = self.process_num
        dicts["iter_num"] = self.iter_num
        dicts["temp"] = self.temp
        dicts["pressure"] = self.pressure
        dicts["traj_step"] = self.traj_step
        dicts["md_dt"] = self.md_dt
        dicts["ensemble"] = self.ensemble
        dicts["lmp_iso"] = self.lmp_iso
        dicts["kspacing"] = self.kspacing
        dicts["silent_mode"] = self.silent_mode
        dicts["num_select_per_group"] = self.num_select_per_group
        dicts["psp_dir"] = self.psp_dir
        dicts["etot_file"] = self.etot_file
        dicts["struct_dir"] = self.struct_dir
        dicts["ff_file"] = self.ff_file
        dicts["node_num"] = self.node_num
        dicts["atom_type"] = self.atom_type
        dicts["success_bar"] = self.success_bar
        dicts["candidate_bar"] = self.candidate_bar
        dicts["lmp_damp"] = self.lmp_damp
        dicts["lmp_nprocs"] = self.lmp_nprocs
        dicts["is_single_node"] = self.is_single_node
        dicts["lmp_partition_name"] = self.lmp_partition_name
        dicts["lmp_ntask_per_node"] = self.lmp_ntask_per_node
        dicts["lmp_wall_time"] = self.lmp_wall_time
        dicts["lmp_custom_lines"] = self.lmp_custom_lines
        return dicts
    
