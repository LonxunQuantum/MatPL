#!/usr/bin/env python
import json
import os
import sys
import argparse
from src.user.nep_work import nep_train, nep_test, nep_test_ckpt, togpumd
from src.user.dp_work import dp_train, dp_test
# from src.user.cheby_work import cheby_train, cheby_test
from src.user.envs import comm_info, matpl_help
from src.utils.json_operation import get_parameter, get_required_parameter
from src.user.ckpt_extract import extract_force_field, script_model
from src.user.ckpt_compress import compress_force_field
from src.user.infer_main import infer_main, model_devi

if __name__ == "__main__":

    num_nodes  = os.environ.get("SLURM_NNODES", None)
    if num_nodes is not None and int(num_nodes) > 1: # multi node training and start by slurm srun
        rank       = int(os.environ["SLURM_PROCID"])
    else:
        rank = 0
    if rank == 0:
        comm_info()
    
    if len(sys.argv) == 1 or "-h".upper() == sys.argv[1].upper() or \
        "help".upper() == sys.argv[1].upper() or "-help".upper() == sys.argv[1].upper() or "--help".upper() == sys.argv[1].upper():
        if rank == 0:
            matpl_help()
    else:
        cmd_type = sys.argv[1].upper()
        # cmd_type = "test".upper()
        # cmd_type = "train".upper()
        # cmd_type = "infer".upper()
        # cmd_type = "explore".upper()
        if cmd_type.lower() not in ["train", "test", "extract_ff", "compress", "totxt", "script", "infer", "model_devi", "kpu", "nep_profile"]:
            raise Exception("Error! The input command {} can not be recognized, please use 'MatPL -h' to query all available commands!".format(cmd_type))
        if cmd_type == "extract_ff".upper():
            ckpt_file = sys.argv[2]
            extract_force_field(ckpt_file, cmd_type)
        elif cmd_type == "nep_profile".upper():
            json_path = sys.argv[2]
            extra = sys.argv[3:]
            os.chdir(os.path.dirname(os.path.abspath(json_path)))
            json_file = json.load(open(json_path))
            from src.user.nep_profile_work import nep_profile
            nep_profile(json_file, extra)
        elif cmd_type == "compress".upper():
            ckpt_file = sys.argv[2]
            compress_force_field(ckpt_file)
        # elif cmd_type == "toneplmps".upper():
        #     toneplmps(sys.argv[2:])
        elif cmd_type == "totxt".upper():
            togpumd(sys.argv[2:])

        # elif cmd_type == "topwmlff".upper():
        #     tonepckpt(sys.argv[2:])

        elif cmd_type == "script".upper():
            ckpt_file = sys.argv[2]
            if len(sys.argv) > 3:
                script_save_name = sys.argv[3]
            else:
                script_save_name = None
            script_model(ckpt_file, script_save_name)
        elif cmd_type == "infer".upper():
            infer_main(sys.argv[2:]) # config or poscar
        elif cmd_type == "model_devi".upper():
            parser = argparse.ArgumentParser()
            parser.add_argument('-m', '--model_list', help='specify input model files', nargs='+', type=str, default=None)
            parser.add_argument('-t', '--atom_type', help='specify the atom type of configs for lammps/dump file', nargs='+', type=str, default=None)
            parser.add_argument('-s', '--savename', help='specify stored file name', type=str, default='matpl_model_devi.out')
            parser.add_argument('-f', '--format', help="specify input structure format, supports all data format defined in pwdata, such as 'pwmat/movement', 'lammps/dump', or 'extxyz'", type=str, default="pwmat/config")
            parser.add_argument('-c', '--config', help='specify structure dir', type=str, default=None)
            parser.add_argument('-w', '--work_dir', help='specify work dir', type=str, default='./')
            args = parser.parse_args(sys.argv[2:])
            print(args.work_dir)
            os.chdir(args.work_dir)

            model_devi(args.model_list, args.config, format=args.format, save_path=args.savename, atom_names=args.atom_type) # config or poscar

        # elif cmd_type == "kpu".upper():
        #     parser = argparse.ArgumentParser()
        #     parser.add_argument('-m', '--model_path', help='specify input model file', type=str, default="dp_model.ckpt")
        #     parser.add_argument('-c', '--config', help='specify structure dir', type=str, default='traj')
        #     parser.add_argument('-f', '--format', help="specify input structure format, 'outcar', 'config', 'dump'", type=str, default="dump")
        #     parser.add_argument('-a', '--atom_type', help="file or string list for atom types, example '.../atom_type.txt', or 'Li Si ...' ", nargs='+', type=str, default="dump")
        #     parser.add_argument('-s', '--savepath', help='specify stored directory', type=str, default='kpu_model_devi.out')
        #     parser.add_argument('-w', '--work_dir', help='specify work dir', type=str, default='./')
            
        #     # parser.add_argument("-e", "--etotkpu", dest="etotkpu", action="store_true", help="calculate etotkpu")
        #     # parser.add_argument("-f", "--forcekpu", dest="forcekpu", action="store_true", help="calculate forcekpu")
        #     parser.add_argument("-d", "--forcedetail", dest="forcedetail", action="store_true", help="save force kpu detail")
        #     # parser.add_argument("-h", "--help",help="Example:\nPWMLFF kpu -m model.ckpt -c traj -f dump -a Li Si -s model_devi.out -d", nargs=0)  
    
        #     args = parser.parse_args(sys.argv[2:])
        #     print(args.work_dir)
        #     os.chdir(args.work_dir)
        #     from src.user.kpu_dp import KPU_CALCULATE
        #     kpu = KPU_CALCULATE(args.model_path)
        #     kpu.kpu_dp(structure_dir=args.config, format=args.format, atom_names=args.atom_type, savepath=args.savepath, \
        #         is_etot_kpu=True, is_force_kpu=True, force_kpu_detail=args.forcedetail)

        else:
            json_path = sys.argv[2]
            # cmd_type = "test".upper()
            # json_path = "/data/home/hfhuang/2_MLFF/1-NN/7-json/4-CH4-dbg/nn_new.json"
            os.chdir(os.path.dirname(os.path.abspath(json_path)))
            json_file = json.load(open(json_path))
            model_type = get_required_parameter("model_type", json_file).upper()  # model type : dp or nn or linear
            model_num = get_parameter("model_num", json_file, 1)
            if model_num > 1 and cmd_type == "train".upper():
                # for multi train, need to input slurm file
                slurm_file = sys.argv[3]
                from src.utils.gen_multi_train import multi_train
                multi_train(json_path, cmd_type, slurm_file)

            if cmd_type == "train".upper():
                if model_type == "DP".upper():
                    dp_train(json_file, cmd_type)
                elif model_type == "NN".upper():
                    from src.user.nn_work import nn_train
                    nn_train(json_file, cmd_type)
                elif model_type == "Linear".upper():
                    from src.user.linear_work import linear_train
                    linear_train(json_file, cmd_type)
                elif model_type == "NEP".upper():
                    nep_train(json_file, cmd_type)
                # elif model_type == "CHEBY".upper():
                #     cheby_train(json_file, cmd_type)
                else:
                    raise Exception("Error! the model_type param in json file does not existent, you could use [DP/NN/LINEAR/NEP]")
            elif cmd_type == "test".upper():
                if model_type == "DP".upper():
                    dp_test(json_file, cmd_type)
                elif model_type == "NN".upper():
                    from src.user.nn_work import nn_test
                    nn_test(json_file, cmd_type)
                elif model_type == "Linear".upper():
                    from src.user.linear_work import linear_test
                    linear_test(json_file, cmd_type)
                elif model_type == "NEP".upper():
                    nep_test(json_file, cmd_type)
                    # nep_test_ckpt(json_file, cmd_type) 用于debug，做ckpt文件的推理，测试pytorch端正确性
                    pass
                # elif model_type == "CHEBY".upper():
                #     cheby_test(json_file, cmd_type)
                else:
                    raise Exception("Error! the model_type param in json file does not existent, you could use [DP/NN/LINEAR/NEP]")
            
            elif cmd_type == "gen_feat".upper():
                if model_type == "DP".upper():
                    pass
                elif model_type == "NN".upper():
                    from src.user.nn_work import gen_nn_feature
                    gen_nn_feature(json_file, cmd_type)
                    # gen_nep_feature(json_file, cmd_type)
                else:
                    raise Exception("Error! the model_type param in json file does not existent, you could use [DP/NN/LINEAR/NEP]")
            elif cmd_type == "explore".upper():
                # for now, only support explore for DP model
                from src.user.active_work import ff2lmps_explore
                ff2lmps_explore(json_file)
    
        