#!/usr/bin/env python
import json
import os, sys
import argparse
from src.user.nep_work import nep_train, nep_test, nep_test_ckpt, togpumd
from src.user.envs import comm_info, matpl_help
from utils.json_operation import get_parameter, get_required_parameter
from src.user.infer_main import infer_main, model_devi

if __name__ == "__main__":
    comm_info()

    if len(sys.argv) == 1 or "-h".upper() == sys.argv[1].upper() or \
        "help".upper() == sys.argv[1].upper() or "-help".upper() == sys.argv[1].upper() or "--help".upper() == sys.argv[1].upper():
        matpl_help()
    else:
        cmd_type = sys.argv[1].upper()
        # cmd_type = "test".upper()
        # cmd_type = "train".upper()
        # cmd_type = "infer".upper()
        # cmd_type = "explore".upper()
        if cmd_type.lower() not in ["train", "test", "extract_ff", "compress", "totxt", "script", "infer", "model_devi", "kpu"]:
            raise Exception("Error! The input command {} can not be recognized, please use 'MatPL -h' to query all available commands!".format(cmd_type))
        # elif cmd_type == "toneplmps".upper():
        #     toneplmps(sys.argv[2:])
        elif cmd_type == "totxt".upper():
            togpumd(sys.argv[2:])

        elif cmd_type == "infer".upper():
            infer_main(sys.argv[2:]) # config or poscar
        elif cmd_type == "model_devi".upper():
            parser = argparse.ArgumentParser()
            parser.add_argument('-m', '--model_list', help='specify input model files', nargs='+', type=str, default=None)
            parser.add_argument('-t', '--atom_type', help='specify the atom type of configs for lammps/dump file', nargs='+', type=str, default=None)
            parser.add_argument('-f', '--format', help="specify input structure format, default is 'lammps/dump'", type=str, default="lammps/dump")
            parser.add_argument('-s', '--savename', help='specify stored file name', type=str, default='matpl_model_devi.out')
            parser.add_argument('-c', '--config', help='specify structure dir', type=str, default='trajs')
            parser.add_argument('-w', '--work_dir', help='specify work dir', type=str, default='./')
            args = parser.parse_args(sys.argv[2:])
            print(args.work_dir)
            os.chdir(args.work_dir)
            model_devi(args.model_list, args.config, format=args.format, save_path=args.savename, atom_names=args.atom_type) # config or poscar
        else:
            json_path = sys.argv[2]
            # cmd_type = "test".upper()
            # json_path = "/data/home/hfhuang/2_MLFF/1-NN/7-json/4-CH4-dbg/nn_new.json"
            os.chdir(os.path.dirname(os.path.abspath(json_path)))
            json_file = json.load(open(json_path))
            model_type = get_required_parameter("model_type", json_file).upper()  # model type : dp or nn or linear
            if cmd_type == "train".upper():
                nep_train(json_file, cmd_type)
            elif cmd_type == "test".upper():
                if model_type == "NEP".upper():
                    nep_test(json_file, cmd_type)

    
        