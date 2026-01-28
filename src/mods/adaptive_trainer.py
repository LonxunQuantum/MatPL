"""
    Adaptive trainer for DPKF
    2023.3, L. Wang  
"""

import os
import pathlib
import sys
import subprocess 
import multiprocessing as mp 
import numpy as np 
import random 
import time
from src.user.lmps_param import LmpParam
codepath = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(codepath)

sys.path.append(codepath+'/../PWmatMLFF')
sys.path.append(codepath+'/../aux')
sys.path.append(codepath+'/../pre_data')

#import dp_network

from poscar2lammps import p2l 
# from poscar2lammps import elem2idx
from slice import slice


class adaptive_trainer():
    
    def __init__(self, lmp_param:LmpParam):  
        """
            DIR: train
                All the training stuff
            
            DIR: seed 
                seed configs for exploration
                there will be groups with different ratio, box, etc.
                num of atom is the same with in a group 
                
            DIR: explore 
                You need 3 directories
                * models: 1.ff 2.ff ... n.ff. Force fields 
                * results: candidates that will be used for train. 
                * subsys: all configurations to be explored 
                *** which contian sys1, sys2, sys3, ...         
                *** in each of them, generate trajectory, and  
        """
        self.num_cand_per_cfg =lmp_param.num_cand_per_cfg                #  number of candidate per config
        self.silent_mode = lmp_param.silent_mode
        self.process_num = lmp_param.process_num

        self.success_bar = lmp_param.success_bar 
        self.candidate_bar = lmp_param.candidate_bar

        self.num_select_per_group = lmp_param.num_select_per_group
        self.is_single_node = lmp_param.is_single_node

        self.iter_num = lmp_param.iter_num
        # lammps related
        self.temperature = lmp_param.temp
        self.pressure = lmp_param.pressure
        self.ensemble = lmp_param.ensemble
        self.lmp_iso = lmp_param.lmp_iso
        self.traj_step = lmp_param.traj_step
        self.md_dt = lmp_param.md_dt
        # dir for sub systems explore
        # ./subsys/***
        
        # for traj generation
        self.ksapcing  = lmp_param.kspacing
        self.model_num = lmp_param.model_num

        self.working_dir = lmp_param.json_dir
        self.psp_dir = lmp_param.psp_dir
        self.etot_file = lmp_param.etot_file
        self.struct_dir = lmp_param.struct_dir

        self.train_path = os.path.dirname(self.working_dir)+"/00.train"
           
        self.ff_file = lmp_param.ff_file

        self.explore_path = os.path.abspath(os.path.join(self.working_dir,"explore"))
        self.seed_path = os.path.abspath(os.path.join(self.working_dir,"seed"))

        self.explore_model_dir  = self.explore_path + "/model"
        self.explore_result_dir = self.explore_path + "/result"
        self.explore_subsys_dir = self.explore_path + "/subsys"

        self.write_lmp_config = {   "init_config":"lmp.init",
                                    "ff_name":self.explore_model_dir+"/1.ff",
                                    "init_temp":300,
                                    "end_temp":300,
                                    "ensemble":self.ensemble,
                                    "lmp_iso":self.lmp_iso,
                                    "timestep":self.md_dt,
                                    "output_name":"traj",
                                    "is_traj":True,
                                    "traj_step":self.traj_step}
        
        # stats in each iteration 
        self.stats_list = []
        # list of trainer instance 
        self.trainer_list = []
         
        self.atom_type = lmp_param.atom_type
        
        self.node_num = lmp_param.node_num

        self.lmp_damp = lmp_param.lmp_damp
        self.lmp_nprocs = lmp_param.lmp_nprocs
        self.lmp_partition_name = lmp_param.lmp_partition_name
        self.lmp_wall_time = lmp_param.lmp_wall_time
        self.lmp_custom_lines = lmp_param.lmp_custom_lines 
        self.lmp_ntask_per_node = lmp_param.lmp_ntask_per_node

        if self.is_single_node is False:
            if self.lmp_partition_name is None:
                raise Exception("lmp_partition_name must be specified under multi-node mode")
            if self.lmp_ntask_per_node is None:
                raise Exception("lmp_ntask_per_node must be specified under multi-node mode")


    def initialize(self):
        # housekeeping. Set up dirs etc. 

        # if not os.path.exists(self.train_path):
        #     print("creating dir \'train\'")
        #     os.mkdir(self.train_path)
        
        if not os.path.exists(self.explore_path):
            print("creating dir \'explore\'")
            os.mkdir(self.explore_path)
        
        if not os.path.exists(self.seed_path):
            print("creating dir \'seed\'")
            os.mkdir(self.seed_path)

        if not os.path.exists(self.explore_model_dir):
            print("creating dir \'explore/model\'")
            os.mkdir(self.explore_model_dir)

        if not os.path.exists(self.explore_result_dir):
            print("creating dir \'explore/result\'")
            os.mkdir(self.explore_result_dir)

        if not os.path.exists(self.explore_subsys_dir):
            print("creating dir \'explore/subsys\'")
            os.mkdir(self.explore_subsys_dir)
        
        #print (os.path.exists(current_dir+"/explore"))
    
    """
       ****************************************** 
               selecting configurations 
       ****************************************** 
    """
    def select_init_config(self):
        """
            move all data from /seed to /subsys    
            add new data to /seed step by step 
            Note: will create directoroes in both /subsys and /result 
        """
        #remove all the subsys 
        #subprocess.run(["rm "+self.explore_subsys_dir+" * -r"],shell=True)
        
        # Delete the old link file (if existence)
        for filename in os.listdir(self.seed_path):
            file_path = os.path.join(self.seed_path, filename)
            if os.path.islink(file_path):
                os.remove(file_path)
        # link all the config files in /structures to /seed
        for path, dirList, fileList in os.walk(os.path.join(self.struct_dir, 'structures')):
            for filename in fileList:
                if filename.endswith(".config"):
                    source_file = os.path.abspath(os.path.join(path, filename))
                    target_file = os.path.join(self.seed_path, filename)
                    os.symlink(source_file, target_file)

        for a, b, c in os.walk(self.seed_path):
                cfg_names = c
                break
        
        for cfg in cfg_names: 
            # all tempratures 
            for temp in self.temperature:
                for pres in self.pressure:
                    
                    tmp_path_subsys = self.explore_subsys_dir+"/"+cfg + "_T="+str(temp) + "_P="+str(pres)
                    tmp_path_result = self.explore_result_dir+"/"+cfg + "_T="+str(temp) + "_P="+str(pres)
                    
                    if not os.path.exists(tmp_path_subsys):
                        os.mkdir(tmp_path_subsys)

                    #if not os.path.exists(tmp_path_result):
                    #    os.mkdir(tmp_path_result)
                    
                    # copy configuration
                    cmd = "cp "+self.seed_path+"/"+cfg + " " + tmp_path_subsys+"/atom.config"
                    subprocess.run([cmd],shell=True)

                    # generate lmp.cfg
                    os.chdir(tmp_path_subsys)
                    subprocess.run(["config2poscar.x atom.config > /dev/null"], shell = True)
                    p2l(output_name = "lmp.init")
                    subprocess.run(["rm","atom.config","POSCAR"])

                    print("lmp input config generated in "+tmp_path_subsys)

                    write_cfg = self.write_lmp_config  
                    write_cfg["init_temp"] = temp
                    write_cfg["end_temp"] = temp 
                    write_cfg["init_pres"] = pres
                    write_cfg["end_pres"] = pres

                    self.write_lmp_in(write_cfg)
        """
        # collect all dirs in /seed
        for a, b, c in os.walk(self.seed_path):
            dirs = b
            break
        
        for subdir in dirs:
            # create dirs in /subsys
            if not os.path.exists(self.explore_subsys_dir+"/"+subdir):
                os.mkdir(self.explore_subsys_dir+"/"+subdir)

            # create dirs in /result
            if not os.path.exists(self.explore_result_dir+"/"+subdir):
                os.mkdir(self.explore_result_dir+"/"+subdir)
            
            for a,b, c in os.walk(self.seed_path+"/"+subdir):
                cfg_names = c
                print (cfg_names)
                break
            
            cfg_root = self.seed_path+"/"+subdir

            for cfg in cfg_names: 
                # all tempratures 
                for temp in self.temperature:
                    for pres in self.pressure:
                        
                        tmp_path_subsys = self.explore_subsys_dir+"/"+subdir+"/"+cfg + "_T="+str(temp) + "_P="+str(pres)
                        tmp_path_result = self.explore_result_dir+"/"+subdir+"/"+cfg + "_T="+str(temp) + "_P="+str(pres)
                        
                        if not os.path.exists(tmp_path_subsys):
                            os.mkdir(tmp_path_subsys)

                        #if not os.path.exists(tmp_path_result):
                        #    os.mkdir(tmp_path_result)
                        
                        # copy configuration
                        cmd = "cp "+cfg_root+"/"+cfg + " " + tmp_path_subsys+"/atom.config"
                        subprocess.run([cmd],shell=True)

                        # generate lmp.cfg
                        os.chdir(tmp_path_subsys)
                        subprocess.run(["config2poscar.x atom.config > /dev/null"], shell = True)
                        p2l(output_name = "lmp.init")
                        subprocess.run(["rm","atom.config","POSCAR"])

                        print("lmp input config generated in "+tmp_path_subsys)

                        write_cfg = self.write_lmp_config  
                        write_cfg["init_temp"] = temp
                        write_cfg["end_temp"] = temp 
                        write_cfg["init_pres"] = pres
                        write_cfg["end_pres"] = pres

                        self.write_lmp_in(write_cfg)
                        """
            

    def write_lmp_in(self,write_config):
        # write lammps.in 
        import time 

        out_name = "lammps.in"

        head  =  ["units           metal\n",
                   "boundary        p p p\n",
                   "atom_style      atomic\n",
                   "processors    * * *\n",
                   "neighbor        2.0 bin\n",
                   "neigh_modify    every 10 delay 0 check no\n"]
        
        init_config = write_config["init_config"]

        #seed for initial velocity 
        lmp_seed = int(time.time())%356178 

        vline1 = "velocity         all create " 
        #vline2 = " 500000 dist gaussian\n"
        vline2 = " "+str(lmp_seed)+ " dist gaussian\n"

        dumpline1 = "dump             1 all custom 1 "
        dumpline2 = " id type x y z  vx vy vz fx fy fz\n"

        # lammps.in 
        f = open(out_name,"w")
 
        for line in head:
            f.writelines(line)
        
        f.writelines("read_data   "+init_config+"\n")

        f.writelines("pair_style  pwmatmlff\n")
        #f.writelines("pair_coeff  * * 5 5 3 14 "+write_config["ff_name"]+"\n")
        f.writelines("pair_coeff  * * 5 "+str(self.model_num)+" ")
        
        for i in range(1,self.model_num+1):
            write_config["ff_name"] = self.explore_model_dir + "/" + str(i) +  ".ff "
            f.writelines(write_config["ff_name"])

        atom_type_str = ' '.join(str(atom) for atom in self.atom_type)
        f.writelines("%s %.2f %.2f" % (atom_type_str, self.success_bar,self.candidate_bar))
        f.writelines("\n")
        
        # initial velocity gen
        f.writelines(vline1+str(write_config["init_temp"])+vline2)
        
        f.writelines("timestep         "+str(write_config["timestep"]) + "\n") 

        if write_config["ensemble"] == "nvt":
            line = "fix              1 all nvt temp "
            line = line + str(write_config["init_temp"]) + " "
            line = line + str(write_config["end_temp"]) + " "
            line  = line + str(write_config["timestep"]*self.lmp_damp) + "\n"
            f.writelines(line)
        elif write_config["ensemble"] == "npt":
            line = "fix              1 all npt temp "
            line = line + str(write_config["init_temp"]) + " "
            line = line + str(write_config["end_temp"]) + " "
            line  = line + str(write_config["timestep"]*self.lmp_damp) + " "
            line = line + str(write_config["lmp_iso"]) + " " +  str(write_config["init_pres"]) + " " + str(write_config["end_pres"])+ " "+ str(write_config["timestep"]*100) + "\n"
            f.writelines(line)

        f.writelines("thermo_style     custom step pe ke etotal temp vol press\n")
        f.writelines("thermo           1\n")

        f.writelines(dumpline1+write_config["output_name"]+dumpline2)
        
        # how many lammps steps 
        f.writelines("run              ")
        
        if write_config["is_traj"] is True:
            f.writelines(str(write_config["traj_step"])+"\n")
        else:
            f.writelines("0\n")

        f.close()

        # bars for success and candidate
        #f = open("bars","w")
        #f.writelines(str(self.success_bar)+" "+str(self.candidate_bar))
        #f.close()
        
    """
       ****************************************** 
          running lmp to generate trajectories 
       ****************************************** 
    """
    def write_lmp_sbatch(self,target_dirs, idx, lmp_unique_name):
        """
            example: 
                
                #!/bin/sh
                #SBATCH --partition=cpu
                #SBATCH --job-name=lmp
                #SBATCH --output=log
                #SBATCH --nodes=1
                #SBATCH --ntasks=2
                
                dir_array=("268_T=1200_P=1.0" 
                           "268_T=1200_P=10" 
                           "268_T=1200_P=100" 
                           "268_T=1200_P=1000")

                for dir in ${dir_array[@]};
                do
                    cd ${dir}
                    srun --exclusive -n 1 -c 1 lmp_mpi -in lammps.in &
                    cd ..
                done
                wait
                echo "done"
        """
        file = open(str(idx)+".sh","w")
        
        file.writelines("#!/bin/sh\n")
        file.writelines("#SBATCH --partition="+self.lmp_partition_name+"\n")
        file.writelines("#SBATCH --job-name="+lmp_unique_name+"\n")
        file.writelines("#SBATCH --nodes=1\n")
        file.writelines("#SBATCH --ntasks="+str(self.lmp_ntask_per_node)+"\n")
        
        # user defined lines 
        for line in self.lmp_custom_lines:
            file.writelines(line+"\n")
        
        # dir array and loop
        
        file.writelines("dir_array=(")
        for dir in target_dirs:
            file.writelines("\"")
            file.writelines(dir) 
            file.writelines("\" ")
        file.writelines(")\n")
        
        file.writelines("for dir in ${dir_array[@]};\n")
        file.writelines("do\n")
        file.writelines("cd ${dir}\n")
        file.writelines("srun --exclusive -n 1 -c 1 lmp_mpi -in lammps.in > /dev/null &\n")
        file.writelines("done\n")
        file.writelines("wait\n")
        
        file.close()                   

    def run_lmp_mp(self,tgt_dirs):
        import time

        for dir in tgt_dirs:
            print ("generating trajectory for " + dir)
            os.chdir(dir)
            
            begin = time.time()

            if self.silent_mode is True:
                subprocess.run(["srun --exclusive -n %d lmp_mpi -in lammps.in > /dev/null"%self.lmp_nprocs], shell=True)   
            else:
                subprocess.run(["srun --exclusive -n %d lmp_mpi -in lammps.in"%self.lmp_nprocs], shell=True)
            
            end = time.time()

            print("trajectory generated for "+dir+" in", end-begin, "s\n")
            
    def link_ff_files(self):     
        """
            link the force field files to the model dir
        """

        os.chdir(self.working_dir)
        if not os.path.exists(self.explore_model_dir):
            os.makedirs(self.explore_model_dir)

        if len(self.ff_file) == self.model_num:
            # Delete the old link file (if existence)
            for file in os.listdir(self.explore_model_dir):
                ff_name = os.path.join(self.explore_model_dir, file)
                if os.path.islink(ff_name):
                    os.unlink(ff_name)

            i = 0
            for file_path in self.ff_file:
                i += 1
                os.symlink(os.path.abspath(file_path), self.explore_model_dir+"/"+str(i)+".ff")

                if i == self.model_num:
                    break
        else:
            raise ValueError("the number of force field files is not equal to the number of models")

    def run_lmp(self):
        """
            using LAMMPS to generate trajectories 

            single_node : on one node using pool 
            multi_node  : use sbatch
        """
        is_complete = False
        lmp_dirs = []
        if self.process_num % self.lmp_nprocs != 0:
            raise Exception("process_num must can be divided by lmp_nprocs!")
        else:
            run_nums = int(self.process_num / self.lmp_nprocs)
        #lmp_dir_mp = [[] for i in range(self.process_num)] 
        lmp_dir_mp = [[] for i in range(run_nums)]

        # look for sub dirs
        for a, b, c in os.walk(self.explore_subsys_dir):
            # exclude those not at the lowest level 
            if len(b) == 0:
                lmp_dirs.append(a)

        num_dirs = len(lmp_dirs)

        print ("number of target dirs:", num_dirs)

        if self.is_single_node is True:
            for i in range(num_dirs):
                #lmp_dir_mp[i%ntasks_running].append(lmp_dirs[i])
                lmp_dir_mp[i%run_nums].append(lmp_dirs[i])
            
            start = time.time()  
            # distribute across processes
            #pool = mp.Pool(self.self.process_num)
            with mp.Pool(run_nums) as pool:
                pool.map(self.run_lmp_mp,lmp_dir_mp) 

            is_complete = True
        else:
            """
                use a single bacth file to submit multiple lmp jobs
                create sbacth files in explore/subsys
            """
            idx = 0 

            lmp_name_unique = "AJUFVC" 
            
            os.chdir(self.explore_subsys_dir)

            for start in range(0,num_dirs,self.lmp_ntask_per_node):
                
                if start + self.lmp_ntask_per_node > num_dirs:
                    end = num_dirs
                else:
                    end = start + self.lmp_ntask_per_node
                print ("start, end:",start,end-1)
                self.write_lmp_sbatch(lmp_dirs[start:end],idx,lmp_name_unique)
                
                print ("submitting "+str(idx)+".sh")
                cmd = "sbatch "+str(idx)+".sh"
                
                subprocess.run([cmd],shell=True)
                idx+=1

            # use squeue -l | grep RUNNING -c to monitor progress
            # also set a wall time of 2 hours
            start = time.time()     

            # how to determine when to stop? 
            while True:
                num_ongoing = int(subprocess.check_output(["squeue -n "+lmp_name_unique+" | wc -l"],shell=True))
                #num_pending = int(subprocess.check_output(["squeue -h -t pending -r | wc -l"],shell=True))
                
                if num_ongoing == 1:
                    is_complete = True
                    break 
                
                print(num_ongoing-1, " batches are running or pending")
                #print(num_pending, " batches are waiting")
                print("**************************************\n")
                if (time.time() - start > self.lmp_wall_time):
                    print("Wall time for traj gen is reached. Quitting")
                    break 
                
                time.sleep(10)
        
        if is_complete is True:
            
            time_elapsed =  time.time() - start 
            print (num_dirs, " explorations done in ", time_elapsed, "s")
        # do not proceed if not completed
        else:
            raise Exception("traj gen not accomplished. Will not proceed")

        num_success = 0 
        num_fail = 0
        num_cand = 0 
        
        # collect stats  
        print ("*****************SUMMARY OF EXPLORATION*****************")

        for dir in lmp_dirs: 
            
            if not os.path.exists(os.path.join(dir,"explr.stat")):
                print("exploration stat not found in", dir)
                print("\n")
                continue 
            
            f = open(os.path.join(dir,"explr.stat"),"r")
            raw = f.readlines()[-1].split()  
            f.close()
            
            tmp_success = int(raw[0])
            tmp_cand = int(raw[1])
            tmp_fail = int(raw[2])
            
            tmp_total = tmp_cand+ tmp_success+ tmp_fail

            print("in",dir)
            print ("ratio of success:", float(tmp_success)/tmp_total)
            print ("ratio of candidate:", float(tmp_cand)/tmp_total)
            print ("ratio of failure:", float(tmp_fail)/tmp_total)
            print("\n")

            num_success += int(raw[0])
            num_cand += int(raw[1])
            num_fail += int(raw[2])
        
        num_total = num_success+num_cand+num_fail 

        with open(os.path.join(self.working_dir,"explore.ratio"),"w") as f:
            f.write("num of all imgs: {}\n".format(num_success+num_cand+num_fail))
            f.write("ratio of success: {}\n".format(float(num_success)/num_total))
            f.write("ratio of candidate: {}\n".format(float(num_cand)/num_total))
            f.write("ratio of failure: {}\n".format(float(num_fail)/num_total))

        '''
        print ("********************************************************")

        print ("num of all img:", num_success+num_cand+num_fail)
        print ("ratio of success:", float(num_success)/num_total)
        print ("ratio of candidate:", float(num_cand)/num_total)
        print ("ratio of failure:", float(num_fail)/num_total)
        
        print ("********************************************************\n")
        '''
    
    def lmp_get_err(self,tgt_dir = None):
        lmp_dirs = [] 
        
        if tgt_dir is None:
            tgt_dir = self.explore_subsys_dir

        # obtain leaf dirs
        for a, b, c in os.walk(tgt_dir):
            if len(b) == 0:
                lmp_dirs.append(a)

        num_success = 0 
        num_cand = 0
        num_fail = 0

        err = [] 
        print(len(lmp_dirs))
        print ("in",tgt_dir)
        print ("*****************SUMMARY OF EXPLORATION*****************")

        for dir in lmp_dirs: 
            
            if not os.path.exists(os.path.join(dir,"explr.error")):
                print("exploration error not found in", dir)
                continue 
            
            f = open(os.path.join(dir,"explr.error"),"r")
            raw = f.readlines()  
            f.close()

            for line in raw:
                tmp = float(line)
                if tmp <= self.success_bar:
                    num_success += 1
                
                if tmp <= self.candidate_bar and tmp > self.success_bar:
                    num_cand +=1 
                
                if tmp > self.candidate_bar:
                    num_fail +=1 

                err.append(tmp)
        
        num_tot = num_success + num_cand + num_fail

        print("ratio of success (err <0.15)", num_success/num_tot)
        print("ratio of candidate (0.15 < err < 0.35)",num_cand/num_tot)
        print("ratio of failure (err > 0.35)", num_fail/num_tot)
        
        #np.save("err.npy",err)
        
        #print (num_success, num_cand, num_fail)

        print ("********************************************************")

        # save      
    def lmp_dbg(self):
        
        lmp_dirs = [] 

        for a, b, c in os.walk(self.explore_subsys_dir):
            # exclude those not at the lowest level 
            if len(b) == 0:
                lmp_dirs.append(a)

        num_success = 0 
        num_cand = 0
        num_fail = 0

        print(len(lmp_dirs))

        print ("*****************SUMMARY OF EXPLORATION*****************")

        for dir in lmp_dirs: 
            
            if not os.path.exists(os.path.join(dir,"explr.stat")):
                print("exploration stat not foudn in", dir)
                
                continue 
            
            f = open(os.path.join(dir,"explr.stat"),"r")
            raw = f.readlines()[0].split()  
            f.close()
            
            tmp_success = float(raw[0])
            tmp_cand = float(raw[1])
            tmp_fail = float(raw[2])
            
            print(dir)
            print(tmp_success, tmp_cand, tmp_fail)
            tmp_total = tmp_cand+ tmp_success+ tmp_fail
            print(tmp_success/tmp_total, tmp_cand/tmp_total, tmp_fail/tmp_total)
            
            print("\n")

            num_success = num_success + tmp_success 
            num_cand = num_cand + tmp_cand
            num_fail = num_fail + tmp_fail 

            continue
            tmp_total = tmp_cand+tmp_success+tmp_fail

            print("in",dir)
            print ("ratio of success:", float(tmp_success)/tmp_total)
            print ("ratio of candidate:", float(tmp_cand)/tmp_total)
            print ("ratio of failure:", float(tmp_fail)/tmp_total)
            print("\n")

            num_success += int(raw[0])
            num_cand += int(raw[1])
            num_fail += int(raw[2])
        
        #num_total = num_success+num_cand+num_fail 
        print (num_success, num_cand, num_fail)
        
        print ("********************************************************")
        # save 
    """
       ****************************************** 
                        run scf 
       ****************************************** 
    """

    def run_scf(self):
        """
            use selected structures for SCF. 
        """
        self.collect_cfgs() 
        
        self.write_scf_in_and_run() 
        
        #self.collect_mvmt()

    def write_etot_input(self, dir, num_atom, norm_b_list, atom_type):
        """
            write etot.input
        """
        from math import floor 

        psp_path = self.psp_dir

        etot_input = self.etot_file

        mp_line = "mp_n123 = "

        round_up = lambda x: round(x) if round(x)>0 else 1

        if num_atom > 200:
            mp_line = "mp_n123 = 1 1 1 0 0 0"
        else:
            mp_line = mp_line + str(round_up(norm_b_list[0]/self.ksapcing)) + " "
            mp_line = mp_line + str(round_up(norm_b_list[1]/self.ksapcing)) + " "
            mp_line = mp_line + str(round_up(norm_b_list[2]/self.ksapcing)) + " "
            mp_line = mp_line + "0 0 0"
        
        # idx_tabel = elem2idx()
        # elem_type = [elem for elem, order in idx_tabel.items() if order in atom_type]

        # psp_lines = []
        # for i, elem in enumerate(elem_type, start=1):
        #     psp_lines.append(f"in.psp{i} = {psp_path}/{elem}.SG15.PBE.UPF\n")
            # psp_lines.append("in.psp"+str(i)+" = "+psp_path+"/"+elem+".SG15.PBE.UPF\n")

        if etot_input is not None:
            with open(dir+"/etot.input","w") as file:
                with open(etot_input,"r") as etot_file:
                    etot_lines = etot_file.readlines()
                for line in etot_lines:
                    line = line.strip()
                    file.writelines(line + "\n")
                # file.writelines("job = scf\n")
                # file.writelines("in.atom = atom.config\n")
                # for line in psp_lines:
                #     file.writelines(line)
                file.writelines(mp_line)
        else:
            raise FileNotFoundError("template (etot_input) does not exist")

    def calc_recip_latt(self,a1,a2,a3):
        """
            return norms of reciprocal lattice vector
        """
        from numpy.linalg import norm
        from numpy import dot
        from numpy import cross

        pi = 3.14159268

        denominator = 1.0/dot(a1,cross(a2,a3))
        
        b1 = norm(2*pi*denominator*cross(a2,a3))
        b2 = norm(2*pi*denominator*cross(a3,a1)) 
        b3 = norm(2*pi*denominator*cross(a1,a2))

        return b1,b2,b3

    def write_scf_in_and_run(self):
        """
            For etot.input: 
            1. check atom number. Use single K-point when > 200
            2. use kspacing para to determine K mesh
        """    
        import time 

        atom_type = self.atom_type

        to_array = lambda x: np.array([float(i) for i in x.split()])

        scf_dirs = [] 

        for a, b, c in os.walk(self.explore_result_dir):
            # exclude those not at the lowest level 
            if len(b) == 0:
                scf_dirs.append(a)   

        for dir in scf_dirs:
            print ("writting etot.input in",dir)

            with open(dir+"/atom.config","r") as file:
                raw = file.readlines()[:5]

            num_atom = int(raw[0])
            
            a1 = to_array(raw[2])
            a2 = to_array(raw[3])
            a3 = to_array(raw[4])
            
            norm_b1,norm_b2,norm_b3 = self.calc_recip_latt(a1,a2,a3)
            
            #print (norm_b1,norm_b2,norm_b3)
            self.write_etot_input(dir, num_atom, [norm_b1,norm_b2,norm_b3], atom_type)

            print("etot.input written in",dir)

            # print ("SCF starts in", dir)
            
            # os.chdir(dir)
            # begin = time.time()
            # # silent mode 
            # # pwmat_cmd = "mpirun -np $SLURM_NPROCS -iface ib0 PWmat -host 10.0.0.2 50002 > /dev/null"
            # pwmat_cmd = "mpirun -np $SLURM_NPROCS -iface ib0 PWmat -host 10.0.0.2 50002 | tee output"
            # subprocess.run(pwmat_cmd,shell=True) 
            
            # end = time.time() 
            
            # print ("SCF ends successfully in", end-begin, "s")
            
    def collect_cfgs(self):
        """
            select configs in /subsys to /result
        """
        from random import shuffle

        # for all groups pick out all config.x in lmp dir 
        # for root, dirs, files in os.walk(self.explore_subsys_dir, topdown=False):
        #     group_dirs = dirs        
        
        # for group in group_dirs:
            # print("collecting configs in",group)
            # walk in this group 
        cfg_dot_dirs = [] 
            
        for root, dirs, files in os.walk(self.explore_subsys_dir):
            """
                some lmp run will just die out. 
                No config.(x) is generated
            """
            for file in files:
                if file.endswith(".config"):
                    cfg_dot_dirs.append(os.path.join(root, file))
            # for name in files:
            #     tmp = os.path.join(root, name)
            #     if ".config" in tmp:
            #         cfg_dot_dirs.append(tmp)

        shuffle(cfg_dot_dirs)

        # result dir 
            # tgt = (self.explore_result_dir + "/"+group).replace("subsys","result")
        tgt = (self.explore_result_dir).replace("subsys","result")

        # randomly select self.num_select_per_group
        for idx, cfg in enumerate(cfg_dot_dirs[:self.num_select_per_group]):
            
            tgt_sub = tgt+ "/" + str(idx)
            print("creating:",tgt_sub)

            if not os.path.exists(tgt_sub):
                os.mkdir(tgt_sub)
            
            cmd = "cp "+cfg+" "+tgt_sub+"/atom.config"
            subprocess.run([cmd], shell=True) 

    def collect_mvmt(self):
        """
            collect and concatenate movements from all scf dirs
            all OUT.MLMD -> to group_(x)/MOVEMENT  
        """
        scf_dirs = [] 
        tgt_name = "OUT.MLMD" 


        for root, dirs, files in os.walk(self.explore_result_dir):
            group_dirs = dirs 
            break
        
        for group in group_dirs:
            num_img = 0 
            cat_cmd = "cat "
            
            # walk in this group 
            for root, dirs, files in os.walk(os.path.join(self.explore_result_dir,group)):
                cand_dirs = dirs 
                break 
                
            for cand in cand_dirs:
                # check if OUT.MLMD exist 
                if os.path.exists(os.path.join(self.explore_result_dir,group,cand,tgt_name)):
                    num_img +=1
                    cat_cmd = cat_cmd + os.path.join(self.explore_result_dir,group,cand,tgt_name) + " " 
                #print (type(os.path.join(self.explore_result_dir,group,cand,tgt_name))
            
            if cat_cmd != "cat ":
                cat_cmd = cat_cmd + " > " + os.path.join(self.explore_result_dir, group, "MOVEMENT")
                #print(cat_cmd) 
                subprocess.run([cat_cmd],shell=True)
                print ("num image:",num_img)
                print ("MOVEMENT generated in", group)

    """
       ****************************************** 
                 training preparations
       ****************************************** 
    """
    def copy_mvmt_to_train(self):
        """
            copy movements from /explore/result
            also clean the old data in /train /valid 
        """

        return 

    """
       ****************************************** 
                    exploration core  
       ****************************************** 
    """
    def explore(self):
        
        # prepare directories
        self.initialize()
        # prepare initial configs
        self.select_init_config() 
        # copy force field files
        self.link_ff_files() 
        # generate trajectories and select candidates
        self.run_lmp()    
        # select a certain number of configs and move into ./result
        self.collect_cfgs() 
        # plot sigma from lammmps - explr.error
        self.plot_sigma()
        # prepare for scf
        self.write_scf_in_and_run() 

    """
       ****************************************** 
                    training core  
       ****************************************** 
    """
    def train_wrapper(self,idx):
        """
            foo bar foo bar ... 
        """
        print ("trainer",idx,"starts")
        
        os.chdir(str(idx))

        self.trainer_list[idx].generate_data(is_real_Ep = True) 
        
        self.trainer_list[idx].load_and_train() 
        
        # lmp inputs start from 1 
        self.trainer_list[idx].extract_force_field(name = str(idx+1)+".ff")

    def train(self):
        from src.PWMLFF.dp_network import dp_network
        import time
        """
            a self-contained training process
            automatically determine whether to resume
            requires that subdirs for each model and data have been in place 
        """  
        os.chdir(self.train_path)
        """
        for idx in range(self.model_num):
            if not os.path.exists(str(idx)):
                os.mkdir(str(idx))
                os.mkdir(str(idx)+"/PWdata")
        """
        # create trainer instances
        for i in range(self.model_num):
            
            mk = False 
            
            # check whether to resume training 
            if os.path.exists(str(i)+"/"+"record_"+str(i)):
                mk = True 
                print ("\nFound trained model in ", str(i)+"/"+"record_"+str(i)+". Will resume.",) 
                print ("cleaning previous data")

                input_dir = str(i)+"/input -r"
                output_dir = str(i)+"/output -r"
                pwdata_dir = str(i)+"/PWdata/*.dat"

                subprocess.run(["rm "+input_dir],shell=True)
                subprocess.run(["rm "+output_dir],shell=True)
                subprocess.run(["rm "+pwdata_dir],shell=True)
                
                #clean previous data
            self.trainer_list.append( dp_network(        
                                            atom_type = self.atom_type,   
                                            optimizer = "LKF",       
                                            gpu_id = i,   
                                            session_dir = "record_"+str(i),  
                                            n_epoch = 1,             
                                            batch_size = 10,          
                                            Rmax = 5.0,              
                                            Rmin = 0.5,              
                                            M2 = 16,                 
                                            dataset_size = 1000,     
                                            block_size= 10240,       
                                            is_virial = False,        
                                            is_resume = mk, 
                                            workers_dataload= 0, #ensure single process loading
                                            pre_fac_force = 1.0,     
                                            pre_fac_etot = 0.5,      
                                        ))
        
        #self.trainer_list[0].generate_data(is_real_Ep = True) 
        
        trainer_pool = mp.Pool(self.model_num)
        
        trainer_pool.map(self.train_wrapper,[i for i in range(self.model_num)]) 
        
        # !!! should be a barrier here !!!
        print ("\ntraining done\n")
        
    """
       ****************************************** 
                    plot_sigma core  
       ****************************************** 
    """
    def plot_sigma(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        mpl.use("Agg")
        fontsize = 20

        md_step = self.traj_step + 1   # idx from 1
        md_dt = int(self.md_dt * 1000) # fs for 'units metal' of lammps.in 
        T_range = [self.temperature]
        P_range = [self.pressure]
        plot_dir = self.working_dir + '/%s_explr_plot' % self.ensemble
        if not os.path.isdir(plot_dir):
            os.mkdir(plot_dir)
        xticks = range(0, md_step*md_dt, int(md_step//5)*md_dt)
        for idx in os.listdir(self.seed_path):
            for P in P_range[0]:
                if P == 1:
                    P = '1.0'
                else:
                    P = str(P)
                fig = plt.figure(figsize=(6, 8))
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                for T in T_range[0]:
                    error_data = np.zeros([md_step])
                    explr_error_file = '%s/%s_T=%d_P=%s/explr.error' % (self.explore_subsys_dir, idx, T, P)
                    if os.path.isfile(explr_error_file):
                        error_tmp = np.loadtxt(explr_error_file)
                        error_data[:len(error_tmp)] = error_tmp[:, 1]
                        y, x = np.histogram(error_data, bins=200, range=(0., .4))
                        xx = [(x[i]+x[i+1])/2 for i in range(len(x)-1)]
                        ax1.plot(xx, y*500./len(error_data), lw=2, ls="--", label="T=%d K" % T)
                        ax2.plot(np.arange(error_data.shape[0])*md_dt, error_data, lw=2, ls="--", label="T=%d K" % T)

                        ax1_data = np.column_stack((xx, y*500./len(error_data)))
                        ax2_data = np.column_stack((np.arange(error_data.shape[0])*md_dt, error_data))
                        data_file1 = f"{plot_dir}/error_data_T{T}_P{P}.txt"
                        data_file2 = f"{plot_dir}/error_data_T{T}_P{P}_Step.txt"
                        np.savetxt(data_file1, ax1_data, header="x y", fmt='%.6f')
                        np.savetxt(data_file2, ax2_data, header="x y", fmt='%d %.6f')
                        
                # ax1.set_title(" Iteration %s" % self.iter_num, fontsize=fontsize+4)
                ax1.set_xlabel(r'$\sigma_f^{\mathrm{max}}$ (eV/$\mathrm{\AA}$)', size=fontsize)
                ax1.set_ylabel('Distribution (%)', size=fontsize)
                ax1.set_xticks([0.,.1,.2,.3,.4])
                ax1.set_yticks(range(0,41,10))
                ax1.set_xticklabels([0.,.1,.2,.3,.4])
                ax1.set_yticklabels(range(0,41,10))
                ax1.tick_params(axis='x', labelsize=fontsize-4)
                ax1.tick_params(axis='y', labelsize=fontsize-4)
                ax1.set_ylim([0,45])
                ax1.set_xlim([0,.4])
                ax1.legend(fontsize=fontsize-6, labels=['T=%d K' % T for T in T_range[0]])
                ax2.set_ylabel(r'$\sigma_f^{\mathrm{max}}$ (eV/$\mathrm{\AA}$)', size=fontsize)
                ax2.set_xlabel('MD time', size=fontsize)
                ax2.set_yticks([0.,.1,.2,.3,.4])
                ax2.set_xticks(xticks)
                ax2.set_yticklabels([0.,.1,.2,.3,.4])
                ax2.set_xticklabels(xticks)
                ax2.tick_params(axis='x', labelsize=fontsize-4)
                ax2.tick_params(axis='y', labelsize=fontsize-4)
                ax2.set_ylim([0,.4])
                ax2.set_xlim([0,md_step*md_dt])
                plt.tight_layout()
                plt.savefig(plot_dir+'/dev_fmax.png', dpi=300)
                plt.close()

    def my_dbg(self):

        for root, dirs, files in os.walk(self.explore_subsys_dir, topdown=False):
            group_dirs = dirs
        
        for group in group_dirs:
            # walk in this group 
            cfg_dot_dirs = [] 

            for root, dirs, files in os.walk(os.path.join(self.explore_subsys_dir,group), topdown=False):
                """
                    some lmp run will just die out. 
                    No config.(x) is generated
                """
                for name in files:
                    tmp = os.path.join(root, name)
                    if "config." in tmp:
                        cfg_dot_dirs.append(tmp)
    
        
    def dbg_explore(self):
        
        # prepare directories
        self.select_init_config() 

        # generate trajectories and select candidates
        self.run_lmp()    
        
        # select a certain number of configs and move into ./result
        self.collect_cfgs() 
        
        # prepare for scf
        self.write_scf_in_and_run() 

if __name__ == "__main__":
    """
        1. as long as the seed doesn't change, only need to update .ff at every iteration 
        2. OUT.MLMD should be processed to shift the atomic energies
    """
    temp_range = [400,600,800,1000]
    pressure_range= [1.0,10,100,1000]   

    adpt_trainer = adaptive_trainer(
                                    temp = temp_range,
                                    pressure= pressure_range, 
                                    process_num = 64,           # for lmp traj gen 
                                    model_num = 4,              
                                    kspacing = 0.16, 
                                    ensemble= "npt",
                                    traj_step = 3000, 
                                    num_select_per_group = 50,
                                    atom_type = [3,14],
                                    success_bar = 0.15,
                                    candidate_bar = 0.35,       
                                   )
    #adpt_trainer.initialize() 
    adpt_trainer.explore()
    #adpt_trainer.dbg_explore()  
    #adpt_trainer.run_scf()
    #adpt_trainer.train() 
    #adpt_trainer.lmp_get_err()
