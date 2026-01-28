import os,sys
import shutil
import pathlib

codepath = str(pathlib.Path(__file__).parent.resolve())

#for model.mlff 
sys.path.append(codepath+'/../model')

#for default_para, data_loader_2type dfeat_sparse
sys.path.append(codepath+'/../pre_data')

#for optimizer
sys.path.append(codepath+'/..')
sys.path.append(codepath+'/../aux')
sys.path.append(codepath+'/../lib')
sys.path.append(codepath+'/../..')

from src.user.input_param_tmp import InputParam
# from src.pre_data.nn_mlff_hybrid import get_cluster_dirs, make_work_dir, mv_featrues, copy_file
from utils.file_operation import write_arrays_to_file
from src.aux.inference_plot import inference_plot
from utils.file_operation import copy_movements_to_work_dir, reset_pm_params, combine_movement, copy_tree
from src.aux.plot_evaluation import plot_new, read_dft_movement
import fortran_fitting as ff 
import default_para as pm 
class linear_regressor:
    
    def __init__(   
                    self,
                    dp_param: InputParam
                ):

        # atom_type = None, 
        # feature_type = None,
        # max_neigh_num = 100, 
        
        # etot_weight = 0.5, 
        # force_weight = 0.5,
        # ei_weight = 0.5, 
        self.dp_params = dp_param
        if self.dp_params.atom_type == None:
            raise Exception("atom types not specifed")
        
        if self.dp_params.descriptor.feature_type is None:
            raise Exception("feature type not specified")

        pm.atomType = self.dp_params.atom_type

        pm.use_Ftype = self.dp_params.descriptor.feature_type
        pm.nfeat_type = len(pm.use_Ftype)

        pm.atomTypeNum = len(pm.atomType)       
        pm.ntypes = len(pm.atomType)

        pm.maxNeighborNum = self.dp_params.max_neigh_num 
        
        # weights for regression 
        pm.fortranFitWeightOfEtot = self.dp_params.optimizer_param.pre_fac_etot
        pm.fortranFitWeightOfForce = self.dp_params.optimizer_param.pre_fac_force
        pm.fortranFitWeightOfEnergy = self.dp_params.optimizer_param.pre_fac_ei

    def evaluate_prepare_data(self, movement_path:list[str]=None, work_dir:str=None):
        # copy movement file to MD/ dir
        target_dir = os.path.join(work_dir, "MD")
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)
        combine_movement(movement_path, 
                         os.path.join(target_dir, self.dp_params.file_paths.movement_name))
        # copy forcefild files from forcefield dir
        source_dir = os.path.join(self.dp_params.file_paths.json_dir, os.path.basename(self.dp_params.file_paths.forcefield_dir))
        target_dir = work_dir
        # copy fread_dfeat  input  output to target dir
        copy_tree(os.path.join(source_dir, "fread_dfeat"), os.path.join(target_dir, "fread_dfeat"))
        copy_tree(os.path.join(source_dir, "input"), os.path.join(target_dir, "input"))
        copy_tree(os.path.join(source_dir, "output"), os.path.join(target_dir, "output"))

    def generate_data(self, movement_path:list[str]=None, feature_type:str=None):
        
        if os.path.exists(self.dp_params.file_paths.nn_work) is True: # work_dir/feature dir
            shutil.rmtree(self.dp_params.file_paths.nn_work)
        os.makedirs(self.dp_params.file_paths.nn_work)
        pwdata_work_dir = copy_movements_to_work_dir(movement_path,
                            os.path.abspath(os.path.join(self.dp_params.file_paths.nn_work, feature_type)), 
                              "PWdata", 
                                self.dp_params.file_paths.movement_name)
        # generate feature
        cwd = os.getcwd()
        os.chdir(os.path.dirname(pwdata_work_dir))
        reset_pm_params(pm, os.path.dirname(pwdata_work_dir))
        print("data generation starts")
        # calculate features
        pm.isCalcFeat = True 
        import mlff 
        pm.isCalcFeat = False
        os.chdir(cwd)
        
    def train(self, work_path):
        print ("training starts")
        
        pm.isFitLinModel = True     

        #pp.prepare_dir_info()   

        #os.system('cp '+pm.fbinListPath+' ./input/')
        #pp.writeGenFeatInput()
        #pp.collectAllSourceFiles()

        #pp.prepare_novdw()
        # change dir to feature dir and do fitting
        os.chdir(work_path)
        ff.fit() 

        pm.isFitLinModel = False 
        """
        pm.isFitLinModel = True    
        import mlff
        pm.isFitLinModel = False 
        """
    
    def evaluate(self, work_path, num_thread = 1, plot_elem = False, save_data = False):
        """
            evaluate a model w.r.t AIMD
            put a MOVEMENT in /MD and run MD100 
        """
        cwd = os.getcwd()
        os.chdir(work_path)
        if not os.path.exists("MD/MOVEMENT"):
            raise Exception("MD/MOVEMENT not found. It should be an Ab Initio MD result")
        if os.path.exists("MOVEMENT"):
            os.remove("MOVEMENT")        
        import md100    
        md100.run_md100(imodel = 1, atom_type = pm.atomType, num_process = num_thread)
        self.plot_evaluation(plot_elem = plot_elem, save_data = save_data)
        os.chdir(cwd)

    def plot_evaluation(self, plot_elem, save_data):
        if not os.path.exists("MOVEMENT"):
            raise Exception("MOVEMENT not found. It should be force field MD result")

        # plot_ei = True if pm.fortranFitWeightOfEnergy != 0 else False
        # plot_new(atom_type = pm.atomType, plot_elem = plot_elem, save_data = save_data, plot_ei = plot_ei)
        from pwdata import Config
        dft_image = Config(format="pwmat/movement", data_path="MD/MOVEMENT")
        nn_image  = Config(format="pwmat/movement", data_path="MOVEMENT")
        num_atoms = []
        dft_total_energy = []
        dft_atomic_energy = []
        dft_force = []
        nn_total_energy = []
        nn_atomic_energy = []
        nn_force = []
        for dft, nn in zip(dft_image.images, nn_image.images):
            num_atoms.append(dft.force.shape[0])
            dft_total_energy.append(dft.Ep)
            nn_total_energy.append(nn.Ep)
            dft_atomic_energy.append(dft.atomic_energy)
            nn_atomic_energy.append(nn.atomic_energy)
            dft_force.append(dft.force)
            nn_force.append(nn.force)

        # print infos
        inference_path = os.path.join(os.getcwd(),"plot_data")
        if os.path.exists(inference_path) is False:
            os.makedirs(inference_path)
        write_arrays_to_file(os.path.join(inference_path, "image_atom_nums.txt"), num_atoms)
        write_arrays_to_file(os.path.join(inference_path, "dft_total_energy.txt"), dft_total_energy)
        write_arrays_to_file(os.path.join(inference_path, "inference_total_energy.txt"), nn_total_energy)

        write_arrays_to_file(os.path.join(inference_path, "dft_atomic_energy.txt"), dft_atomic_energy)
        write_arrays_to_file(os.path.join(inference_path, "inference_atomic_energy.txt"), nn_atomic_energy)

        write_arrays_to_file(os.path.join(inference_path, "dft_force.txt"), dft_force)
        write_arrays_to_file(os.path.join(inference_path, "inference_force.txt"), nn_force)

        rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2 = inference_plot(inference_path)
        inference_cout = ""
        inference_cout += "For {} images: \n".format(len(num_atoms))
        inference_cout += "Average RMSE of Etot per atom: {} R2: {}\n".format(rmse_E, e_r2)
        inference_cout += "Average RMSE of Force: {} R2: {}\n".format(rmse_F, f_r2)
        inference_cout += "Average RMSE of Virial per atom: {} R2: {}\n".format(rmse_V, v_r2)
        tmp_log = "\nMore details can be found under the file directory:\n{}\n".format(os.path.join(self.dp_params.file_paths.json_dir, os.path.basename(self.dp_params.file_paths.test_dir)))
        inference_cout += tmp_log
        print(tmp_log)

        with open(os.path.join(inference_path, "inference_summary.txt"), 'w') as wf:
            wf.writelines(inference_cout)


    def extract_force_field(self, name= "myforcefield.ff"):

        from extract_ff import extract_ff
        extract_ff(ff_name = name, model_type = 1, atom_type = pm.atomType)
        
    def run_md(self, init_config = "atom.config", md_details = None, num_thread = 1, follow = False):

        import subprocess 
        from poscar2lammps import idx2mass
        
        # remove existing MOVEMENT file for not 
        if follow == False:
            os.system('rm -f MOVEMENT')     
        
        if md_details is None:
            raise Exception("md detail is missing")
        
        md_detail_line = str(md_details)[1:-1]+"\n"
        
        if os.path.exists(init_config) is not True: 
            raise Exception("initial config for MD is not found")
        
        # preparing md.input 
        idx_tabel = idx2mass()
        mass_type = []
        for idx in pm.atomType:
            if idx in idx_tabel:
                mass_type.append(idx_tabel[idx])
                
        f = open('md.input', 'w')
        f.write(init_config+"\n")

        f.write(md_detail_line) 
        f.write('F\n')
        f.write("1\n")     # imodel=1,2,3.    {1:linear;  2:VV;   3:NN;}
        f.write('1\n')               # interval for MOVEMENT output
        f.write('%d\n' % len(pm.atomType)) 
        
        for i in range(len(pm.atomType)):
            f.write('%d %.3f\n' % (pm.atomType[i], mass_type[i]))
        f.close()    
        
        # creating md.input for main_MD.x 
        command = r'mpirun -n ' + str(num_thread) + r' main_MD.x'
        print (command)
        subprocess.run(command, shell=True) 

"""
if __name__ == "__main__":

    atom_type = [29,8]
    feature_type = [1,2] 

    linReg = linear_regressor(atom_type = atom_type, feature_type = feature_type)
    
    #linReg.generate_data() 
    
    #linReg.train() 

    md_detail = [1,1000,1,500,500]
    
    kfnn_trainer.run_md(md_details = md_detail, num_thread = 12, follow = False) 
"""

