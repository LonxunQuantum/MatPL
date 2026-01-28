import os
import shutil
from utils.file_operation import copy_file
from src.user.gpumd_param import GPUmdParam
from pwdata import Config
from pwdata.calculators.const import elements

# from pwdata.extendedxyz import save_to_extxyz

class GPUMD(object):
    def __init__(self, input_param: GPUmdParam) -> None:
        self.input_param = input_param

    def run_md(self):
        # set work dir 
        self.set_work_files()
        # run work
        cwd = os.getcwd()
        os.chdir(self.input_param.working_dir)
        result = os.system("gpumd")
        if result == 0:
            print("gpumd running successfully!")
        else:
            raise Exception("ERROR! gpumd run error!")
        os.chdir(cwd)
        # collect result

    def set_work_files(self):
        if os.path.exists(self.input_param.working_dir):
            shutil.rmtree(self.input_param.working_dir)
        os.makedirs(self.input_param.working_dir)
        # copy files to work dir
        copy_file(self.input_param.potential_file, 
                  os.path.join(self.input_param.working_dir, os.path.basename(self.input_param.potential_file)))
        
        copy_file(self.input_param.run_in_file, 
                  os.path.join(self.input_param.working_dir, os.path.basename(self.input_param.run_in_file)))
        
        if self.input_param.basis_in_file is not None:
            copy_file(self.input_param.basis_in_file,
                os.path.join(self.input_param.working_dir, os.path.basename(self.input_param.basis_in_file)))

        if self.input_param.kpoints_in_file is not None:
            copy_file(self.input_param.kpoints_in_file,
                os.path.join(self.input_param.working_dir, os.path.basename(self.input_param.kpoints_in_file)))

        # if the md init file is not xyz format, convert it
        self.copy_md_xyz_file(os.path.join(self.input_param.working_dir, "model.xyz"))

    def copy_md_xyz_file(self, target_file:str):
        if "xyz" in os.path.basename(self.input_param.md_init_config_file).lower()\
            or "xyz" in self.input_param.md_init_config_format.lower():
            copy_file(self.input_param.md_init_config_file, target_file)
        else:
            config = Config.read(
                format=self.input_param.md_init_config_format, 
                data_path=self.input_param.md_init_config_file
                )
            self.save_to_extxyz(image_data_all = [config], 
                            output_path = os.path.dirname(target_file), 
                            data_name = os.path.basename(target_file), 
                            write_patthen='w')

    def save_to_extxyz(self, image_data_all: list, output_path: str, data_name: str, write_patthen='w'):
        data_name = open(os.path.join(output_path, data_name), write_patthen)
        for i in range(len(image_data_all)):
            image_data = image_data_all[i]
            if not image_data.cartesian:
                image_data._set_cartesian()
            data_name.write("%d\n" % image_data.atom_nums)
            # data_name.write("Iteration: %s\n" % image_data.iteration)
            if image_data.Ep is not None:
                output_head = 'Lattice="%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" Properties=species:S:1:pos:R:3:force:R:3:local_energy:R:1 pbc="T T T" energy={}\n'.format(image_data.Ep)
                output_extended = (image_data.lattice[0][0], image_data.lattice[0][1], image_data.lattice[0][2], 
                                        image_data.lattice[1][0], image_data.lattice[1][1], image_data.lattice[1][2], 
                                        image_data.lattice[2][0], image_data.lattice[2][1], image_data.lattice[2][2])
            else:
                output_head = 'Lattice="%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f" Properties=species:S:1:pos:R:3 pbc="T T T"\n'
                output_extended = (image_data.lattice[0][0], image_data.lattice[0][1], image_data.lattice[0][2], 
                                        image_data.lattice[1][0], image_data.lattice[1][1], image_data.lattice[1][2], 
                                        image_data.lattice[2][0], image_data.lattice[2][1], image_data.lattice[2][2])
            data_name.write(output_head % output_extended)

            for j in range(image_data.atom_nums):
                if image_data.Ep is not None:
                    properties_format = "%s %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f %14.8f\n"
                    properties = (elements[image_data.atom_types_image[j]], image_data.position[j][0], image_data.position[j][1], image_data.position[j][2], 
                                    image_data.force[j][0], image_data.force[j][1], image_data.force[j][2], 
                                    image_data.atomic_energy[j])
                else:
                    properties_format = "%s %14.8f %14.8f %14.8f\n"
                    properties = (elements[image_data.atom_types_image[j]], image_data.position[j][0], image_data.position[j][1], image_data.position[j][2])
                data_name.write(properties_format % properties)
        data_name.close()
        print("Convert to %s successfully!" % data_name)
        
        # config.to(output_path=os.path.dirname(target_file), 
        #         data_name  =os.path.basename(target_file),
        #         save_format="xyz",
        #         direct     =True, 
        #         sort       =True, 
        #         wrap       =False
        #         )
    
        # if "config".upper() in self.input_param.md_init_config_file.upper():
        #     # atom.config to xyz format
        #     atomconfig2xyz(source_file, target_file)

        # elif "outcar".upper() in self.input_param.md_init_config_file.upper():
        #     POSCAR_OUTCAR2xyz(source_file, target_file, "OUTCAR")

        # elif "poscar".upper() in self.input_param.md_init_config_file.upper():
        #     POSCAR_OUTCAR2xyz(source_file, target_file, "POSCAR")

        # elif "xyz".upper() in self.input_param.md_init_config_file.upper():
        #     # copy xyz file
        #     copy_file(source_file, target_file)

        