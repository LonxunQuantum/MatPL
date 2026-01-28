#!/usr/bin/env python3
# image class, again, hahaha
# all forces in python variables are correct one f_atom = -dE/dR
# please add minus sign when you read and write MOVEMENT
import numpy as np
import numpy.linalg as LA
import dpdata
import argparse
import os

element = ["0", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg"]

class Image():
    def __init__(self, num_atoms=None, lattice=None, type_atom=None, x_atom=None, f_atom=None, e_atom=None, ddde=None, e_potential=None, virial=None):
        self.num_atoms = num_atoms
        self.lattice = lattice
        self.type_atom = type_atom
        self.x_atom = x_atom
        self.f_atom = f_atom
        self.e_atom = e_atom
        self.e_potential = e_potential
        self.ddde = ddde
        self.virial = virial

        self.egroup = np.zeros((num_atoms), dtype=float)
        
def write_image(fout, image:Image):
    fout.write(" %d atoms, Iteration (fs) = %16.10E, Etot,Ep,Ek (eV) = %16.10E  %16.10E   %16.10E, SCF = 1\n"\
                % (image.num_atoms, 0.0, image.e_potential, image.e_potential, 0.0))
    fout.write(" MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K) \n")
    fout.write("          1    0.5000000000E+00   0.59978E+03   0.30000E+03   0.59978E+03   0.50000E+02   0.59978E+03\n")
    fout.write(" MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)\n")
    fout.write("          -0.1971547257E+05\n")
    fout.write("Lattice vector (Angstrom)\n")
    for i in range(3):
        if image.virial is None:
            fout.write("  %16.10E    %16.10E    %16.10E\n" % (image.lattice[i][0], image.lattice[i][1], image.lattice[i][2]))
        else:
            fout.write("  %16.10E    %16.10E    %16.10E     stress (eV): %16.10E    %16.10E    %16.10E\n" % \
                       (image.lattice[i][0], image.lattice[i][1], image.lattice[i][2], image.virial[i][0], image.virial[i][1], image.virial[i][2]))
    fout.write("  Position (normalized), move_x, move_y, move_z\n")
    for i in range(image.num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F    1 1 1\n"\
                 % (image.type_atom[i], image.x_atom[i][0], image.x_atom[i][1], image.x_atom[i][2]))
    fout.write("  Force (-force, eV/Angstrom)\n")
    for i in range(image.num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                 % (image.type_atom[i], -image.f_atom[i][0], -image.f_atom[i][1], -image.f_atom[i][2]))  # minus sign here
    # fout.write("  Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  %20.15E\n " % image.ddde)
    # for i in range(image.num_atoms):
    #     fout.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
    #              % (image.type_atom[i], image.e_atom[i], 0.0, 0.0))
    fout.write(' -------------------------------------\n')
    # idtk = i don't know


# def outcar2raw():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='OUTCAR')
#     parser.add_argument('-n', '--number', help='specify number of samples per set', type=int, default=2000)
#     parser.add_argument('-d', '--directory', help='specify stored directory of raw data', type=str, default='.')
#     args = parser.parse_args()

#     dpdata.LabeledSystem(args.input, fmt='vasp/outcar').to('deepmd/raw', args.directory)

# def movement2raw():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-i', '--input', help='specify input movement filename', type=str, default='MOVEMENT')
#     parser.add_argument('-n', '--number', help='specify number of samples per set', type=int, default=2000)
#     parser.add_argument('-d', '--directory', help='specify stored directory of raw data', type=str, default='.')
#     args = parser.parse_args()

#     dpdata.LabeledSystem(args.input, fmt='pwmat/movement').to('deepmd/raw', args.directory)

def dpdata2movement():
    # "/data/home/wuxingxing/datas/pwmat_mlff_workdir/dpkit/datasets/iter.000000/02.fp/data.000/"
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--type', help='specify type of dpdata: raw or npy', type=str, default='raw')
    # parser.add_argument('-s', '--save-name', help='specify movement save name', type=str, default="movement_res")
    # parser.add_argument('-d', '--directory', help='specify directory of raw data or npy data', type=str, default='.')
    # args = parser.parse_args()

    # save_file = args.save_name
    # work_type = args.type.upper()
    # os.chdir(args.directory)

    save_file = "movement_test"
    work_type = "raw".upper()
    os.chdir("/data/home/wuxingxing/datas/pwmat_mlff_workdir/dpkit/datasets/iter.000000/02.fp/data.000/")

    print("work dir: ", os.getcwd())
    if work_type == "raw".upper():
        type_raw = np.loadtxt(r'type.raw', dtype=int)
        fin = open(r'type_map.raw')
        box_raw = np.loadtxt(r'box.raw')
        coord_raw = np.loadtxt(r'coord.raw')
        energy_raw = np.loadtxt(r'energy.raw')
        force_raw = np.loadtxt(r'force.raw')
        if os.path.exists("virial.raw"):
            virial_raw = np.loadtxt(r'virial.raw')
    else:
        type_raw = np.loadtxt(r'../type.raw', dtype=int)
        fin = open(r'../type_map.raw')
        box_raw = np.load(r'box.npy')
        coord_raw = np.load(r'coord.npy')
        energy_raw = np.load(r'energy.npy')
        force_raw = np.load(r'force.npy')
        if os.path.exists("virial.npy"):
            virial_raw = np.load(r'virial.npy')


    type_map_txt = fin.readlines()
    fin.close()

    print ("raw data reading completed\n")	
    num_type = len(type_map_txt)
    num_atom = type_raw.shape[0]
    num_image = coord_raw.shape[0]
    type_map = [ 'H' for tmp in range(num_type)]
    type_atom = np.zeros((num_atom), dtype=int)

    for i in range(num_type):
        type_map[i] = type_map_txt[i].split()[0]
    for i in range(num_atom):
        type_atom[i] = element.index(type_map[type_raw[i]])
    print(type_map)
    print(list(dict.fromkeys(type_atom)))

    all_images = []
    for i in range(num_image):
        lattice = box_raw[i].reshape(3,3)
        virial = virial_raw[i].reshape(3,3)
        x_atom = np.dot(coord_raw[i].reshape(num_atom,3),LA.inv(lattice))
        f_atom = force_raw[i].reshape(num_atom,3)
        tmp_eatom = energy_raw[i] / num_atom
        e_atom = np.array([tmp_eatom for i in range(num_atom)])
        ddde = 0.0
        e_potential = energy_raw[i]
        tmp_image = Image(num_atom, lattice, type_atom, x_atom, f_atom, e_atom, ddde, e_potential, virial)
        all_images.append(tmp_image)

    fout = open(save_file, 'w')
    for i in range(num_image):
        write_image(fout, all_images[i])
    fout.close()

if __name__ == '__main__':
    dpdata2movement()