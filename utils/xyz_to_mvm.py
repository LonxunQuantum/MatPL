#!/usr/bin/env python3
import sys
import numpy as np
import ase
from ase.io import extxyz

def read_one_xyz(atoms):
    num_atoms = len(atoms.get_atomic_numbers())
    f_atom = atoms.get_forces()
    lattice = atoms.get_cell()
    pos = atoms.get_scaled_positions() # for normalized position
    type_atoms = atoms.numbers
    e_atom = np.zeros((num_atoms), dtype=float)
    ene = atoms.get_potential_energy()
    stress = atoms.get_stress(voigt=False)
    return num_atoms, ene, f_atom, lattice, pos, type_atoms, stress

def write_image(fout, num_atoms, type_atoms,iterid, dftb_energy, pos, lattice, f_atom , stress):
    fout.write(" %d atoms, Iteration %d (fs) = %16.10E, Etot,Ep,Ek (eV) = %16.10E  %16.10E   %16.10E\n"\
                % (num_atoms, iterid, 0.0, dftb_energy, dftb_energy, 0.0))
    fout.write(" MD_INFO: METHOD(1-VV,2-NH,3-LV,4-LVPR,5-NHRP) TIME(fs) TEMP(K) DESIRED_TEMP(K) AVE_TEMP(K) TIME_INTERVAL(fs) TOT_TEMP(K) \n")
    fout.write("          1    0.5000000000E+00   0.59978E+03   0.30000E+03   0.59978E+03   0.50000E+02   0.59978E+03\n")
    fout.write(" MD_VV_INFO: Basic Velocity Verlet Dynamics (NVE), Initialized total energy(Hartree)\n")
    fout.write("          -0.1971547257E+05\n")
    fout.write("Lattice vector (Angstrom)\n")
    for i in range(3):
        fout.write("  %16.10E    %16.10E    %16.10E stress (eV): %16.10E    %16.10E    %16.10E  \n" % (lattice[i][0], lattice[i][1], lattice[i][2], stress[i][0], stress[i][1], stress[i][2]))
    fout.write("  Position (normalized), move_x, move_y, move_z\n")
    for i in range(num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F    1 1 1\n"\
                 % (type_atoms[i], pos[i][0], pos[i][1], pos[i][2]))
    fout.write("  Force (-force, eV/Angstrom)\n")
    for i in range(num_atoms):
        fout.write(" %4d    %20.15F    %20.15F    %20.15F\n"\
                 % (type_atoms[i], f_atom[i][0], f_atom[i][1], f_atom[i][2]))  # minus sign here
    fout.write(' -------------------------------------\n')
    # idtk = i don't know

def convert_extxyz2movement(infile, outfile):
    xyzfile = open(infile, 'r')
    configurations = extxyz.read_extxyz(xyzfile, index=slice(0,200))
    num_iters = 1
    ep = []
    f_appended=[]

    fout = open(outfile, 'w')
    for atoms in configurations:
        num_atoms, dftb_energy, f_atom, lattice, pos, type_atoms, stress= read_one_xyz(atoms)
        write_image(fout, num_atoms, type_atoms, num_iters, dftb_energy, pos, lattice, f_atom, stress)
        num_iters = num_iters + 1
    fout.close()
    xyzfile.close()


if __name__ == '__main__':
    import os
    os.chdir("/data/home/wuxingxing/datas/PWMLFF_library_data/nep-data/2023_Wang_Si_GAP2018_better_force")
    convert_extxyz2movement("train.xyz", "si_MOVEMENT")

