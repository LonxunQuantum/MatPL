#!/usr/bin/env python3
import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
import sys

class PWmatImage():
    def __init__(self, num_atoms, lattice, type_atom, x_atom, f_atom, e_atom, ddde):
        self.num_atoms = num_atoms
        self.lattice = lattice
        self.type_atom = type_atom
        self.x_atom = x_atom
        self.f_atom = f_atom
        self.e_atom = e_atom
        self.egroup = np.zeros((num_atoms), dtype=float)
        self.ddde = ddde
        self.dRRR = np.zeros((self.num_atoms, self.num_atoms, 3), dtype=float)
        self.distance_matrix = np.zeros((self.num_atoms, self.num_atoms), dtype=float)
    def calc_distance(self):
        for i in range(self.num_atoms):
            d0 = self.x_atom - self.x_atom[i]
            d1 = np.where(d0<-0.5, d0+1.0, d0)
            dd = np.where(d1>0.5, d1-1.0, d1)
            d_cart = np.matmul(dd, self.lattice)
            self.dRRR[i] = d_cart
            self.distance_matrix[i] = np.array([ LA.norm(kk) for kk in d_cart])
        print(self.dRRR.shape)
        print(self.distance_matrix.shape)

    def find_neighbor(self, rcut=5.0):
        self.calc_distance()
        neighbor_list = [ [] for i in range(self.num_atoms)]
        neighbor_dR = [ [] for i in range(self.num_atoms)]
        for i in range(self.num_atoms):
            # np.where(self.distance_matrix[i] < rcut, 1, 0)  is neighbor?
            # np.where(xxx == 1) neighbor list get.
            # remove(i)       distance[i][i] equals 0, remove itself
            neighbor_list[i] = np.where(np.where(self.distance_matrix[i] < rcut, 1, 0) == 1)[0].tolist()
            neighbor_list[i].remove(i)
            neighbor_dR[i] = self.dRRR[i][neighbor_list[i]]
        return neighbor_list, neighbor_dR


# read all images in a MOVEMENT file
# create one image from text section of MOVEMENT, start from " num atoms,Iteration = ..."
# end before next " num atoms,Iteration = ..."
def read_image(text):
    num_atoms = int(text[0].split()[0])
    lattice = np.zeros((3,3), dtype=float)
    type_atom = np.zeros((num_atoms), dtype=int)
    x_atom = np.zeros((num_atoms,3), dtype=float)
    f_atom = np.zeros((num_atoms,3), dtype=float)
    e_atom = np.zeros((num_atoms), dtype=float)
    egroup = np.zeros((num_atoms), dtype=float)
    ddde = 0.0
    for idx, line in enumerate(text):
        if 'Lattice' in line:
            lattice[0] = np.array([float(kk) for kk in text[idx+1].split()])
            lattice[1] = np.array([float(kk) for kk in text[idx+2].split()])
            lattice[2] = np.array([float(kk) for kk in text[idx+3].split()])

        if 'Position' in line:
            for i in range(num_atoms):
                tmp = text[idx+1+i].split()
                type_atom[i] = int(tmp[0])
                x_atom[i] = np.array([float(tmp[1]), float(tmp[2]), float(tmp[3])])

        if 'Atomic-Energy' in line:
            ddde = float(line.split('Q_atom:dE(eV)=')[1].split()[0])
            for i in range(num_atoms):
                e_atom[i] = float(text[idx+1+i].split()[1])

        if 'Force' in line:
            for i in range(num_atoms):
                tmp = text[idx+1+i].split()
                f_atom[i] = np.array([-float(tmp[1]), -float(tmp[2]), -float(tmp[3])])
    return PWmatImage(num_atoms, lattice, type_atom, x_atom, f_atom, e_atom, ddde)
        
    
# example
# return almost all the informations in MOVEMENT
def read_dft_movement(file_dft=r'PWdata/MOVEMENT', output_dir=r'PWdata', rcut=5.0):
    f = open(file_dft, 'r')
    txt = f.readlines()
    f.close()

    num_iters = 0   # num_images
    natom_all_images = []
    ep_all_images = []
    dE_all_images = []
    atomic_energy_all_images = []
    xatom_all_images = []
    force_all_images = []
    image_starts = []
    neighbor_list_all_images = []
    neighbor_dR_all_images = []
    for idx, line in enumerate(txt):
        if 'Iteration' in line:
            natom_all_images.append(int(line.split()[0]))
            ep_all_images.append(float(line.split('Etot,Ep,Ek (eV) =')[1].split()[1]))
            num_iters += 1
            image_starts.append(idx)
    image_starts.append(len(txt))
    
    for i in range(num_iters):
        print('iter: %d' % i)
        image = read_image(txt[image_starts[i]:image_starts[i+1]])

        dE_all_images.append(image.ddde)  # ddde is a float number
        neighbor_list, neighbor_dR = image.find_neighbor(rcut=5.0)
        neighbor_list_all_images += neighbor_list
        neighbor_dR_all_images += neighbor_dR
        atomic_energy_all_images += image.e_atom.tolist()
        xatom_all_images += image.x_atom.tolist()
        force_all_images += image.f_atom.tolist()

    # shapes
    # natom.npy             (num_images)
    # atomic_energy.npy     (num_images, natoms)
    # xatom.npy             (num_images, natoms, 3)
    # force.npy             (num_images, natoms, 3)
    # etot.npy              (num_images)
    # neighbor_list.npy     (num_images, natoms, nneighbors[index_nimage][index_natom])
    # neighbor_dR.npy       (num_images, natoms, nneighbors[index_nimage][index_natom], 3)
    np.save(output_dir+r'/natom.npy', np.array(natom_all_images))
    np.save(output_dir+r'/atomic_energy.npy', np.array(atomic_energy_all_images))
    np.save(output_dir+r'/xatom.npy', np.array(xatom_all_images))
    np.save(output_dir+r'/force.npy', np.array(force_all_images))
    np.save(output_dir+r'/etot.npy', np.array(ep_all_images))
    np.save(output_dir+r'/neighbor_list.npy', np.array(neighbor_list_all_images))
    np.save(output_dir+r'/neighbor_dR.npy', np.array(neighbor_dR_all_images))
    return num_iters, np.array(natom_all_images), np.array(atomic_energy_all_images), np.array(force_all_images), np.array(ep_all_images)



    
if __name__ == '__main__':
    read_dft_movement(file_dft=r'PWdata/data1/MOVEMENT', output_dir=r'PWdata', rcut=3.0)
    exit()
