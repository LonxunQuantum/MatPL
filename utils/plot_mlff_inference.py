#!/usr/bin/env python3
import sys
import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt

class Image():
    def __init__(self, num_atoms, lattice, type_atom, x_atom, f_atom, e_atom, ddde):
        self.num_atoms = num_atoms
        self.lattice = lattice
        self.type_atom = type_atom
        self.x_atom = x_atom
        self.f_atom = f_atom
        self.e_atom = e_atom
        self.egroup = np.zeros((num_atoms), dtype=float)
        self.ddde = ddde

    def calc_egroup(self):
        # to line 40, get e_atom0
        f = open(r'fread_dfeat/feat.info', 'r')
        txt = f.readlines()
        f.close()
        iflag_pca = int(txt[0].split()[0])
        num_feat_type = int(txt[1].split()[0])
        for i in range(num_feat_type):
            ifeat_type = int(txt[2+i].split()[0])
        num_atomtype = int(txt[2+num_feat_type].split()[0])
        itype_atom = np.zeros((num_atomtype), dtype=int)
        nfeat1 = np.zeros((num_atomtype), dtype=int)
        nfeat2 = np.zeros((num_atomtype), dtype=int)
        nfeat2_integral = np.zeros((num_atomtype), dtype=int)
        nfeat = np.zeros((num_feat_type, num_atomtype), dtype=int)
        ipos_feat = np.zeros((num_feat_type, num_atomtype), dtype=int)
        for i in range(num_atomtype):
            tmp = [int(kk) for kk in txt[3+num_feat_type+i].split()]
            itype_atom[i] = tmp[0]
            nfeat1[i] = tmp[1]
            nfeat2[i] = tmp[2]
        for i in range(num_atomtype):
            nfeat2_integral[i] = np.sum(nfeat2[0:i+1])

        # read fit_linearMM.input
        f = open(r'fread_dfeat/fit_linearMM.input', 'r')
        txt = f.readlines()
        f.close()
        tmp_ntype = int(txt[0].split()[0].split(',')[0])
        tmp_m_neigh = int(txt[0].split()[1].split(',')[0])
        type_map = [ 0 for i in range(tmp_ntype)]
        for i in range(tmp_ntype):
            type_map[i] = int(txt[1+i].split()[0].split(',')[0])

        dwidth = float(txt[tmp_ntype+2].split()[0])

        # read linear_fitB.ntype
        f = open(r'fread_dfeat/linear_fitB.ntype', 'r')
        txt = f.readlines()
        f.close()
        e_atom0 = np.zeros((num_atomtype), dtype=float)
        for i in range(num_atomtype):
            e_atom0[i] = float(txt[nfeat2_integral[i]].split()[1])

        # calc distance
        # num_atoms, x_atom, lattice
        distance_matrix = np.zeros((self.num_atoms, self.num_atoms), dtype=float)
        for i in range(self.num_atoms):
            d0 = self.x_atom - self.x_atom[i]
            d1 = np.where(d0<-0.5, d0+1.0, d0)
            dd = np.where(d1>0.5, d1-1.0, d1)
            d_cart = np.matmul(dd, self.lattice)
            distance_matrix[i] = np.array([ LA.norm(kk) for kk in d_cart])

        #print(distance_matrix[0]*distance_matrix[0])
        fact_matrix = np.exp(-distance_matrix**2/dwidth**2)
        e_atom0_array = np.zeros((self.num_atoms), dtype=float)
        for i in range(self.num_atoms):
            e_atom0_array[i] = e_atom0[type_map.index(self.type_atom[i])]

        for i in range(self.num_atoms):
            esum1 = ((self.e_atom - e_atom0_array)*fact_matrix[i]).sum()
            self.egroup[i] = esum1 / fact_matrix[i].sum()
        #sys.exit(0)
        #print(fact_matrix)
        #print(self.egroup)

def read_image(text):
    num_atoms = int(text[0].split()[0])
    lattice = np.zeros((3,3), dtype=float)
    type_atom = np.zeros((num_atoms), dtype=int)
    x_atom = np.zeros((num_atoms,3), dtype=float)
    f_atom = np.zeros((num_atoms,3), dtype=float)
    e_atom = np.zeros((num_atoms), dtype=float)
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
    return Image(num_atoms, lattice, type_atom, x_atom, f_atom, e_atom, ddde)

        
    
def read_dft_movement(file_dft=r'MD/MOVEMENT'):
    f = open(file_dft, 'r')
    txt = f.readlines()
    f.close()
    num_atoms = int(txt[0].split()[0])
    num_iters = 0
    ep = []
    e_dE = []
    atomic_energy = []
    group_energy = []
    f_appended = []
    image_starts = []
    for idx, line in enumerate(txt):
        if 'Iteration' in line:
            ep.append(float(line.split('Etot,Ep,Ek (eV) =')[1].split()[1]))
            num_iters += 1
            image_starts.append(idx)
    image_starts.append(len(txt))
    
    print('reading image')
    for i in range(num_iters):
        if i % 50 == 0:
            print('reading image %d' % i)
        image = read_image(txt[image_starts[i]:image_starts[i+1]])
        image.calc_egroup()
        e_dE.append(image.ddde)  # ddde is a float number
        group_energy += image.egroup.tolist()  # combine lists
        atomic_energy += image.e_atom.tolist()
        f_appended += image.f_atom.tolist()
    return num_atoms, num_iters, np.array(atomic_energy), np.array(f_appended), np.array(group_energy)
    
def read_nn_movement(file_nn=r'MD/md/MOVEMENT'):
    f = open(file_nn, 'r')
    txt = f.readlines()
    f.close()
    
    num_atoms = int(txt[0].split()[0])
    atomic_energy = []
    group_energy = []
    f_appended = []
    for idx, line in enumerate(txt):
        if 'POSITION' in line:
            for i in range(num_atoms):
                tmp = txt[idx+1+i].split()
                atomic_energy.append(float(tmp[7]))
                fx = (float(tmp[8]))
                fy = (float(tmp[9]))
                fz = (float(tmp[10]))
                if len(tmp) > 11:
                    group_energy.append(float(tmp[11]))
                else:
                    group_energy.append(0.0)
                f_appended.append([fx,fy,fz])

    return np.array(atomic_energy), np.array(f_appended), np.array(group_energy)

    
if __name__ == '__main__':
    num_atoms, num_iters, dft_atomic_energy, dft_force, dft_group_energy = read_dft_movement()
    tmp_natom, tmp_niters, nn_atomic_energy, nn_force, nn_group_energy = read_dft_movement(file_dft='MOVEMENT')

    #nn_atomic_energy, nn_force, nn_group_energy = read_nn_movement()
    #print(dft_group_energy[0:100])

    e_min = min(min(dft_atomic_energy), min(nn_atomic_energy))
    e_max = max(max(dft_atomic_energy), max(nn_atomic_energy))
    lim_min = e_min - (e_max - e_min) * 0.1
    lim_max = e_max + (e_max - e_min) * 0.1

    egroup_min = min(min(dft_group_energy), min(nn_group_energy))
    egroup_max = max(max(dft_group_energy), max(nn_group_energy))
    lim_group_min = egroup_min - (egroup_max - egroup_min) * 0.1
    lim_group_max = egroup_max + (egroup_max - egroup_min) * 0.1

    dft_total_energy = np.zeros(num_iters, dtype=float)
    nn_total_energy = np.zeros(num_iters, dtype=float)
    for i in range(num_iters):
        dft_total_energy[i] = dft_atomic_energy[i*num_atoms:(i+1)*num_atoms].sum()
        nn_total_energy[i] = nn_atomic_energy[i*num_atoms:(i+1)*num_atoms].sum()

    e_tot_min = min(min(dft_total_energy), min(nn_total_energy))
    e_tot_max = max(max(dft_total_energy), max(nn_total_energy))
    lim_tot_min = e_tot_min - (e_tot_max - e_tot_min) * 0.1
    lim_tot_max = e_tot_max + (e_tot_max - e_tot_min) * 0.1


    e_atomic_rms = LA.norm(dft_atomic_energy - nn_atomic_energy) / np.sqrt(len(dft_atomic_energy))
    print('E_atomic, e_rmse: %.3E' % e_atomic_rms)
    e_group_rms = LA.norm(dft_group_energy - nn_group_energy) / np.sqrt(len(dft_group_energy))
    print('E_group, e_rmse: %.3E' % e_group_rms)
    e_tot_rms = LA.norm(dft_total_energy - nn_total_energy) / np.sqrt(len(dft_total_energy))
    print('E_tot, e_rmse: %.3E' % e_tot_rms)
    print('E_tot/N_atom, e_rmse: %.3E' % (e_tot_rms/num_atoms))
    # calculate force
    f_dft_plot = dft_force.reshape(3*num_atoms*num_iters)
    f_nn_plot = nn_force.reshape(3*num_atoms*num_iters)
    fmax = max(f_dft_plot.max(), f_nn_plot.max())
    fmin = min(f_dft_plot.min(), f_nn_plot.min())
    lim_f_min = fmin - (fmax-fmin)*0.1
    lim_f_max = fmax + (fmax-fmin)*0.1

    f_rms = LA.norm(f_dft_plot - f_nn_plot) / np.sqrt(3*num_atoms*num_iters)
    print('f_rmse: %.3E' % f_rms)

    print('#'*10)
    etot_mse = e_tot_rms**2
    force_mse = f_rms**2
    print('Energy L2err: %.3E' % etot_mse)
    print('Energy L2err/Natoms: %.3E' % (etot_mse / num_atoms))
    print('Force L2err: %.3E' % force_mse)

    # plot atomic energy
    #plt.subplot(2,2,1)
    plt.title('atomic energy',fontsize=12)
    plt.scatter(dft_atomic_energy, nn_atomic_energy, s=0.1)
    plt.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.xlabel('DFT atomic energy (eV)',fontsize = 12)
    plt.ylabel('MLFF atomic energy (eV)',fontsize = 12)
    plt.text(lim_min, lim_max-(lim_max-lim_min)*0.1,'rmse: %.3E' % (e_atomic_rms),fontsize = 12)
    plt.savefig('atomic_energy.png', format='png')

    # save and show
    #plt.savefig('dft_nn.png', format='png')
    #plt.xticks(fontsize = 8) 
    #plt.yticks(fontsize = 8)
    # plot group energy
    #plt.subplot(2,2,2)
    plt.title('group energy',fontsize=12)
    plt.scatter(dft_group_energy, nn_group_energy, s=0.1)
    plt.plot((lim_group_min, lim_group_max), (lim_group_min, lim_group_max), ls='--', color='red')
    plt.xlim(lim_group_min, lim_group_max)
    plt.ylim(lim_group_min, lim_group_max)
    plt.xlabel('DFT group energy (eV)',fontsize=12)
    plt.ylabel('MLFF group energy (eV)',fontsize=12)
    plt.text(lim_min, lim_max-(lim_max-lim_min)*0.1,'rmse: %.3E' % (e_group_rms),fontsize = 12)
    #plt.savefig('dft_nn.png', format='png')
    plt.xticks(fontsize = 8) 
    plt.yticks(fontsize = 8)
    plt.savefig('group_energy.png', format='png')

    # plot total energy
        
    #plt.clf()
    #plt.subplot(2,2,3)
    plt.title('total energy',fontsize = 12)
    plt.scatter(dft_total_energy, nn_total_energy, s=0.1)
    plt.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='red')
    plt.xlim(lim_tot_min, lim_tot_max)
    plt.ylim(lim_tot_min, lim_tot_max)
    plt.text(e_tot_min, e_tot_max,'rmse: %.3E' %(e_tot_rms),fontsize = 12)
    plt.text(e_tot_min, e_tot_max - (e_tot_max - e_tot_min)*0.1,'rmse/natom: %.3E' %(e_tot_rms/num_atoms),fontsize = 12)
    plt.xlabel('DFT energy (eV)',fontsize = 12)
    plt.ylabel('MLFF energy (eV)', fontsize = 12)
    plt.xticks(fontsize = 8) 
    plt.yticks(fontsize = 8)
    plt.savefig('total_energy.png', format='png')

    # plot force
    plt.title('force',fontsize=12)
    plt.xlim(lim_f_min, lim_f_max)
    plt.ylim(lim_f_min, lim_f_max)
    plt.text(fmin, fmax,'rmse: %.3E' %(f_rms),fontsize = 12)
    plt.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='red')
    plt.scatter(f_dft_plot, f_nn_plot, s=0.1)
    plt.ylabel('MLFF Force(ev/A)',fontsize = 12)
    plt.xlabel('DFT Force(ev/A)',fontsize = 12)
    # save and show
    plt.xticks(fontsize = 8) 
    plt.yticks(fontsize = 8)
    plt.savefig('force.png', format='png')
    
    #plt.subplots_adjust(wspace = 0.4, hspace =0.4)
    #plt.savefig('MLFF_inference.png', format='png')
    #if len(sys.argv) < 2:
    #    plt.show()
