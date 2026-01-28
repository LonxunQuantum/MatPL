#!/usr/bin/env python3
import sys
import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt

def read_dft_movement(file_dft=r'MD/MOVEMENT'):
    f = open(file_dft, 'r')
    txt = f.readlines()
    f.close()
    num_atoms = int(txt[0].split()[0])
    num_iters = 0
    ep = []
    e_dE = []
    atomic_energy = []
    f_atom = []
    for idx, line in enumerate(txt):
        if 'Iteration' in line:
            ep.append(float(line.split('Etot,Ep,Ek (eV) =')[1].split()[1]))
            num_iters += 1
    
        if 'Atomic-Energy' in line:
            e_dE.append(float(line.split('Q_atom:dE(eV)=')[1].split()[0]))
            for i in range(num_atoms):
                atomic_energy.append(float(txt[idx+1+i].split()[1]))

        if 'Force' in line:
            for i in range(num_atoms):
                tmp = txt[idx+1+i].split()
                f_atom.append([-float(tmp[1]), -float(tmp[2]), -float(tmp[3])])

    return num_atoms, num_iters, np.array(atomic_energy), np.array(f_atom)
    
def read_nn_movement(file_nn='MOVEMENT'):
    f = open(file_nn, 'r')
    txt = f.readlines()
    f.close()
        
    num_atoms = int(txt[0].split()[0])
    print (num_atoms)
    atomic_energy = []
    f_atom = []
    for idx, line in enumerate(txt):
        if 'POSITION' in line:
            for i in range(num_atoms):
                atomic_energy.append(float(txt[idx+1+i].split()[7]))
                fx = (float(txt[idx+1+i].split()[8]))
                fy = (float(txt[idx+1+i].split()[9]))
                fz = (float(txt[idx+1+i].split()[10]))
                f_atom.append([fx,fy,fz])

    return np.array(atomic_energy), np.array(f_atom)

    
if __name__ == '__main__':
    num_atoms, num_iters, dft_atomic_energy, dft_force = read_dft_movement()
    tmp_atoms, tmp_iters, nn_atomic_energy,  nn_force  = read_dft_movement(file_dft='MOVEMENT') 
    #nn_atomic_energy, nn_force = read_nn_movement()
    #print (dft_atomic_energy)
    #print (nn_atomic_energy)
    e_min = min(min(dft_atomic_energy), min(nn_atomic_energy))
    e_max = max(max(dft_atomic_energy), max(nn_atomic_energy))
    lim_min = e_min - (e_max - e_min) * 0.1
    lim_max = e_max + (e_max - e_min) * 0.1

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


    # plot atomic energy
    plt.title('atomic energy')
    plt.scatter(dft_atomic_energy, nn_atomic_energy, s=0.1)
    plt.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red')
    plt.xlim(lim_min, lim_max)
    plt.ylim(lim_min, lim_max)
    plt.xlabel('DFT energy (eV)')
    plt.ylabel('MLFF energy (eV)')
    plt.text(lim_min, lim_max-(lim_max-lim_min)*0.1,'Ei, rmse: %.3E' % (e_atomic_rms))
    # save and show
    plt.savefig('atomic_energy.png', format='png')
    #if len(sys.argv) < 2:
    #    plt.show()

    # plot total energy
    #plt.clf()
    #plt.subplot(1,2,1)
    plt.title('total energy')
    plt.scatter(dft_total_energy, nn_total_energy, s=0.1)
    plt.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='red')
    plt.xlim(lim_tot_min, lim_tot_max)
    plt.ylim(lim_tot_min, lim_tot_max)
    plt.text(e_tot_min, e_tot_max,'Etot, rmse: %.3E' %(e_tot_rms))
    plt.text(e_tot_min, e_tot_max - (e_tot_max - e_tot_min)*0.1,'Etot/Natom, rmse: %.3E' %(e_tot_rms/num_atoms))
    plt.xlabel('DFT energy (eV)')
    plt.ylabel('MLFF energy (eV)')
    plt.savefig('total_energy.png', format='png')

    # plot force
    #plt.subplot(1,2,2)
    plt.title('force')
    plt.xlim(lim_f_min, lim_f_max)
    plt.ylim(lim_f_min, lim_f_max)
    plt.text(fmin, fmax,'Force, rmse: %.3E' %(f_rms))
    plt.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='red')
    plt.scatter(f_dft_plot, f_nn_plot, s=0.1)
    plt.ylabel('MLFF Force')
    plt.xlabel('DFT Force')
    plt.savefig('force.png', format='png')
    #if len(sys.argv) < 2:
    #    plt.show()
