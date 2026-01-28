#!/usr/bin/env python3
from codecs import raw_unicode_escape_decode
import sys
import numpy as np
import numpy.linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def read_dft_movement(file_dft=r'MD/MOVEMENT', atom_type=[22,8]):
    f = open(file_dft, 'r')
    txt = f.readlines()
    f.close()
    num_atoms = int(txt[0].split()[0])
    num_iters = 0
    ep = []
    e_dE = []
    atomic_energy = []
    f_atom = []
    atomic_energy_per_elem = {atype: [] for atype in atom_type}
    force_per_elem = {atype: [] for atype in atom_type}
    is_atomic = False
    order = [
    1, 3, 4, 5, 6, 7, 8, 9, 
    11, 12, 13, 14, 15, 16, 17, 
    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 
    55, 56, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83
    ]

    atomic_E = [
    -45.140551665, -210.0485218888889, -321.1987119, -146.63024691666666, -399.0110205833333, -502.070125, -879.0771215, -1091.0652775, 
    -1275.295054, -2131.9724644444445, -2412.581311, -787.3439924999999, -1215.4995769047619, -1705.5754946875, -557.9141695, 
    -1544.3553605, -1105.0024515, -1420.574128, -1970.9374273333333, -2274.598644, -2331.976294, -2762.3960913793107, -3298.6401545, -3637.624857, -4140.3502, -5133.970898611111, -5498.13054, -2073.70436625, -2013.83114375, -463.783827, -658.83885375, -495.05260075, 
    -782.22601375, -1136.1897344444444, -1567.6510633333335, -2136.8407, -2568.946113, -2845.9228975, -3149.6645705, -3640.458547, -4080.81555, -4952.347355, -5073.703895555555, -4879.3604305, -2082.8865266666667, -2051.94076125, -2380.010715, -2983.2449, -3478.003375, 
    -1096.984396724138, -969.538106, -2433.925215, -2419.015324, -2872.458516, -4684.01374, -5170.37679, -4678.720765, -5133.04942, -5055.7201, -5791.21431, -1412.194369, -2018.85905225, -2440.8732966666666
    ]

    order2atomic_E = {k: v for k, v in zip(order, atomic_E)}
    
    for idx, line in enumerate(txt):
        if 'Iteration' in line:
            ep.append(float(line.split('Etot,Ep,Ek (eV) =')[1].split()[1]))
            num_iters += 1
    
        if 'Atomic-Energy' in line:
            is_atomic = True
            
            e_dE.append(float(line.split('Q_atom:dE(eV)=')[1].split()[0]))
            for i in range(num_atoms):
                energy = float(txt[idx+1+i].split()[1])
                if energy > 0:
                    energy = energy + order2atomic_E.get(int(txt[idx+1+i].split()[0]))
                atomic_energy.append(energy)
                # atomic_energy.append(float(txt[idx+1+i].split()[1]))
                atype = int(txt[idx+1+i].split()[0])
                if atype in atom_type:
                    atomic_energy_per_elem[atype].append(energy)

        if 'Force' in line:
            for i in range(num_atoms):
                tmp = txt[idx+1+i].split()
                f_atom.append([-float(tmp[1]), -float(tmp[2]), -float(tmp[3])])
                atype = int(txt[idx+1+i].split()[0])
                if atype in atom_type:
                    force_per_elem[atype].append([-float(tmp[1]), -float(tmp[2]), -float(tmp[3])])
        
    
    return num_atoms, num_iters, np.array(ep), np.array(atomic_energy), np.array(f_atom), atomic_energy_per_elem, force_per_elem, is_atomic

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
    '''
def plot():
    
    print("Plotting of evaluation starts. Make sure MD/MOVEMENT and MOVEMENT are of EXACTLY THE SAME LENGTH")
    
    num_atoms, num_iters, dft_total_energy, dft_atomic_energy, dft_force = read_dft_movement(file_dft="MD/MOVEMENT")
    tmp_atoms, tmp_iters, nn_total_energy,  nn_atomic_energy,  nn_force  = read_dft_movement(file_dft='MOVEMENT')
    
    if num_atoms != tmp_atoms:
        raise Exception("number of atoms in two files does not match")

    if num_iters != tmp_iters:
        raise Exception("number of images in two files does not match")

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

    # Set up matplotlib parameters for better-looking plots
    plt.rcParams['figure.figsize'] = (8, 6) # Increase figure size 
    plt.rcParams['font.size'] = 15  # Increase font size 
    plt.rcParams["axes.linewidth"] = 1.5  # Increase axes linewidth 
    plt.rcParams["lines.linewidth"] = 2.5  # Increase lines linewidth 

    plt.rcParams['figure.figsize'] = (8, 6) # Increase figure size 
    plt.figure()
    plt.rcParams['font.size'] = 15  # Increase font size 
    plt.scatter(dft_atomic_energy, nn_atomic_energy, s=2, label="Evaluation data")
    plt.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red', label="Fit data")

    # plot atomic energy
    plt.figure()
    plt.title('atomic energy')
    plt.legend(fontsize='small', bbox_to_anchor=(0, 0.8), loc='upper left')
    plt.scatter(dft_atomic_energy, nn_atomic_energy, s=2, label="Evaluation data")
    plt.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red', label="Fit data")
    plt.savefig('atomic_energy.png', dpi=300, bbox_inches='tight', format='png')
    plt.ylim(lim_min, lim_max)
    plt.xlabel('DFT energy (eV)')
    plt.ylabel('MLFF energy (eV)')
    plt.legend(fontsize='small', bbox_to_anchor=(0, 0.8), loc='upper left')
    plt.text(lim_min, lim_max-(lim_max-lim_min)*0.1,'Ei, rmse: %.3E' % (e_atomic_rms))
    # save and show
    plt.figure()
    plt.savefig('atomic_energy.png', dpi=300, bbox_inches='tight', format='png')
    plt.scatter(dft_total_energy, nn_total_energy, s=10, label="Evaluation data")
    plt.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='red', label="Fit data")
    
    # plot total energy
    #plt.clf()
    #plt.subplot(1,2,1)
    plt.figure()
    plt.title('total energy')
    plt.legend(fontsize='small', bbox_to_anchor=(0, 0.8), loc='upper left')
    plt.savefig('total_energy.png', dpi=300, bbox_inches='tight', format='png')
    plt.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='red', label="Fit data")
    plt.xlim(lim_tot_min, lim_tot_max)
    plt.ylim(lim_tot_min, lim_tot_max)
    plt.figure()
    plt.text(e_tot_min, e_tot_max,'Etot, rmse: %.3E' %(e_tot_rms))
    plt.text(e_tot_min, e_tot_max - (e_tot_max - e_tot_min)*0.1,'Etot/Natom, rmse: %.3E' %(e_tot_rms/num_atoms))
    plt.xlabel('DFT energy (eV)')
    plt.ylabel('MLFF energy (eV)')
    plt.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='red', label="Fit data")
    plt.scatter(f_dft_plot, f_nn_plot, s=2, label="Evaluation data")
    plt.ylabel('MLFF Force (eV/$\mathrm{\AA}$)')
    plt.xlabel('DFT Force (eV/$\mathrm{\AA}$)')
    plt.legend(fontsize='small', bbox_to_anchor=(0, 0.8), loc='upper left')
    plt.savefig('force.png', dpi=300, bbox_inches='tight', format='png')
    plt.figure()
    plt.title('force')
    '''
def plot_new(atom_type, plot_elem = False, save_data = False, plot_ei = False):
    """
        Plots the evaluation of the MLFF model against DFT data.

        Args:
            atom_type (str): The type of atom to plot.
            plot_elem (bool): Whether to plot the atomic energy per element.
            save_data (bool): Whether to save the plotted data.
            plot_ei (bool): Whether to plot the atomic energy. Only plot if it has been trained. But someone may put a new movement file without atomic energy for evaluation.
    """
    import os    
    inference_summary_str = ""
    print("Plotting of evaluation starts. Make sure MD/MOVEMENT and MOVEMENT are of EXACTLY THE SAME LENGTH")
    
    num_atoms, num_iters, dft_total_energy, dft_atomic_energy, dft_force, dft_atomic_energy_per_elem, dft_atomic_force_per_elem, dft_is_atomic = read_dft_movement(file_dft="MD/MOVEMENT", atom_type=atom_type)
    tmp_atoms, tmp_iters, nn_total_energy,  nn_atomic_energy,  nn_force, nn_atomic_energy_per_elem, nn_atomic_force_per_elem, _  = read_dft_movement(file_dft='MOVEMENT', atom_type=atom_type)
    
    if num_atoms != tmp_atoms:
        raise Exception("number of atoms in two files does not match")

    if num_iters != tmp_iters:
        raise Exception("number of images in two files does not match")

    if dft_is_atomic:    
        dft_total_energy = np.zeros(num_iters, dtype=float)
        nn_total_energy = np.zeros(num_iters, dtype=float)
        
        for i in range(num_iters):
            dft_total_energy[i] = dft_atomic_energy[i*num_atoms:(i+1)*num_atoms].sum()
            nn_total_energy[i] = nn_atomic_energy[i*num_atoms:(i+1)*num_atoms].sum()
        
    e_tot_min = min(min(dft_total_energy), min(nn_total_energy))
    e_tot_max = max(max(dft_total_energy), max(nn_total_energy))

    lim_tot_min = e_tot_min - (e_tot_max - e_tot_min) * 0.1
    lim_tot_max = e_tot_max + (e_tot_max - e_tot_min) * 0.1

    if plot_ei and dft_is_atomic:
        e_min = min(min(dft_atomic_energy), min(nn_atomic_energy))
        e_max = max(max(dft_atomic_energy), max(nn_atomic_energy))
            
        lim_min = e_min - (e_max - e_min) * 0.1
        lim_max = e_max + (e_max - e_min) * 0.1

        e_atomic_rms = LA.norm(dft_atomic_energy - nn_atomic_energy) / np.sqrt(len(dft_atomic_energy))
        print('E_atomic, e_rmse: %.3E' % e_atomic_rms)
        inference_summary_str += 'E_atomic, e_rmse: %.3E\n' % e_atomic_rms
    
    e_tot_rms = LA.norm(dft_total_energy - nn_total_energy) / np.sqrt(len(dft_total_energy))
    e_r2 = r2_score(dft_total_energy, nn_total_energy)
    print('E_tot, e_rmse: %.3E, R2: %.3f' % (e_tot_rms, e_r2))
    print('E_tot/N_atom, e_rmse: %.3E' % (e_tot_rms/num_atoms))

    inference_summary_str += 'E_tot, e_rmse: %.3E R2: %.3f\n' % (e_tot_rms, e_r2)
    inference_summary_str += 'E_tot/N_atom, e_rmse: %.3E\n' % (e_tot_rms/num_atoms)

    # calculate force
    f_dft_plot = dft_force.reshape(3*num_atoms*num_iters)
    f_nn_plot = nn_force.reshape(3*num_atoms*num_iters)
    fmax = max(f_dft_plot.max(), f_nn_plot.max())
    fmin = min(f_dft_plot.min(), f_nn_plot.min())
    lim_f_min = fmin - (fmax-fmin)*0.1
    lim_f_max = fmax + (fmax-fmin)*0.1

    f_rms = LA.norm(f_dft_plot - f_nn_plot) / np.sqrt(3*num_atoms*num_iters)
    f_r2 = r2_score(f_dft_plot, f_nn_plot)
    print('f_rmse: %.3E R2: %.3f' % (f_rms, f_r2))
    inference_summary_str += 'f_rmse: %.3E R2:%.3E\n' % (f_rms, f_r2)
    # Set up matplotlib parameters for better-looking plots
    # plt.rcParams['figure.figsize'] = (19, 5) # Increase figure size 
    plt.rcParams['font.size'] = 12  # Increase font size 
    plt.rcParams["axes.linewidth"] = 1.5  # Increase axes linewidth 
    plt.rcParams["lines.linewidth"] = 2.5  # Increase lines linewidth 

    # Create figure and axes objects
    if plot_ei and dft_is_atomic :
        fig, axs = plt.subplots(1, 3, figsize=(19, 5))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # Plot atomic energy
    if plot_ei and dft_is_atomic:
        ax = axs[0]
        ax.set_title('Atomic energy')
        ax.scatter(dft_atomic_energy, nn_atomic_energy, s=30, label="MLFF data")
        ax.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red', label="DFT data")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel('DFT energy (eV)')
        ax.set_ylabel('MLFF energy (eV)')
        ax.legend(fontsize='small', loc='upper left')
        # ax.text(lim_min, lim_max-(lim_max-lim_min)*0.1,'Ei, rmse: %.3E' % (e_atomic_rms))
        ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s='Ei, rmse: %.3E' % (e_atomic_rms), transform=ax.transAxes)

    # Plot total energy
    ax = axs[1 if plot_ei and dft_is_atomic else 0]
    ax.set_title('Total energy')
    ax.scatter(dft_total_energy, nn_total_energy, s=30, label="MLFF data")
    ax.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='red', label="DFT data")
    ax.set_xlim(lim_tot_min, lim_tot_max)
    ax.set_ylim(lim_tot_min, lim_tot_max)
    ax.set_xlabel('DFT energy (eV)')
    ax.set_ylabel('MLFF energy (eV)')
    ax.legend(fontsize='small', loc='upper left')
    ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s='Etot/Natom, rmse: %.3E' % (e_tot_rms / num_atoms), transform=ax.transAxes)
    ax.text(0.02, 0.75, horizontalalignment='left', verticalalignment='top', s='Etot, rmse: %.3E, R2: %3E' % (e_tot_rms, e_r2), transform=ax.transAxes)

    # Plot force
    ax = axs[2 if plot_ei and dft_is_atomic else 1]
    ax.set_title('Force')
    ax.scatter(f_dft_plot, f_nn_plot, s=2, label="MLFF data")
    ax.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='red', label="DFT data")
    ax.set_xlim(lim_f_min, lim_f_max)
    ax.set_ylim(lim_f_min, lim_f_max)
    ax.set_ylabel('MLFF Force (eV/Å)')
    ax.set_xlabel('DFT Force (eV/Å)')
    ax.legend(fontsize='small', loc='upper left')
    ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s='Force, rmse: %.3E, R2: %3E' % (f_rms, f_r2), transform=ax.transAxes)

    # Save figure
    if not os.path.exists('plot_data'):
        os.makedirs('plot_data')
    plt.savefig('plot_data/evaluation_plots.png', dpi=300, bbox_inches='tight', format='png')
    #if len(sys.argv) < 2:
    #    plt.show()
    
    if save_data:
        if plot_ei and dft_is_atomic:
            np.savetxt('plot_data/dft_atomic_energy.txt', dft_atomic_energy)
            np.savetxt('plot_data/inference_atomic_energy.txt', nn_atomic_energy)
        np.savetxt('plot_data/dft_total_energy.txt', dft_total_energy)
        np.savetxt('plot_data/inference_total_energy.txt', nn_total_energy)
        np.savetxt('plot_data/dft_force.txt', f_dft_plot)
        np.savetxt('plot_data/inference_force.txt', f_nn_plot)
        # res_pd_save_path = os.path.join(inf_dir, "inference_loss.csv")

    if plot_elem:

        from poscar2lammps import elem2idx
        # Create figure and axes for each atom type
        # fig, axs = plt.subplots(1, len(atom_type))
        figs = []
        axes = []
        
        for i, atom in enumerate(atom_type):

            if plot_ei and dft_is_atomic:
                fig, axs = plt.subplots(1, 2, figsize=(9, 6))
                
                e_min = min(min(dft_atomic_energy_per_elem[atom]), min(nn_atomic_energy_per_elem[atom]))
                e_max = max(max(dft_atomic_energy_per_elem[atom]), max(nn_atomic_energy_per_elem[atom]))
                lim_min = e_min - (e_max - e_min) * 0.1
                lim_max = e_max + (e_max - e_min) * 0.1
            else:
                fig, axs = plt.subplots(1, 1, figsize=(5, 4))
                axs = [axs]     # Convert to list for consistent indexing


            dft_atomic_force_per_elemx = np.array(dft_atomic_force_per_elem[atom])
            nn_atomic_force_per_elemy = np.array(nn_atomic_force_per_elem[atom])
            f_dft_plot = dft_atomic_force_per_elemx.reshape(-1)
            f_nn_plot = nn_atomic_force_per_elemy.reshape(-1)
            fmax = max(f_dft_plot.max(), f_nn_plot.max())
            fmin = min(f_dft_plot.min(), f_nn_plot.min())
            lim_f_min = fmin - (fmax-fmin)*0.1
            lim_f_max = fmax + (fmax-fmin)*0.1

            # Convert atom order to element type
            idx_tabel = elem2idx()
            elem_type = [elem for elem, idx in idx_tabel.items() if idx == atom][0]            

            # Plot atomic energy
            if plot_ei and dft_is_atomic:
                ax = axs[0]
                ax.set_title(f'Atomic energy ({elem_type})')
                ax.scatter(dft_atomic_energy_per_elem[atom], nn_atomic_energy_per_elem[atom], s=30, label="MLFF data")
                ax.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red', label="DFT data")
                ax.set_xlim(lim_min, lim_max)
                ax.set_ylim(lim_min, lim_max)
                ax.set_xlabel('DFT energy (eV)')
                ax.set_ylabel('MLFF energy (eV)')
                ax.legend(fontsize='small', loc='upper left')
                # ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s=f'Ei ({atom}), rmse: %.3E' % (e_atomic_rms), transform=ax.transAxes)

            # Plot force
            ax = axs[1 if plot_ei and dft_is_atomic else 0]
            ax.set_title(f'Force ({elem_type})')
            ax.scatter(dft_atomic_force_per_elemx, nn_atomic_force_per_elemy, s=2, label="MLFF data")
            ax.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='red', label="DFT data")
            ax.set_xlim(lim_f_min, lim_f_max)
            ax.set_ylim(lim_f_min, lim_f_max)
            ax.set_ylabel('MLFF Force (eV/Å)')
            ax.set_xlabel('DFT Force (eV/Å)')
            ax.legend(fontsize='small', loc='upper left')

            figs.append(fig)
            axes.append(ax)

            if save_data:
                if plot_ei and dft_is_atomic:
                    np.savetxt(f'plot_data/dft_atomic_energy_{elem_type}.txt', dft_atomic_energy_per_elem[atom])
                    np.savetxt(f'plot_data/inference_atomic_energy_{elem_type}.txt', nn_atomic_energy_per_elem[atom])
                np.savetxt(f'plot_data/dft_atomic_force_{elem_type}.txt', dft_atomic_force_per_elemx)
                np.savetxt(f'plot_data/inference_atomic_force_{elem_type}.txt', nn_atomic_force_per_elemy)
        
        for i, fig in enumerate(figs):
            fig.savefig(f'plot_data/evaluation_plots_{atom_type[i]}.png', dpi=300, bbox_inches='tight', format='png')

    with open(f"plot_data/inference_summary.txt", 'w') as wf:
        wf.writelines(inference_summary_str)


