#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os

def plot(dir = "./", plot_ei = False):
    """
        Plots the evaluation of the MLFF model against DFT data.

        Args:
            plot_ei (bool): Whether to plot the atomic energy. Only plot if it has been trained. But someone may put a new movement file without atomic energy for evaluation.
    """
    dft_total_energy = np.loadtxt(os.path.join(dir,"dft_total_energy.txt"))
    nn_total_energy = np.loadtxt(os.path.join(dir,"inference_total_energy.txt"))
    dft_force = np.loadtxt(os.path.join(dir,"dft_force.txt"))
    nn_force = np.loadtxt(os.path.join(dir,"inference_force.txt"))

    e_tot_min = min(min(dft_total_energy), min(nn_total_energy))
    e_tot_max = max(max(dft_total_energy), max(nn_total_energy))

    lim_tot_min = e_tot_min - (e_tot_max - e_tot_min) * 0.1
    lim_tot_max = e_tot_max + (e_tot_max - e_tot_min) * 0.1

    if plot_ei:
        dft_atomic_energy = np.loadtxt(os.path.join(dir,"dft_atomic_energy.txt"))
        nn_atomic_energy = np.loadtxt(os.path.join(dir,"inference_atomic_energy.txt"))
        dft_atomic_energy = dft_atomic_energy.reshape(-1)
        nn_atomic_energy = nn_atomic_energy.reshape(-1)
        e_min = min(min(dft_atomic_energy), min(nn_atomic_energy))
        e_max = max(max(dft_atomic_energy), max(nn_atomic_energy))
            
        lim_min = e_min - (e_max - e_min) * 0.1
        lim_max = e_max + (e_max - e_min) * 0.1

    # calculate force
    f_dft_plot = dft_force.reshape(-1)
    f_nn_plot = nn_force.reshape(-1)
    fmax = max(f_dft_plot.max(), f_nn_plot.max())
    fmin = min(f_dft_plot.min(), f_nn_plot.min())
    lim_f_min = fmin - (fmax-fmin)*0.1
    lim_f_max = fmax + (fmax-fmin)*0.1

    # Set up matplotlib parameters for better-looking plots
    # plt.rcParams['figure.figsize'] = (19, 5) # Increase figure size 
    plt.rcParams['font.size'] = 12  # Increase font size 
    plt.rcParams["axes.linewidth"] = 1.5  # Increase axes linewidth 
    plt.rcParams["lines.linewidth"] = 2.5  # Increase lines linewidth 

    # Create figure and axes objects
    if plot_ei:
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot atomic energy
    if plot_ei:
        ax = axs[0]
        ax.set_title('Atomic energy')
        ax.scatter(dft_atomic_energy, nn_atomic_energy, s=30, label="MLFF data")
        ax.plot((lim_min, lim_max), (lim_min, lim_max), ls='--', color='red', label="DFT data")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel('DFT energy (eV)')
        ax.set_ylabel('MLFF energy (eV)')
        ax.legend(fontsize='small', loc='upper left')
        # ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s='Ei, rmse: %.3E' % (e_atomic_rms), transform=ax.transAxes)

    # Plot total energy
    ax = axs[1 if plot_ei else 0]
    ax.set_title('Total energy')
    ax.scatter(dft_total_energy, nn_total_energy, s=30, label="MLFF data")
    ax.plot((lim_tot_min, lim_tot_max), (lim_tot_min, lim_tot_max), ls='--', color='red', label="DFT data")
    ax.set_xlim(lim_tot_min, lim_tot_max)
    ax.set_ylim(lim_tot_min, lim_tot_max)
    ax.set_xlabel('DFT energy (eV)')
    ax.set_ylabel('MLFF energy (eV)')
    ax.legend(fontsize='small', loc='upper left')
    # ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s='Etot/Natom, rmse: %.3E' % (e_tot_rms / num_atoms), transform=ax.transAxes)
    # ax.text(0.02, 0.75, horizontalalignment='left', verticalalignment='top', s='Etot, rmse: %.3E' % (e_tot_rms), transform=ax.transAxes)

    # Plot force
    ax = axs[2 if plot_ei else 1]
    ax.set_title('Force')
    ax.scatter(f_dft_plot, f_nn_plot, s=2, label="MLFF data")
    ax.plot((lim_f_min, lim_f_max), (lim_f_min, lim_f_max), ls='--', color='red', label="DFT data")
    ax.set_xlim(lim_f_min, lim_f_max)
    ax.set_ylim(lim_f_min, lim_f_max)
    ax.set_ylabel('MLFF Force (eV/Å)')
    ax.set_xlabel('DFT Force (eV/Å)')
    ax.legend(fontsize='small', loc='upper left')
    # ax.text(0.02, 0.82, horizontalalignment='left', verticalalignment='top', s='Force, rmse: %.3E' % (f_rms), transform=ax.transAxes)

    # Save figure
    plt.savefig(os.path.join(dir,"evaluation_plots.png"), dpi=300, bbox_inches='tight', format='png')
    #if len(sys.argv) < 2:
    #    plt.show()
    
