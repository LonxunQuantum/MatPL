# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys

size = 20
def lammps_heatflux(path):
    start = 500
    dt = 0.001  # ps 
    Ns = 1000  # Sample interval
    thermo = np.loadtxt(path + "/compute_Energy_Temp.out")
    jp = np.loadtxt(path + "/compute_HeatFlux.out")
    BLOCK_LENGTH = 426
    Ein = thermo[start:, 1]
    Eout = thermo[start:, 2]
    Etol = (Eout - Ein) / 2 / 1000 #in units of KeV
    Etol = Etol - Etol[0]
    t = dt * np.arange(1, len(Etol) + 1) * Ns / 1000 #unit in ns
    jpy = jp[start:, 2] - jp[start:, 5]
    jpy = jpy / BLOCK_LENGTH / 10 * 1000 #in units of eV/ns
    accum_jpy = np.cumsum(jpy) * 0.001 / 1000 #in units of KeV
    return t, accum_jpy, Etol

path = sys.argv[1] if len(sys.argv) > 1 else "."
t_nep, jp_nep, etol_nep = lammps_heatflux(path)

plt.figure(figsize=(10,6))
plt.plot(t_nep, -1 * jp_nep, lw = 3, ls = "-", label = "NEP, from atoms")
plt.plot(t_nep, etol_nep, lw = 3, ls = "--", label = "NEP, from thermostats")
plt.xlim([0, 1.5])
plt.xticks(np.linspace(0, 1.5, 4),size=size-4,fontweight='bold')
plt.ylim([0, 2])
plt.yticks(np.linspace(0, 2, 5),fontsize=size-4,fontweight='bold')
plt.xlabel('Time (ns)',size=size,fontweight='bold')
plt.ylabel('Accumulated heat (keV)',size=size,fontweight='bold')
plt.title("MatPL-NEP",size=size+4,fontweight='bold')
plt.legend(prop={'size':size-6,'weight':'bold'})

bwith=2
ax = plt.gca()
ax.spines['bottom'].set_linewidth(bwith)
ax.spines['left'].set_linewidth(bwith)
ax.spines['top'].set_linewidth(bwith)
ax.spines['right'].set_linewidth(bwith)

plt.tight_layout()
plt.savefig("HeatFlux.png",dpi=360,bbox_inches='tight')
plt.close()
