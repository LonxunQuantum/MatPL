import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score

def _safe_r2(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    return r2_score(y_true, y_pred)

def _load_flat_values(file_path):
    values = []
    with open(file_path, "r") as rf:
        for line in rf:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values.extend(float(_) for _ in line.split())
    return np.array(values)

def inference_plot(data_dir:str, return_extra=False):
    size = 20
    num_atom = np.loadtxt(os.path.join(data_dir, "image_atom_nums.txt"))
    dft_E   = np.loadtxt(os.path.join(data_dir, "dft_total_energy.txt")) / num_atom
    MLFF_E  = np.loadtxt(os.path.join(data_dir, "inference_total_energy.txt")) / num_atom
    rmse_E = np.sqrt(np.square(dft_E - MLFF_E).mean())
    E_min = min(dft_E.min(), MLFF_E.min())
    E_max = max(dft_E.max(), MLFF_E.max())

    dft_F   = np.loadtxt(os.path.join(data_dir, "dft_force.txt")).flatten()
    MLFF_F  = np.loadtxt(os.path.join(data_dir, "inference_force.txt")).flatten()
    rmse_F = np.sqrt(np.square(dft_F - MLFF_F).mean())
    F_min = min(dft_F.min(), MLFF_F.min())
    F_max = max(dft_F.max(), MLFF_F.max())

    if os.path.exists(os.path.join(data_dir, "dft_virial.txt")):
        _dft_V  = np.loadtxt(os.path.join(data_dir, "dft_virial.txt"))
        _MLFF_V = np.loadtxt(os.path.join(data_dir, "inference_virial.txt"))

        filtered_indices = (_dft_V > -1e6).all(axis=1)
        dft_V = _dft_V[filtered_indices,:]
        MLFF_V= _MLFF_V[filtered_indices,:]

        if len(dft_V) > 0:
            atom_idx = np.repeat(num_atom, 6).reshape(num_atom.shape[0], 6)[filtered_indices, :]
            dft_V = dft_V/atom_idx
            MLFF_V = MLFF_V/atom_idx
            rmse_V = np.sqrt(np.square(dft_V - MLFF_V).mean())
            V_min = min(dft_V.min(), MLFF_V.min())
            V_max = max(dft_V.max(), MLFF_V.max())
        else:
            rmse_V = None
            dft_V = []
    else:
        rmse_V = None
        dft_V = []
    plt.plot(dft_E.flatten(), MLFF_E.flatten(),"o",markersize=3,c="C0")
    plt.plot([E_min,E_max],[E_min,E_max],"--",lw=1.2,c="C1")
    # plt.axis([E_min*1.02,E_max*0.98,E_min*1.02,E_max*0.98])
    e_r2 = _safe_r2(dft_E.flatten(), MLFF_E.flatten())
    s = "RMSE of Energy is %.1f meV/atom" % (rmse_E * 1e3)
    sr2=r"R$^2$ = %.3f" % e_r2
    ax = plt.gca()
    plt.text(.4,.125,sr2,fontsize=size-2,transform=ax.transAxes)
    plt.text(.14,.03,s,fontsize=size-2,transform=ax.transAxes)
    plt.xticks(size=size-4)
    plt.yticks(size=size-4)
    plt.xlabel("DFT Energy (eV/atom)",size=size)
    plt.ylabel("MLFF Energy (eV/atom)",size=size)
    #plt.title(title,size=size+4)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "Energy.png"), dpi=360)
    plt.close()

    plt.plot(dft_F, MLFF_F,"o",markersize=3,c="C0")
    plt.plot([F_min,F_max],[F_min,F_max],"--",lw=1.2,c="C1")
    # plt.axis([F_min*1.02,F_max*1.02,F_min*1.02,F_max*1.02])
    f_r2 = _safe_r2(dft_F, MLFF_F)
    s = r"RMSE of Force is %.3f eV/$\mathrm{\AA}$" % rmse_F
    sr2=r"R$^2$ = %.3f" % f_r2
    ax = plt.gca()
    plt.text(.4,.125,sr2,fontsize=size-2,transform=ax.transAxes)
    plt.text(.14,.03,s,fontsize=size-2,transform=ax.transAxes)
    plt.xticks(size=size-4)
    plt.yticks(size=size-4)
    plt.xlabel(r"DFT Force (eV/$\mathrm{\AA}$)",size=size)
    plt.ylabel(r"MLFF Force (eV/$\mathrm{\AA}$)",size=size)
    #plt.title(title,size=size+4)
    plt.tight_layout()
    plt.savefig(os.path.join(data_dir, "Force.png"), dpi=360)
    plt.close()

    if rmse_V is not None and len(dft_V) > 0 :
        plt.plot(dft_V, MLFF_V,"o",markersize=3,c="C0")
        plt.plot([V_min,V_max],[V_min,V_max],"--",lw=1.2,c="C1")
        # plt.axis([V_min*1.02,V_max*1.02,V_min*1.02,V_max*1.02])
        v_r2 = _safe_r2(dft_V.flatten(), MLFF_V.flatten())
        s = r"RMSE of Virial is %.3f eV/atom" % rmse_V
        sr2=r"R$^2$ = %.3f" % v_r2
        ax = plt.gca()
        plt.text(.4,.125,sr2,fontsize=size,transform=ax.transAxes)
        plt.text(.14,.03,s,fontsize=size-2,transform=ax.transAxes)
        plt.xticks(size=size-4)
        plt.yticks(size=size-4)
        plt.xlabel(r"DFT Virial (eV/atom)",size=size)
        plt.ylabel(r"MLFF Virial (eV/atom)",size=size)
        #plt.title(title,size=size+4)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, "Virial.png"), dpi=360)
        plt.close()
    else:
        v_r2 = 0.000
        rmse_V = 0.000

    rmse_charge = None
    charge_r2 = None
    charge_dft_path = os.path.join(data_dir, "dft_charge.txt")
    charge_pred_path = os.path.join(data_dir, "inference_charge.txt")
    if os.path.exists(charge_dft_path) and os.path.exists(charge_pred_path):
        dft_charge = _load_flat_values(charge_dft_path)
        MLFF_charge = _load_flat_values(charge_pred_path)
        if len(dft_charge) > 0 and len(dft_charge) == len(MLFF_charge):
            charge_mask = np.isfinite(dft_charge) & np.isfinite(MLFF_charge)
            dft_charge = dft_charge[charge_mask]
            MLFF_charge = MLFF_charge[charge_mask]
        if len(dft_charge) > 0 and len(dft_charge) == len(MLFF_charge):
            rmse_charge = np.sqrt(np.square(dft_charge - MLFF_charge).mean())
            charge_r2 = _safe_r2(dft_charge, MLFF_charge)
            charge_min = min(dft_charge.min(), MLFF_charge.min())
            charge_max = max(dft_charge.max(), MLFF_charge.max())
            plt.plot(dft_charge, MLFF_charge, "o", markersize=3, c="C0")
            plt.plot([charge_min, charge_max], [charge_min, charge_max], "--", lw=1.2, c="C1")
            ax = plt.gca()
            plt.text(.4, .125, r"R$^2$ = %.3f" % charge_r2, fontsize=size-2, transform=ax.transAxes)
            plt.text(.14, .03, "RMSE of Charge is %.3f e" % rmse_charge, fontsize=size-2, transform=ax.transAxes)
            plt.xticks(size=size-4)
            plt.yticks(size=size-4)
            plt.xlabel("DFT Charge (e)", size=size)
            plt.ylabel("MLFF Charge (e)", size=size)
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, "Charge.png"), dpi=360)
            plt.close()

    rmse_bec = None
    bec_r2 = None
    bec_dft_path = os.path.join(data_dir, "dft_bec.txt")
    bec_pred_path = os.path.join(data_dir, "inference_bec.txt")
    if os.path.exists(bec_dft_path) and os.path.exists(bec_pred_path):
        _dft_bec = np.loadtxt(bec_dft_path)
        _MLFF_bec = np.loadtxt(bec_pred_path)
        _dft_bec = np.atleast_2d(_dft_bec)
        _MLFF_bec = np.atleast_2d(_MLFF_bec)
        filtered_indices = (_dft_bec > -1e6).all(axis=1)
        dft_bec = _dft_bec[filtered_indices, :].flatten()
        MLFF_bec = _MLFF_bec[filtered_indices, :].flatten()
        if len(dft_bec) > 0 and len(dft_bec) == len(MLFF_bec):
            rmse_bec = np.sqrt(np.square(dft_bec - MLFF_bec).mean())
            bec_r2 = _safe_r2(dft_bec, MLFF_bec)
            bec_min = min(dft_bec.min(), MLFF_bec.min())
            bec_max = max(dft_bec.max(), MLFF_bec.max())
            plt.plot(dft_bec, MLFF_bec, "o", markersize=3, c="C0")
            plt.plot([bec_min, bec_max], [bec_min, bec_max], "--", lw=1.2, c="C1")
            ax = plt.gca()
            plt.text(.4, .125, r"R$^2$ = %.3f" % bec_r2, fontsize=size-2, transform=ax.transAxes)
            plt.text(.14, .03, "RMSE of BEC is %.3f e" % rmse_bec, fontsize=size-2, transform=ax.transAxes)
            plt.xticks(size=size-4)
            plt.yticks(size=size-4)
            plt.xlabel("DFT BEC (e)", size=size)
            plt.ylabel("MLFF BEC (e)", size=size)
            plt.tight_layout()
            plt.savefig(os.path.join(data_dir, "BEC.png"), dpi=360)
            plt.close()

    if return_extra:
        return rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2, rmse_charge, charge_r2, rmse_bec, bec_r2
    return rmse_E, rmse_F, rmse_V, e_r2, f_r2, v_r2

if __name__=="__main__":
    inference_plot("/data/home/wuxingxing/datas/pwmat_mlff_workdir/fec/std/test/test_result")
