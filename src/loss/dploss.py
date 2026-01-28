import numpy as np
from src.user.input_param import InputParam
# def dp_loss(
#     start_lr,
#     real_lr,
#     has_fi,
#     lossFi,
#     has_etot,
#     loss_Etot,
#     has_virial,
#     loss_Virial,
#     has_egroup,
#     loss_Egroup,
#     has_ei,
#     loss_Ei,
#     natoms_sum,
# ):
def dp_loss(dp_param:InputParam, start_lr, real_lr, stat, *args):

    if stat == 1:   
        has_fi, lossFi, has_etot, loss_Etot, has_virial, loss_Virial, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum = args
    elif stat == 2: # no virial
        has_fi, lossFi, has_etot, loss_Etot, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum = args
    elif stat == 3: # no egroup
        has_fi, lossFi, has_etot, loss_Etot, has_virial, loss_Virial, has_ei, loss_Ei, natoms_sum = args
    else:   # no virial and egroup
        has_fi, lossFi, has_etot, loss_Etot, has_ei, loss_Ei, natoms_sum = args

    start_pref_egroup, limit_pref_egroup = dp_param.optimizer_param.start_pre_fac_egroup, dp_param.optimizer_param.end_pre_fac_egroup
    start_pref_F, limit_pref_F = dp_param.optimizer_param.start_pre_fac_force, dp_param.optimizer_param.end_pre_fac_force # 1000, 1.0
    start_pref_etot, limit_pref_etot = dp_param.optimizer_param.start_pre_fac_etot, dp_param.optimizer_param.end_pre_fac_etot # 0.02, 1.0
    start_pref_virial, limit_pref_virial = dp_param.optimizer_param.start_pre_fac_virial, dp_param.optimizer_param.end_pre_fac_virial # 50.0, 1
    start_pref_ei, limit_pref_ei =dp_param.optimizer_param.start_pre_fac_ei, dp_param.optimizer_param.end_pre_fac_ei # 0.1, 2.0

    pref_fi = has_fi * (
        limit_pref_F + (start_pref_F - limit_pref_F) * real_lr / start_lr
    )
    pref_etot = has_etot * (
        limit_pref_etot + (start_pref_etot - limit_pref_etot) * real_lr / start_lr
    )
    if stat == 1 or stat == 3:
        pref_virial = has_virial * (
            limit_pref_virial + (start_pref_virial - limit_pref_virial) * real_lr / start_lr
        )
    if stat == 1 or stat == 2:
        pref_egroup = has_egroup * (
            limit_pref_egroup + (start_pref_egroup - limit_pref_egroup) * real_lr / start_lr
        )
    pref_ei = has_ei * (
        limit_pref_ei + (start_pref_ei - limit_pref_ei) * real_lr / start_lr
    )
    l2_loss = 0
    if has_fi:
        l2_loss += pref_fi * lossFi
    if has_etot:
        l2_loss += 1.0 / natoms_sum * pref_etot * loss_Etot
    if stat == 1 or stat == 3:
        if has_virial:
            l2_loss += 1.0 / natoms_sum * pref_virial * loss_Virial
            # import ipdb;ipdb.set_trace()
    if stat == 1 or stat == 2:
        if has_egroup:
            l2_loss += pref_egroup * loss_Egroup
    if has_ei:
        l2_loss += pref_ei * loss_Ei
    return l2_loss, pref_fi, pref_etot


def adjust_lr(iter, start_lr, stop_step, decay_step, stop_lr=3.51e-8):
    # stop_step = 1000000
    # decay_step = 5000
    if iter > stop_step: # or real_lr < stop_lr
        return stop_lr

    decay_rate = np.exp(
        np.log(stop_lr / start_lr) / (stop_step / decay_step)
    )  # 0.9500064099092085
    real_lr = start_lr * np.power(decay_rate, (iter // decay_step))
    return real_lr
