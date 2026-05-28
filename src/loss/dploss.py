import math
import numpy as np
from src.user.input_param import InputParam

def calc_loss(input_param:InputParam, start_lr, real_lr, stat, *args):

    if stat == 1:   
        has_fi, lossFi, has_etot, loss_Etot, has_virial, loss_Virial, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum = args
    elif stat == 2: # no virial
        has_fi, lossFi, has_etot, loss_Etot, has_egroup, loss_Egroup, has_ei, loss_Ei, natoms_sum = args
    elif stat == 3: # no egroup
        has_fi, lossFi, has_etot, loss_Etot, has_virial, loss_Virial, has_ei, loss_Ei, natoms_sum = args
    else:   # no virial and egroup
        has_fi, lossFi, has_etot, loss_Etot, has_ei, loss_Ei, natoms_sum = args

    start_pref_egroup, limit_pref_egroup = input_param.optimizer_param.start_pre_fac_egroup, input_param.optimizer_param.end_pre_fac_egroup
    start_pref_F, limit_pref_F = input_param.optimizer_param.start_pre_fac_force, input_param.optimizer_param.end_pre_fac_force # 1000, 1.0
    start_pref_etot, limit_pref_etot = input_param.optimizer_param.start_pre_fac_etot, input_param.optimizer_param.end_pre_fac_etot # 0.02, 1.0
    start_pref_virial, limit_pref_virial = input_param.optimizer_param.start_pre_fac_virial, input_param.optimizer_param.end_pre_fac_virial # 50.0, 1
    start_pref_ei, limit_pref_ei =input_param.optimizer_param.start_pre_fac_ei, input_param.optimizer_param.end_pre_fac_ei # 0.1, 2.0

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

    decay_rate = np.exp(np.log(stop_lr / start_lr) / (stop_step / decay_step))  # 0.9500064099092085
    real_lr = start_lr * np.power(decay_rate, (iter // decay_step))
    return real_lr

"""
预热阶段，线性增加学习率
"""
def warmup_lr(iter, iternum, cur_epoch, warm_epochs, start_lr, end_lr):
    if cur_epoch <= warm_epochs:
        cur_epoch = cur_epoch - 1 # epoch 从1开始计数
        return start_lr + (cur_epoch * iternum + iter) / (warm_epochs * iternum) * (end_lr - start_lr)
    else:
        raise Exception(f"ERROR! The current epochs {cur_epoch} > warmepoch nums {warm_epochs}")


def wsd_lr(global_step, total_steps, peak_lr, stop_lr,
           warmup_steps=0, stable_frac=0.9, decay_kind="cosine"):
    """Warmup-Stable-Decay LR schedule.

    Three phases over ``total_steps``:
      * ``[0, warmup_steps)`` — linear ramp from ``stop_lr`` to ``peak_lr``.
      * ``[warmup_steps, decay_start)`` — flat at ``peak_lr`` where
        ``decay_start = warmup_steps + (total_steps - warmup_steps) * stable_frac``.
      * ``[decay_start, total_steps]`` — decay from ``peak_lr`` to ``stop_lr``
        with either linear or cosine profile.
    """

    if total_steps <= 0:
        return peak_lr
    step = max(0, min(int(global_step), int(total_steps)))
    warmup = max(0, int(warmup_steps))
    if warmup > total_steps:
        warmup = total_steps

    if step < warmup:
        ratio = step / warmup
        return stop_lr + (peak_lr - stop_lr) * ratio

    decay_start = warmup + int((total_steps - warmup) * stable_frac)
    decay_start = min(max(decay_start, warmup), total_steps)
    if step < decay_start:
        return peak_lr

    decay_total = total_steps - decay_start
    if decay_total <= 0:
        return stop_lr
    progress = (step - decay_start) / decay_total
    progress = min(max(progress, 0.0), 1.0)
    if decay_kind == "linear":
        return peak_lr + (stop_lr - peak_lr) * progress
    # cosine: smooth peak->stop transition
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return stop_lr + (peak_lr - stop_lr) * cosine