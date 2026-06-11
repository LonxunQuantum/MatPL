import numpy as np
import torch
from src.user.input_param import InputParam

def get_adam_loss_prefactor(start_prefactor, end_prefactor, real_lr, start_lr=0.001):
    lr_ratio = real_lr / start_lr
    lr_ratio = min(max(lr_ratio, 0.0), 1.0)
    return end_prefactor + (start_prefactor - end_prefactor) * lr_ratio


def get_loss(
    args: InputParam,
    real_lr,
    avg_atom_number,
    loss_F_val,
    loss_Etot_val,
    loss_Virial_val=None,
    loss_Egroup_val=None,
    loss_Charge_val=None,
    loss_BEC_val=None,
    train_virial=False,
):
    optimizer_param = args.optimizer_param
    loss = torch.zeros_like(loss_F_val)

    if optimizer_param.train_force:
        pref_force = get_adam_loss_prefactor(
            optimizer_param.start_pre_fac_force,
            optimizer_param.end_pre_fac_force,
            real_lr,
        )
        loss = loss + pref_force * loss_F_val

    if optimizer_param.train_energy:
        pref_etot = get_adam_loss_prefactor(
            optimizer_param.start_pre_fac_etot,
            optimizer_param.end_pre_fac_etot,
            real_lr,
        )
        loss = loss + pref_etot * loss_Etot_val / avg_atom_number

    if train_virial and loss_Virial_val is not None:
        pref_virial = get_adam_loss_prefactor(
            optimizer_param.start_pre_fac_virial,
            optimizer_param.end_pre_fac_virial,
            real_lr,
        )
        loss = loss + pref_virial * loss_Virial_val / avg_atom_number

    if optimizer_param.train_egroup and loss_Egroup_val is not None:
        pref_egroup = get_adam_loss_prefactor(
            optimizer_param.start_pre_fac_egroup,
            optimizer_param.end_pre_fac_egroup,
            real_lr,
        )
        loss = loss + pref_egroup * loss_Egroup_val

    if getattr(optimizer_param, "train_charge", False) and loss_Charge_val is not None:
        pref_charge = get_adam_loss_prefactor(
            optimizer_param.start_pre_fac_charge,
            optimizer_param.end_pre_fac_charge,
            real_lr,
        )
        loss = loss + pref_charge * loss_Charge_val / avg_atom_number

    if getattr(optimizer_param, "train_bec", False) and loss_BEC_val is not None:
        pref_bec = get_adam_loss_prefactor(
            optimizer_param.start_pre_fac_bec,
            optimizer_param.end_pre_fac_bec,
            real_lr,
        )
        loss = loss + pref_bec * loss_BEC_val

    return loss


def print_l1_l2(model):
    params = model.parameters()
    dtype = next(params).dtype
    device = next(params).device
    L1 = torch.tensor(0.0, device=device, dtype=dtype).detach().requires_grad_(False)
    L2 = torch.tensor(0.0, device=device, dtype=dtype).detach().requires_grad_(False)
    nums_param = 0
    for p in params:
        L1 += torch.sum(torch.abs(p))
        L2 += torch.sum(p**2)
        nums_param += p.nelement()
    L1 = L1 / nums_param
    L2 = L2 / nums_param
    return L1, L2


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