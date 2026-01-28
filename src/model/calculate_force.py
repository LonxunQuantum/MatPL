from torch import nn
import torch
from torch.autograd import Function
import op

class CalculateCompress(Function):
    @staticmethod
    def forward(ctx, f2, coefficient):
        sij_num = coefficient.shape[0]
        layer_node = coefficient.shape[1]
        coe_num = coefficient.shape[2]
        ctx.save_for_backward(f2, coefficient)
        G = torch.empty(
            [sij_num, layer_node],
            device=f2.device,
            dtype=f2.dtype,
        )
        op.calculate_compress(f2, coefficient, sij_num, layer_node, coe_num, G)
        return G

    @staticmethod
    def backward(ctx, grad_output): # the grad_output is dE/dG
        # print("IN compress gradout:\n", grad_output.shape)
        # print(grad_output[0,:], "\n\n")
        inputs = ctx.saved_tensors
        f2 = inputs[0]
        coefficient = inputs[1]
        sij_num = coefficient.shape[0]
        layer_node = coefficient.shape[1]
        coe_num = coefficient.shape[2]
        Grad = torch.zeros([sij_num, layer_node], device=f2.device, dtype=f2.dtype)
        op.calculate_compress_grad(f2, coefficient, grad_output, sij_num, layer_node, coe_num, Grad)
        grad_out = torch.sum(grad_output*Grad, dim=1).unsqueeze(-1) #对应坐标位置相乘，然后算加法
        # print("OUT compress Grad:\n", Grad.shape)
        # print(Grad[0,:], "\n\n")
        # print("OUT compress grad_out:\n", Grad.shape)
        # print(grad_out[0,:], "\n\n")
        # assert(1 == 0)
        return (grad_out, None)
    
class CalculateForce(Function):
    @staticmethod
    def forward(ctx, list_neigh, dE, Ri_d, F):
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        ctx.save_for_backward(list_neigh, dE, Ri_d)
        op.calculate_force(list_neigh, dE, Ri_d, batch_size, natoms, neigh_num, F)
        return F

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        list_neigh = inputs[0]
        dE = inputs[1]
        Ri_d = inputs[2]
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        grad = torch.zeros_like(dE)
        op.calculate_force_grad(list_neigh, Ri_d, grad_output, batch_size, natoms, neigh_num, grad)
        return (None, grad, None, None)


class CalculateVirialForce(Function):
    @staticmethod
    def forward(ctx, list_neigh, dE, Rij, Ri_d):
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        ctx.save_for_backward(list_neigh, dE, Rij, Ri_d)
        virial = torch.zeros(batch_size, 9, dtype=dE.dtype, device=dE.device)
        atom_virial = torch.zeros(batch_size, natoms, 9, dtype=dE.dtype, device=dE.device)
        op.calculate_virial_force(list_neigh, dE, Rij, Ri_d, batch_size, natoms, neigh_num, virial, atom_virial)
        return virial

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        list_neigh = inputs[0]
        dE = inputs[1]
        Rij = inputs[2]
        Ri_d = inputs[3]
        dims = list_neigh.shape
        batch_size = dims[0]
        natoms = dims[1]
        neigh_num = dims[2] * dims[3]
        grad = torch.zeros_like(dE)
        op.calculate_virial_force_grad(list_neigh, Rij, Ri_d, grad_output, batch_size, natoms, neigh_num, grad)
        return (None, grad, None, None, None)