#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

void torch_launch_calculate_nepvirial(
    const torch::Tensor &nblist,
    const torch::Tensor &dE,
    const torch::Tensor &Rij,
    const torch::Tensor &Ri_d,
    const torch::Tensor &num_atom,
    int64_t batch_num,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor &virial_force,
    torch::Tensor &atom_virial_force
){
    int device_id = virial_force.device().index();
    launch_calculate_nepvirial(
        (const int64_t *) nblist.data_ptr(),
        (const double *) dE.data_ptr(),
        (const double *) Rij.data_ptr(),
        (const double *) Ri_d.data_ptr(),
        (const int64_t *) num_atom.data_ptr(),
        batch_num, natoms, neigh_num,
        (double *) virial_force.data_ptr(),
        (double *) atom_virial_force.data_ptr(),
        device_id
    );
}

void torch_launch_calculate_nepvirial_grad(
    const torch::Tensor &nblist,
    const torch::Tensor &Rij,
    const torch::Tensor &Ri_d,
    const torch::Tensor &net_grad,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor &grad
){
    int device_id = nblist.device().index();
    launch_calculate_nepvirial_grad(
        (const int64_t *) nblist.data_ptr(),
        (const double *) Rij.data_ptr(),
        (const double *) Ri_d.data_ptr(),
        (const double *) net_grad.data_ptr(),
        natoms, neigh_num,
        (double *) grad.data_ptr(),
        device_id
    );
}

