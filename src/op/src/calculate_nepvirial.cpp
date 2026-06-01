#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

// Phase B: AT_DISPATCH_FLOATING_TYPES for virial ops.

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
    AT_DISPATCH_FLOATING_TYPES(dE.scalar_type(), "calculate_nepvirial", [&] {
        launch_calculate_nepvirial<scalar_t>(
            nblist.data_ptr<int64_t>(),
            dE.data_ptr<scalar_t>(),
            Rij.data_ptr<scalar_t>(),
            Ri_d.data_ptr<scalar_t>(),
            num_atom.data_ptr<int64_t>(),
            batch_num, natoms, neigh_num,
            virial_force.data_ptr<scalar_t>(),
            atom_virial_force.data_ptr<scalar_t>(),
            device_id
        );
    });
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
    AT_DISPATCH_FLOATING_TYPES(Rij.scalar_type(), "calculate_nepvirial_grad", [&] {
        launch_calculate_nepvirial_grad<scalar_t>(
            nblist.data_ptr<int64_t>(),
            Rij.data_ptr<scalar_t>(),
            Ri_d.data_ptr<scalar_t>(),
            net_grad.data_ptr<scalar_t>(),
            natoms, neigh_num,
            grad.data_ptr<scalar_t>(),
            device_id
        );
    });
}
