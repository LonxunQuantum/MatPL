#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

void torch_launch_calculate_nepforce(
    const torch::Tensor &nblist,
    const torch::Tensor &dE,
    const torch::Tensor &Ri_d,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor &force
) 
{
    int device_id = force.device().index();
    launch_calculate_nepforce(
        (const int64_t *) nblist.data_ptr(),
        (const double *) dE.data_ptr(),
        (const double *) Ri_d.data_ptr(),
        natoms, neigh_num,
        (double *) force.data_ptr(),
        device_id
    );
}

void torch_launch_calculate_nepforce_grad(
    const torch::Tensor &nblist,
    const torch::Tensor &Ri_d,
    const torch::Tensor &net_grad,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor &grad
) 
{
    int device_id = nblist.device().index();
    launch_calculate_nepforce_grad(
        (const int64_t *) nblist.data_ptr(),
        (const double *) Ri_d.data_ptr(),
        (const double *) net_grad.data_ptr(),
        natoms, neigh_num,
        (double *) grad.data_ptr(),
        device_id
    );
}
