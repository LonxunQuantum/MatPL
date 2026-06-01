#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

// Phase B: AT_DISPATCH_FLOATING_TYPES for force ops.

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
    AT_DISPATCH_FLOATING_TYPES(dE.scalar_type(), "calculate_nepforce", [&] {
        launch_calculate_nepforce<scalar_t>(
            nblist.data_ptr<int64_t>(),
            dE.data_ptr<scalar_t>(),
            Ri_d.data_ptr<scalar_t>(),
            natoms, neigh_num,
            force.data_ptr<scalar_t>(),
            device_id
        );
    });
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
    AT_DISPATCH_FLOATING_TYPES(Ri_d.scalar_type(), "calculate_nepforce_grad", [&] {
        launch_calculate_nepforce_grad<scalar_t>(
            nblist.data_ptr<int64_t>(),
            Ri_d.data_ptr<scalar_t>(),
            net_grad.data_ptr<scalar_t>(),
            natoms, neigh_num,
            grad.data_ptr<scalar_t>(),
            device_id
        );
    });
}
