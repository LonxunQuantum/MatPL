#include <torch/extension.h>
#include "../include/calculate_force.h"

void torch_launch_calculate_force(torch::Tensor &nblist,
                       const torch::Tensor &dE,
                       const torch::Tensor &Ri_d,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &force,
                       int64_t nghost
) 
{
    auto dtype = dE.dtype();
    assert(Ri_d.dtype() == dtype);
    assert(force.dtype() == dtype);
    int device_id = force.device().index();
    if (dtype == torch::kFloat32)
    {
        launch_calculate_force<float>(
            (const int *) nblist.data_ptr(),
            (const float *) dE.data_ptr(),
            (const float *) Ri_d.data_ptr(),
            batch_size, natoms, neigh_num,
            (float *) force.data_ptr(),
            nghost, device_id
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_force<double>(
            (const int *) nblist.data_ptr(),
            (const double *) dE.data_ptr(),
            (const double *) Ri_d.data_ptr(),
            batch_size, natoms, neigh_num,
            (double *) force.data_ptr(),
            nghost, device_id
        );
    }
    else
        printf("data type error!");
}
