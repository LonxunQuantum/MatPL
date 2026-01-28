#include <torch/extension.h>
#include "../include/calculate_force_grad.h"

void torch_launch_calculate_force_grad(torch::Tensor &nblist,
                       const torch::Tensor &Ri_d,
                       const torch::Tensor &net_grad,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &grad
) 
{
    auto dtype = Ri_d.dtype();
    assert(net_grad.dtype() == dtype);
    assert(grad.dtype() == dtype);
    if (dtype == torch::kFloat32)
    {
        launch_calculate_force_grad(
            (const int *) nblist.data_ptr(),
            (const float *) Ri_d.data_ptr(),
            (const float *) net_grad.data_ptr(),
            batch_size, natoms, neigh_num,
            (float *) grad.data_ptr()
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_force_grad(
            (const int *) nblist.data_ptr(),
            (const double *) Ri_d.data_ptr(),
            (const double *) net_grad.data_ptr(),
            batch_size, natoms, neigh_num,
            (double *) grad.data_ptr()
        );
    } else
        printf("data type error!");
    
}
