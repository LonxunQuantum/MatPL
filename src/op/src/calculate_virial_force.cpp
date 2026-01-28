#include <torch/extension.h>
#include "../include/calculate_force.h"

void torch_launch_calculate_virial_force(torch::Tensor &nblist,
                       const torch::Tensor &dE,
                       const torch::Tensor &Rij,
                       const torch::Tensor &Ri_d,
                       int64_t batch_size,
                       int64_t natoms,
                       int64_t neigh_num,
                       const torch::Tensor &virial_force,
                       const torch::Tensor &atom_virial_force,
                       int64_t nghost
) 
{
    auto dtype = dE.dtype();
    assert(Rij.dtype() == dtype);
    assert(Ri_d.dtype() == dtype);
    assert(virial_force.dtype() == dtype);
    int device_id = virial_force.device().index();
    if (dtype == torch::kFloat32)
    {
        launch_calculate_virial_force<float>(
            (const int *) nblist.data_ptr(),
            (const float *) dE.data_ptr(),
            (const float *) Rij.data_ptr(),
            (const float *) Ri_d.data_ptr(),
            batch_size, natoms, neigh_num,
            (float *) virial_force.data_ptr(),
            (float *) atom_virial_force.data_ptr(),
            nghost, device_id
        );
    } else if (dtype == torch::kFloat64)
    {
        launch_calculate_virial_force<double>(
            (const int *) nblist.data_ptr(),
            (const double *) dE.data_ptr(),
            (const double *) Rij.data_ptr(),
            (const double *) Ri_d.data_ptr(),
            batch_size, natoms, neigh_num,
            (double *) virial_force.data_ptr(),
            (double *) atom_virial_force.data_ptr(),
            nghost, device_id
        );
    }
    else
        printf("data type error!");
}
