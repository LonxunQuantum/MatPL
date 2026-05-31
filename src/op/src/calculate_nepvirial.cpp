#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

// Phase A: tighten data_ptr casts and assert fp64 dtype.
// Same rationale as calculate_nepfeat.cpp.

#define MATPL_REQUIRE_DOUBLE(t, name)                                            \
    TORCH_CHECK((t).scalar_type() == torch::kDouble,                             \
                name " currently only supports float64 inputs; got ",            \
                (t).scalar_type())

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
    MATPL_REQUIRE_DOUBLE(dE, "calculate_nepvirial");
    MATPL_REQUIRE_DOUBLE(Rij, "calculate_nepvirial");
    MATPL_REQUIRE_DOUBLE(Ri_d, "calculate_nepvirial");
    int device_id = virial_force.device().index();
    launch_calculate_nepvirial(
        nblist.data_ptr<int64_t>(),
        dE.data_ptr<double>(),
        Rij.data_ptr<double>(),
        Ri_d.data_ptr<double>(),
        num_atom.data_ptr<int64_t>(),
        batch_num, natoms, neigh_num,
        virial_force.data_ptr<double>(),
        atom_virial_force.data_ptr<double>(),
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
    MATPL_REQUIRE_DOUBLE(Rij, "calculate_nepvirial_grad");
    MATPL_REQUIRE_DOUBLE(Ri_d, "calculate_nepvirial_grad");
    MATPL_REQUIRE_DOUBLE(net_grad, "calculate_nepvirial_grad");
    int device_id = nblist.device().index();
    launch_calculate_nepvirial_grad(
        nblist.data_ptr<int64_t>(),
        Rij.data_ptr<double>(),
        Ri_d.data_ptr<double>(),
        net_grad.data_ptr<double>(),
        natoms, neigh_num,
        grad.data_ptr<double>(),
        device_id
    );
}

#undef MATPL_REQUIRE_DOUBLE
