#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

// Phase A: tighten data_ptr casts and assert fp64 dtype.
// Same rationale as calculate_nepfeat.cpp — silently reinterpreting
// fp32 as fp64 was UB. Phase B will replace these with template
// dispatch; here we only harden the fp64 path.

#define MATPL_REQUIRE_DOUBLE(t, name)                                            \
    TORCH_CHECK((t).scalar_type() == torch::kDouble,                             \
                name " currently only supports float64 inputs; got ",            \
                (t).scalar_type())

void torch_launch_calculate_nepforce(
    const torch::Tensor &nblist,
    const torch::Tensor &dE,
    const torch::Tensor &Ri_d,
    int64_t natoms,
    int64_t neigh_num,
    torch::Tensor &force
)
{
    MATPL_REQUIRE_DOUBLE(dE, "calculate_nepforce");
    MATPL_REQUIRE_DOUBLE(Ri_d, "calculate_nepforce");
    MATPL_REQUIRE_DOUBLE(force, "calculate_nepforce");
    int device_id = force.device().index();
    launch_calculate_nepforce(
        nblist.data_ptr<int64_t>(),
        dE.data_ptr<double>(),
        Ri_d.data_ptr<double>(),
        natoms, neigh_num,
        force.data_ptr<double>(),
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
    MATPL_REQUIRE_DOUBLE(Ri_d, "calculate_nepforce_grad");
    MATPL_REQUIRE_DOUBLE(net_grad, "calculate_nepforce_grad");
    int device_id = nblist.device().index();
    launch_calculate_nepforce_grad(
        nblist.data_ptr<int64_t>(),
        Ri_d.data_ptr<double>(),
        net_grad.data_ptr<double>(),
        natoms, neigh_num,
        grad.data_ptr<double>(),
        device_id
    );
}

#undef MATPL_REQUIRE_DOUBLE
