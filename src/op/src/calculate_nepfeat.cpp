#include <torch/extension.h>
#include "../include/calculate_nepfeat.h"

// Phase B: AT_DISPATCH_FLOATING_TYPES routes to the correct template
// instantiation based on the input tensor's dtype at runtime.

//2b
void torch_launch_calculate_nepfeat(
                        const torch::Tensor &coeff2,
                        const torch::Tensor &d12_radial,
                        const torch::Tensor &NL_radial,
                        const torch::Tensor &atom_map,
                        const double rcut_radial,
                        torch::Tensor &feat_2b,
                        torch::Tensor &dfeat_c2,
                        torch::Tensor &dfeat_2b,
                        torch::Tensor &dfeat_2b_noc,
                        int64_t natoms,
                        int64_t neigh_num,
                        int64_t n_max,
                        int64_t n_base,
                        int64_t n_types
){
    int device_id = d12_radial.device().index();
    AT_DISPATCH_FLOATING_TYPES(d12_radial.scalar_type(), "calculate_nepfeat", [&] {
        launch_calculate_nepfeat<scalar_t>(
            coeff2.data_ptr<scalar_t>(),
            d12_radial.data_ptr<scalar_t>(),
            NL_radial.data_ptr<int64_t>(),
            atom_map.data_ptr<int64_t>(),
            static_cast<scalar_t>(rcut_radial),
            feat_2b.data_ptr<scalar_t>(),
            dfeat_c2.data_ptr<scalar_t>(),
            dfeat_2b.data_ptr<scalar_t>(),
            dfeat_2b_noc.data_ptr<scalar_t>(),
            natoms, neigh_num, n_max, n_base, n_types,
            device_id
        );
    });
}

void torch_launch_calculate_nepfeat_grad(const torch::Tensor &grad_output,
                                const torch::Tensor &dfeat_c2,
                                const torch::Tensor &dfeat_2b,
                                const torch::Tensor atom_map,
                                int64_t atom_nums,
                                int64_t maxneighs,
                                int64_t n_max_2b,
                                int64_t n_base_2b,
                                int64_t n_types,
                                int64_t multi_feat_num,
                                torch::Tensor &grad_coeff2,
                                torch::Tensor &grad_d12_radial)
{
    int device_id = dfeat_c2.device().index();
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "calculate_nepfeat_grad", [&] {
        launch_calculate_nepfeat_grad<scalar_t>(
            grad_output.data_ptr<scalar_t>(),
            dfeat_c2.data_ptr<scalar_t>(),
            dfeat_2b.data_ptr<scalar_t>(),
            atom_map.data_ptr<int64_t>(),
            grad_coeff2.data_ptr<scalar_t>(),
            grad_d12_radial.data_ptr<scalar_t>(),
            atom_nums, maxneighs, n_max_2b, n_base_2b, n_types, multi_feat_num,
            device_id
        );
    });
}


void torch_launch_calculate_nepfeat_secondgradout(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &dfeat_b,
                        const int64_t atom_nums,
                        const int64_t maxneighs,
                        const int64_t n_max,
                        torch::Tensor &gradsecond_gradout)
{
    int device_id = dfeat_b.device().index();
    AT_DISPATCH_FLOATING_TYPES(grad_second.scalar_type(), "calculate_nepfeat_secondgradout", [&] {
        launch_calculate_nepfeat_secondgradout<scalar_t>(
            grad_second.data_ptr<scalar_t>(),
            dfeat_b.data_ptr<scalar_t>(),
            gradsecond_gradout.data_ptr<scalar_t>(),
            atom_nums, maxneighs, n_max, device_id
        );
    });
}


void torch_launch_calculate_nepfeat_secondgradout_c2(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &de_feat,
                        const torch::Tensor &dfeat_2b_noc,
                        const torch::Tensor &atom_map,
                        const torch::Tensor &NL_radial,
                        const int64_t atom_nums,
                        const int64_t maxneighs,
                        const int64_t n_max_2b,
                        const int64_t n_base_2b,
                        const int64_t atom_types,
                        const int64_t multi_feat_num,
                        torch::Tensor &gradsecond_c2
)
{
    int device_id = de_feat.device().index();
    AT_DISPATCH_FLOATING_TYPES(grad_second.scalar_type(), "calculate_nepfeat_secondgradout_c2", [&] {
        launch_calculate_nepfeat_secondgradout_c2<scalar_t>(
            grad_second.data_ptr<scalar_t>(),
            de_feat.data_ptr<scalar_t>(),
            dfeat_2b_noc.data_ptr<scalar_t>(),
            atom_map.data_ptr<int64_t>(),
            NL_radial.data_ptr<int64_t>(),
            gradsecond_c2.data_ptr<scalar_t>(),
            atom_nums, maxneighs, n_max_2b, n_base_2b, atom_types, multi_feat_num, device_id
        );
    });
}

//multi feats
void torch_launch_calculate_nepmbfeat(
                        const torch::Tensor &coeff3,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,
                        const torch::Tensor &atom_map,
                        torch::Tensor &feat_3b,
                        torch::Tensor &dfeat_c3,
                        torch::Tensor &dfeat_3b,
                        torch::Tensor &dfeat_3b_noc,
                        torch::Tensor &sum_fxyz,
                        const double rcut,
                        int64_t natoms,
                        int64_t neigh_num,
                        int64_t n_max_3b,
                        int64_t n_base_3b,
                        int64_t lmax_3,
                        int64_t lmax_4,
                        int64_t lmax_5,
                        int64_t n_types
){
    int device_id = d12.device().index();
    AT_DISPATCH_FLOATING_TYPES(d12.scalar_type(), "calculate_nepmbfeat", [&] {
        launch_calculate_nepmbfeat<scalar_t>(
            coeff3.data_ptr<scalar_t>(),
            d12.data_ptr<scalar_t>(),
            NL.data_ptr<int64_t>(),
            atom_map.data_ptr<int64_t>(),
            feat_3b.data_ptr<scalar_t>(),
            dfeat_c3.data_ptr<scalar_t>(),
            dfeat_3b.data_ptr<scalar_t>(),
            dfeat_3b_noc.data_ptr<scalar_t>(),
            sum_fxyz.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rcut),
            natoms, neigh_num, n_max_3b, n_base_3b, lmax_3, lmax_4, lmax_5, n_types,
            device_id
        );
    });
}

void torch_launch_calculate_nepmbfeat_grad(
                        const torch::Tensor &grad_output,
                        const torch::Tensor &coeff3,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,
                        const torch::Tensor &atom_map,
                        const double rcut_angular,
                        int64_t atom_nums,
                        int64_t maxneighs,
                        int64_t feat_2b_num,
                        int64_t n_max_3b,
                        int64_t n_base_3b,
                        int64_t lmax_3,
                        int64_t lmax_4,
                        int64_t lmax_5,
                        int64_t n_types,
                        torch::Tensor &sum_fxyz,
                        torch::Tensor &grad_coeff3,
                        torch::Tensor &grad_d12_3b,
                        torch::Tensor &dsnlm_dc,
                        torch::Tensor &dfeat_drij
){
    int device_id = d12.device().index();
    AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "calculate_nepmbfeat_grad", [&] {
        launch_calculate_nepmbfeat_grad<scalar_t>(
            grad_output.data_ptr<scalar_t>(),
            coeff3.data_ptr<scalar_t>(),
            d12.data_ptr<scalar_t>(),
            NL.data_ptr<int64_t>(),
            atom_map.data_ptr<int64_t>(),
            sum_fxyz.data_ptr<scalar_t>(),
            grad_coeff3.data_ptr<scalar_t>(),
            grad_d12_3b.data_ptr<scalar_t>(),
            dsnlm_dc.data_ptr<scalar_t>(),
            dfeat_drij.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rcut_angular),
            atom_nums, maxneighs, feat_2b_num,
            n_max_3b, n_base_3b, lmax_3, lmax_4, lmax_5, n_types,
            device_id
        );
    });
}

void torch_launch_calculate_nepmbfeat_secondgradout(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &dfeat_b,
                        const int64_t atom_nums,
                        const int64_t maxneighs,
                        const int64_t feat_mu_nums,
                        torch::Tensor &gradsecond_gradout)
{
    int device_id = dfeat_b.device().index();
    AT_DISPATCH_FLOATING_TYPES(grad_second.scalar_type(), "calculate_nepmbfeat_secondgradout", [&] {
        launch_calculate_nepmbfeat_secondgradout<scalar_t>(
            grad_second.data_ptr<scalar_t>(),
            dfeat_b.data_ptr<scalar_t>(),
            gradsecond_gradout.data_ptr<scalar_t>(),
            atom_nums, maxneighs, feat_mu_nums, device_id
        );
    });
}

void torch_launch_calculate_nepmbfeat_secondgradout_c3(
                        const torch::Tensor &grad_second,
                        const torch::Tensor &d12,
                        const torch::Tensor &NL,
                        const torch::Tensor &de_feat,
                        const torch::Tensor &dsnlm_dc,
                        const torch::Tensor &sum_fxyz,
                        const torch::Tensor &atom_map,
                        const torch::Tensor &coeff3,
                        const double rcut_angular,
                        const int64_t atom_nums,
                        const int64_t maxneighs,
                        const int64_t n_max_3b,
                        const int64_t n_base_3b,
                        const int64_t atom_types,
                        const int64_t lmax_3,
                        const int64_t lmax_4,
                        const int64_t lmax_5,
                        const int64_t feat_2b_num,
                        const int64_t multi_feat_num,
                        torch::Tensor &gradsecond_c3
){
    int device_id = de_feat.device().index();
    AT_DISPATCH_FLOATING_TYPES(grad_second.scalar_type(), "calculate_nepmbfeat_secondgradout_c3", [&] {
        launch_calculate_nepmbfeat_secondgradout_c3<scalar_t>(
            grad_second.data_ptr<scalar_t>(),
            d12.data_ptr<scalar_t>(),
            NL.data_ptr<int64_t>(),
            de_feat.data_ptr<scalar_t>(),
            dsnlm_dc.data_ptr<scalar_t>(),
            sum_fxyz.data_ptr<scalar_t>(),
            atom_map.data_ptr<int64_t>(),
            coeff3.data_ptr<scalar_t>(),
            gradsecond_c3.data_ptr<scalar_t>(),
            static_cast<scalar_t>(rcut_angular),
            atom_nums, maxneighs, n_max_3b, n_base_3b, atom_types,
            lmax_3, lmax_4, lmax_5, feat_2b_num, multi_feat_num, device_id
        );
    });
}
