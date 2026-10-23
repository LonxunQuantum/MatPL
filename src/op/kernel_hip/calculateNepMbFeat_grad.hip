#include <hip/hip_runtime.h>
#include "./utilities/common.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep3_small_box.cuh"
#include <iostream>
#include <cstdlib>
#include <c10/hip/HIPStream.h>
#include "./utilities/nep_angular_grad_demand.cuh"
#include "./utilities/nep_angular_hiprtc.hpp"

void launch_calculate_nepmbfeat_grad(
            const double * grad_output,
            const double * coeff3,
            const double * r12,
            const int64_t * NL,
            const int64_t * atom_map,
            double * sum_fxyz,
            double * grad_coeff3,
            double * grad_d12_3b,
            double * dsnlm_dc, // dsnlm/dc_NK_IJ used in second grad mb c
            double * dfeat_drij,
            const int rcut_angular,
            const int atom_nums,
            const int neigh_num,
            const int feat_2b_num,
            const int n_max_3b,
            const int n_base_3b,
            const int lmax_3,
            const int lmax_4,
            const int lmax_5,
            const int n_types,
            const int device_id
) {
    hipSetDevice(device_id);
    const int BLOCK_SIZE = 64;
    const int N = atom_nums;
    const int num_types_sq = n_types * n_types;
    double rcinv_angular = 1.0 / rcut_angular;

    int feat_3b_num = 0;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;

    // 只读取本次调用所需的输出；保留旧实现用于数值和性能对照。
    const char* prune_env = std::getenv("MATPL_NEP_ANGULAR_PRUNE");
    if (prune_env == nullptr || prune_env[0] != '0') {
        if (N == 0 || neigh_num == 0) return;
        const char* use_fused=std::getenv("MATPL_NEP_ANGULAR_FUSED");
        const char* use_rtc=std::getenv("MATPL_NEP_ANGULAR_HIPRTC");
        if ((!use_fused || use_fused[0]!='0') && (!use_rtc || use_rtc[0]!='0')) {
            const unsigned outputs=(grad_coeff3?1:0)|(dsnlm_dc?2:0)|(dfeat_drij?4:0);
            TORCH_CHECK(n_max_3b>=1 && n_max_3b<=MAX_NUM_N && n_base_3b>=1 && n_base_3b<=MAX_NUM_N &&
                n_types>0 && lmax_3==4 && (lmax_4==0 || lmax_4==2) && (lmax_5==0 || lmax_5==1) && (!lmax_5 || lmax_4),
                "NEP angular fused HIPRTC requires the implemented CUDA NEP3/4/5 domain: L3=4, L4=0/2, L5=0/1 with L5 implying L4, 1<=NN,KB<=20");
            matpl_angular_rtc::launch(N,n_types,n_max_3b,n_base_3b,lmax_4,lmax_5,outputs,device_id,
                neigh_num,feat_2b_num,feat_3b_num,rcut_angular,rcinv_angular,NL,r12,coeff3,atom_map,
                grad_output,sum_fxyz,dfeat_drij,grad_d12_3b,grad_coeff3,dsnlm_dc);
            return;
        }
        // 首先验证 16_metal 形状的编译期输出裁剪和寄存器资源。
        if (n_types == 16 && n_max_3b == 5 && n_base_3b == 9 && lmax_3 == 4 && lmax_4 == 2 && lmax_5 == 0) {
            const auto stream = c10::hip::getCurrentHIPStream(device_id);
            const unsigned outputs = (grad_coeff3 ? 1 : 0) | (dsnlm_dc ? 2 : 0) | (dfeat_drij ? 4 : 0);
            const char* fused_env=std::getenv("MATPL_NEP_ANGULAR_FUSED");
            if(fused_env==nullptr || fused_env[0]!='0') {
#define MATPL_ANGULAR_FUSED(M) matpl_angular_demand::find_angular_grad_phase_fused<9,5,16,M><<<N,32,0,stream.stream()>>>(neigh_num,feat_2b_num,feat_3b_num,rcut_angular,rcinv_angular,NL,r12,coeff3,atom_map,grad_output,sum_fxyz,dfeat_drij,grad_d12_3b,grad_coeff3,dsnlm_dc)
                switch(outputs) {
                    case 0:MATPL_ANGULAR_FUSED(0);break;
                    case 1:MATPL_ANGULAR_FUSED(1);break;
                    case 4:MATPL_ANGULAR_FUSED(4);break;
                    case 6:MATPL_ANGULAR_FUSED(6);break;
                    default:goto generic_output_prune;
                }
#undef MATPL_ANGULAR_FUSED
                CUDA_CHECK_KERNEL
                return;
            }

#define MATPL_ANGULAR_LAUNCH(MODE) \
 matpl_angular_demand::find_angular_grad_spatial_split<9,5,16,2,0,(MODE==4)><<<dim3(N,5),64,0,stream.stream()>>>( \
 neigh_num,feat_2b_num,feat_3b_num,rcut_angular,rcinv_angular,NL,r12,coeff3,atom_map,grad_output,sum_fxyz,dfeat_drij,grad_d12_3b)
            switch (outputs) {
                case 0: MATPL_ANGULAR_LAUNCH(0); break;
                case 1:
                    MATPL_ANGULAR_LAUNCH(0);
                    matpl_angular_demand::find_angular_grad_coeff_split<9,5,16,2,0><<<dim3(N,5),64,0,stream.stream()>>>(
                        N,neigh_num,feat_2b_num,feat_3b_num,rcut_angular,rcinv_angular,NL,r12,atom_map,grad_output,sum_fxyz,grad_coeff3);
                    break;
                case 4: MATPL_ANGULAR_LAUNCH(4); break;
                case 6:
                    MATPL_ANGULAR_LAUNCH(4);
                    matpl_angular_demand::find_angular_grad_dsnlm_split<9,16><<<dim3(N,9),64,0,stream.stream()>>>(
                        N,neigh_num,rcut_angular,rcinv_angular,NL,r12,atom_map,dsnlm_dc);
                    break;
                default: goto generic_output_prune;
            }
#undef MATPL_ANGULAR_LAUNCH
            CUDA_CHECK_KERNEL
            return;
        }
        generic_output_prune:
        // 低类型数时优先块内归约；不改变输出裁剪的语义。
        const size_t coeff_len = static_cast<size_t>(n_types) * n_max_3b * n_base_3b;
        const size_t lds_bytes = (MAX_DIM_ANGULAR + n_max_3b * NUM_OF_ABC + coeff_len +
            (grad_coeff3 != nullptr ? coeff_len : 0) +
            (dsnlm_dc != nullptr ? static_cast<size_t>(n_types) * n_base_3b * NUM_OF_ABC : 0)) * sizeof(double);
        const auto stream = c10::hip::getCurrentHIPStream(device_id);
        const bool use_pruned_lds = n_types <= 4 && n_max_3b <= MAX_NUM_N &&
            n_base_3b <= MAX_NUM_N && lmax_3 == 4 && (lmax_4 == 0 || lmax_4 == 2) &&
            (lmax_5 == 0 || lmax_5 == 1) && n_max_3b * (lmax_3 + (lmax_4 > 0) + (lmax_5 > 0)) <= MAX_DIM_ANGULAR &&
            lds_bytes <= 16 * 1024;
        if (use_pruned_lds) {
            matpl_angular_lds_demand::find_angular_grad_output_pruned_lds<<<N, BLOCK_SIZE, lds_bytes, stream.stream()>>>(
                N, n_types, num_types_sq, neigh_num, lmax_3, lmax_4, lmax_5,
                feat_2b_num, feat_3b_num, rcut_angular, rcinv_angular,
                n_max_3b, n_base_3b, NL, r12, coeff3, atom_map,
                grad_output - feat_2b_num, sum_fxyz, dsnlm_dc, grad_coeff3,
                dfeat_drij, grad_d12_3b);
            CUDA_CHECK_KERNEL
            return;
        }
        size_t shared_bytes = (MAX_DIM_ANGULAR + n_max_3b * NUM_OF_ABC) * sizeof(double);
        shared_bytes = (shared_bytes + 255) & ~size_t(255);
        matpl_angular_demand::find_angular_grad_output_pruned<<<N, BLOCK_SIZE, shared_bytes, stream.stream()>>>(
            N, n_types, num_types_sq, neigh_num, lmax_3, lmax_4, lmax_5,
            feat_2b_num, feat_3b_num, rcut_angular, rcinv_angular,
            n_max_3b, n_base_3b, NL, r12, coeff3, atom_map,
            grad_output - feat_2b_num, sum_fxyz, dsnlm_dc, grad_coeff3,
            dfeat_drij, grad_d12_3b);
        CUDA_CHECK_KERNEL
        return;
    }

    GPU_Vector<double> dfeat_c3(N * n_types * n_max_3b * n_base_3b, 0.0);

    constexpr size_t MAX_SMALL_BOX_OPT_SHARED_BYTES = 48 * 1024;
    const size_t small_box_opt_shared_bytes =
        static_cast<size_t>(MAX_DIM_ANGULAR + n_max_3b * NUM_OF_ABC +
                            2 * n_types * n_max_3b * n_base_3b +
                            n_types * n_base_3b * NUM_OF_ABC) *
        sizeof(double);
    const bool use_small_box_opt =
        n_max_3b <= MAX_NUM_N && n_base_3b <= MAX_NUM_N &&
        lmax_3 == 4 && lmax_4 == 2 &&
        (lmax_5 == 0 || lmax_5 == 1) &&
        n_max_3b * (lmax_3 + (lmax_4 > 0) + (lmax_5 > 0)) <= MAX_DIM_ANGULAR &&
        small_box_opt_shared_bytes <= MAX_SMALL_BOX_OPT_SHARED_BYTES;
    const size_t active_types_shared_bytes =
        static_cast<size_t>(MAX_DIM_ANGULAR + n_max_3b * NUM_OF_ABC +
                            2 * n_max_3b * n_base_3b +
                            n_base_3b * NUM_OF_ABC) *
        sizeof(double);
    const bool use_active_types_opt =
        n_types <= 128 && n_max_3b <= MAX_NUM_N && n_base_3b <= MAX_NUM_N &&
        lmax_3 == 4 && lmax_4 == 2 &&
        (lmax_5 == 0 || lmax_5 == 1) &&
        n_max_3b * (lmax_3 + (lmax_4 > 0) + (lmax_5 > 0)) <= MAX_DIM_ANGULAR &&
        active_types_shared_bytes <= MAX_SMALL_BOX_OPT_SHARED_BYTES;
    if (use_active_types_opt && !use_small_box_opt) {
        find_angular_gard_small_box_active_types<<<
            N, BLOCK_SIZE, static_cast<int>(active_types_shared_bytes)>>>(
            N, n_types, num_types_sq, neigh_num, lmax_3, lmax_4, lmax_5,
            feat_2b_num, feat_3b_num, rcut_angular, rcinv_angular,
            n_max_3b, n_base_3b, NL, r12, coeff3, atom_map,
            grad_output - feat_2b_num, sum_fxyz, dsnlm_dc, dfeat_c3.data(),
            dfeat_drij, grad_d12_3b);
    } else if (use_small_box_opt) {
        const int shared_bytes =
            (MAX_DIM_ANGULAR + n_max_3b * NUM_OF_ABC +
             2 * n_types * n_max_3b * n_base_3b +
             n_types * n_base_3b * NUM_OF_ABC) *
            static_cast<int>(sizeof(double));
        find_angular_gard_small_box_opt<<<N, BLOCK_SIZE, shared_bytes>>>(
            N, n_types, num_types_sq, neigh_num, lmax_3, lmax_4, lmax_5,
            feat_2b_num, feat_3b_num, rcut_angular, rcinv_angular,
            n_max_3b, n_base_3b, NL, r12, coeff3, atom_map,
            grad_output - feat_2b_num, sum_fxyz, dsnlm_dc, dfeat_c3.data(),
            dfeat_drij, grad_d12_3b);
    } else {
        const int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        find_angular_gard_small_box<<<grid_size, BLOCK_SIZE>>>(
            N, n_types, num_types_sq, neigh_num, lmax_3, lmax_4, lmax_5,
            feat_2b_num, feat_3b_num, rcut_angular, rcinv_angular,
            n_max_3b, n_base_3b, NL, r12, coeff3, atom_map,
            grad_output - feat_2b_num, sum_fxyz, dsnlm_dc, dfeat_c3.data(),
            dfeat_drij, grad_d12_3b);
    }
    CUDA_CHECK_KERNEL

    // print_dfeat_c3(dfeat_c3.data(), N, n_types, n_max_3b, n_base_3b);

    int total_elements = N * n_max_3b * n_base_3b;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<<<num_blocks, threads_per_block>>>(
    dfeat_c3.data(),
    atom_map,
    grad_coeff3,
    N,
    n_types,
    n_max_3b,
    n_base_3b);
    CUDA_CHECK_KERNEL

    // print_grad_coeff3(grad_coeff3, n_types, n_max_3b, n_base_3b);

}
