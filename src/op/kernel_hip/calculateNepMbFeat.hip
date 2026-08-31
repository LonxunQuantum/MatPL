#include <hip/hip_runtime.h>
#include "./utilities/error.cuh"
#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep3_small_box.cuh"
#include <iostream>

void launch_calculate_nepmbfeat(
    const double * coeff3,
    const double * d12,
    const int64_t * NL,
    const int64_t * atom_map,
    double * feat_3b,
    double * dfeat_c3,
    double * dfeat_3b,
    double * dfeat_3b_noc,
    double * sum_fxyz,
    const double rcut_angular,
    const int natoms,
    const int neigh_num,
    const int n_max_3b,
    const int n_base_3b,
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int n_types,
    const int device_id
){
    hipSetDevice(device_id);
    const int N = natoms;// N = natoms * batch_size
    const int num_types_sq = n_types * n_types;
    double rcinv_angular = 1.0 / rcut_angular;
    int feat_3b_num = 0;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;

    // 优化版本使用两阶段kernel
    // 1. 计算 s (需要先清零 sum_fxyz)
    const int total_s_size = N * n_max_3b * NUM_OF_ABC;
    hipMemset(sum_fxyz, 0, total_s_size * sizeof(double));

    const int BLOCK_SIZE_STAGE1 = 256;
    const int total_threads_stage1 = N * n_max_3b * neigh_num;
    const int grid_size_stage1 = (total_threads_stage1 + BLOCK_SIZE_STAGE1 - 1) / BLOCK_SIZE_STAGE1;

    compute_s_optimized<<<grid_size_stage1, BLOCK_SIZE_STAGE1>>>(
        N,
        neigh_num,
        n_max_3b,
        n_types,
        n_base_3b,
        NL,
        d12,
        atom_map,
        coeff3,
        rcut_angular,
        rcinv_angular,
        sum_fxyz);
    CUDA_CHECK_KERNEL

    // 2. 从 s 计算 q 和 feats
    const int BLOCK_SIZE_STAGE2 = 256;
    const int grid_size_stage2 = (N + BLOCK_SIZE_STAGE2 - 1) / BLOCK_SIZE_STAGE2;

    compute_q_and_feats_optimized<<<grid_size_stage2, BLOCK_SIZE_STAGE2>>>(
        N,
        n_max_3b,
        feat_3b_num,
        lmax_3,
        lmax_4,
        lmax_5,
        sum_fxyz,
        feat_3b);
    CUDA_CHECK_KERNEL
}
