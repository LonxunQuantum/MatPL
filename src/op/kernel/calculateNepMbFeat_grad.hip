#include <hip/hip_runtime.h>
#include "./utilities/common.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep3_small_box.cuh"
#include <iostream>

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
    const int BLOCK_SIZE = 256;
    const int N = atom_nums;
    const int num_types_sq = n_types * n_types;
    double rcinv_angular = 1.0 / rcut_angular;
    
    int feat_3b_num = 0;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;
    
    // 优化版本使用两阶段 kernel
    GPU_Vector<double> local_dfeat_c3(N * neigh_num * n_types * n_max_3b * n_base_3b, 0.0);
    GPU_Vector<double> dfeat_c3(N * n_types * n_max_3b * n_base_3b, 0.0);
    
    // 1. 并行计算局部梯度
    find_angular_gard_small_box_optimized<<<N, BLOCK_SIZE>>>(
        N,
        n_types,
        num_types_sq,
        neigh_num,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num, 
        feat_3b_num,
        rcut_angular,
        rcinv_angular,
        n_max_3b, 
        n_base_3b,
        NL,
        r12,
        coeff3,
        atom_map,
        grad_output - feat_2b_num,
        sum_fxyz,
        dsnlm_dc,
        local_dfeat_c3.data(),
        dfeat_drij,
        grad_d12_3b
    );
    CUDA_CHECK_KERNEL
    
    // 2. 规约求和
    int total_threads = N * neigh_num;
    int grid_size = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_local_dfeat_c3<<<grid_size, BLOCK_SIZE>>>(
        NL,
        local_dfeat_c3.data(),
        dfeat_c3.data(),
        N,
        neigh_num,
        n_types,
        n_max_3b,
        n_base_3b
    );
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
