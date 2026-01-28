#include "./utilities/common.cuh"
#include "./utilities/nep3_small_box.cuh"
#include "./utilities/nep3_small_box_mbgrad.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <iostream>
#include <cuda_runtime.h>

__global__ void compute_gradsecond_mbgradout(
    const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
    const double *dfeat_drij,    // Shape: [batch_size, atom_nums, maxneighs, feat_mb_num, 4]
    double *gradsecond_gradout, // Shape: [batch_size, atom_nums, feat_mb_num]
    int atom_nums,
    int maxneighs,
    int feat_mb_num)
{
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (atom_idx < atom_nums && feat_idx < feat_mb_num) {
        float sum = 0.0f;
        for (int neigh_idx = 0; neigh_idx < maxneighs; ++neigh_idx) {
            for (int dim = 0; dim < 4; ++dim) {
                int dfeat_index = ((atom_idx * maxneighs + neigh_idx) * feat_mb_num + feat_idx) * 4 + dim;
                int grad_index = (atom_idx * maxneighs + neigh_idx) * 4 + dim;
                sum += dfeat_drij[dfeat_index] * grad_second[grad_index];
            }
        }
        gradsecond_gradout[atom_idx * feat_mb_num + feat_idx] = sum;
    }
}

void launch_calculate_nepmbfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_gradout,
    const int atom_nums, 
    const int maxneighs, 
    const int feat_mb_nums, 
    const int device
) {
    cudaSetDevice(device);
    dim3 blockDim(16, 16);
    dim3 gridDim((atom_nums + blockDim.x - 1) / blockDim.x, (feat_mb_nums + blockDim.y - 1) / blockDim.y);
    compute_gradsecond_mbgradout<<<gridDim, blockDim>>>(
        grad_second, 
        dfeat_b, 
        gradsecond_gradout,
        atom_nums, 
        maxneighs, 
        feat_mb_nums
        );

    CUDA_CHECK_KERNEL
}

void launch_calculate_nepmbfeat_secondgradout_c3(
    const double * grad_second,
    const double * d12,
    const int64_t * NL,
    const double * de_dfeat,
    const double * dsnlm_dc,
    const double * sum_fxyz,
    const int64_t * atom_map,
    const double * coeff3,
    double * gradsecond_c3,
    const double rcut_angular,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_3b, 
    const int n_base_3b, 
    const int atom_types, 
    const int lmax_3,
    const int lmax_4,
    const int lmax_5,
    const int feat_2b_num,
    const int multi_feat_num,
    const int device
){
    cudaSetDevice(device);
    // // 每个线程块的线程数 (这里选择 8 * 16 * 2 = 256，保证不会超过 1024)
    double rcinv_angular = 1.0 / rcut_angular;
    int atom_types_sq = atom_types * atom_types;
    // 按照邻居粒度存在一个问题，线程内分配的数组太多了，可能炸显存，但是速度会快，后面可以试试

    // const int BLOCK_SIZE = 64;
    // const int N = batch_size * atom_nums; // N = natoms * batch_size
    // const int grid_size = (N - 1) / BLOCK_SIZE + 1;

    // GPU_Vector<double> dfeat_c3(N * atom_types * n_max_3b * n_base_3b, 0.0);
    // find_angular_gardc_small_box<<<grid_size, BLOCK_SIZE>>>(
    //     N,
    //     grad_second,
    //     d12, 
    //     NL,
    //     de_dfeat-feat_2b_num, 
    //     dsnlm_dc,
    //     sum_fxyz,
    //     atom_map,
    //     coeff3,
    //     dfeat_c3.data(),
    //     rcut_angular,
    //     rcinv_angular,
    //     batch_size, 
    //     atom_nums, 
    //     maxneighs, 
    //     n_max_3b,
    //     n_base_3b,
    //     atom_types,
    //     atom_types_sq,
    //     lmax_3,
    //     lmax_4,
    //     lmax_5,
    //     feat_2b_num,
    //     multi_feat_num
    //     );
    // cudaDeviceSynchronize();
    // printf("==== tmp_c3 =====\n");
    // std::vector<double> tmp_c3(N * atom_types * n_max_3b * n_base_3b);
    // dfeat_c3.copy_to_host(tmp_c3.data());
    // for(int i = 0; i < N; i++) {
    //     for(int j = 0; j < atom_types; j++) {
    //         for (int n = 0; n < n_max_3b; n++) {
    //             printf("tmp_c3[i %d][J %d][n %d][k:] = ", i, j, n);
    //             for(int k = 0; k < n_base_3b; k++) {
    //                 printf("%f ", tmp_c3[i * atom_types * n_max_3b * n_base_3b + j * n_max_3b * n_base_3b + n * n_base_3b + k]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }

    int total_elements = atom_nums * maxneighs;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    const int N = atom_nums;
    GPU_Vector<double> dfeat_c3(N * maxneighs * atom_types * n_max_3b * n_base_3b, 0.0);
    find_angular_gardc_neigh<<<num_blocks, threads_per_block>>>(
        total_elements,
        grad_second,
        d12, 
        NL,
        de_dfeat-feat_2b_num, 
        dsnlm_dc,
        sum_fxyz,
        atom_map,
        coeff3,
        dfeat_c3.data(),
        rcut_angular,
        rcinv_angular,
        atom_nums, 
        maxneighs, 
        n_max_3b,
        n_base_3b,
        atom_types,
        atom_types_sq,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num,
        multi_feat_num
        );
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
    
    GPU_Vector<double> tmp_dfeat_c3(N * atom_types * n_max_3b * n_base_3b, 0.0);

    const int BLOCK_SIZE = 64;
    const int grid_size = (N - 1) / BLOCK_SIZE + 1;
    aggregate_dfeat_c3<<<grid_size, BLOCK_SIZE>>>(
        NL,
        atom_map,
        dfeat_c3.data(),
        tmp_dfeat_c3.data(),
        N,
        atom_nums, 
        maxneighs, 
        atom_types,
        n_max_3b,
        n_base_3b
    );
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();

    total_elements = N * n_max_3b * n_base_3b;
    threads_per_block = 256;
    num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    aggregate_features<<<num_blocks, threads_per_block>>>(
    tmp_dfeat_c3.data(), 
    atom_map, 
    gradsecond_c3, 
    N, 
    atom_types, 
    n_max_3b, 
    n_base_3b);   
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
}
