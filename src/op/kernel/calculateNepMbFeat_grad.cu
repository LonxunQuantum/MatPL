#include "./utilities/error.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep_feature.cuh"
#include <c10/cuda/CUDAStream.h>

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
    cudaSetDevice(device_id);
    const int N = atom_nums; // N = natoms * batch_size
    const int num_types_sq = n_types * n_types;
    double rcinv_angular = 1.0 / rcut_angular;

    int feat_3b_num = 0;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;

    // 计算共享内存大小
    const int Fp_size = MAX_DIM_ANGULAR;
    const int sum_fxyz_size = n_max_3b * NUM_OF_ABC;
    size_t shared_mem_size = Fp_size * sizeof(double) + sum_fxyz_size * sizeof(double);
    // 对齐到256字节边界
    shared_mem_size = (shared_mem_size + 255) & ~255;
    // Neighbors stride over two warps, keeping the register-heavy block small.
    constexpr int threads_per_block = 64;
    if (N == 0 || neigh_num == 0) return;
    const auto stream = c10::cuda::getCurrentCUDAStream(device_id);
    find_angular_gard<<<N, threads_per_block, shared_mem_size, stream.stream()>>>(
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
        grad_coeff3,
        dfeat_drij,//[batch*atom, neighbornum, 3b_feat_num, 4]
        grad_d12_3b
    );
    CUDA_CHECK_KERNEL
}
