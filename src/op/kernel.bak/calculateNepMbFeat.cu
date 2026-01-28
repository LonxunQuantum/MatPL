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
    cudaSetDevice(device_id);
    const int BLOCK_SIZE = 64;
    const int N = natoms;// N = natoms * batch_size
    const int grid_size = (N - 1) / BLOCK_SIZE + 1;
    const int num_types_sq = n_types * n_types;
    double rcinv_angular = 1.0 / rcut_angular;
    int feat_3b_num = 0;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;
    find_mb_descriptor_small_box<<<grid_size, BLOCK_SIZE>>>(
        N,
        n_types,
        num_types_sq,
        neigh_num,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_3b_num,
        rcut_angular,
        rcinv_angular,
        n_max_3b, 
        n_base_3b,
        NL,
        coeff3,
        feat_3b,
        atom_map,
        d12,
        sum_fxyz);
    CUDA_CHECK_KERNEL
}
