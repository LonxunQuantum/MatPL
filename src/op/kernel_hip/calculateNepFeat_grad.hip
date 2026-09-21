#include "hip/hip_runtime.h"
#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include <iostream>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                             __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

__global__ void dfeat_2c_calc(
            const double * grad_output,
            const double * dfeat_c2,
            const int64_t* atom_map,
            double * grad_coeff2,
            int64_t atoms,
            int64_t n_max,
            int64_t n_base,
            int64_t n_types,
            int64_t n_types_sq,
            int64_t multi_feat_num)
{
    const uint atom_idx = blockIdx.x;
    const uint n_type_idx = threadIdx.x;
    const uint n_max_idx = threadIdx.y;
    const uint n_base_idx = threadIdx.z;
    if (atom_idx >= atoms || n_max_idx >= n_max ||
    n_type_idx >= n_types || n_base_idx >= n_base) return;
    const uint atom_type = atom_map[atom_idx];
    const uint A_idx = atom_idx * (n_max + multi_feat_num) + n_max_idx;
    const uint B_idx = atom_idx * n_types * n_base + n_type_idx * n_base + n_base_idx;
    const uint C_idx = atom_type * n_types * n_max * n_base + n_type_idx * n_max * n_base + n_max_idx * n_base + n_base_idx;

    atomicAdd(grad_coeff2+C_idx, grad_output[A_idx] * dfeat_c2[B_idx]);
}

__global__ void dfeat_2c_calc_large(
            const double * grad_output,
            const double * dfeat_c2,
            const int64_t* atom_map,
            double * grad_coeff2,
            int64_t natoms,
            int64_t n_max,
            int64_t n_base,
            int64_t n_types,
            int64_t n_types_sq,
            int64_t multi_feat_num)
{
    const int atom_idx = blockIdx.x;
    if (atom_idx >= natoms) return;

    extern __shared__ double atom_data[];
    double* const cached_grad = atom_data;
    double* const cached_dfeat = atom_data + n_max;
    const int dfeat_size = n_types * n_base;

    for (int n = threadIdx.x; n < n_max; n += blockDim.x) {
        cached_grad[n] = grad_output[atom_idx * (n_max + multi_feat_num) + n];
    }
    for (int index = threadIdx.x; index < dfeat_size; index += blockDim.x) {
        cached_dfeat[index] = dfeat_c2[atom_idx * dfeat_size + index];
    }
    __syncthreads();

    const int type_i = atom_map[atom_idx];
    const int output_size = n_max * dfeat_size;
    const int output_start = type_i * output_size;
    for (int index = threadIdx.x; index < output_size; index += blockDim.x) {
        const int k = index % n_base;
        const int n_j = index / n_base;
        const int n = n_j % n_max;
        const int j = n_j / n_max;
        atomicAdd(grad_coeff2 + output_start + index,
                  cached_grad[n] * cached_dfeat[j * n_base + k]);
    }
}

__global__ void dfeat_2b_calc(
            const double * grad_output,
            const double * dfeat_2b,
            double * grad_d12_radial,
            int64_t natoms,
            int64_t neigh_num,
            int64_t n_max,
            int64_t multi_feat_num
            )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < natoms * neigh_num) {
        int atom_idx = index / neigh_num;
        int neigh_idx = index % neigh_num;
        int grad_out_idx = atom_idx * (n_max + multi_feat_num);
        double sum = 0.0;
        for (int i = 0; i < n_max; ++i) {
            sum += grad_output[grad_out_idx + i] *
                   dfeat_2b[atom_idx * neigh_num * n_max + neigh_idx * n_max + i];
        }

        grad_d12_radial[atom_idx * neigh_num * 4 + neigh_idx * 4] = sum;
    }
}

void launch_calculate_nepfeat_grad(
            const double * grad_output,
            const double * dfeat_c2,
            const double * dfeat_2b,
            const int64_t * atom_map,
            double * grad_coeff2,
            double * grad_d12_radial,
            const int natoms,
            const int neigh_num,
            const int n_max_2b,
            const int n_base_2b,
            const int n_types,
            const int multi_feat_num,
            const int device
) {
    hipSetDevice(device);
    int n_types_sq = n_types * n_types;
    const int BLOCK_SIZE = 64;

    if (n_max_2b * n_types * n_base_2b > 1000) {
        const int LARGE_BLOCK_SIZE = 256;
        const size_t shared_bytes =
            static_cast<size_t>(n_max_2b + n_types * n_base_2b) * sizeof(double);
        dfeat_2c_calc_large<<<natoms, LARGE_BLOCK_SIZE, shared_bytes>>>(
            grad_output, dfeat_c2, atom_map, grad_coeff2,
            natoms, n_max_2b, n_base_2b, n_types, n_types_sq, multi_feat_num);
    } else {
        dim3 threads(n_types, n_max_2b, n_base_2b);
        dim3 blocks(natoms);
        dfeat_2c_calc<<<blocks, threads>>>(
            grad_output, dfeat_c2, atom_map, grad_coeff2,
            natoms, n_max_2b, n_base_2b, n_types, n_types_sq, multi_feat_num);
    }
    const int grid_size = (natoms * neigh_num - 1) / BLOCK_SIZE + 1;
    dfeat_2b_calc<<<grid_size, BLOCK_SIZE>>>(
            grad_output, dfeat_2b, grad_d12_radial,
            natoms, neigh_num, n_max_2b, multi_feat_num);
}
