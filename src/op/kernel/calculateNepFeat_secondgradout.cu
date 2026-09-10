#include "./utilities/nep_utilities.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <iostream>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

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

__global__ void compute_gradsecond_gradout(
    const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
    const double *dfeat_2b,    // Shape: [batch_size, atom_nums, maxneighs, n_max_2b]
    double *gradsecond_gradout, // Shape: [batch_size, atom_nums, n_max_2b]
    int atom_nums,
    int maxneighs,
    int n_max_2b)
{
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom_idx < atom_nums) {
        for (int neigh = 0; neigh < maxneighs; ++neigh) {
            for (int n = 0; n < n_max_2b; ++n) {
                // 取 grad_second 的最后一个维度的第0列
                double grad_second_val = grad_second[atom_idx * maxneighs * 4 + neigh * 4];
                // dfeat_2b 的最后一个维度
                double dfeat_2b_val = dfeat_2b[atom_idx * maxneighs * n_max_2b + neigh * n_max_2b + n];
                // 累加到 gradsecond_gradout
                gradsecond_gradout[atom_idx * n_max_2b + n] += grad_second_val * dfeat_2b_val;
            }
        }
    }
}

__global__ void compute_gradsecond_c2_fused(
    const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
    const double *de_feat, // Shape: [batch_size, atom_nums, n_max_2b]
    const double *dfeat_2b_noc, // Shape: [batch_size, atom_nums, maxneighs, n_base_2b, 4]
    const int64_t *atom_map,
    const int64_t *NL_radial,
    double *gradsecond_c2, // Shape: [atom_types, atom_types, n_max_2b, n_base_2b]
    int atom_nums,
    int maxneighs,
    int n_max_2b,
    int n_base_2b,
    int atom_types,
    int multi_feat_num)
{
    const int64_t task = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t pair_count = static_cast<int64_t>(atom_nums) * maxneighs;
    if (task >= pair_count * n_base_2b) return;

    const int basis = task % n_base_2b;
    const int64_t pair = task / n_base_2b;
    const int64_t neighbor = NL_radial[pair];
    if (neighbor < 0) return;

    const int atom = pair / maxneighs;
    const int atom_type_i = static_cast<int>(atom_map[atom]);
    const int atom_type_j = static_cast<int>(atom_map[neighbor]);
    const int64_t grad_start = pair * 4;
    const int64_t noc_start = (pair * n_base_2b + basis) * 4;
    const double contraction =
        dfeat_2b_noc[noc_start] * grad_second[grad_start] +
        dfeat_2b_noc[noc_start + 1] * grad_second[grad_start + 1] +
        dfeat_2b_noc[noc_start + 2] * grad_second[grad_start + 2] +
        dfeat_2b_noc[noc_start + 3] * grad_second[grad_start + 3];

    const int64_t dfeat_start =
        static_cast<int64_t>(atom) * (n_max_2b + multi_feat_num);
    const int64_t output_start =
        (atom_type_i * atom_types + atom_type_j) * n_max_2b * n_base_2b + basis;
    for (int n = 0; n < n_max_2b; ++n) {
        atomicAdd(&gradsecond_c2[output_start + n * n_base_2b],
                  de_feat[dfeat_start + n] * contraction);
    }
}

// grad_second dim is [batch, atoms, neighs, 4]
// dfeat_b dim is [batch, atoms, neighs, n_max_2b], dfeat/drij  the x, y, z is 0.
// do grad_second * dfeat_b
// the out tensor gradsecond_gradout is [batch, atoms, n_max_2b]
void launch_calculate_nepfeat_secondgradout(
    const double * grad_second,
    const double * dfeat_b,
    double * gradsecond_gradout,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max, 
    const int device
) {
    cudaSetDevice(device);
    dim3 threadsPerBlock(16);
    dim3 numBlocks((atom_nums + threadsPerBlock.x - 1) / threadsPerBlock.x);

    compute_gradsecond_gradout<<<numBlocks, threadsPerBlock>>>(
        grad_second, 
        dfeat_b, 
        gradsecond_gradout,
        atom_nums, 
        maxneighs, 
        n_max
        );

    CUDA_CHECK_KERNEL
}



void launch_calculate_nepfeat_secondgradout_c2(
    const double * grad_second,
    const double * de_feat,
    const double * dfeat_2b_noc,
    const int64_t* atom_map,
    const int64_t* NL_radial,
    double * gradsecond_c2,
    const int atom_nums, 
    const int maxneighs, 
    const int n_max_2b, 
    const int n_base_2b, 
    const int atom_types,
    const int multi_feat_num, 
    const int device
) {
    cudaSetDevice(device);
    constexpr int threads_per_block = 256;
    const int64_t total_tasks =
        static_cast<int64_t>(atom_nums) * maxneighs * n_base_2b;
    const int num_blocks = static_cast<int>(
        (total_tasks + threads_per_block - 1) / threads_per_block);
    const auto stream = c10::cuda::getCurrentCUDAStream(device);
    compute_gradsecond_c2_fused<<<num_blocks, threads_per_block, 0, stream.stream()>>>(
        grad_second, 
        de_feat, 
        dfeat_2b_noc,
        atom_map,
        NL_radial,
        gradsecond_c2,
        atom_nums, 
        maxneighs, 
        n_max_2b,
        n_base_2b,
        atom_types,
        multi_feat_num
        );
    CUDA_CHECK_KERNEL
}
