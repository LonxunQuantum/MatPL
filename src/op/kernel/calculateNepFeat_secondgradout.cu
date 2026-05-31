#include "./utilities/nep_utilities.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <iostream>
#include <cuda_runtime.h>

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

template <typename T>
__global__ void compute_gradsecond_gradout(
    const T *grad_second,
    const T *dfeat_2b,
    T *gradsecond_gradout,
    int atom_nums,
    int maxneighs,
    int n_max_2b)
{
    int atom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (atom_idx < atom_nums) {
        for (int neigh = 0; neigh < maxneighs; ++neigh) {
            for (int n = 0; n < n_max_2b; ++n) {
                T grad_second_val = grad_second[atom_idx * maxneighs * 4 + neigh * 4];
                T dfeat_2b_val = dfeat_2b[atom_idx * maxneighs * n_max_2b + neigh * n_max_2b + n];
                gradsecond_gradout[atom_idx * n_max_2b + n] += grad_second_val * dfeat_2b_val;
            }
        }
    }
}

template <typename T>
__global__ void compute_gradsecond_c2(
    const T *grad_second,
    const T *de_feat,
    const T *dfeat_2b_noc,
    T *tmp_grad,
    int atom_nums,
    int maxneighs,
    int n_max_2b,
    int n_base_2b,
    int multi_feat_num)
{
    int total_elements = atom_nums * maxneighs;
    int elem_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (elem_idx < total_elements) {
        int atom_idx = elem_idx / maxneighs;
        int maxneigh_idx = elem_idx % maxneighs;

        int dfeat_start = atom_idx * (n_max_2b + multi_feat_num);
        int dnoc_start = atom_idx * maxneighs * n_base_2b * 4 + maxneigh_idx * n_base_2b * 4;
        int grad2_start = atom_idx * maxneighs * 4 + maxneigh_idx * 4;
        int tmp_grad_start = atom_idx * maxneighs * n_max_2b * n_base_2b + maxneigh_idx * n_max_2b * n_base_2b;

        T noc0 = T(0.0), noc1 = T(0.0), noc2 = T(0.0), noc3 = T(0.0);
        T dfeat_val = T(0.0);

        T grad0 = grad_second[grad2_start];
        T grad1 = grad_second[grad2_start + 1];
        T grad2 = grad_second[grad2_start + 2];
        T grad3 = grad_second[grad2_start + 3];

        for (int n = 0; n < n_max_2b; ++n) {
            dfeat_val = de_feat[dfeat_start + n];
            for (int k = 0; k < n_base_2b; ++k) {
                noc0 = dfeat_2b_noc[dnoc_start + k * 4];
                noc1 = dfeat_2b_noc[dnoc_start + k * 4 + 1];
                noc2 = dfeat_2b_noc[dnoc_start + k * 4 + 2];
                noc3 = dfeat_2b_noc[dnoc_start + k * 4 + 3];

                tmp_grad[tmp_grad_start + n * n_base_2b + k] += dfeat_val * (noc0 * grad0 + noc1 * grad1 + noc2 * grad2 + noc3 * grad3);
            }
        }
    }
}

template <typename T>
__global__ void reduce_kernel(
    T *tmp_grad,
    const int64_t *atom_map,
    const int64_t *NL_radial,
    const int atom_nums,
    const int maxneighs,
    const int n_max_2b,
    const int n_base_2b,
    const int atom_types,
    T *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / maxneighs;
    int maxneighs_j = idx % maxneighs;
    if (i < atom_nums && maxneighs_j < maxneighs) {
        int n2 = NL_radial[i * maxneighs + maxneighs_j];
        if (n2 < 0) return;
        int tmp_grad_start = i * maxneighs * n_max_2b * n_base_2b
                                + maxneighs_j * n_max_2b * n_base_2b;

        int atom_type_j = atom_map[n2];
        int atom_type_i = atom_map[i];

        for (int k = 0; k < n_max_2b; ++k) {
            for (int l = 0; l < n_base_2b; ++l) {
                atomicAdd(&output[atom_type_i * atom_types * n_max_2b * n_base_2b
                                + atom_type_j * n_max_2b * n_base_2b
                                + k * n_base_2b
                                + l],
                                tmp_grad[tmp_grad_start + k * n_base_2b + l]
                );
            }
        }
    }
}

template <typename T>
void launch_calculate_nepfeat_secondgradout(
    const T * grad_second,
    const T * dfeat_b,
    T * gradsecond_gradout,
    const int atom_nums,
    const int maxneighs,
    const int n_max,
    const int device
) {
    cudaSetDevice(device);
    dim3 threadsPerBlock(16);
    dim3 numBlocks((atom_nums + threadsPerBlock.x - 1) / threadsPerBlock.x);

    compute_gradsecond_gradout<T><<<numBlocks, threadsPerBlock>>>(
        grad_second,
        dfeat_b,
        gradsecond_gradout,
        atom_nums,
        maxneighs,
        n_max
        );

    CUDA_CHECK_KERNEL
}

template <typename T>
void launch_calculate_nepfeat_secondgradout_c2(
    const T * grad_second,
    const T * de_feat,
    const T * dfeat_2b_noc,
    const int64_t* atom_map,
    const int64_t* NL_radial,
    T * gradsecond_c2,
    const int atom_nums,
    const int maxneighs,
    const int n_max_2b,
    const int n_base_2b,
    const int atom_types,
    const int multi_feat_num,
    const int device
) {
    cudaSetDevice(device);
    int total_elements = atom_nums * maxneighs;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    GPU_Vector<T> tmp_grad(atom_nums * maxneighs * n_max_2b * n_base_2b, T(0.0));
    compute_gradsecond_c2<T><<<num_blocks, threads_per_block>>>(
        grad_second,
        de_feat,
        dfeat_2b_noc,
        tmp_grad.data(),
        atom_nums,
        maxneighs,
        n_max_2b,
        n_base_2b,
        multi_feat_num
        );
    cudaDeviceSynchronize();

    reduce_kernel<T><<<num_blocks, threads_per_block>>>(
        tmp_grad.data(), atom_map, NL_radial,
        atom_nums, maxneighs,
        n_max_2b, n_base_2b, atom_types, gradsecond_c2
    );

    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
}

// Explicit instantiations
template void launch_calculate_nepfeat_secondgradout<double>(
    const double*, const double*, double*, const int, const int, const int, const int);
template void launch_calculate_nepfeat_secondgradout<float>(
    const float*, const float*, float*, const int, const int, const int, const int);

template void launch_calculate_nepfeat_secondgradout_c2<double>(
    const double*, const double*, const double*, const int64_t*, const int64_t*,
    double*, const int, const int, const int, const int, const int, const int, const int);
template void launch_calculate_nepfeat_secondgradout_c2<float>(
    const float*, const float*, const float*, const int64_t*, const int64_t*,
    float*, const int, const int, const int, const int, const int, const int, const int);
