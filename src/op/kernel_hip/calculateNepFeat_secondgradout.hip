#include "hip/hip_runtime.h"
#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/error.cuh"
#include "./utilities/gpu_vector.cuh"
#include <iostream>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

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

// 每个线程处理一个(atom, neighbor)对
__global__ void compute_gradsecond_gradout_optimized(
    const double *grad_second, // Shape: [atom_nums, maxneighs, 4]
    const double *dfeat_2b,    // Shape: [atom_nums, maxneighs, n_max_2b]
    double *gradsecond_gradout, // Shape: [atom_nums, n_max_2b]
    int atom_nums,
    int maxneighs,
    int n_max_2b)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= atom_nums * maxneighs) return;

    int atom_idx = global_idx / maxneighs;
    int neigh = global_idx % maxneighs;

    double grad_second_val = grad_second[atom_idx * maxneighs * 4 + neigh * 4];
    int dfeat_base = atom_idx * maxneighs * n_max_2b + neigh * n_max_2b;
    int output_base = atom_idx * n_max_2b;

    for (int n = 0; n < n_max_2b; ++n) {
        double dfeat_2b_val = dfeat_2b[dfeat_base + n];
        atomicAdd(&gradsecond_gradout[output_base + n], grad_second_val * dfeat_2b_val);
    }
}

__global__ void compute_gradsecond_c2(
    const double *grad_second, // Shape: [batch_size, atom_nums, maxneighs, 4]
    const double *de_feat, // Shape: [batch_size, atom_nums, n_max_2b]
    const double *dfeat_2b_noc, // Shape: [batch_size, atom_nums, maxneighs, n_base_2b, 4]
    double *tmp_grad, // Shape: [batch_size, atom_nums, maxneighs, n_base_2b]
    int atom_nums,
    int maxneighs,
    int n_max_2b,
    int n_base_2b,
    int multi_feat_num)
{
    // 计算总元素数目
    int total_elements = atom_nums * maxneighs;
    int elem_idx = threadIdx.x + blockIdx.x * blockDim.x; // 网格中的元素索引

    if (elem_idx < total_elements) {
        // 计算对应的 atom_idx 和 maxneigh_idx
        int atom_idx = elem_idx / maxneighs;
        int maxneigh_idx = elem_idx % maxneighs;

        int dfeat_start = atom_idx * (n_max_2b + multi_feat_num);
        int dnoc_start = atom_idx * maxneighs * n_base_2b * 4 + maxneigh_idx * n_base_2b * 4;
        int grad2_start = atom_idx * maxneighs * 4 + maxneigh_idx * 4;
        int tmp_grad_start = atom_idx * maxneighs * n_max_2b * n_base_2b + maxneigh_idx * n_max_2b * n_base_2b;

        double noc0 = 0.0, noc1 = 0.0, noc2 = 0.0, noc3 = 0.0;
        double dfeat_val = 0.0;

        double grad0 = grad_second[grad2_start];
        double grad1 = grad_second[grad2_start + 1];
        double grad2 = grad_second[grad2_start + 2];
        double grad3 = grad_second[grad2_start + 3];

        for (int n = 0; n < n_max_2b; ++n) {
            dfeat_val = de_feat[dfeat_start + n];
            for (int k = 0; k < n_base_2b; ++k) {
                noc0 = dfeat_2b_noc[dnoc_start + k * 4];       // 2b is 0
                noc1 = dfeat_2b_noc[dnoc_start + k * 4 + 1];   // 2b is 0
                noc2 = dfeat_2b_noc[dnoc_start + k * 4 + 2];   // 2b is 0
                noc3 = dfeat_2b_noc[dnoc_start + k * 4 + 3];   // 2b is 0

                // 更新tmp_grad数组
                tmp_grad[tmp_grad_start + n * n_base_2b + k] += dfeat_val * (noc0 * grad0 + noc1 * grad1 + noc2 * grad2 + noc3 * grad3);

                // if (batch_idx == 0 and atom_idx == 1 and maxneigh_idx < 20){
                //     printf("tmp_grad[b %d i %d j %d n %d k %d] = %f dfeat_val %f secondgrad %f %f %f %f noc0-4 %f %f %f %f\n",
                //         batch_idx, atom_idx, maxneigh_idx, n, k,
                //             tmp_grad[tmp_grad_start + n * n_base_2b + k], dfeat_val, grad0, grad1, grad2, grad3, noc0, noc1, noc2,noc3 );
                // }
            }
        }
    }
}

// Map one lane to one basis function. This makes neighboring lanes access
// adjacent dfeat_2b_noc entries while preserving the original accumulation.
__global__ void compute_gradsecond_c2_opt(
    const double *grad_second,
    const double *de_feat,
    const double *dfeat_2b_noc,
    double *tmp_grad,
    int atom_nums,
    int maxneighs,
    int n_max_2b,
    int n_base_2b,
    int multi_feat_num)
{
    const int total = atom_nums * maxneighs * n_base_2b;
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total) return;

    const int k = idx % n_base_2b;
    const int pair = idx / n_base_2b;
    const int maxneigh_idx = pair % maxneighs;
    const int atom_idx = pair / maxneighs;
    const int dfeat_start = atom_idx * (n_max_2b + multi_feat_num);
    const int dnoc = pair * n_base_2b * 4 + k * 4;
    const int grad2 = pair * 4;
    const double contracted =
        dfeat_2b_noc[dnoc] * grad_second[grad2] +
        dfeat_2b_noc[dnoc + 1] * grad_second[grad2 + 1] +
        dfeat_2b_noc[dnoc + 2] * grad_second[grad2 + 2] +
        dfeat_2b_noc[dnoc + 3] * grad_second[grad2 + 3];

    int out = pair * n_max_2b * n_base_2b + k;
    for (int n = 0; n < n_max_2b; ++n) {
        tmp_grad[out] = de_feat[dfeat_start + n] * contracted;
        out += n_base_2b;
    }
}

__global__ void compute_gradsecond_c2_fused(
    const double *grad_second,
    const double *de_feat,
    const double *dfeat_2b_noc,
    const int64_t *atom_map,
    const int64_t *NL_radial,
    double *gradsecond_c2,
    int atom_nums,
    int maxneighs,
    int n_max_2b,
    int n_base_2b,
    int atom_types,
    int multi_feat_num)
{
    const int64_t task =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
        (atom_type_i * atom_types + atom_type_j) * n_max_2b * n_base_2b +
        basis;
    for (int n = 0; n < n_max_2b; ++n) {
        atomicAdd(&gradsecond_c2[output_start + n * n_base_2b],
                  de_feat[dfeat_start + n] * contraction);
    }
}

__global__ void reduce_kernel(
    double *tmp_grad,          // 输入的tmp_grad数组，维度为(batch_size, atom_nums, maxneighs, n_max_2b, n_base_2b)
    const int64_t *atom_map,            // atom_map数组，维度为(atom_nums)
    const int64_t *NL_radial,           // NL_radial数组，维度为(batch_size, atom_nums, maxneighs)
    const int atom_nums,            // atom_nums的大小
    const int maxneighs,            // maxneighs的大小
    const int n_max_2b,             // n_max_2b的大小
    const int n_base_2b,            // n_base_2b的大小
    const int atom_types,           // atom_types的大小
    double *output             // 输出数组，维度为(atom_types, atom_types, n_max_2b, n_base_2b)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // 线程索引
    // 计算对应的 batch_idx, atom_idx 和 maxneigh_idx
    int i = idx / maxneighs;
    int maxneighs_j = idx % maxneighs;
    if (i < atom_nums && maxneighs_j < maxneighs) {
        int n2 = NL_radial[i * maxneighs + maxneighs_j];
        if (n2 < 0) return;
        int tmp_grad_start = i * maxneighs * n_max_2b * n_base_2b
                                + maxneighs_j * n_max_2b * n_base_2b;
        // 临时变量用于存储计算结果
        // double result[MAX_NUM_BEADS] = {0.0};

        // 第一步：根据NL_radial规约tmp_grad的maxneighs维度
        // 获取NL_radial中对应邻居的元素类型

        int atom_type_j = atom_map[n2];  // 使用NL_radial索引获取邻居的元素类型
        int atom_type_i = atom_map[i];

        for (int k = 0; k < n_max_2b; ++k) {
            for (int l = 0; l < n_base_2b; ++l) {
                // if (b == 0 and i == 0 and maxneighs_j < 10){
                // printf("t1 %d t2 %d tmp_grad[b %d i %d j %d n %d k %d] = %f\n",
                //     atom_type_i, atom_type_j, b, i, maxneighs_j, k, l,
                //         tmp_grad[tmp_grad_start + k * n_base_2b + l]);
                // }

                // 将之前的结果累加到输出
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

// 优化版本使用wavefront级别的规约消除atomicAdd
__global__ void reduce_kernel_warp_optimized(
    double *tmp_grad,          // [atom_nums, maxneighs, n_max_2b, n_base_2b]
    const int64_t *atom_map,   // [atom_nums]
    const int64_t *NL_radial,  // [atom_nums, maxneighs]
    const int atom_nums,
    const int maxneighs,
    const int n_max_2b,
    const int n_base_2b,
    const int atom_types,
    double *output)            // [atom_types, atom_types, n_max_2b, n_base_2b]
{
    const int WAVEFRONT_SIZE = 64;

    // 每个wavefront负责计算output的一个元素
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WAVEFRONT_SIZE;
    int lane_id = threadIdx.x % WAVEFRONT_SIZE;

    int total_outputs = atom_types * atom_types * n_max_2b * n_base_2b;
    if (warp_id >= total_outputs) return;

    // 解码output索引: [atom_type_i, atom_type_j, k, l]
    int atom_type_i = warp_id / (atom_types * n_max_2b * n_base_2b);
    int remainder = warp_id % (atom_types * n_max_2b * n_base_2b);
    int atom_type_j = remainder / (n_max_2b * n_base_2b);
    int kl = remainder % (n_max_2b * n_base_2b);
    int k = kl / n_base_2b;
    int l = kl % n_base_2b;

    // 每个lane负责扫描部分(atom, neighbor)对
    int total_pairs = atom_nums * maxneighs;
    double local_sum = 0.0;

    for (int pair_idx = lane_id; pair_idx < total_pairs; pair_idx += WAVEFRONT_SIZE) {
        int i = pair_idx / maxneighs;
        int j = pair_idx % maxneighs;

        // 过滤: 只累加匹配的原子类型
        if (atom_map[i] != atom_type_i) continue;

        int n2 = NL_radial[i * maxneighs + j];
        if (n2 < 0) continue;
        if (atom_map[n2] != atom_type_j) continue;

        // 累加tmp_grad的对应元素
        int tmp_grad_idx = pair_idx * n_max_2b * n_base_2b + k * n_base_2b + l;
        local_sum += tmp_grad[tmp_grad_idx];
    }

    // wavefront内规约
    // 使用 __shfl_down 树形规约
    for (int offset = WAVEFRONT_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down(local_sum, offset, WAVEFRONT_SIZE);
    }

    // 每个wavefront的第一个lane写入结果 (无竞争)
    if (lane_id == 0) {
        output[warp_id] = local_sum;
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
    hipSetDevice(device);
    const int BLOCK_SIZE = 256;
    int total_threads = atom_nums * maxneighs;
    int numBlocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const auto stream = c10::hip::getCurrentHIPStream(device);
    compute_gradsecond_gradout_optimized<<<numBlocks, BLOCK_SIZE, 0, stream.stream()>>>(
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
    hipSetDevice(device);
    constexpr int threads_per_block = 256;
    const int64_t total_tasks =
        static_cast<int64_t>(atom_nums) * maxneighs * n_base_2b;
    const int num_blocks = static_cast<int>(
        (total_tasks + threads_per_block - 1) / threads_per_block);
    const auto stream = c10::hip::getCurrentHIPStream(device);
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
        multi_feat_num);

    CUDA_CHECK_KERNEL
}
