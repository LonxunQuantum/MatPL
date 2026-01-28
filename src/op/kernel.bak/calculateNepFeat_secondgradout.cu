#include "./utilities/common.cuh"
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
    // 每个线程块的线程数 (这里选择 8 * 16 * 2 = 256，保证不会超过 1024)
    int total_elements = atom_nums * maxneighs;
    int threads_per_block = 256;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    GPU_Vector<double> tmp_grad(atom_nums * maxneighs * n_max_2b * n_base_2b, 0.0);
    compute_gradsecond_c2<<<num_blocks, threads_per_block>>>(
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

    reduce_kernel<<<num_blocks, threads_per_block>>>(
        tmp_grad.data(), atom_map, NL_radial,
        atom_nums, maxneighs,
        n_max_2b, n_base_2b, atom_types, gradsecond_c2
    );
    
    CUDA_CHECK_KERNEL
    cudaDeviceSynchronize();
}
