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
    const uint atom_type = atom_map[atom_idx];  // 获取原子类型
    // if (n_type_idx != atom_type) return; // 只累加对应的类型
    const uint A_idx = atom_idx * (n_max + multi_feat_num) + n_max_idx;
    const uint B_idx = atom_idx * n_types * n_base + n_type_idx * n_base + n_base_idx;
    const uint C_idx = atom_type * n_types * n_max * n_base + n_type_idx * n_max * n_base + n_max_idx * n_base + n_base_idx;
    
    // 将 A 和 B 的元素相乘并累加到 C 中
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
    int global_atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算批次和原子索引
    int atom_idx = global_atom_index;
    if (atom_idx >= natoms) return;
    const uint type_i = atom_map[atom_idx];
    uint A_idx = 0;
    uint B_idx_start = atom_idx * n_types * n_base;
    uint C_idx_start = type_i * n_types * n_max * n_base;
    uint C_idx = 0;
    for (int n = 0; n < n_max; n++) {
        A_idx = atom_idx * (n_max + multi_feat_num) + n;
        for (int j = 0; j < n_types; j++) {
            for (int k = 0; k < n_base; k++) {
                C_idx = C_idx_start + j * n_max * n_base + n * n_base + k;
                atomicAdd(grad_coeff2 + C_idx, grad_output[A_idx] * dfeat_c2[B_idx_start + j * n_base + k]);
            }
        }
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
        // 对 n_max 维度加和，逐元素乘法并加和
        for (int i = 0; i < n_max; ++i) {
            // if ((atom_idx == 0 or atom_idx == 1) and (neigh_idx < 10)) {
            //     printf("atom %d j %d grad_out[0][%d][%d] = %f dfeat_%d_drij = %f\n", atom_idx, neigh_idx, atom_idx, i,
            //             grad_output[grad_out_idx + i],
            //             i, dfeat_2b[batch_idx * natoms * neigh_num * n_max + atom_idx * neigh_num * n_max + neigh_idx * n_max + i]
            //             );
            // }
            sum += grad_output[grad_out_idx + i] * 
                   dfeat_2b[atom_idx * neigh_num * n_max + neigh_idx * n_max + i];
        }

        grad_d12_radial[atom_idx * neigh_num * 4 + neigh_idx * 4] = sum;
    }
}

// grad_output shape is [batch_size, natoms, (n_max_2b+multifeature)]
// dfeat_c2 shape is    [batch_size, n_types_J, n_base_2b]
// dfeat_2b shape is    [batch_size, natoms, n_max_2b, maxneighs]
// atom_map shape is    [natoms]
// grad_coeff2 shape is [n_types_I, n_max_2b, n_types_J, n_base_2b]
// grad_d12_radial shape is [batch_size, natoms, n_max_2b, maxneighs]
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
    cudaSetDevice(device);
    int n_types_sq = n_types * n_types;
    int BLOCK_SIZE = 64; //common value
    int grid_size = (natoms - 1) / BLOCK_SIZE + 1;//common value
    
    if (n_max_2b * n_types * n_base_2b > 1000) {
        dfeat_2c_calc_large<<<grid_size, BLOCK_SIZE>>>(
            grad_output, dfeat_c2, atom_map, grad_coeff2, 
                        natoms, n_max_2b, n_base_2b, n_types, n_types_sq, multi_feat_num);
    } else {
        dim3 threads(n_types, n_max_2b, n_base_2b);
        dim3 blocks(natoms);
        dfeat_2c_calc<<<blocks, threads>>>(
                    grad_output, dfeat_c2, atom_map, grad_coeff2, 
                                natoms, n_max_2b, n_base_2b, n_types, n_types_sq, multi_feat_num);
    }
    grid_size = (natoms * neigh_num - 1) / BLOCK_SIZE + 1;
    dfeat_2b_calc<<<grid_size, BLOCK_SIZE>>>(
            grad_output, dfeat_2b, grad_d12_radial, 
                        natoms, neigh_num, n_max_2b, multi_feat_num);
}
