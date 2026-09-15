#include "./utilities/nep_utilities.cuh"
#include <iostream>
#include <c10/cuda/CUDAStream.h>
#include "./utilities/error.cuh"

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

// Adjacent lanes consume adjacent basis/type values for one center atom.
__global__ void dfeat_2c_calc_flat(
    const double* grad_output, const double* dfeat_c2,
    const int64_t* atom_map, double* grad_coeff2,
    int64_t natoms, int64_t n_max, int64_t n_base,
    int64_t n_types, int64_t multi_feat_num)
{
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= natoms * n_types * n_base) return;
    const double basis_sum = dfeat_c2[index];
    // Most element types are absent from a center's neighbor list.
    if (basis_sum == 0.0) return;
    const int64_t atom = index / (n_types * n_base);
    const int64_t type_j = (index / n_base) % n_types;
    const int64_t basis = index % n_base;
    const int64_t output = (atom_map[atom] * n_types + type_j) * n_max * n_base + basis;
    for (int n = 0; n < n_max; ++n) {
        atomicAdd(grad_coeff2 + output + n * n_base,
                  grad_output[atom * (n_max + multi_feat_num) + n] * basis_sum);
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
    if (natoms == 0) return;
    constexpr int block_size = 256;
    const auto stream = c10::cuda::getCurrentCUDAStream(device);
    if (grad_coeff2 != nullptr) {
        const int64_t tasks = int64_t(natoms) * n_types * n_base_2b;
        dfeat_2c_calc_flat<<<(tasks + block_size - 1) / block_size,
                            block_size, 0, stream.stream()>>>(
            grad_output, dfeat_c2, atom_map, grad_coeff2,
            natoms, n_max_2b, n_base_2b, n_types, multi_feat_num);
        CUDA_CHECK_KERNEL
    }
    if (neigh_num > 0) {
        const int64_t tasks = int64_t(natoms) * neigh_num;
        dfeat_2b_calc<<<(tasks + block_size - 1) / block_size,
                        block_size, 0, stream.stream()>>>(
            grad_output, dfeat_2b, grad_d12_radial,
            natoms, neigh_num, n_max_2b, multi_feat_num);
        CUDA_CHECK_KERNEL
    }
}
