#include <iostream>
#include "../include/calculate_force.h"
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
//virial
__global__ void atom_nepvirial_reduction(
    double *virial,           // 输出数组，维度为 [num_structures, 9]
    const double *atom_virial, // 输入数组，维度为 [nalls, 9]
    const int64_t *num_atom,  // 每个结构的原子数，维度为 [num_structures]
    const int nalls)          // 总原子数
{
    unsigned int bid = blockIdx.x;  // 当前处理的维度 (0-8)
    unsigned int tid = threadIdx.x; // 线程索引

    // 计算当前结构在 atom_virial 中的起始和结束位置
    int64_t start = 0;
    int64_t end = 0;
    for (int i = 0; i < blockIdx.y; ++i) {
        start += num_atom[i];
    }
    end = start + num_atom[blockIdx.y];

    __shared__ double data[256];
    data[tid] = 0.0;

    // 对当前结构的原子进行归约
    for (int64_t ii = start + tid; ii < end; ii += 256) {
        data[tid] += atom_virial[ii * 9 + bid];
    }
    __syncthreads();

    // 在共享内存中进行归约
    for (int ii = 256 >> 1; ii > 0; ii >>= 1) {
        if (tid < ii) {
            data[tid] += data[tid + ii];
        }
        __syncthreads();
    }

    // 将结果写回全局内存
    if (tid == 0) {
        virial[blockIdx.y * 9 + bid] = data[0];
    }
}

__global__ void nepvirial_deriv_wrt_neighbors_a(
    double * atom_virial,
    const double * dE,
    const double * Ri_d,
    const double * Rij,
    const int64_t * nlist,
    const int natoms,
    const int neigh_num) 
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int virial_index = threadIdx.y;
    // const int ndescrpt = neigh_num * 4;

    if (neigh_index >= neigh_num)
        return;

    const unsigned int nlist_offset = atom_id * neigh_num + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    double virial_tmp = 0.0;
    const unsigned int dE_offset = atom_id * neigh_num * 4 + neigh_index * 4;
    const unsigned int Ri_d_offset = atom_id * neigh_num * 12 + neigh_index * 12;
    const unsigned int Rij_offset = atom_id * neigh_num * 3 + neigh_index * 3;

    for (int idw = 0; idw < 4; ++idw) {
        virial_tmp += dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + virial_index / 3] * Rij[Rij_offset + virial_index % 3];
    }

    const uint index = neigh_id * 9 + virial_index;

    atomicAdd(atom_virial + index, virial_tmp);
}

//virial grad
__device__ inline double nepdev_dot9(
    const double * arr1, 
    const double * arr2) 
{
    double result = 0.0;
    for(int ii = 0; ii < 9; ii++){
        result += arr1[ii] * arr2[ii];
    }
    return result;
}

__global__ void nepvirial_grad_wrt_neighbors_a(
    double * grad_output,    // natoms * neigh_num * 4
    const double * net_grad, // 9
    const double * Ri_d,     // Ri_d
    const double * Rij,
    const int64_t * nlist,
    const int natoms,
    const int neigh_num)
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;

    if (neigh_index >= neigh_num)
        return;

    const unsigned int index_xyzw = threadIdx.y;
    const unsigned int tid = threadIdx.x;

    __shared__ double grad_one[9];
    if(tid < 9){
        grad_one[tid] = net_grad[tid];
    }
    __syncthreads();

    if (atom_id >= natoms) {
        return;
    }

    int j_idx = nlist[atom_id * neigh_num + neigh_index];

    if (j_idx < 0) {
        return;
    }

    double tmp[9];
    const int Rij_offset = atom_id * neigh_num * 3 + neigh_index * 3;
    const int Ri_d_offset = atom_id * neigh_num * 4 * 3 + neigh_index * 4 * 3 + index_xyzw * 3;

    for (int dd0 = 0; dd0 < 3; ++dd0){
        for (int dd1 = 0; dd1 < 3; ++dd1){
            tmp[dd0 * 3 + dd1] = Rij[Rij_offset + dd1] * Ri_d[Ri_d_offset + dd0];
        }
    }

    const int grad_output_offset = atom_id * neigh_num * 4 + neigh_index * 4 + index_xyzw;
    grad_output[grad_output_offset] += nepdev_dot9(grad_one, tmp);
}


void launch_calculate_nepvirial(
    const int64_t * nblist,
    const double * dE,
    const double * Rij,
    const double * Ri_d,
    const int64_t * num_atom,
    const int batch_num,
    const int natoms,
    const int neigh_num,
    double * virial,
    double * atom_virial,
    const int device_id
)
{
    cudaSetDevice(device_id);
    const int LEN = 16;
    int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms);
    dim3 thread_grid(LEN, 9);
    // compute virial of a frame
    nepvirial_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(
        atom_virial, 
        dE, 
        Ri_d, 
        Rij, 
        nblist, 
        natoms, 
        neigh_num
        );

    block_grid = dim3(9, batch_num);
    // reduction atom_virial to virial
    atom_nepvirial_reduction<<<block_grid, 256>>>(
        virial, 
        atom_virial, 
        num_atom,
        natoms
        );
}

void launch_calculate_nepvirial_grad(
    const int64_t * nblist,
    const double * Rij,
    const double * Ri_d,
    const double * net_grad,
    const int natoms,
    const int neigh_num,
    double * grad_output,
    const int device_id
)
{
    cudaSetDevice(device_id);
    int LEN = 128;
    // const int ndesc = neigh_num * 4;

    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms);
    dim3 thread_grid(LEN, 4);
    
    nepvirial_grad_wrt_neighbors_a<<<block_grid, thread_grid>>>(
        grad_output, 
        net_grad, 
        Ri_d, 
        Rij, 
        nblist, 
        natoms, 
        neigh_num
        );
}
