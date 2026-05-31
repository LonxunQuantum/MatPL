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

template <typename T>
__global__ void atom_nepvirial_reduction(
    T *virial,
    const T *atom_virial,
    const int64_t *num_atom,
    const int nalls)
{
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    int64_t start = 0;
    int64_t end = 0;
    for (int i = 0; i < blockIdx.y; ++i) {
        start += num_atom[i];
    }
    end = start + num_atom[blockIdx.y];

    __shared__ T data[256];
    data[tid] = T(0.0);

    for (int64_t ii = start + tid; ii < end; ii += 256) {
        data[tid] += atom_virial[ii * 9 + bid];
    }
    __syncthreads();

    for (int ii = 256 >> 1; ii > 0; ii >>= 1) {
        if (tid < ii) {
            data[tid] += data[tid + ii];
        }
        __syncthreads();
    }

    if (tid == 0) {
        virial[blockIdx.y * 9 + bid] = data[0];
    }
}

template <typename T>
__global__ void nepvirial_deriv_wrt_neighbors_a(
    T * atom_virial,
    const T * dE,
    const T * Ri_d,
    const T * Rij,
    const int64_t * nlist,
    const int natoms,
    const int neigh_num)
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int virial_index = threadIdx.y;

    if (neigh_index >= neigh_num)
        return;

    const unsigned int nlist_offset = atom_id * neigh_num + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    T virial_tmp = T(0.0);
    const unsigned int dE_offset = atom_id * neigh_num * 4 + neigh_index * 4;
    const unsigned int Ri_d_offset = atom_id * neigh_num * 12 + neigh_index * 12;
    const unsigned int Rij_offset = atom_id * neigh_num * 3 + neigh_index * 3;

    for (int idw = 0; idw < 4; ++idw) {
        virial_tmp += dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + virial_index / 3] * Rij[Rij_offset + virial_index % 3];
    }

    const uint index = neigh_id * 9 + virial_index;

    atomicAdd(atom_virial + index, virial_tmp);
}

template <typename T>
__device__ inline T nepdev_dot9(
    const T * arr1,
    const T * arr2)
{
    T result = T(0.0);
    for(int ii = 0; ii < 9; ii++){
        result += arr1[ii] * arr2[ii];
    }
    return result;
}

template <typename T>
__global__ void nepvirial_grad_wrt_neighbors_a(
    T * grad_output,
    const T * net_grad,
    const T * Ri_d,
    const T * Rij,
    const int64_t * nlist,
    const int natoms,
    const int neigh_num)
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;

    if (neigh_index >= neigh_num)
        return;

    const unsigned int nlist_offset = atom_id * neigh_num + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    const unsigned int dE_offset = atom_id * neigh_num * 4 + neigh_index * 4;
    const unsigned int Ri_d_offset = atom_id * neigh_num * 12 + neigh_index * 12;
    const unsigned int Rij_offset = atom_id * neigh_num * 3 + neigh_index * 3;

    for (int idw = 0; idw < 4; ++idw) {
        T tmp[9];
        for (int v = 0; v < 9; ++v) {
            tmp[v] = Ri_d[Ri_d_offset + idw * 3 + v / 3] * Rij[Rij_offset + v % 3];
        }
        grad_output[dE_offset + idw] += nepdev_dot9(net_grad, tmp);
    }
}

template <typename T>
void launch_calculate_nepvirial(
    const int64_t * nblist,
    const T * dE,
    const T * Rij,
    const T * Ri_d,
    const int64_t * num_atom,
    const int batch_num,
    const int natoms,
    const int neigh_num,
    T * virial,
    T * atom_virial,
    const int device_id
)
{
    cudaSetDevice(device_id);
    const int LEN = 16;
    int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms);
    dim3 thread_grid(LEN, 9);
    nepvirial_deriv_wrt_neighbors_a<T><<<block_grid, thread_grid>>>(
        atom_virial,
        dE,
        Ri_d,
        Rij,
        nblist,
        natoms,
        neigh_num
        );

    block_grid = dim3(9, batch_num);
    atom_nepvirial_reduction<T><<<block_grid, 256>>>(
        virial,
        atom_virial,
        num_atom,
        natoms
        );
}

template <typename T>
void launch_calculate_nepvirial_grad(
    const int64_t * nblist,
    const T * Rij,
    const T * Ri_d,
    const T * net_grad,
    const int natoms,
    const int neigh_num,
    T * grad_output,
    const int device_id
)
{
    cudaSetDevice(device_id);
    int LEN = 128;

    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms);
    dim3 thread_grid(LEN, 4);

    nepvirial_grad_wrt_neighbors_a<T><<<block_grid, thread_grid>>>(
        grad_output,
        net_grad,
        Ri_d,
        Rij,
        nblist,
        natoms,
        neigh_num
        );
}

// Explicit instantiations
template void launch_calculate_nepvirial<double>(
    const int64_t*, const double*, const double*, const double*, const int64_t*,
    const int, const int, const int, double*, double*, const int);
template void launch_calculate_nepvirial<float>(
    const int64_t*, const float*, const float*, const float*, const int64_t*,
    const int, const int, const int, float*, float*, const int);

template void launch_calculate_nepvirial_grad<double>(
    const int64_t*, const double*, const double*, const double*, const int, const int, double*, const int);
template void launch_calculate_nepvirial_grad<float>(
    const int64_t*, const float*, const float*, const float*, const int, const int, float*, const int);
