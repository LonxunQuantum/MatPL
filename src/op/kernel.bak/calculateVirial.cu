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
template <typename DType, int THREADS_PER_BLOCK>
__global__ void atom_virial_reduction(
    DType * virial, 
    const DType * atom_virial,
    const int nall)
{
    unsigned int batch_index = blockIdx.y;
    unsigned int bid = blockIdx.x;
    unsigned int tid = threadIdx.x;

    __shared__ DType data[THREADS_PER_BLOCK];
    
    data[tid] = (DType)0.;
    for (int ii = tid; ii < nall; ii += THREADS_PER_BLOCK) {
        data[tid] += atom_virial[batch_index * nall * 9 + ii * 9 + bid];
    }
    __syncthreads(); 
    // do reduction in shared memory
    for (int ii = THREADS_PER_BLOCK >> 1; ii > 0; ii >>= 1) {
        if (tid < ii) {
            data[tid] += data[tid + ii];
        }
        __syncthreads();
    }
    // write result for this block to global memory
    if (tid == 0) virial[batch_index * 9 +  bid] = data[0];
}

template<typename DType>
__global__ void virial_deriv_wrt_neighbors_a(
    DType * atom_virial,
    const DType * dE,
    const DType * Ri_d,
    const DType * Rij,
    const int * nlist,
    const int natoms,
    const int neigh_num) 
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int batch_id = blockIdx.z;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int virial_index = threadIdx.y;
    const int ndescrpt = neigh_num * 4;

    if (neigh_index >= neigh_num)
        return;

    const unsigned int nlist_offset = batch_id * natoms * neigh_num + atom_id * neigh_num + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    DType virial_tmp = (DType)0.;
    const unsigned int dE_offset = batch_id * natoms * ndescrpt + atom_id * ndescrpt + neigh_index * 4;
    const unsigned int Ri_d_offset = batch_id * natoms * ndescrpt * 3 + atom_id * ndescrpt * 3 + neigh_index * 12;
    const unsigned int Rij_offset = batch_id * natoms * neigh_num * 3 + atom_id * neigh_num * 3 + neigh_index * 3;

    for (int idw = 0; idw < 4; ++idw) {
        // virial_tmp += dE[idx * ndescrpt + idy * 4 + idw] * Ri[idx * neigh_num * 3 + idy * 3 + idz % 3] * Ri_d[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz / 3];
        virial_tmp += dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + virial_index / 3] * Rij[Rij_offset + virial_index % 3];
    }

    const uint index = batch_id * natoms * 9 + neigh_id * 9 + virial_index;

    atomicAdd(atom_virial + index, virial_tmp);
}

template<typename DType>
void launch_calculate_virial_force(
    const int * nblist,
    const DType * dE,
    const DType * Rij,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * virial,
    DType * atom_virial,
    const int nghost,
    const int device_id
)
{
    cudaSetDevice(device_id);
    // cudaMemset(virial, 0, sizeof(DTYPE) * 9 * batch_size);
    // cudaMemset(atom_virial, 0, sizeof(DTYPE) * 9 * natoms * batch_size);
        
    const int LEN = 16;
    int nblock = (neigh_num + LEN - 1) / LEN;

    dim3 block_grid(nblock, natoms, batch_size);
    dim3 thread_grid(LEN, 9);
    // compute virial of a frame
    virial_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(atom_virial, dE, Ri_d, Rij, nblist, natoms, neigh_num);

    block_grid = dim3(9, batch_size);
    // reduction atom_virial to virial
    const int nall = natoms + nghost;
    atom_virial_reduction<DType, 256> <<<block_grid, 256>>>(virial, atom_virial, nall);

}
/*template <typename DType>
void atom_virial_reduction(
    DType * virial,
    const DType * atom_virial,
    const int nall,
    const int batch_size)
{
    for (int virial_index = 0; virial_index < 9; ++virial_index) {
        for (int atom_id = 0; atom_id < nall; ++atom_id) {
            for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
                const int index = ibatch * 9 + virial_index;
                const int atom_virial_index = ibatch * nall * 9 + atom_id * 9 + virial_index;
                printf("atom_virial[%d]: %f\n", atom_virial_index, atom_virial[atom_virial_index]);
                virial[index] += atom_virial[atom_virial_index];
            }
        }
        printf("virial[%d]: %f\n", virial_index, virial[virial_index]);
    }
}

template<typename DType>
void virial_deriv_wrt_neighbors_a(
    DType * atom_virial,
    const DType * dE,
    const DType * Ri_d,
    const DType * Rij,
    const int * nlist,
    const int natoms,
    const int neigh_num,
    const int batch_size)
{
    const int ndescrpt = neigh_num * 4;

    for (int ibatch = 0; ibatch < batch_size; ++ibatch) {
        for (int atom_id = 0; atom_id < natoms; ++atom_id) {
            for (int neigh_index = 0; neigh_index < neigh_num; ++neigh_index) {
                const unsigned int nlist_offset = ibatch * natoms * neigh_num + atom_id * neigh_num + neigh_index;
                const int neigh_id = nlist[nlist_offset];
                if (neigh_id < 0) {
                    continue;
                }
                DType dE_Ri_d_temp[3];
                DType Rij_temp[3];
                DType virial_tmp = (DType)0.;
                const int dE_offset = ibatch * natoms * ndescrpt + atom_id * ndescrpt + neigh_index * 4;
                const int Ri_d_offset = ibatch * natoms * ndescrpt * 3 + atom_id * ndescrpt * 3 + neigh_index * 12;
                const int Rij_offset = ibatch * natoms * neigh_num * 3 + atom_id * neigh_num * 3 + neigh_index * 3;
                // for (int idw = 0; idw < 4; ++idw) {
                //     virial_tmp += dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + virial_index % 3] * Rij[Rij_offset + virial_index / 3];
                // }
                for (int virial_index = 0; virial_index < 9; ++virial_index) {
                    Rij_temp[0] = Rij[Rij_offset];
                    Rij_temp[1] = Rij[Rij_offset + 1];
                    Rij_temp[2] = Rij[Rij_offset + 2];
                    for (int idw = 0; idw < 4; ++idw) {
                        dE_Ri_d_temp[0] = dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3];
                        dE_Ri_d_temp[1] = dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + 1];
                        dE_Ri_d_temp[2] = dE[dE_offset + idw] * Ri_d[Ri_d_offset + idw * 3 + 2];
                        virial_tmp += dE_Ri_d_temp[virial_index % 3] * Rij_temp[virial_index / 3];
                    }
                
                    const int index = ibatch * natoms * 9 + neigh_id * 9 + virial_index;
                    atom_virial[index] += virial_tmp;
                }
            }
        }
    }
}

template<typename DType>
void launch_calculate_virial_force(
    const int * nblist,
    const DType * dE,
    const DType * Rij,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * virial,
    DType * atom_virial,
    const int nghost
)
{   
    const int nall = natoms + nghost;
    int* h_nblist = new int[neigh_num * natoms * batch_size];
    DType * h_dE = new DType[neigh_num * 4 * natoms * batch_size];
    DType * h_Rij = new DType[neigh_num * 3 * natoms * batch_size];
    DType * h_Ri_d = new DType[neigh_num * 4 * 3 * natoms * batch_size];
    DType * h_virial = new DType[9 * batch_size];
    DType * h_atom_virial = new DType[9 * nall * batch_size];
    cudaMemcpy(h_nblist, nblist, neigh_num * natoms * batch_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dE, dE, neigh_num * 4 * natoms * batch_size * sizeof(DType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Rij, Rij, neigh_num * 3 * natoms * batch_size * sizeof(DType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Ri_d, Ri_d, neigh_num * 4 * 3 * natoms * batch_size * sizeof(DType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_virial, virial, 9 * batch_size * sizeof(DType), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_atom_virial, atom_virial, 9 * nall * batch_size * sizeof(DType), cudaMemcpyDeviceToHost);
    // compute virial of a frame
    virial_deriv_wrt_neighbors_a(h_atom_virial, h_dE, h_Ri_d, h_Rij, h_nblist, natoms, neigh_num, batch_size);
    // reduction atom_virial to virial
    atom_virial_reduction(h_virial, h_atom_virial, nall, batch_size);
    delete[] h_nblist;
    delete[] h_dE;
    delete[] h_Rij;
    delete[] h_Ri_d;
    delete[] h_virial;
    delete[] h_atom_virial;
}*/


template void launch_calculate_virial_force(
    const int * nblist,
    const float * dE,
    const float * Rij,
    const float * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    float * virial,
    float * atom_virial,
    const int nghost,
    const int device_id
);

template void launch_calculate_virial_force(
    const int * nblist,
    const double * dE,
    const double * Rij,
    const double * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * virial,
    double * atom_virial,
    const int nghost,
    const int device_id
);