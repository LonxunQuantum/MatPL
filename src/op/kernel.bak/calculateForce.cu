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
__global__ void force_deriv_wrt_neighbors_a(
    double * force, 
    const double * net_deriv,
    const double * in_deriv,
    const int * nlist,
    const int nloc,
    const int nnei)
{  
    // idy -> nnei
    const unsigned int idx = blockIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.x + threadIdx.x;
    const unsigned int idz = threadIdx.y;
    const int ndescrpt = nnei * 4;
    if (idy >= nnei) {
        return;
    }
    // deriv wrt neighbors
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    double force_tmp = 0.f;
    for (int idw = 0; idw < 4; ++idw) {
        force_tmp += net_deriv[idx * ndescrpt + idy * 4 + idw] * in_deriv[idx * ndescrpt * 3 + (idy * 4 + idw) * 3 + idz];
    }
    const uint force_index = j_idx * 3 + idz;
    if (force_index >= 324) {
        printf("error index %d\n", j_idx);
    }
    atomicAdd(force + force_index, force_tmp);
}

template<typename DType>
__global__ void force_calc(
    DType * force, 
    const DType * net_deriv,
    const DType * in_deriv,
    const int * nlist,
    const int nloc,
    const int nnei)
{  
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int batch_id = blockIdx.z;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int xyz_index = threadIdx.y;
    const int ndescrpt = nnei * 4;

    if (neigh_index >= nnei)
        return;
    const unsigned int nlist_offset = batch_id * nloc * nnei + atom_id * nnei + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    DType temp_a[4], temp_b[4];

    const unsigned int net_offset = batch_id * nloc * ndescrpt + atom_id * ndescrpt + neigh_index * 4;
    const unsigned int in_offset = batch_id * nloc * ndescrpt * 3 + atom_id * ndescrpt * 3 + neigh_index * 12;

    for (int i=0; i<4; i++) {
        temp_a[i] = net_deriv[net_offset + i];
        temp_b[i] = in_deriv[in_offset + i*3 + xyz_index];
    }

    DType res = 0.f;

    for (int i=0; i<4; i++) {
        res += temp_a[i] * temp_b[i];
    }

    const uint force_index = batch_id * nloc * 3 + neigh_id * 3 + xyz_index;

    atomicAdd(force + force_index, res);
}

template<typename DType>
void launch_calculate_force(
    const int * nblist,
    const DType * dE,
    const DType * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * force,
    const int nghost,
    const int device_id
) {

#if 0
    const int LEN = 64;
    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(natoms, nblock);
    dim3 thread_grid(LEN, 3);
    force_deriv_wrt_neighbors_a<<<block_grid, thread_grid>>>(force, dE, Ri_d, nblist, natoms, neigh_num);
#endif

#if 1
    cudaSetDevice(device_id);
    const int LEN = 256;
    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms, batch_size);
    dim3 thread_grid(LEN, 3);
    force_calc<<<block_grid, thread_grid>>>(force, dE, Ri_d, nblist, natoms, neigh_num);

    // std::cout << "batch_size: " << batch_size << std::endl;
    // std::cout << "natoms: " << natoms << std::endl;
    // std::cout << "neigh_num: " << neigh_num << std::endl;
    // std::vector<int> h_nblist(neigh_num * natoms * batch_size);
    // std::vector<DType> h_dE(neigh_num * 4 * natoms * batch_size);
    // std::vector<DType> h_Ri_d(neigh_num * 4 * 3 * natoms * batch_size);
    // std::vector<DType> h_force(natoms * 3 * batch_size);
    // cudaMemcpy(h_nblist.data(), nblist, sizeof(int) * neigh_num * natoms * batch_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_dE.data(), dE, sizeof(DType) * neigh_num * 4 * natoms * batch_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_Ri_d.data(), Ri_d, sizeof(DType) * neigh_num * 4 * 3 * natoms * batch_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_force.data(), force, sizeof(DType) * natoms * 3 * batch_size, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < neigh_num * natoms * batch_size; i++) {
    //     std::cout << "nblist[" << i << "]: " << h_nblist[i] << std::endl;
    // }
    // for (int i = 0; i < natoms * 3 * batch_size; i++) {
    //     std::cout << "force[" << i << "]: " << h_force[i] << std::endl;
    // }
#endif
}

template void launch_calculate_force(const int * nblist,
    const float * dE,
    const float * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    float * force,
    const int nghost,
    const int device_id);

template void launch_calculate_force(const int * nblist,
    const double * dE,
    const double * Ri_d,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * force,
    const int nghost,
    const int device_id);