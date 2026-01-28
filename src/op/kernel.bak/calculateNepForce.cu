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

__device__ inline double nep_force_dev_dot(
    const double * arr1, 
    const double * arr2) 
{
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

__global__ void nepforce_calc(
    double * force, 
    const double * net_deriv,
    const double * in_deriv,
    const int64_t * nlist,
    const int nloc,
    const int nnei)
{  
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    const unsigned int xyz_index = threadIdx.y;
    const int ndescrpt = nnei * 4;

    if (neigh_index >= nnei)
        return;

    // 偏移量计算
    const unsigned int nlist_offset = atom_id * nnei + neigh_index;  // 不再有 batch_id

    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    double temp_a[4], temp_b[4];

    const unsigned int net_offset = atom_id * ndescrpt + neigh_index * 4;  // 不再有 batch_id
    const unsigned int in_offset = atom_id * ndescrpt * 3 + neigh_index * 12;

    for (int i = 0; i < 4; i++) {
        temp_a[i] = net_deriv[net_offset + i];
        temp_b[i] = in_deriv[in_offset + i * 3 + xyz_index];
    }

    double res = 0.0;

    for (int i = 0; i < 4; i++) {
        res += temp_a[i] * temp_b[i];
    }

    const uint force_index = neigh_id * 3 + xyz_index;  // 不再有 batch_id

    atomicAdd(force + force_index, res);
}



__global__ void nepforce_grad_wrt_center_atom(
    double * grad_net,
    const double * grad, 
    const double * env_deriv, 
    const int natoms,
    const int ndescrpt)
{
    unsigned int center_idx = blockIdx.x; // 使用原子索引，不需要 batch_id
    unsigned int tid = threadIdx.x;
    __shared__ double grad_one[3];

    if (tid < 3) {
        grad_one[tid] = grad[center_idx * 3 + tid]; // 不再使用 batch_id
    }
    __syncthreads();
    
    unsigned int descrpt_idx = blockIdx.y * blockDim.x + tid;
    if (descrpt_idx < ndescrpt) {
        grad_net[center_idx * ndescrpt + descrpt_idx] -= 
            nep_force_dev_dot(grad_one, env_deriv + center_idx * ndescrpt * 3 + descrpt_idx * 3); // 不再使用 batch_id
    }
}

__global__ void nepforce_grad_wrt_neighbors_a(
    double * grad_net, 
    const double * grad, 
    const double * env_deriv, 
    const int64_t * nlist, 
    const int nloc,
    const int nnei)
{
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idw = threadIdx.y;
    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    if (j_idx >= nloc) j_idx = j_idx % nloc;

    const unsigned int grad_net_offset = idx * nnei * 4 + idy * 4 + idw;
    const unsigned int grad_offset = j_idx * 3;
    const unsigned int env_deriv_offset = idx * nnei * 4 * 3 + idy * 4 * 3 + idw * 3;
    
    grad_net[grad_net_offset] += nep_force_dev_dot(grad + grad_offset, env_deriv + env_deriv_offset);
}

void launch_calculate_nepforce(
    const int64_t * nblist,
    const double * dE,
    const double * Ri_d,
    const int natoms,
    const int neigh_num,
    double * force,
    const int device_id
) {
    cudaSetDevice(device_id);
    const int LEN = 256;
    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms);  // 不再需要 batch_size 维度
    dim3 thread_grid(LEN, 3);
    nepforce_calc<<<block_grid, thread_grid>>>(force, dE, Ri_d, nblist, natoms, neigh_num);
}

void launch_calculate_nepforce_grad(
    const int64_t * nblist,
    const double * Ri_d,
    const double * net_grad,
    const int natoms,
    const int neigh_num,
    double * grad,
    const int device_id
) {
    cudaSetDevice(device_id);
    int LEN = 256;
    const int ndesc = neigh_num * 4;

    const int nblock = (ndesc + LEN - 1) / LEN;
    dim3 block_grid(natoms, nblock);
    dim3 thread_grid(LEN, 1);
    nepforce_grad_wrt_center_atom<<<block_grid, thread_grid>>>(
        grad,
        net_grad, 
        Ri_d, 
        natoms, 
        ndesc);

    LEN = 128;
    const int nblock_ = (natoms + LEN - 1) / LEN;
    dim3 block_grid_(nblock_, neigh_num);
    dim3 thread_grid_(LEN, 4);
    nepforce_grad_wrt_neighbors_a<<<block_grid_, thread_grid_>>>(
        grad,
        net_grad, 
        Ri_d, 
        nblist, 
        natoms, 
        neigh_num);
}
