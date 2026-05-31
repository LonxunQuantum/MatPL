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

template <typename T>
__device__ inline T nep_force_dev_dot(
    const T * arr1,
    const T * arr2)
{
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template <typename T>
__global__ void nepforce_calc(
    T * force,
    const T * net_deriv,
    const T * in_deriv,
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

    const unsigned int nlist_offset = atom_id * nnei + neigh_index;
    const int neigh_id = nlist[nlist_offset];

    if (neigh_id < 0) {
        return;
    }

    T temp_a[4], temp_b[4];

    const unsigned int net_offset = atom_id * ndescrpt + neigh_index * 4;
    const unsigned int in_offset = atom_id * ndescrpt * 3 + neigh_index * 12;

    for (int i = 0; i < 4; i++) {
        temp_a[i] = net_deriv[net_offset + i];
        temp_b[i] = in_deriv[in_offset + i * 3 + xyz_index];
    }

    T res = T(0.0);

    for (int i = 0; i < 4; i++) {
        res += temp_a[i] * temp_b[i];
    }

    const uint force_index = neigh_id * 3 + xyz_index;

    atomicAdd(force + force_index, res);
}

template <typename T>
__global__ void nepforce_grad_wrt_center_atom(
    T * grad_net,
    const T * grad,
    const T * env_deriv,
    const int natoms,
    const int ndescrpt)
{
    unsigned int center_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    __shared__ T grad_one[3];

    if (tid < 3) {
        grad_one[tid] = grad[center_idx * 3 + tid];
    }
    __syncthreads();

    unsigned int descrpt_idx = blockIdx.y * blockDim.x + tid;
    if (descrpt_idx < ndescrpt) {
        grad_net[center_idx * ndescrpt + descrpt_idx] -=
            nep_force_dev_dot(grad_one, env_deriv + center_idx * ndescrpt * 3 + descrpt_idx * 3);
    }
}

template <typename T>
__global__ void nepforce_grad_wrt_neighbors_a(
    T * grad_net,
    const T * grad,
    const T * env_deriv,
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

template <typename T>
void launch_calculate_nepforce(
    const int64_t * nblist,
    const T * dE,
    const T * Ri_d,
    const int natoms,
    const int neigh_num,
    T * force,
    const int device_id
) {
    cudaSetDevice(device_id);
    const int LEN = 256;
    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms);
    dim3 thread_grid(LEN, 3);
    nepforce_calc<T><<<block_grid, thread_grid>>>(force, dE, Ri_d, nblist, natoms, neigh_num);
}

template <typename T>
void launch_calculate_nepforce_grad(
    const int64_t * nblist,
    const T * Ri_d,
    const T * net_grad,
    const int natoms,
    const int neigh_num,
    T * grad,
    const int device_id
) {
    cudaSetDevice(device_id);
    int LEN = 256;
    const int ndesc = neigh_num * 4;

    const int nblock = (ndesc + LEN - 1) / LEN;
    dim3 block_grid(natoms, nblock);
    dim3 thread_grid(LEN, 1);
    nepforce_grad_wrt_center_atom<T><<<block_grid, thread_grid>>>(
        grad,
        net_grad,
        Ri_d,
        natoms,
        ndesc);

    LEN = 128;
    const int nblock_ = (natoms + LEN - 1) / LEN;
    dim3 block_grid_(nblock_, neigh_num);
    dim3 thread_grid_(LEN, 4);
    nepforce_grad_wrt_neighbors_a<T><<<block_grid_, thread_grid_>>>(
        grad,
        net_grad,
        Ri_d,
        nblist,
        natoms,
        neigh_num);
}

// Explicit instantiations
template void launch_calculate_nepforce<double>(
    const int64_t*, const double*, const double*, const int, const int, double*, const int);
template void launch_calculate_nepforce<float>(
    const int64_t*, const float*, const float*, const int, const int, float*, const int);

template void launch_calculate_nepforce_grad<double>(
    const int64_t*, const double*, const double*, const int, const int, double*, const int);
template void launch_calculate_nepforce_grad<float>(
    const int64_t*, const float*, const float*, const int, const int, float*, const int);
