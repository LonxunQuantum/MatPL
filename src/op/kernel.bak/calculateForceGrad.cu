template<typename DType>
__device__ inline DType dev_dot(
    const DType * arr1, 
    const DType * arr2) 
{
    return arr1[0] * arr2[0] + arr1[1] * arr2[1] + arr1[2] * arr2[2];
}

template<typename DType>
__global__ void force_grad_wrt_center_atom(
    DType * grad_net,
    const DType * grad, 
    const DType * env_deriv, 
    const int natoms,
    const int ndescrpt)
{
    unsigned int batch_id = blockIdx.z;
    __shared__ DType grad_one[3];
    unsigned int center_idx = blockIdx.x;
    unsigned int tid = threadIdx.x;
    if(tid < 3){
        grad_one[tid] = grad[batch_id * natoms * 3 + center_idx * 3 + tid];
    }
    __syncthreads();
    unsigned int descrpt_idx = blockIdx.y * blockDim.x + tid;
    const int env_deriv_offset = batch_id * natoms * ndescrpt * 3 + center_idx * ndescrpt * 3 + descrpt_idx * 3;
    if(descrpt_idx < ndescrpt){
        grad_net[batch_id * natoms * ndescrpt + center_idx * ndescrpt + descrpt_idx] -= 
            dev_dot(grad_one, env_deriv + env_deriv_offset);
    }
}

template<typename DType>
__global__ void force_grad_wrt_neighbors_a(
    DType * grad_net, 
    const DType * grad, 
    const DType * env_deriv, 
    const int * nlist, 
    const int nloc,
    const int nnei)
{
    // idy -> nnei
    const unsigned int batch_id = blockIdx.z;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y;
    const unsigned int idw = threadIdx.y;
    if (idx >= nloc) {
        return;
    }
    int j_idx = nlist[batch_id * nloc * nnei + idx * nnei + idy];
    if (j_idx < 0) {
        return;
    }
    if (j_idx >= nloc) j_idx = j_idx % nloc;
    const unsigned int grad_net_offset = batch_id * nloc * nnei * 4 + idx * nnei * 4 + idy * 4 + idw;
    const unsigned int grad_offset = batch_id * nloc * 3 + j_idx * 3;
    const unsigned int env_deriv_offset = batch_id * nloc * nnei * 4 * 3 + idx * nnei * 4 * 3 + idy * 4 * 3 + idw * 3;
    grad_net[grad_net_offset] += dev_dot(grad + grad_offset, env_deriv + env_deriv_offset);
}

template<typename DType>
void launch_calculate_force_grad(
    const int * nblist,
    const DType * Ri_d,
    const DType * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * grad
) {

    int LEN = 256;
    const int ndesc = neigh_num * 4;
    cudaMemset(grad, 0.0, sizeof(DType) * batch_size * natoms * ndesc);

    const int nblock = (ndesc + LEN - 1) / LEN;
    dim3 block_grid(natoms, nblock, batch_size);
    dim3 thread_grid(LEN, 1);
    force_grad_wrt_center_atom<<<block_grid, thread_grid>>>(
        grad,
        net_grad, Ri_d, natoms, ndesc);

    LEN = 128;
    const int nblock_ = (natoms + LEN -1) / LEN;
    dim3 block_grid_(nblock_, neigh_num, batch_size);
    dim3 thread_grid_(LEN, 4);
    force_grad_wrt_neighbors_a<<<block_grid_, thread_grid_>>>(
        grad,
        net_grad, Ri_d, nblist, natoms, neigh_num);
}

template void launch_calculate_force_grad(
    const int * nblist,
    const float * Ri_d,
    const float * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    float * grad
);

template void launch_calculate_force_grad(
    const int * nblist,
    const double * Ri_d,
    const double * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * grad
);