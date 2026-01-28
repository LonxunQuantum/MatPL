#include <iostream>

template<typename FPTYPE>
__device__ inline FPTYPE dev_dot9(
    const FPTYPE * arr1, 
    const FPTYPE * arr2) 
{
    FPTYPE result = (FPTYPE)0.0;
    for(int ii=0; ii<9; ii++){
        result += arr1[ii] * arr2[ii];
    }
    return result;
}

template<typename DType>
__global__ void virial_grad_wrt_neighbors_a(
    DType * grad_output,    // batch * natoms * neigh_num * 4
    const DType * net_grad,  // batch * 9
    const DType * Ri_d, // Ri_d
    const DType * Rij,
    const int * nlist,
    const int natoms,
    const int neigh_num)
{
    const unsigned int block_id = blockIdx.x;
    const unsigned int atom_id = blockIdx.y;
    const unsigned int batch_id = blockIdx.z;
    const unsigned int neigh_index = threadIdx.x + block_id * blockDim.x;
    
    if (neigh_index >= neigh_num)
        return; 

    const unsigned int index_xyzw = threadIdx.y;
    const unsigned int tid = threadIdx.x;

    __shared__ DType grad_one[9];
    if(tid < 9){
        grad_one[tid] = net_grad[batch_id * 9 + tid];
    }
    __syncthreads();

    if (atom_id >= natoms) {
        return;
    }

    int j_idx = nlist[batch_id * natoms * neigh_num + atom_id * neigh_num + neigh_index];

    if (j_idx < 0) {
        return;
    }

    DType tmp[9];
    const int Rij_offset = batch_id * natoms * neigh_num * 3 + atom_id * neigh_num * 3 + neigh_index * 3;
    const int Ri_d_offset = batch_id * natoms * neigh_num * 4 * 3 + atom_id * neigh_num * 4 * 3 + neigh_index * 4 * 3 + index_xyzw * 3;

    
    for (int dd0 = 0; dd0 < 3; ++dd0){
        for (int dd1 = 0; dd1 < 3; ++dd1){
            tmp[dd0 * 3 + dd1] = Rij[Rij_offset + dd1] * Ri_d[Ri_d_offset + dd0];
        }
    }

    const int grad_output_offset = batch_id * natoms * neigh_num * 4 + atom_id * neigh_num * 4 + neigh_index * 4 + index_xyzw;
    grad_output[grad_output_offset] -= (DType) - 1.0 * dev_dot9(grad_one, tmp);
    
}

template <typename DType>
void launch_calculate_virial_force_grad(
    const int * nblist,
    const DType * Rij,
    const DType * Ri_d,
    const DType * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    DType * grad_output
)
{
    
    int LEN = 128;
    const int ndesc = neigh_num * 4;
    cudaMemset(grad_output, 0.0, sizeof(DType) * batch_size * natoms * ndesc);

    const int nblock = (neigh_num + LEN - 1) / LEN;
    dim3 block_grid(nblock, natoms, batch_size);
    dim3 thread_grid(LEN, 4);
    
    virial_grad_wrt_neighbors_a<<<block_grid, thread_grid>>>(
        grad_output,
        net_grad, Ri_d, Rij, nblist, natoms, neigh_num);
}

template void launch_calculate_virial_force_grad(
    const int * nblist,
    const float * Rij,
    const float * Ri_d,
    const float * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    float * grad
);

template void launch_calculate_virial_force_grad(
    const int * nblist,
    const double * Rij,
    const double * Ri_d,
    const double * net_grad,
    const int batch_size,
    const int natoms,
    const int neigh_num,
    double * grad
);
