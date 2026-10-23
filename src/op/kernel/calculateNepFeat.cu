#include "./utilities/nep_utilities.cuh"
#include <iostream>
#include <c10/cuda/CUDAStream.h>
#include "./utilities/error.cuh"

__global__ void feat_2b_calc(
        const double * coeff2,
        const double * d12_radial,
        const int64_t * NL_radial,
        const int64_t * atom_map,
        const double rcut_radial,
        const double rcinv_radial,
        double * feat_2b,
        double * dfeat_c2,
        double * dfeat_2b,
        double * dfeat_2b_noc,
        const int natoms,
        const int neigh_num,
        const int n_max,
        const int n_base,
        const int num_types,
        const int num_types_sq, const int noc_components)
{  
    // 计算全局线程索引，每个线程处理一个中心原子
    int global_atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算批次和原子索引
    int atom_id = global_atom_index;
    int c_index = 0;
    // if (atom_id == 0){
    //     for(int i = 0; i < natoms; i++) {
    //         if (i > 0) continue;
    //         for(int j = 0; j < neigh_num; j++) {
    //             int ti = atom_map[i];
    //             int nj = NL_radial[i * neigh_num + j];
    //             int tt2 = -1;
    //             if (nj >= 0){
    //                 tt2 = atom_map[nj];
    //             }
    //             printf("i %d ti %d idj %d j %d tj %d\n", i, ti, j, nj, tt2);
    //         }
    //     }
    // }
    if (atom_id < natoms) {
        int t1 = atom_map[atom_id];
        int neigh_start_idx = atom_id * neigh_num;
        int r12_start_idx =  atom_id * neigh_num * 4;
        int feat_start_idx = atom_id * n_max; 
        int dfeat_c_start_idx = atom_id * num_types * n_base;
        int dfeat_2b_start_idx = atom_id * neigh_num * n_max;
        int dfeat_2b_noc_start_idx=atom_id * neigh_num * n_base * noc_components;
        int c_start_idx = t1 * num_types * n_max * n_base;

        for (int i1=0; i1 < neigh_num; ++i1) {
            int n2 = NL_radial[neigh_start_idx + i1];
            if (n2 < 0) return;
            int t2 = atom_map[n2];
            int c_I_J_idx = c_start_idx + t2 * n_max * n_base;
            int rij_idx = r12_start_idx + i1*4;
            int d2b_idx = dfeat_2b_start_idx + i1 * n_max;
            int d2bnoc_idx=dfeat_2b_noc_start_idx + i1 * n_base * noc_components;
            double d12 = d12_radial[rij_idx]; // [rij, x, y, z]
            double fc12, fcp12;
            find_fc_and_fcp(rcut_radial, rcinv_radial, d12, fc12, fcp12);
            double fn12[MAX_NUM_N];
            double fnp12[MAX_NUM_N];
            find_fn_and_fnp(
                n_base, rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
            for (int n = 0; n < n_max; ++n) {
                double gn12 = 0.0;
                for (int k = 0; k < n_base; ++k) {
                    // c2的维度为[Nmax, Nbas, I, J]对c的索引会更方便
                    c_index =  c_I_J_idx + n * n_base + k;
                    gn12 += fn12[k] * coeff2[c_index];
                    dfeat_2b[d2b_idx + n] += fnp12[k]*coeff2[c_index];
                    // if (atom_id==23 and i1==6){
                    //     printf("i %d t %d j %d t %d n %d k %d c %f cid %d rij %f rid %d fn12[%d]=%f max_NN_radial=%d\n", 
                    //         atom_id, t1, i1, t2, n, k, coeff2[c_index], c_index, d12, rij_idx, k, fn12[k], neigh_num);
                    // }
                    if (n == 0) {
                        dfeat_c2[dfeat_c_start_idx + t2 * n_base + k] += fn12[k]; //[batch, n_atom, J_Ntypes, N_base]
                        dfeat_2b_noc[d2bnoc_idx + k * noc_components] += fnp12[k];
                    }
                }
                feat_2b[feat_start_idx + n] += gn12;
                // if (atom_id==23){
                //     printf("gn i %d t %d j %d t %d n %d gn12=%f feat_2b[%d+%d=%d]=%f\n", 
                //         atom_id, t1, i1, t2, n, gn12, feat_start_idx, n, feat_start_idx + n, feat_2b[feat_start_idx + n]);
                // }
            }
        }//neighs
        // if(atom_id == 23) {
        //     printf("atom %d feat [%f %f %f %f %f] dfc [%f %f]\n", atom_id, 
        //     feat_2b[feat_start_idx], feat_2b[feat_start_idx+1], feat_2b[feat_start_idx+2], feat_2b[feat_start_idx+3], feat_2b[feat_start_idx+4],
        //         dfeat_c2[dfeat_c_start_idx], dfeat_c2[dfeat_c_start_idx+1*n_base]);
        // }
    }
}

void launch_calculate_nepfeat(
        const double * coeff2,
        const double * d12_radial,
        const int64_t * NL_radial,
        const int64_t * atom_map,
        const double rcut_radial,
        double * feat_2b,
        double * dfeat_c2,
        double * dfeat_2b,
        double * dfeat_2b_noc,
        const int natoms,
        const int neigh_num,
        const int n_max,
        const int n_base,
        const int num_types,
        const int device_id, const int noc_components
) {
    cudaSetDevice(device_id);
    if (natoms == 0 || neigh_num == 0) return;
    const auto stream = c10::cuda::getCurrentCUDAStream(device_id);
    int num_types_sq = num_types * num_types;
    int BLOCK_SIZE = 64;
    int grid_size = (natoms - 1) / BLOCK_SIZE + 1;
    double rcinv_radial = 1.0 / rcut_radial;
    feat_2b_calc<<<grid_size, BLOCK_SIZE, 0, stream.stream()>>>(
                coeff2, d12_radial, NL_radial, atom_map, 
                    rcut_radial, rcinv_radial,
                        feat_2b, dfeat_c2, dfeat_2b, dfeat_2b_noc,
                            natoms, neigh_num, 
                                n_max, n_base, num_types, num_types_sq, noc_components
                            );
                            
    CUDA_CHECK_KERNEL
}
