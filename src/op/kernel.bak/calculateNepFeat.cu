#include "./utilities/common.cuh"
#include "./utilities/nep_utilities.cuh"
#include <iostream>

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
        const int num_types_sq)
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
        int dfeat_2b_noc_start_idx=atom_id * neigh_num * n_base * 4;
        int c_start_idx = t1 * num_types * n_max * n_base;

        for (int i1=0; i1 < neigh_num; ++i1) {
            int n2 = NL_radial[neigh_start_idx + i1];
            if (n2 < 0) return;
            int t2 = atom_map[n2];
            int c_I_J_idx = c_start_idx + t2 * n_max * n_base;
            int rij_idx = r12_start_idx + i1*4;
            int d2b_idx = dfeat_2b_start_idx + i1 * n_max;
            int d2bnoc_idx=dfeat_2b_noc_start_idx + i1 * n_base * 4;
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
                        dfeat_2b_noc[d2bnoc_idx + k * 4] += fnp12[k];
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
        const int device_id
) {
    cudaSetDevice(device_id);
    int num_types_sq = num_types * num_types;
    int BLOCK_SIZE = 64;
    int grid_size = (natoms - 1) / BLOCK_SIZE + 1;
    double rcinv_radial = 1.0 / rcut_radial;
    feat_2b_calc<<<grid_size, BLOCK_SIZE>>>(
                coeff2, d12_radial, NL_radial, atom_map, 
                    rcut_radial, rcinv_radial,
                        feat_2b, dfeat_c2, dfeat_2b, dfeat_2b_noc,
                            natoms, neigh_num, 
                                n_max, n_base, num_types, num_types_sq
                            );
                            
    // cudaDeviceSynchronize();
    // 打印 dfeat_c2 数据 (将 dfeat_c2 从设备内存复制到主机内存)
    // DType * h_dfeat_c2 = new DType[batch_size * natoms * num_types * n_base];  // 主机内存中的副本
    // cudaMemcpy(h_dfeat_c2, dfeat_c2, batch_size * natoms * num_types * n_base * sizeof(DType), cudaMemcpyDeviceToHost);
    
    // // 打印 dfeat_c2 的一部分 (例如前 5 个元素)
    // printf("dfeat_c2 (first few values):\n");
    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < natoms; ++j) {
    //         for (int k = 0; k < num_types; ++k) {
    //             for (int p = 0; p < n_base; ++p) {
    //                 printf("dfeat_c2[%d][%d][%d][%d] = %f\n", i, j, k, p, h_dfeat_c2[i * natoms * num_types * n_base + j* num_types * n_base + k * n_base + p]);
    //             }
    //         }
    //     }
    // }

    // 打印 dfeat_2b 数据 (将 dfeat_2b 从设备内存复制到主机内存)
    // DType * h_dfeat_2b = new DType[batch_size * natoms * neigh_num * n_max];  // 主机内存中的副本
    // cudaMemcpy(h_dfeat_2b, dfeat_2b, batch_size * natoms * neigh_num * n_max * sizeof(DType), cudaMemcpyDeviceToHost);
    
    // 打印 dfeat_2b 的一部分 (例如前 5 个元素)
    // printf("dfeat_2b (first few values):\n");
    // for (int i = 0; i < batch_size; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         for (int k = 0; k < neigh_num; ++k) {
    //             printf("dfeat_2b[%d][%d][%d] = ", i, j, k);
    //             for (int l = 0; l < n_max; ++l) {
    //                 printf(" %f ", h_dfeat_2b[i * natoms * neigh_num * n_max + j * neigh_num * n_max + k * n_max + l]);
    //             }
    //             printf("\n");
    //         }
    //     }
    // }
}
