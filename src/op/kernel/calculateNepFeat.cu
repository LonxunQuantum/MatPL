#include "./utilities/nep_utilities.cuh"
#include <iostream>

template <typename T>
__global__ void feat_2b_calc(
        const T * coeff2,
        const T * d12_radial,
        const int64_t * NL_radial,
        const int64_t * atom_map,
        const T rcut_radial,
        const T rcinv_radial,
        T * feat_2b,
        T * dfeat_c2,
        T * dfeat_2b,
        T * dfeat_2b_noc,
        const int natoms,
        const int neigh_num,
        const int n_max,
        const int n_base,
        const int num_types,
        const int num_types_sq)
{
    int global_atom_index = blockIdx.x * blockDim.x + threadIdx.x;
    int atom_id = global_atom_index;
    int c_index = 0;
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
            T d12 = d12_radial[rij_idx];
            T fc12, fcp12;
            find_fc_and_fcp(rcut_radial, rcinv_radial, d12, fc12, fcp12);
            T fn12[MAX_NUM_N];
            T fnp12[MAX_NUM_N];
            find_fn_and_fnp(
                n_base, rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
            for (int n = 0; n < n_max; ++n) {
                T gn12 = T(0.0);
                for (int k = 0; k < n_base; ++k) {
                    c_index =  c_I_J_idx + n * n_base + k;
                    gn12 += fn12[k] * coeff2[c_index];
                    dfeat_2b[d2b_idx + n] += fnp12[k]*coeff2[c_index];
                    if (n == 0) {
                        dfeat_c2[dfeat_c_start_idx + t2 * n_base + k] += fn12[k];
                        dfeat_2b_noc[d2bnoc_idx + k * 4] += fnp12[k];
                    }
                }
                feat_2b[feat_start_idx + n] += gn12;
            }
        }
    }
}

template <typename T>
void launch_calculate_nepfeat(
        const T * coeff2,
        const T * d12_radial,
        const int64_t * NL_radial,
        const int64_t * atom_map,
        const T rcut_radial,
        T * feat_2b,
        T * dfeat_c2,
        T * dfeat_2b,
        T * dfeat_2b_noc,
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
    T rcinv_radial = T(1.0) / rcut_radial;
    feat_2b_calc<T><<<grid_size, BLOCK_SIZE>>>(
                coeff2, d12_radial, NL_radial, atom_map,
                    rcut_radial, rcinv_radial,
                        feat_2b, dfeat_c2, dfeat_2b, dfeat_2b_noc,
                            natoms, neigh_num,
                                n_max, n_base, num_types, num_types_sq
                            );
}

// Explicit instantiations
template void launch_calculate_nepfeat<double>(
    const double*, const double*, const int64_t*, const int64_t*,
    const double, double*, double*, double*, double*,
    const int, const int, const int, const int, const int, const int);
template void launch_calculate_nepfeat<float>(
    const float*, const float*, const int64_t*, const int64_t*,
    const float, float*, float*, float*, float*,
    const int, const int, const int, const int, const int, const int);
