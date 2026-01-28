
#include "./utilities/error.cuh"
#include "./utilities/nep_utilities.cuh"
#include "./utilities/nep3_small_box.cuh"
#include <iostream>
#include <c10/cuda/CUDAStream.h>

static __global__ void gpu_find_neighbor_number(
    int64_t const* Na, 
    int64_t const* Na_sum, 
    double const g_rc_radial, 
    double const g_rc_angular, 
    double const* g_box,
    double const* g_box_original, 
    int64_t const* __restrict__ g_num_cell, 
    double const* pos, 
    int64_t* NN_radial,
    int64_t* NN_angular)
{
    int N2 = Na_sum[blockIdx.x];
    int N1 = N2 - Na[blockIdx.x];
    for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x)
    {
        double const* __restrict__ box = g_box + 18 * blockIdx.x;
        double const* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
        int64_t const* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
        double x1 = pos[n1 * 3];
        double y1 = pos[n1 * 3 + 1];
        double z1 = pos[n1 * 3 + 2];
        // printf("n1 %d box[0,1,2,17]=%f %f %f %f box_org[0,1,2,8]=%f %f %f %f pos=%f %f %f num_cell=%lld %lld %lld g_num_cell=%lld %lld %lld\n", \
        //         n1, box[0], box[1], box[2], box[17], \
        //         box_original[0], box_original[1], box_original[2], box_original[8],\
        //         x1, y1, z1, num_cell[0], num_cell[1], num_cell[2]\
        //         , g_num_cell[0], g_num_cell[1], g_num_cell[2]);
        int count_radial = 0;
        int count_angular = 0;
        for (int n2 = N1; n2 < N2; ++n2)
        {
            for (int ia = 0; ia < num_cell[0]; ++ia)
            {
                for (int ib = 0; ib < num_cell[1]; ++ib)
                {
                    for (int ic = 0; ic < num_cell[2]; ++ic)
                    {
                        if (ia == 0 && ib == 0 && ic == 0 && n1 == n2)
                        {
                            continue; // exclude self
                        }

                        double delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
                        double delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
                        double delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;

                        double x12 = pos[n2 * 3] + delta_x - x1;
                        double y12 = pos[n2 * 3 + 1] + delta_y - y1;
                        double z12 = pos[n2 * 3 + 2] + delta_z - z1;

                        dev_apply_mic(box, x12, y12, z12);

                        double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
                        double rc_radial = g_rc_radial;
                        double rc_angular = g_rc_angular;

                        if (distance_square < rc_radial * rc_radial)
                        {
                            count_radial++;
                        }

                        if (distance_square < rc_angular * rc_angular)
                        {
                            count_angular++;
                        }
                    }
                }
            }
        }
        NN_radial[n1] = count_radial;
        NN_angular[n1] = count_angular;
    }
}


static __global__ void gpu_find_neighbor_number_with_type(
    int64_t const* Na, 
    int64_t const* Na_sum, 
    double const g_rc_radial, 
    double const g_rc_angular, 
    double const* g_box,
    double const* g_box_original, 
    int64_t const* __restrict__ g_num_cell, 
    double const* pos, 
    int64_t* NN_radial,
    int64_t* NN_angular,
    int64_t  atom_type_num,
    const int64_t* atom_type_map
    )
{
    int N2 = Na_sum[blockIdx.x];
    int N1 = N2 - Na[blockIdx.x];
    for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x)
    {
        double const* __restrict__ box = g_box + 18 * blockIdx.x;
        double const* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
        int64_t const* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
        double x1 = pos[n1 * 3];
        double y1 = pos[n1 * 3 + 1];
        double z1 = pos[n1 * 3 + 2];
        // printf("n1 %d box[0,1,2,17]=%f %f %f %f box_org[0,1,2,8]=%f %f %f %f pos=%f %f %f num_cell=%lld %lld %lld g_num_cell=%lld %lld %lld\n", \
        //         n1, box[0], box[1], box[2], box[17], \
        //         box_original[0], box_original[1], box_original[2], box_original[8],\
        //         x1, y1, z1, num_cell[0], num_cell[1], num_cell[2]\
        //         , g_num_cell[0], g_num_cell[1], g_num_cell[2]);
        int64_t count_radial[TYPES] = {0};
        int64_t count_angular[TYPES] = {0};
        for (int n2 = N1; n2 < N2; ++n2)
        {   
            int t2 = atom_type_map[n2];
            for (int ia = 0; ia < num_cell[0]; ++ia)
            {
                for (int ib = 0; ib < num_cell[1]; ++ib)
                {
                    for (int ic = 0; ic < num_cell[2]; ++ic)
                    {
                        if (ia == 0 && ib == 0 && ic == 0 && n1 == n2)
                        {
                            continue; // exclude self
                        }

                        double delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
                        double delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
                        double delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;

                        double x12 = pos[n2 * 3] + delta_x - x1;
                        double y12 = pos[n2 * 3 + 1] + delta_y - y1;
                        double z12 = pos[n2 * 3 + 2] + delta_z - z1;

                        dev_apply_mic(box, x12, y12, z12);

                        double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
                        double rc_radial = g_rc_radial;
                        double rc_angular = g_rc_angular;

                        if (distance_square < rc_radial * rc_radial)
                        {
                            count_radial[t2]++;
                        }

                        if (distance_square < rc_angular * rc_angular)
                        {
                            count_angular[t2]++;
                        }
                    }
                }
            }
        }

        for (int _t = 0; _t< atom_type_num; _t++){
            NN_radial[ n1 * atom_type_num + _t] = count_radial[_t];
            NN_angular[n1 * atom_type_num + _t] = count_angular[_t];
        }
        // NN_radial[n1] = count_radial;
        // NN_angular[n1] = count_angular;
    }
}

static __global__ void gpu_find_neighbor_list(
    int64_t const* Na, 
    int64_t const* Na_sum, 
    int64_t const* g_type,
    int64_t const* g_atomic_numbers, 
    double const g_rc_radial, 
    double const g_rc_angular, 
    double const* __restrict__ g_box,
    double const* __restrict__ g_box_original, 
    int64_t const* __restrict__ g_num_cell, 
    double const* pos, 
    int64_t max_NN_radial,
    int64_t max_NN_angular, 
    int64_t* NN_radial, 
    int64_t* NL_radial, 
    int64_t* NN_angular, 
    int64_t* NL_angular, 
    double* Ri_radial,
    double* Ri_angular,
    bool with_rij=false)
{
    // Ri_radial [num_atoms, max_NN_radial, 3]
    // Ri_angular [num_atoms, max_NN_angular, 3]
    int N2 = Na_sum[blockIdx.x];
    int N1 = N2 - Na[blockIdx.x];

    for (int n1 = N1 + threadIdx.x; n1 < N2; n1 += blockDim.x)
    {
        double const* __restrict__ box = g_box + 18 * blockIdx.x;
        double const* __restrict__ box_original = g_box_original + 9 * blockIdx.x;
        int64_t const* __restrict__ num_cell = g_num_cell + 3 * blockIdx.x;
        double x1 = pos[n1 * 3];
        double y1 = pos[n1 * 3 + 1];
        double z1 = pos[n1 * 3 + 2];
        int count_radial = 0;
        int count_angular = 0;
        for (int n2 = N1; n2 < N2; ++n2)
        {
            for (int ia = 0; ia < num_cell[0]; ++ia)
            {
                for (int ib = 0; ib < num_cell[1]; ++ib)
                {
                    for (int ic = 0; ic < num_cell[2]; ++ic)
                    {
                        if (ia == 0 && ib == 0 && ic == 0 && n1 == n2)
                        {
                            continue; // exclude self
                        }

                        double delta_x = box_original[0] * ia + box_original[1] * ib + box_original[2] * ic;
                        double delta_y = box_original[3] * ia + box_original[4] * ib + box_original[5] * ic;
                        double delta_z = box_original[6] * ia + box_original[7] * ib + box_original[8] * ic;

                        double x12 = pos[n2 * 3] + delta_x - x1;
                        double y12 = pos[n2 * 3 + 1] + delta_y - y1;
                        double z12 = pos[n2 * 3 + 2] + delta_z - z1;

                        dev_apply_mic(box, x12, y12, z12);

                        double distance_square = x12 * x12 + y12 * y12 + z12 * z12;
                        double rc_radial = g_rc_radial;
                        double rc_angular = g_rc_angular;
                        
                        if (!with_rij){
                            if (distance_square < rc_radial * rc_radial)
                            {
                                NL_radial[n1 * max_NN_radial + count_radial] = n2;
                                Ri_radial[n1 * max_NN_radial * 3 + count_radial * 3] = x12;
                                Ri_radial[n1 * max_NN_radial * 3 + count_radial * 3 + 1] = y12;
                                Ri_radial[n1 * max_NN_radial * 3 + count_radial * 3 + 2] = z12;
                                count_radial++;
                            }

                            if (distance_square < rc_angular * rc_angular)
                            {
                                NL_angular[n1 * max_NN_angular + count_angular] = n2;
                                Ri_angular[n1 * max_NN_angular * 3 + count_angular * 3] = x12;
                                Ri_angular[n1 * max_NN_angular * 3 + count_angular * 3 + 1] = y12;
                                Ri_angular[n1 * max_NN_angular * 3 + count_angular * 3 + 2] = z12;
                                count_angular++;
                            }
                        }
                        else {
                            if (distance_square < rc_radial * rc_radial)
                            {
                                NL_radial[n1 * max_NN_radial + count_radial] = n2;
                                Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4] = sqrt(distance_square);
                                Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4 + 1] = x12;
                                Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4 + 2] = y12;
                                Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4 + 3] = z12;
                                count_radial++;
                                // if(n1 == 0) {
                                //     printf("n1 %lld n1_t %lld n2 %lld n2_t %lld nmax %lld nl2 %lld count_radial %lld\n",
                                //      n1, g_type[n1], n2, g_type[n2], max_NN_radial, NL_radial[n1 * max_NN_radial + count_radial], count_radial);
                                // }
                            }

                            if (distance_square < rc_angular * rc_angular)
                            {
                                NL_angular[n1 * max_NN_angular + count_angular] = n2;
                                Ri_angular[n1 * max_NN_angular * 4 + count_angular * 4] = sqrt(distance_square);
                                Ri_angular[n1 * max_NN_angular * 4 + count_angular * 4 + 1] = x12;
                                Ri_angular[n1 * max_NN_angular * 4 + count_angular * 4 + 2] = y12;
                                Ri_angular[n1 * max_NN_angular * 4 + count_angular * 4 + 3] = z12;
                                count_angular++;
                            }                            
                        }
                    }
                }
            }
        }

        NN_radial[n1] = count_radial;
        NN_angular[n1] = count_angular;
    }
}

void launch_calculate_maxneigh(
    const int64_t * num_atoms,
    const int64_t * num_atoms_sum,
    const double  * box,
    const double  * box_orig, 
    const int64_t * num_cell, 
    const double  * position,
    const double   cutoff_2b,
    const double   cutoff_3b,
    const int64_t   total_frames,
    const int64_t   total_atoms,
    int64_t * NN_radial, 
    int64_t * NN_angular,
    const int64_t atom_type_num,
    const bool with_type,
    const int64_t * atom_type_map

){
    auto stream = c10::cuda::getCurrentCUDAStream();
    if (with_type == false) {
        gpu_find_neighbor_number<<<total_frames, 256, 0, stream.stream()>>>(
            num_atoms,
            num_atoms_sum, 
            cutoff_2b, 
            cutoff_3b, 
            box, 
            box_orig,
            num_cell, 
            position, 
            NN_radial, 
            NN_angular
        );
    } else {
        gpu_find_neighbor_number_with_type<<<total_frames, 256, 0, stream.stream()>>>(
            num_atoms,
            num_atoms_sum, 
            cutoff_2b, 
            cutoff_3b, 
            box, 
            box_orig,
            num_cell, 
            position, 
            NN_radial, 
            NN_angular,
            atom_type_num,
            atom_type_map
            );        
    }
    CUDA_CHECK_KERNEL
}

void launch_calculate_neighbor(
    const int64_t * num_atoms,
    const int64_t * num_atoms_sum,
    const int64_t * atom_type_map,
    const int64_t * atom_types, 
    const double  * box,
    const double  * box_orig, 
    const int64_t * num_cell, 
    const double  * position,
    const double   cutoff_2b,
    const double   cutoff_3b,
    const int64_t  max_NN_radial,
    const int64_t  max_NN_angular,
    const int64_t  total_frames,
    const int64_t  total_atoms,
    int64_t * NN_radial, 
    int64_t * NL_radial,
    int64_t * NN_angular, 
    int64_t * NL_angular,
    double  * Ri_radial, 
    double  * Ri_angular,
    bool with_rij = false
){
    auto stream = c10::cuda::getCurrentCUDAStream();
    gpu_find_neighbor_list<<<total_frames, 256, 0, stream.stream()>>>(
        num_atoms,
        num_atoms_sum, 
        atom_type_map, 
        atom_types,
        cutoff_2b, 
        cutoff_3b, 
        box, 
        box_orig,
        num_cell, 
        position,
        max_NN_radial,
        max_NN_angular,
        NN_radial, 
        NL_radial,
        NN_angular,
        NL_angular,
        Ri_radial,
        Ri_angular,
        with_rij
        );
    CUDA_CHECK_KERNEL
}

void launch_calculate_descriptor(
    const double * coeff2,
    const double * coeff3,
    const double * r12,
    const int64_t * NL,
    const int64_t * atom_map,
    const double rcut_radial,
    const double rcut_angular,
    double * feats,
    const int64_t total_atoms,
    const int64_t neigh_num,
    const int64_t n_max_2b,
    const int64_t n_base_2b,
    const int64_t n_max_3b,
    const int64_t n_base_3b,
    const int64_t lmax_3,
    const int64_t lmax_4,
    const int64_t lmax_5,
    const int64_t n_types
) {
    const int BLOCK_SIZE = 64;
    const int N = total_atoms;// N = natoms * batch_size
    const int grid_size = (N - 1) / BLOCK_SIZE + 1;
    const int num_types_sq = n_types * n_types;
    double rcinv_radial = 1.0 / rcut_radial;
    double rcinv_angular = 1.0 / rcut_angular;
    
    int feat_2b_num = 0;
    int feat_3b_num = 0;
    feat_2b_num = n_max_2b;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;

    auto stream = c10::cuda::getCurrentCUDAStream();
    find_descriptor<<<grid_size, BLOCK_SIZE, 0, stream.stream()>>>(
        N,
        n_types,
        num_types_sq,
        neigh_num,
        lmax_3,
        lmax_4,
        lmax_5,
        feat_2b_num + feat_3b_num,
        rcut_radial,
        rcinv_radial,
        rcut_angular,
        rcinv_angular,
        n_max_2b,
        n_base_2b,
        n_max_3b,
        n_base_3b,
        NL,
        coeff2,
        coeff3,
        feats,
        atom_map,
        r12);
    CUDA_CHECK_KERNEL
}