#include <torch/extension.h>
#include "../include/nep_cpu.h"
#include "../include/cpu_calculate_nepneighbor.h"
const int TYPES = 100;

void launch_calculate_maxneigh_cpu(
    const int64_t * num_atoms,
    const int64_t * num_atoms_sum,
    const double  * box,
    const double  * box_orig, 
    const int64_t * num_cell, 
    const double  * position,
    const double   cutoff_2b,
    const double   cutoff_3b,
    const int64_t  total_frames,
    const int64_t  total_atoms,
    int64_t      * NN_radial, 
    int64_t      * NN_angular,
    const int64_t  atom_type_num,
    const bool     with_type,
    const int64_t * atom_type_map
) {
    for (int64_t frame = 0; frame < total_frames; ++frame) {
        int64_t N2 = num_atoms_sum[frame];
        int64_t N1 = N2 - num_atoms[frame];

        for (int64_t n1 = N1; n1 < N2; ++n1) {
            double const* box_frame = box + 18 * frame;
            double const* box_original_frame = box_orig + 9 * frame;
            int64_t const* num_cell_frame = num_cell + 3 * frame;

            double x1 = position[n1 * 3];
            double y1 = position[n1 * 3 + 1];
            double z1 = position[n1 * 3 + 2];

            int count_radial = 0;
            int count_angular = 0;

            int count_radial_list[TYPES]  = {0};
            int count_angular_list[TYPES] = {0};

            for (int64_t n2 = N1; n2 < N2; ++n2) {
                int t2 = atom_type_map[n2];
                for (int ia = 0; ia < num_cell_frame[0]; ++ia) {
                    for (int ib = 0; ib < num_cell_frame[1]; ++ib) {
                        for (int ic = 0; ic < num_cell_frame[2]; ++ic) {
                            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
                                continue; // 排除自身
                            }

                            double delta_x = box_original_frame[0] * ia + box_original_frame[1] * ib + box_original_frame[2] * ic;
                            double delta_y = box_original_frame[3] * ia + box_original_frame[4] * ib + box_original_frame[5] * ic;
                            double delta_z = box_original_frame[6] * ia + box_original_frame[7] * ib + box_original_frame[8] * ic;

                            // 应用最小影像条件（MIC）
                            double x12 = position[n2 * 3] + delta_x - x1;
                            double y12 = position[n2 * 3 + 1] + delta_y - y1;
                            double z12 = position[n2 * 3 + 2] + delta_z - z1;

                            cpu_dev_apply_mic(box_frame, x12, y12, z12);

                            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;

                            if(with_type == true){
                                if (distance_square < cutoff_2b * cutoff_2b) {
                                    count_radial_list[t2]++;
                                }
                                if (distance_square < cutoff_3b * cutoff_3b) {
                                    count_angular_list[t2]++;
                                }                                
                            } else {
                                if (distance_square < cutoff_2b * cutoff_2b) {
                                    count_radial++;
                                }
                                if (distance_square < cutoff_3b * cutoff_3b) {
                                    count_angular++;
                                }
                            }
                        }
                    }
                }
            }

            if(with_type == true){
                for (int _t = 0; _t < atom_type_num; _t++) {
                   NN_radial[ n1 * atom_type_num + _t] = count_radial_list[_t];
                   NN_angular[n1 * atom_type_num + _t] = count_angular_list[_t];
                }
            } else {
                NN_radial[n1] = count_radial;
                NN_angular[n1] = count_angular;
            }
        }
    }    
}

void launch_calculate_neighbor_cpu(
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
    int64_t      * NN_radial, 
    int64_t      * NL_radial,
    int64_t      * NN_angular, 
    int64_t      * NL_angular,
    double       * Ri_radial, 
    double       * Ri_angular,
    bool with_rij
) {
    for (int64_t frame = 0; frame < total_frames; ++frame) {
        int64_t N2 = num_atoms_sum[frame];
        int64_t N1 = N2 - num_atoms[frame];

        // 处理每个原子的邻居关系
        for (int64_t n1 = N1; n1 < N2; ++n1) {
            double const* box_frame = box + 18 * frame;
            double const* box_original_frame = box_orig + 9 * frame;
            int64_t const* num_cell_frame = num_cell + 3 * frame;

            double x1 = position[n1 * 3];
            double y1 = position[n1 * 3 + 1];
            double z1 = position[n1 * 3 + 2];

            int count_radial = 0;
            int count_angular = 0;

            // 遍历其他原子计算邻居
            for (int64_t n2 = N1; n2 < N2; ++n2) {
                for (int ia = 0; ia < num_cell_frame[0]; ++ia) {
                    for (int ib = 0; ib < num_cell_frame[1]; ++ib) {
                        for (int ic = 0; ic < num_cell_frame[2]; ++ic) {
                            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
                                continue; // 排除自身
                            }

                            double delta_x = box_original_frame[0] * ia + box_original_frame[1] * ib + box_original_frame[2] * ic;
                            double delta_y = box_original_frame[3] * ia + box_original_frame[4] * ib + box_original_frame[5] * ic;
                            double delta_z = box_original_frame[6] * ia + box_original_frame[7] * ib + box_original_frame[8] * ic;

                            // 应用最小影像条件（MIC）
                            double x12 = position[n2 * 3] + delta_x - x1;
                            double y12 = position[n2 * 3 + 1] + delta_y - y1;
                            double z12 = position[n2 * 3 + 2] + delta_z - z1;

                            cpu_dev_apply_mic(box_frame, x12, y12, z12);

                            double distance_square = x12 * x12 + y12 * y12 + z12 * z12;

                            // 判断是否为径向和角度邻居
                            if (!with_rij) {
                                if (distance_square < cutoff_2b * cutoff_2b) {
                                    NL_radial[n1 * max_NN_radial + count_radial] = n2;
                                    Ri_radial[n1 * max_NN_radial * 3 + count_radial * 3] = x12;
                                    Ri_radial[n1 * max_NN_radial * 3 + count_radial * 3 + 1] = y12;
                                    Ri_radial[n1 * max_NN_radial * 3 + count_radial * 3 + 2] = z12;
                                    count_radial++;
                                }

                                if (distance_square < cutoff_3b * cutoff_3b) {
                                    NL_angular[n1 * max_NN_angular + count_angular] = n2;
                                    Ri_angular[n1 * max_NN_angular * 3 + count_angular * 3] = x12;
                                    Ri_angular[n1 * max_NN_angular * 3 + count_angular * 3 + 1] = y12;
                                    Ri_angular[n1 * max_NN_angular * 3 + count_angular * 3 + 2] = z12;
                                    count_angular++;
                                }
                            } else {
                                if (distance_square < cutoff_2b * cutoff_2b) {
                                    NL_radial[n1 * max_NN_radial + count_radial] = n2;
                                    Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4] = sqrt(distance_square);
                                    Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4 + 1] = x12;
                                    Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4 + 2] = y12;
                                    Ri_radial[n1 * max_NN_radial * 4 + count_radial * 4 + 3] = z12;
                                    count_radial++;
                                }

                                if (distance_square < cutoff_3b * cutoff_3b) {
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
}

void launch_calculate_descriptor_cpu(
    const double  * coeff2,
    const double  * coeff3,
    const double  * rij,
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
){
    const int64_t N = total_atoms; // N = natoms * batch_size

    int feat_2b_num = 0;
    int feat_3b_num = 0;
    feat_2b_num = n_max_2b;
    if (lmax_3 > 0) feat_3b_num += n_max_3b * lmax_3;
    if (lmax_4 > 0) feat_3b_num += n_max_3b;
    if (lmax_5 > 0) feat_3b_num += n_max_3b;

    double rcinv_radial = 1.0 / rcut_radial;
    double rcinv_angular = 1.0 / rcut_angular;

    // CPU parallel loop
    // #pragma omp parallel for
    for (int n1 = 0; n1 < N; ++n1) {
        int t1 = atom_map[n1];
        // Get radial descriptors
        double q[MAX_DIM] = {0.0};
        int neigh_start_idx = n1 * neigh_num;
        int r12_start_idx =  n1 * neigh_num * 3;
        int feat_start_idx = n1 * (feat_2b_num + feat_3b_num); 
        int c2_start_idx = t1 * n_types * n_max_2b * n_base_2b;
        
        for (int i1 = 0; i1 < neigh_num; ++i1) {
            int n2 = NL[neigh_start_idx + i1]; // the data from neighbor list
            if (n2 < 0) break;
            int t2 = atom_map[n2];
            int c_I_J_idx = c2_start_idx + t2 * n_max_2b * n_base_2b;
            int rij_idx = r12_start_idx + i1 * 3;
            double r12[3] = {rij[rij_idx], rij[rij_idx + 1], rij[rij_idx + 2]};
            double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
            double fc12;
            find_fc(rcut_radial, rcinv_radial, d12, fc12);

            double fn12[MAX_NUM_N];
            find_fn(n_base_2b, rcinv_radial, d12, fc12, fn12);
            for (int n = 0; n < n_max_2b; ++n) {
                double gn12 = 0.0;
                for (int k = 0; k < n_base_2b; ++k) {
                    int c_index = c_I_J_idx + n * n_base_2b + k;
                    gn12 += fn12[k] * coeff2[c_index];
                }
                // 2b feats
                q[n] += gn12;
            }
            // printf("2b n1=%d t1=%d i1=%d n2=%d t2=%d neigh_num=%d r12_start_idx=%d rij_idx=%d d12=%f xyz=% lf %lf %lf\n",
            //      n1, t1, i1, n2, t2, neigh_num, r12_start_idx, rij_idx, d12, r12[0], r12[1], r12[2]);
        }

        // Get angular descriptors
        int c3_start_idx = t1 * n_types * n_max_3b * n_base_3b;
        for (int n = 0; n < n_max_3b; ++n) {
            double s[NUM_OF_ABC] = {0.0};
            for (int i1 = 0; i1 < neigh_num; ++i1) {
                int n2 = NL[neigh_start_idx + i1];
                // printf("3b n1=%d i1=%d n2=%d n=%d\n", n1, i1, n2, n);
                if (n2 < 0) continue;
                int t2 = atom_map[n2];
                int rij_idx = r12_start_idx + i1 * 3;
                double r12[3] = {rij[rij_idx], rij[rij_idx + 1], rij[rij_idx + 2]};
                double d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
                if (d12 > rcut_angular) continue;
                // printf("3b n1=%d t1=%d i1=%d n2=%d t2=%d n=%d d12=%f\n", n1, i1, t1, n2, t2, n, d12);
                double fc12;
                find_fc(rcut_angular, rcinv_angular, d12, fc12);
                double fn12[MAX_NUM_N];
                find_fn(n_base_3b, rcinv_angular, d12, fc12, fn12);
                // printf("3b n1=%d t1=%d i1=%d n2=%d t2=%d n=%d d12=%f fn12[0]=%f\n", n1, i1, t1, n2, t2, n, d12, fn12[0]);
                double gn12 = 0.0;
                int c_I_J_idx = c3_start_idx + t2 * n_max_3b * n_base_3b;
                for (int k = 0; k < n_base_3b; ++k) {
                    int c_index = c_I_J_idx + n * n_base_3b + k;
                    gn12 += fn12[k] * coeff3[c_index];
                }
                // printf("3b n1=%d t1=%d i1=%d n2=%d t2=%d n=%d d12=%f fn12[0]=%f gn12=%f\n", n1, i1, t1, n2, t2, n, d12, fn12[0], gn12);
                accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
            }

            if (lmax_5 == 1) {
                find_q_with_5body(n_max_3b, n, s, q + n_max_2b);
            } else if (lmax_4 == 2) {
                find_q_with_4body(n_max_3b, n, s, q + n_max_2b);
            } else {
                find_q(n_max_3b, n, s, q + n_max_2b);
            }
        }

        // Storing results in feats
        for (int n1 = 0; n1 < feat_2b_num + feat_3b_num; ++n1) {
            feats[feat_start_idx + n1] = q[n1];
        }
    }
}