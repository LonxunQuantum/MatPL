/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "common.cuh"
#include "nep_utilities.cuh"
#include "nep_utilities_mb_secondc_opt_legacy.cuh"

namespace nep3_base_fallback {
#include "nep_utilities_mb_secondc_base_fallback_opt_legacy.cuh"
}

static __global__ void find_angular_gardc_small_box(
  const int N,
  const double* grad_second,
  const double* g_d12,
  const int64_t* g_NL,
  const double* de_dfeat,
  const double* dsnlm_dc, //[i, J, nbase, 24]
  const double* g_sum_fxyz,
  const int64_t* g_type,
  const double * coeff3,
  double * dfeat_c3,
  const double rc_angular,
  const double rcinv_angular,
  const int atom_nums,
  const int neigh_num,
  const int max_3b,
  const int base_3b,
  const int num_types,
  const int num_types_sq,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums // 3b + 4b + 5b
  )
{
  // int total_elements = batch_size * atom_nums * neigh_num;
  // int elem_idx = threadIdx.x + blockIdx.x * blockDim.x; // 网格中的元素索引
  // if elem_idx >= total_elements return;

  // int batch_idx = elem_idx / (atom_nums * neigh_num);
  // int remaining = elem_idx % (atom_nums * neigh_num);
  // int n1 = remaining / neigh_num;
  // int il = remaining % neigh_num;

  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int g_sum_start = n1 * max_3b * NUM_OF_ABC;
    int r12_start_idx =  n1 * neigh_num * 4;
    int dc_start_idx = n1 * num_types * max_3b * base_3b;
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
    int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;
    int neigh_start_idx = n1 * neigh_num;
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    int b3_nums = max_3b * L_max3;
    int dd = 0;
    // if (n1 == 0) {
    //   for (int nn=0; nn < 108; nn++) {//all
    //     printf("grad_out_angluar[b0][%d][:] = ", nn);
    //     // printf("grad[%d + %d]=%f\n", de_start, feat_2b_nums + nn, de_dfeat[de_start + feat_2b_nums + nn]);
    //     for (int jj = 0; jj < 25; jj++) {
    //       printf("%f  ", de_dfeat[nn*25 + jj]);
    //     }
    //     printf("\n");
    //   }
    // }
    for (int nn=0; nn < max_3b; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        Fp[dd] = de_dfeat[de_start + feat_2b_nums + ll * max_3b + nn];// i -> nmax_3b*l_max+2?
        // 0 5 10 15
        // 1 6 11 16
        // 2 7 12 17
        // 3 8 13 18
        // 4 9 14 19 the feature order is L*n_max
        // if (n1==0){
        //   printf("3b Fp[%d] = %f from de_dfeat[%d + %d] = %f\n", dd, Fp[dd], de_start,  feat_2b_nums + ll * max_3b + nn, de_dfeat[de_start +  feat_2b_nums + ll * max_3b + nn]);
        // }
        dd++;
      }
    }
    if (L_max4 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        Fp[b3_nums + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + ll];
        // if (n1==0){
        //   printf("4b Fp[%d + %d] = %f from de_dfeat[%d + %d] = %f\n",
        //   b3_nums, ll, Fp[b3_nums + ll], de_start,  feat_2b_nums + b3_nums + ll, de_dfeat[de_start + feat_2b_nums + b3_nums + ll]);
        // }
      }
    }
    if (L_max5 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        Fp[b3_nums + max_3b + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + ll];
        // if (n1==0){
        //   printf("5b Fp[%d + %d] = %f from de_dfeat[%d + %d] = %f\n",
        //   b3_nums, max_3b + ll, Fp[b3_nums + max_3b + ll], de_start, feat_2b_nums + b3_nums + max_3b + ll, de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + ll]);
        // }
      }
    }

    for (int d = 0; d < max_3b * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[g_sum_start + d]; // g_sum is [N, n_max, 24]
    }

    // Fp[MAX_DIM_ANGULAR] = {1.0};

    int t1 = g_type[n1];
    int c3_start_idx = t1 * num_types * max_3b * base_3b;
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL[neigh_start_idx + i1];
      if (n2 < 0) break;
      int t2 = g_type[n2];
      int rij_idx = r12_start_idx + i1*4;
      // int dsnlm_idx = dsnlm_start_idx + t2 * base_3b * NUM_OF_ABC;
      double d12 = g_d12[rij_idx];
      if (d12 > rc_angular) break;
      double r12[3] = {g_d12[rij_idx+1], g_d12[rij_idx+2], g_d12[rij_idx+3]};
      double scd_r12[4] = {grad_second[rij_idx],grad_second[rij_idx+1],grad_second[rij_idx+2],grad_second[rij_idx+3]};// [r x y z]
      // double scd_r12[4] = {1.0};// [r x y z]

      // if(n1==0 and i1 == 0) {
      //   printf("=====scdr12======%f %f %f %f ========\n",scd_r12[0], scd_r12[1], scd_r12[2], scd_r12[3]);
      // }
      double f12[4] = {0.0};

      double fc12, fcp12;
      find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(
        base_3b, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

      int c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
      // double s[NUM_OF_ABC*6] = {0.0}; //[sij/(rij_^L), blm, blm/drij, blm/dx, blm/dy, blm/dz]
      double blm[NUM_OF_ABC] = {0.0};
      double rij_blm[NUM_OF_ABC]= {0.0};
      double dblm_x[NUM_OF_ABC] = {0.0};
      double dblm_y[NUM_OF_ABC] = {0.0};
      double dblm_z[NUM_OF_ABC] = {0.0};
      double dblm_r[NUM_OF_ABC] = {0.0};
      scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2],
          blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
      for (int n = 0; n < max_3b; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k < base_3b; ++k) {
          int c_index = c_I_J_idx + n * base_3b + k;
          gn12 += fn12[k] * coeff3[c_index];
          gnp12 += fnp12[k] * coeff3[c_index];
        }
        // double f12d[MAX_LMAX * 4] = {0.0}; // dfeat/drij [nl+n+n, 4]
        double f12k[TYPES * MAX_NUM_N] = {0.0};// max type is 20
        if (L_max5 > 0) {
          scd_accumulate_f12_with_5body(
            n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              t2, num_types, L_max3,
              max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1);
        } else if (L_max4 > 0) {
          scd_accumulate_f12_with_4body(
            n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              t2, num_types, L_max3,
              max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1);
        } else {
          scd_accumulate_f12(
            n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              t2, num_types, L_max3,
              max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1);
        }
        for (int j = 0; j < num_types; ++j){
          for (int k = 0; k < base_3b; ++k){
            int dc_id = dc_start_idx + j * max_3b * base_3b + n*base_3b + k;
            int k_id = j * base_3b + k;
            dfeat_c3[dc_id] += f12k[k_id];
            // if (n1 == 0){
            //   printf("n1=%d t1=%d n2=%d t2=%d n=%d k=%d dc=%f frxyz = %f %f %f %f\n",n1, t1, i1, t2, n, k,
            //     (f12k[k_id] + f12k[k_id+1] + f12k[k_id+2] + f12k[k_id+3]), f12k[k_id], f12k[k_id+1], f12k[k_id+2], f12k[k_id+3]);
            //   }
          }
        }
        //add f12k [k, 4] -> c[atomI, J_type, nmax, k, 4] -> c[atomI, J_type, nmax, k]
        // 是否把4 在scd时候直接给累加起来？ 还是单独加？
      }
    }
  }
}


static __global__ void find_angular_gardc_neigh(
  const int N,
  const double* grad_second,
  const double* g_d12,
  const int64_t* g_NL,
  const double* de_dfeat,
  const double* dsnlm_dc, //[i, J, nbase, 24]
  const double* g_sum_fxyz,
  const int64_t* g_type,
  const double * coeff3,
  double * dfeat_c3,
  const double rc_angular,
  const double rcinv_angular,
  const int atom_nums,
  const int neigh_num,
  const int max_3b,
  const int base_3b,
  const int num_types,
  const int num_types_sq,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums // 3b + 4b + 5b
  )
{
  // int total_elements = batch_size * atom_nums * neigh_num;
  int elem_idx = threadIdx.x + blockIdx.x * blockDim.x; // 网格中的元素索引
  if (elem_idx >= N) return;

  int n1 = elem_idx / neigh_num;
  int i1 = elem_idx % neigh_num;

  int neigh_start_idx = n1 * neigh_num;

  int t1 = g_type[n1];
  int n2 = g_NL[neigh_start_idx + i1];
  if (n2 < 0) return;
  int t2 = g_type[n2];

    int g_sum_start = n1 * max_3b * NUM_OF_ABC;
    int r12_start_idx =  n1 * neigh_num * 4;
    int dc_start_idx = n1 * neigh_num * num_types * max_3b * base_3b + i1 * num_types * max_3b * base_3b;
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
    int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    int b3_nums = max_3b * L_max3;
    int dd = 0;

    for (int nn=0; nn < max_3b; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        Fp[dd] = de_dfeat[de_start + feat_2b_nums + ll * max_3b + nn];// i -> nmax_3b*l_max+2?
        // 0 5 10 15
        // 1 6 11 16
        // 2 7 12 17
        // 3 8 13 18
        // 4 9 14 19 the feature order is L*n_max
        // if (n1==0){
        //   printf("3b Fp[%d] = %f from de_dfeat[%d + %d] = %f\n", dd, Fp[dd], de_start,  feat_2b_nums + ll * max_3b + nn, de_dfeat[de_start +  feat_2b_nums + ll * max_3b + nn]);
        // }
        dd++;
      }
    }
    if (L_max4 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        Fp[b3_nums + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + ll];
        // if (n1==0){
        //   printf("4b Fp[%d + %d] = %f from de_dfeat[%d + %d] = %f\n",
        //   b3_nums, ll, Fp[b3_nums + ll], de_start,  feat_2b_nums + b3_nums + ll, de_dfeat[de_start + feat_2b_nums + b3_nums + ll]);
        // }
      }
    }
    if (L_max5 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        Fp[b3_nums + max_3b + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + ll];
        // if (n1==0){
        //   printf("5b Fp[%d + %d] = %f from de_dfeat[%d + %d] = %f\n",
        //   b3_nums, max_3b + ll, Fp[b3_nums + max_3b + ll], de_start, feat_2b_nums + b3_nums + max_3b + ll, de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + ll]);
        // }
      }
    }

    for (int d = 0; d < max_3b * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[g_sum_start + d]; // g_sum is [N, n_max, 24]
    }

    int c3_start_idx = t1 * num_types * max_3b * base_3b;

    int rij_idx = r12_start_idx + i1*4;
    // int dsnlm_idx = dsnlm_start_idx + t2 * base_3b * NUM_OF_ABC;
    double d12 = g_d12[rij_idx];
    if (d12 > rc_angular) return;
    double r12[3] = {g_d12[rij_idx+1], g_d12[rij_idx+2], g_d12[rij_idx+3]};
    double scd_r12[4] = {grad_second[rij_idx],grad_second[rij_idx+1],grad_second[rij_idx+2],grad_second[rij_idx+3]};// [r x y z]
    double f12[4] = {0.0};
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

    double fn12[MAX_NUM_N];
    double fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      base_3b, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    int c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
    // double s[NUM_OF_ABC*6] = {0.0}; //[sij/(rij_^L), blm, blm/drij, blm/dx, blm/dy, blm/dz]
    double blm[NUM_OF_ABC] = {0.0};
    double rij_blm[NUM_OF_ABC]= {0.0};
    double dblm_x[NUM_OF_ABC] = {0.0};
    double dblm_y[NUM_OF_ABC] = {0.0};
    double dblm_z[NUM_OF_ABC] = {0.0};
    double dblm_r[NUM_OF_ABC] = {0.0};
    scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2],
        blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
    for (int n = 0; n < max_3b; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      for (int k = 0; k < base_3b; ++k) {
        int c_index = c_I_J_idx + n * base_3b + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }
      // double f12d[MAX_LMAX * 4] = {0.0}; // dfeat/drij [nl+n+n, 4]
      double f12k[TYPES * MAX_NUM_N] = {0.0};// max type is 20
      if (L_max5 > 0) {
        scd_accumulate_f12_with_5body(
          n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
            blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
            f12, f12k, scd_r12, fn12, fnp12,
            t2, num_types, L_max3,
            max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1);
      } else if (L_max4 > 0) {
        scd_accumulate_f12_with_4body(
          n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
            blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
            f12, f12k, scd_r12, fn12, fnp12,
            t2, num_types, L_max3,
            max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1);
      } else {
        scd_accumulate_f12(
          n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
            blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
            f12, f12k, scd_r12, fn12, fnp12,
            t2, num_types, L_max3,
            max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1);
      }
      for (int j = 0; j < num_types; ++j){
        for (int k = 0; k < base_3b; ++k){
          int dc_id = dc_start_idx + j * max_3b * base_3b + n*base_3b + k;
          int k_id = j * base_3b + k;
          dfeat_c3[dc_id] += f12k[k_id];
          // if (n1 == 0){
          //   printf("n1=%d t1=%d n2=%d t2=%d n=%d k=%d dc=%f frxyz = %f %f %f %f\n",n1, t1, i1, t2, n, k,
          //     (f12k[k_id] + f12k[k_id+1] + f12k[k_id+2] + f12k[k_id+3]), f12k[k_id], f12k[k_id+1], f12k[k_id+2], f12k[k_id+3]);
          //   }
        }
      }
      //add f12k [k, 4] -> c[atomI, J_type, nmax, k, 4] -> c[atomI, J_type, nmax, k]
      // 是否把4 在scd时候直接给累加起来？ 还是单独加？
    }
}


// 移除f12k中的多余空间，同时要调整子函数，确保写入数据的索引从0开始。
static __global__ void find_angular_gardc_neigh_optimized(
  const int N,
  const double* grad_second,
  const double* g_d12,
  const int64_t* g_NL,
  const double* de_dfeat,
  const double* dsnlm_dc, //[i, J, nbase, 24]
  const double* g_sum_fxyz,
  const int64_t* g_type,
  const double * coeff3,
  double * dfeat_c3,
  const double rc_angular,
  const double rcinv_angular,
  const int atom_nums,
  const int neigh_num,
  const int max_3b,
  const int base_3b,
  const int num_types,
  const int num_types_sq,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums // 3b + 4b + 5b
  )
{
  int elem_idx = threadIdx.x + blockIdx.x * blockDim.x; // 网格中的元素索引
  if (elem_idx >= N) return;

  int n1 = elem_idx / neigh_num;
  int i1 = elem_idx % neigh_num;

  int neigh_start_idx = n1 * neigh_num;

  int t1 = g_type[n1];
  int n2 = g_NL[neigh_start_idx + i1];
  if (n2 < 0) return;
  int t2 = g_type[n2];

    int g_sum_start = n1 * max_3b * NUM_OF_ABC;
    int r12_start_idx =  n1 * neigh_num * 4;
    int dc_start_idx = n1 * neigh_num * num_types * max_3b * base_3b + i1 * num_types * max_3b * base_3b;
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
    int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;

    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    int b3_nums = max_3b * L_max3;
    int dd = 0;

    for (int nn=0; nn < max_3b; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        Fp[dd] = de_dfeat[de_start + feat_2b_nums + ll * max_3b + nn];// i -> nmax_3b*l_max+2?
        dd++;
      }
    }
    if (L_max4 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        Fp[b3_nums + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + ll];
      }
    }
    if (L_max5 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        Fp[b3_nums + max_3b + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + ll];
      }
    }

    for (int d = 0; d < max_3b * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[g_sum_start + d]; // g_sum is [N, n_max, 24]
    }

    int c3_start_idx = t1 * num_types * max_3b * base_3b;

    int rij_idx = r12_start_idx + i1*4;
    double d12 = g_d12[rij_idx];
    if (d12 > rc_angular) return;
    double r12[3] = {g_d12[rij_idx+1], g_d12[rij_idx+2], g_d12[rij_idx+3]};
    double scd_r12[4] = {grad_second[rij_idx],grad_second[rij_idx+1],grad_second[rij_idx+2],grad_second[rij_idx+3]};// [r x y z]
    double f12[4] = {0.0};
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

    double fn12[MAX_NUM_N];
    double fnp12[MAX_NUM_N];
    find_fn_and_fnp(
      base_3b, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    int c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
    double blm[NUM_OF_ABC] = {0.0};
    double rij_blm[NUM_OF_ABC]= {0.0};
    double dblm_x[NUM_OF_ABC] = {0.0};
    double dblm_y[NUM_OF_ABC] = {0.0};
    double dblm_z[NUM_OF_ABC] = {0.0};
    double dblm_r[NUM_OF_ABC] = {0.0};
    scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2],
        blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
    for (int n = 0; n < max_3b; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      for (int k = 0; k < base_3b; ++k) {
        int c_index = c_I_J_idx + n * base_3b + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }
      // double f12d[MAX_LMAX * 4] = {0.0}; // dfeat/drij [nl+n+n, 4]
      // double f12k[TYPES * MAX_NUM_N] = {0.0};// max type is 20
      for (int j = 0; j < num_types; ++j) {
        double f12k[MAX_NUM_N] = {0.0}; // (20*4)*8=640 Bytes
        bool same_type = (t2 == j);
        if (L_max5 > 0) {
          scd_accumulate_f12_with_5body(
            n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              j, num_types, L_max3,
              max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        } else if (L_max4 > 0) {
          scd_accumulate_f12_with_4body(
            n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              j, num_types, L_max3,
              max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        } else {
          scd_accumulate_f12(
            n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              j, num_types, L_max3,
              max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        }
        for (int k = 0; k < base_3b; ++k){
          int dc_id = dc_start_idx + j * max_3b * base_3b + n*base_3b + k;
          // int k_id = j * base_3b * 4 + k * 4;
          int k_id = k;
          dfeat_c3[dc_id] += f12k[k_id];
        }
      }
    }
}


static __device__ __forceinline__ void find_fn_and_fnp_scalar(
  const int n, const double rcinv, const double d12,
  const double fc12, const double fcp12, double& fn, double& fnp)
{
  if (n == 0) {
    fn = fc12;
    fnp = fcp12;
    return;
  }

  const double shifted = d12 * rcinv - 1.0;
  const double x = 2.0 * shifted * shifted - 1.0;
  if (n == 1) {
    const double raw_fn = (x + 1.0) * 0.5;
    fn = raw_fn * fc12;
    fnp = 2.0 * shifted * rcinv * fc12 + raw_fn * fcp12;
    return;
  }
  double t_nm2 = 1.0;
  double t_nm1 = x;
  for (int order = 2; order <= n; ++order) {
    const double t_n = 2.0 * x * t_nm1 - t_nm2;
    t_nm2 = t_nm1;
    t_nm1 = t_n;
  }

  double u_nm2 = 1.0;
  double u_nm1 = 2.0 * x;
  for (int order = 2; order < n; ++order) {
    const double u_n = 2.0 * x * u_nm1 - u_nm2;
    u_nm2 = u_nm1;
    u_nm1 = u_n;
  }

  const double raw_fn = (t_nm1 + 1.0) * 0.5;
  fnp = n * u_nm1 * 2.0 * shifted * rcinv;
  fnp = fnp * fc12 + raw_fn * fcp12;
  fn = raw_fn * fc12;
}

static __global__ void find_angular_gardc_neigh_fallback(
  const int N,
  const double* grad_second,
  const double* g_d12,
  const int64_t* g_NL,
  const double* de_dfeat,
  const double* dsnlm_dc,
  const double* g_sum_fxyz,
  const int64_t* g_type,
  const double* coeff3,
  double* dfeat_c3,
  const double rc_angular,
  const double rcinv_angular,
  const int atom_nums,
  const int neigh_num,
  const int max_3b,
  const int base_3b,
  const int num_types,
  const int num_types_sq,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums)
{
  int n1 = blockIdx.x;
  if (n1 >= N) return;

  __shared__ double shm_sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
  __shared__ double shm_Fp[MAX_DIM_ANGULAR];

  int neigh_start_idx = n1 * neigh_num;
  int t1 = g_type[n1];
  int g_sum_start = n1 * max_3b * NUM_OF_ABC;
  int r12_start_idx = n1 * neigh_num * 4;
  int de_start = n1 * (feat_3b_nums + feat_2b_nums);
  int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;
  int c3_start_idx = t1 * num_types * max_3b * base_3b;

  int total_s_elements = max_3b * NUM_OF_ABC;
  for (int k = threadIdx.x; k < total_s_elements; k += blockDim.x) {
    shm_sum_fxyz[k] = g_sum_fxyz[g_sum_start + k];
  }

  int b3_nums = max_3b * L_max3;
  int total_Fp_elements = b3_nums + (L_max4 > 0 ? max_3b : 0) +
      (L_max5 > 0 ? max_3b : 0);
  for (int k = threadIdx.x; k < total_Fp_elements; k += blockDim.x) {
    shm_Fp[k] = 0.0;
  }
  for (int k = threadIdx.x; k < b3_nums; k += blockDim.x) {
    int nn = k / L_max3;
    int ll = k % L_max3;
    shm_Fp[k] = de_dfeat[de_start + feat_2b_nums + ll * max_3b + nn];
  }
  if (L_max4 > 0) {
    for (int k = threadIdx.x; k < max_3b; k += blockDim.x) {
      shm_Fp[b3_nums + k] = de_dfeat[de_start + feat_2b_nums + b3_nums + k];
    }
  }
  if (L_max5 > 0) {
    for (int k = threadIdx.x; k < max_3b; k += blockDim.x) {
      shm_Fp[b3_nums + max_3b + k] =
          de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + k];
    }
  }
  __syncthreads();

  for (int i1 = threadIdx.x; i1 < neigh_num; i1 += blockDim.x) {
    int n2 = g_NL[neigh_start_idx + i1];
    if (n2 < 0) continue;
    int t2 = g_type[n2];
    int dc_start_idx =
        (n1 * neigh_num + i1) * num_types * max_3b * base_3b;
    int rij_idx = r12_start_idx + i1 * 4;
    double d12 = g_d12[rij_idx];
    if (d12 > rc_angular) continue;
    double r12[3] = {
        g_d12[rij_idx + 1], g_d12[rij_idx + 2], g_d12[rij_idx + 3]};
    double scd_r12[4] = {
        grad_second[rij_idx], grad_second[rij_idx + 1],
        grad_second[rij_idx + 2], grad_second[rij_idx + 3]};
    double f12[4] = {0.0};
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

    double fn12[MAX_NUM_N];
    double fnp12[MAX_NUM_N];
    find_fn_and_fnp(
        base_3b, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    int c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
    double blm[NUM_OF_ABC] = {0.0};
    double rij_blm[NUM_OF_ABC] = {0.0};
    double dblm_x[NUM_OF_ABC] = {0.0};
    double dblm_y[NUM_OF_ABC] = {0.0};
    double dblm_z[NUM_OF_ABC] = {0.0};
    double dblm_r[NUM_OF_ABC] = {0.0};
    scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2], blm, rij_blm,
        dblm_x, dblm_y, dblm_z, dblm_r);
    for (int n = 0; n < max_3b; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      for (int k = 0; k < base_3b; ++k) {
        int c_index = c_I_J_idx + n * base_3b + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }
      for (int j = 0; j < num_types; ++j) {
        double f12k[MAX_NUM_N * 4] = {0.0};
        bool same_type = (t2 == j);
        if (L_max5 > 0) {
          nep3_base_fallback::scd_accumulate_f12_with_5body(
              n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r, f12, f12k,
              scd_r12, fn12, fnp12, j, num_types, L_max3, max_3b, base_3b,
              dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        } else if (L_max4 > 0) {
          nep3_base_fallback::scd_accumulate_f12_with_4body(
              n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r, f12, f12k,
              scd_r12, fn12, fnp12, j, num_types, L_max3, max_3b, base_3b,
              dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        } else {
          nep3_base_fallback::scd_accumulate_f12(
              n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r, f12, f12k,
              scd_r12, fn12, fnp12, j, num_types, L_max3, max_3b, base_3b,
              dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        }
        for (int k = 0; k < base_3b; ++k) {
          int dc_id = dc_start_idx + j * max_3b * base_3b + n * base_3b + k;
          int k_id = k * 4;
          dfeat_c3[dc_id] += f12k[k_id] + f12k[k_id + 1] +
              f12k[k_id + 2] + f12k[k_id + 3];
        }
      }
    }
  }
}

// 优化 f12k 的基础上，进一步优化。
// 每个 block 处理一个原子，Fp 和 sum_fxyz 移入 shmem。
template <int BASE_3B, int FIXED_MAX_3B = 0, int FIXED_NUM_TYPES = 0,
          int FIXED_L_MAX3 = 0, int FIXED_L_MAX5 = 0>
static __global__ void find_angular_gardc_neigh_opt(
  const int N,
  const double* __restrict__ grad_second,
  const double* __restrict__ g_d12,
  const int64_t* __restrict__ g_NL,
  const double* __restrict__ de_dfeat,
  const double* __restrict__ dsnlm_dc, //[i, J, nbase, 24]
  const double* __restrict__ g_sum_fxyz,
  const int64_t* __restrict__ g_type,
  const double* __restrict__ coeff3,
  double* __restrict__ dfeat_c3,
  const double rc_angular,
  const double rcinv_angular,
  const int atom_nums,
  const int neigh_num,
  const int max_3b,
  const int base_3b,
  const int num_types,
  const int num_types_sq,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums // 3b + 4b + 5b
  )
{
  int n1 = blockIdx.x;
  if (n1 >= N) return;

  // Block 共享存储
  constexpr int shared_max_3b = FIXED_MAX_3B > 0 ? FIXED_MAX_3B : MAX_NUM_N;
  constexpr int shared_l_count = FIXED_L_MAX3 > 0
      ? FIXED_L_MAX3 + (FIXED_L_MAX5 > 0 ? 2 : 1) : 6;
  __shared__ double shm_sum_fxyz[NUM_OF_ABC * shared_max_3b];
  __shared__ double shm_Fp[shared_max_3b * shared_l_count];
  __shared__ double shm_angular_work[14][5 * NUM_OF_ABC - 1];
  __shared__ double shm_neighbor_f12k[14][BASE_3B];
  __shared__ int shm_neighbor_count;

  int neigh_start_idx = n1 * neigh_num;
  int t1 = g_type[n1];

  int g_sum_start = n1 * max_3b * NUM_OF_ABC;
  int r12_start_idx =  n1 * neigh_num * 4;
  int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
  int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;
  int c3_start_idx = t1 * num_types * max_3b * base_3b;

  if (threadIdx.x == 0) {
    int neighbor_count = 0;
    while (neighbor_count < neigh_num &&
           g_NL[neigh_start_idx + neighbor_count] >= 0) {
      ++neighbor_count;
    }
    shm_neighbor_count = neighbor_count;
  }

  // 加载 sum_fxyz 到 shared memory
  int total_s_elements = max_3b * NUM_OF_ABC;
  for (int k = threadIdx.x; k < total_s_elements; k += blockDim.x) {
    shm_sum_fxyz[k] = g_sum_fxyz[g_sum_start + k]; // g_sum is [N, n_max, 24]
  }

  // 加载 Fp 到 shared memory
  int b3_nums = max_3b * L_max3;
  int total_Fp_elements = b3_nums + (L_max4 > 0 ? max_3b : 0) + (L_max5 > 0 ? max_3b : 0);

  for (int k = threadIdx.x; k < total_Fp_elements; k += blockDim.x) {
    shm_Fp[k] = 0.0;
  }

  for (int k = threadIdx.x; k < b3_nums; k += blockDim.x) {
    int nn = k / L_max3;
    int ll = k % L_max3;
    shm_Fp[k] = de_dfeat[de_start + feat_2b_nums + ll * max_3b + nn];
  }
  if (L_max4 > 0) {
    for (int k = threadIdx.x; k < max_3b; k += blockDim.x) {
      shm_Fp[b3_nums + k] = de_dfeat[de_start + feat_2b_nums + b3_nums + k];
    }
  }
  if (L_max5 > 0) {
    for (int k = threadIdx.x; k < max_3b; k += blockDim.x) {
      shm_Fp[b3_nums + max_3b + k] = de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + k];
    }
  }

  __syncthreads();

  const int basis_count = (BASE_3B == MAX_NUM_N) ? base_3b : BASE_3B;

  if constexpr (BASE_3B == 9 || BASE_3B == 13) {
    constexpr int target_max_3b = FIXED_MAX_3B > 0 ? FIXED_MAX_3B : 13;
    constexpr int target_num_types =
        FIXED_NUM_TYPES > 0 ? FIXED_NUM_TYPES : 2;
    constexpr int target_l_max3 = FIXED_L_MAX3 > 0 ? FIXED_L_MAX3 : 4;
    constexpr bool target_has_l5 = FIXED_L_MAX5 > 0;
    constexpr int wave_size = 64;
    constexpr int lanes_per_neighbor = BASE_3B;
    constexpr int neighbors_per_wave = wave_size / lanes_per_neighbor;
    constexpr int neighbors_per_block = 2 * neighbors_per_wave;
    const int wave_id = threadIdx.x / wave_size;
    const int lane_id = threadIdx.x % wave_size;
    const int group_in_wave = lane_id / lanes_per_neighbor;
    const int basis_lane = lane_id % lanes_per_neighbor;
    const bool valid_group = group_in_wave < neighbors_per_wave;
    const int neighbor_group = wave_id * neighbors_per_wave + group_in_wave;

    for (int neighbor_base = 0; neighbor_base < shm_neighbor_count;
         neighbor_base += neighbors_per_block) {
      const int i1 = neighbor_base + neighbor_group;
      bool active = valid_group && i1 < shm_neighbor_count;
      int n2 = -1;
      int t2 = 0;
      int rij_idx = 0;
      double d12 = 0.0;
      double r12[3] = {0.0, 0.0, 0.0};
      double scd_r12[4] = {0.0, 0.0, 0.0, 0.0};
      double fc12 = 0.0;
      double fcp12 = 0.0;

      if (active) {
        n2 = g_NL[neigh_start_idx + i1];
        active = n2 >= 0;
      }
      if (active) {
        t2 = g_type[n2];
        rij_idx = r12_start_idx + i1 * 4;
        d12 = g_d12[rij_idx];
        active = d12 <= rc_angular;
      }
      if (active) {
        r12[0] = g_d12[rij_idx + 1];
        r12[1] = g_d12[rij_idx + 2];
        r12[2] = g_d12[rij_idx + 3];
        scd_r12[0] = grad_second[rij_idx];
        scd_r12[1] = grad_second[rij_idx + 1];
        scd_r12[2] = grad_second[rij_idx + 2];
        scd_r12[3] = grad_second[rij_idx + 3];
        find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);
      }

      if (basis_lane == 0 && active) {
        double* blm = shm_angular_work[neighbor_group];
        double* rij_blm = nullptr;
        double* dblm_x = blm + NUM_OF_ABC;
        double* dblm_y = dblm_x + NUM_OF_ABC;
        double* dblm_z = dblm_y + NUM_OF_ABC;
        double* dblm_r = dblm_z + NUM_OF_ABC;
        scd_accumulate_blm_rij<false>(d12, r12[0], r12[1], r12[2],
            blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
        scd_accumulate_blm_rij_l4_compact(d12, r12[0], r12[1], r12[2],
            blm + 15, rij_blm + 15, dblm_x + 15, dblm_y + 15,
            dblm_z + 15, dblm_r + 15);
      }
      __builtin_amdgcn_wave_barrier();

      double fn12 = 0.0;
      double fnp12 = 0.0;
      if (active) {
        find_fn_and_fnp_scalar(
            basis_lane, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      }
      double basis_gn12 = 0.0;
      double basis_gnp12 = 0.0;
      int c_I_J_idx = 0;
      int dc_start_idx = 0;
      if (active) {
        c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
        dc_start_idx =
            (n1 * neigh_num + i1) * num_types * max_3b * base_3b;
      }

      for (int n = 0; n < target_max_3b; ++n) {
        if (active) {
          const int c_index = c_I_J_idx + n * base_3b + basis_lane;
          basis_gn12 = fn12 * coeff3[c_index];
          basis_gnp12 = fnp12 * coeff3[c_index];
        }
        double gn12 = basis_gn12;
        double gnp12 = basis_gnp12;
        const int group_first_lane =
            valid_group ? group_in_wave * lanes_per_neighbor : 0;
        if (valid_group) {
          double ordered_gn12 = 0.0;
          double ordered_gnp12 = 0.0;
          #pragma unroll
          for (int basis = 0; basis < BASE_3B; ++basis) {
            ordered_gn12 += __shfl(
                basis_gn12, group_first_lane + basis, wave_size);
            ordered_gnp12 += __shfl(
                basis_gnp12, group_first_lane + basis, wave_size);
          }
          gn12 = ordered_gn12;
          gnp12 = ordered_gnp12;
        }

        #pragma unroll
        for (int j = 0; j < target_num_types; ++j) {
          double f12k = 0.0;
          if (active) {
            double* blm = shm_angular_work[neighbor_group];
            double* rij_blm = nullptr;
            double* dblm_x = blm + NUM_OF_ABC;
            double* dblm_y = dblm_x + NUM_OF_ABC;
            double* dblm_z = dblm_y + NUM_OF_ABC;
            double* dblm_r = dblm_z + NUM_OF_ABC;
            const bool same_type = t2 == j;
            const int scalar_dsnlm_start = dsnlm_start_idx +
                (j * BASE_3B + basis_lane) * NUM_OF_ABC;
            if constexpr (target_has_l5) {
              if (same_type) {
                scd_accumulate_f12_5body_only<1>(
                  n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc,
                  shm_sum_fxyz, blm, rij_blm, dblm_x, dblm_y, dblm_z,
                  dblm_r, &f12k, scd_r12, &fn12, &fnp12, 0,
                  target_num_types, target_l_max3, target_max_3b, 1,
                  scalar_dsnlm_start, n1, i1, true);
                scd_accumulate_f12_with_4body<false, 1>(
                  n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                  nullptr, &f12k, scd_r12, &fn12, &fnp12,
                  0, target_num_types, target_l_max3, target_max_3b, 1,
                  dc_start_idx, scalar_dsnlm_start, n1, i1, true);
                scd_accumulate_f12_l4_compact<1>(
                  n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm + 15, rij_blm + 15, dblm_x + 15, dblm_y + 15,
                  dblm_z + 15, dblm_r + 15, &f12k, scd_r12, &fn12, &fnp12,
                  0, target_num_types, target_l_max3, 1, scalar_dsnlm_start,
                  n1, i1, true);
              } else {
                scd_accumulate_f12_5body_only<0>(
                  n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc,
                  shm_sum_fxyz, blm, rij_blm, dblm_x, dblm_y, dblm_z,
                  dblm_r, &f12k, scd_r12, &fn12, &fnp12, 0,
                  target_num_types, target_l_max3, target_max_3b, 1,
                  scalar_dsnlm_start, n1, i1, false);
                scd_accumulate_f12_with_4body<false, 0>(
                  n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                  nullptr, &f12k, scd_r12, &fn12, &fnp12,
                  0, target_num_types, target_l_max3, target_max_3b, 1,
                  dc_start_idx, scalar_dsnlm_start, n1, i1, false);
                scd_accumulate_f12_l4_compact<0>(
                  n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm + 15, rij_blm + 15, dblm_x + 15, dblm_y + 15,
                  dblm_z + 15, dblm_r + 15, &f12k, scd_r12, &fn12, &fnp12,
                  0, target_num_types, target_l_max3, 1, scalar_dsnlm_start,
                  n1, i1, false);
              }
            } else if (same_type) {
              scd_accumulate_f12_with_4body<false, 1>(
                n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                  nullptr, &f12k, scd_r12, &fn12, &fnp12,
                0, target_num_types, target_l_max3, target_max_3b, 1,
                  dc_start_idx, scalar_dsnlm_start, n1, i1, true);
              scd_accumulate_f12_l4_compact<1>(
                n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm + 15, rij_blm + 15, dblm_x + 15, dblm_y + 15,
                  dblm_z + 15, dblm_r + 15, &f12k, scd_r12, &fn12, &fnp12,
                  0, target_num_types, target_l_max3, 1, scalar_dsnlm_start,
                  n1, i1, true);
            } else {
              scd_accumulate_f12_with_4body<false, 0>(
                n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                  nullptr, &f12k, scd_r12, &fn12, &fnp12,
                0, target_num_types, target_l_max3, target_max_3b, 1,
                  dc_start_idx, scalar_dsnlm_start, n1, i1, false);
              scd_accumulate_f12_l4_compact<0>(
                n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                  blm + 15, rij_blm + 15, dblm_x + 15, dblm_y + 15,
                  dblm_z + 15, dblm_r + 15, &f12k, scd_r12, &fn12, &fnp12,
                  0, target_num_types, target_l_max3, 1, scalar_dsnlm_start,
                  n1, i1, false);
            }
          }

          if (valid_group && basis_lane < BASE_3B) {
            shm_neighbor_f12k[neighbor_group][basis_lane] = f12k;
          }
          __syncthreads();
          if (wave_id == 0 && group_in_wave == 0 &&
              neighbor_base < shm_neighbor_count) {
            const int dc_id =
                (n1 * target_num_types * target_max_3b +
                   j * target_max_3b + n) * BASE_3B + basis_lane;
            #pragma unroll
            for (int neighbor_offset = 0;
                 neighbor_offset < neighbors_per_block; ++neighbor_offset) {
              if (neighbor_base + neighbor_offset < shm_neighbor_count) {
                dfeat_c3[dc_id] +=
                    shm_neighbor_f12k[neighbor_offset][basis_lane];
              }
            }
          }
          __syncthreads();
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
    return;
  }

  for (int i1 = threadIdx.x; i1 < neigh_num; i1 += blockDim.x) {
    int n2 = g_NL[neigh_start_idx + i1];
    if (n2 < 0) continue;
    int t2 = g_type[n2];

    // The fixed-shape variants retain one result row per (atom, neighbor),
    // which their wave-level reduction writes without atomics.  The generic
    // BASE_3B == MAX_NUM_N variant instead owns one block per atom and
    // accumulates every neighbor directly into that atom's compact row.
    // This avoids allocating an O(atoms * neighbors * types * n * basis)
    // temporary tensor for otherwise supported descriptor shapes.
    int dc_start_idx = (BASE_3B == MAX_NUM_N)
        ? n1 * num_types * max_3b * base_3b
        : (n1 * neigh_num + i1) * num_types * max_3b * base_3b;
    int rij_idx = r12_start_idx + i1*4;
    double d12 = g_d12[rij_idx];
    if (d12 > rc_angular) continue;
    double r12[3] = {g_d12[rij_idx+1], g_d12[rij_idx+2], g_d12[rij_idx+3]};
    double scd_r12[4] = {grad_second[rij_idx],grad_second[rij_idx+1],grad_second[rij_idx+2],grad_second[rij_idx+3]};// [r x y z]
    double f12[4] = {0.0};
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

    double fn12[BASE_3B];
    double fnp12[BASE_3B];
    find_fn_and_fnp(
      (BASE_3B == MAX_NUM_N) ? base_3b : BASE_3B,
      rcinv_angular, d12, fc12, fcp12, fn12, fnp12);

    int c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
    for (int n = 0; n < max_3b; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      #pragma unroll
      for (int k = 0; k < basis_count; ++k) {
        int c_index = c_I_J_idx + n * base_3b + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }
      // min (1*20*4)*8=640 Bytes, max (20*20*4)*8=12800 Bytes
      // double f12k[TYPES * MAX_NUM_N] = {0.0};// max type is 20
      for (int j = 0; j < num_types; ++j) {
        double f12k[BASE_3B] = {0.0};
        bool same_type = (t2 == j);
        if (L_max5 > 0) {
          double blm[NUM_OF_ABC] = {0.0};
          double rij_blm[NUM_OF_ABC]= {0.0};
          double dblm_x[NUM_OF_ABC] = {0.0};
          double dblm_y[NUM_OF_ABC] = {0.0};
          double dblm_z[NUM_OF_ABC] = {0.0};
          double dblm_r[NUM_OF_ABC] = {0.0};
          scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2],
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
          scd_accumulate_f12_with_5body(
            n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              j, num_types, L_max3,
              max_3b, (BASE_3B == MAX_NUM_N) ? base_3b : BASE_3B,
              dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        } else if (L_max4 > 0) {
          {
            double blm[15] = {0.0};
            double rij_blm[15] = {0.0};
            double dblm_x[15] = {0.0};
            double dblm_y[15] = {0.0};
            double dblm_z[15] = {0.0};
            double dblm_r[15] = {0.0};
            scd_accumulate_blm_rij<false>(d12, r12[0], r12[1], r12[2],
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
            scd_accumulate_f12_with_4body<false>(
              n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                f12, f12k, scd_r12, fn12, fnp12,
                j, num_types, L_max3,
                max_3b, (BASE_3B == MAX_NUM_N) ? base_3b : BASE_3B,
                dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
          }
          {
            double blm[9] = {0.0};
            double rij_blm[9] = {0.0};
            double dblm_x[9] = {0.0};
            double dblm_y[9] = {0.0};
            double dblm_z[9] = {0.0};
            double dblm_r[9] = {0.0};
            scd_accumulate_blm_rij_l4_compact(d12, r12[0], r12[1], r12[2],
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
            scd_accumulate_f12_l4_compact(
              n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
                blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
                f12k, scd_r12, fn12, fnp12, j, num_types, L_max3,
                (BASE_3B == MAX_NUM_N) ? base_3b : BASE_3B,
                dsnlm_start_idx, n1, i1, same_type);
          }
        } else {
          double blm[NUM_OF_ABC] = {0.0};
          double rij_blm[NUM_OF_ABC]= {0.0};
          double dblm_x[NUM_OF_ABC] = {0.0};
          double dblm_y[NUM_OF_ABC] = {0.0};
          double dblm_z[NUM_OF_ABC] = {0.0};
          double dblm_r[NUM_OF_ABC] = {0.0};
          scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2],
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
          scd_accumulate_f12(
            n, d12, r12, gn12, gnp12, shm_Fp, dsnlm_dc, shm_sum_fxyz,
              blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
              f12, f12k, scd_r12, fn12, fnp12,
              j, num_types, L_max3,
              max_3b, (BASE_3B == MAX_NUM_N) ? base_3b : BASE_3B,
              dc_start_idx, dsnlm_start_idx, n1, i1, same_type);
        }
        #pragma unroll
        for (int k = 0; k < basis_count; ++k){
          int dc_id = dc_start_idx + j * max_3b * base_3b + n*base_3b + k;
          // int k_id = j * base_3b * 4 + k * 4;
          int k_id = k;
          if constexpr (BASE_3B == MAX_NUM_N) {
            // Several neighbor threads contribute to the same atom-local C3
            // coefficient. There is one block per atom, but still multiple
            // concurrent writers within that block.
            atomicAdd(&dfeat_c3[dc_id], f12k[k_id]);
          } else {
            dfeat_c3[dc_id] += f12k[k_id];
          }
        }
      }
    }
  }
}


static __global__ void aggregate_dfeat_c3(
  const int64_t* g_NL,
  const int64_t* g_type,
  const double* dfeat_c3,
  double* tmp_dfeat_c3,
  const int N,
  const int atom_nums,
  const int neigh_num,
  const int num_types,
  const int max_3b,
  const int base_3b
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int tmp_start_idx = n1 * num_types * max_3b * base_3b;
    int dc_start_idx = n1 * neigh_num * num_types * max_3b * base_3b;
    int neigh_start_idx = n1 * neigh_num;
    // int t1 = g_type[n1];
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL[neigh_start_idx + i1];
      if (n2 < 0) break;
      // int t2 = g_type[n2];
      int dc_idx = dc_start_idx + i1 * num_types * max_3b * base_3b;
      for (int j = 0; j < num_types; ++j){
        for (int n = 0; n < max_3b; ++n) {
          for (int k = 0; k < base_3b; ++k){
            int dc_id = dc_idx + j * max_3b * base_3b + n*base_3b + k;
            int tmp_dc_id = tmp_start_idx + j * max_3b * base_3b + n*base_3b + k;
            tmp_dfeat_c3[tmp_dc_id] += dfeat_c3[dc_id];
          }
        }
      }
    }
  }
}

// 每个线程处理一个(atom, neighbor)对，将dfeat_c3中的梯度累加到tmp_dfeat_c3中
static __global__ void aggregate_dfeat_c3_optimized(
  const int64_t* g_NL,
  const int64_t* g_type,
  const double* dfeat_c3,
  double* tmp_dfeat_c3,
  const int N,
  const int atom_nums,
  const int neigh_num,
  const int num_types,
  const int max_3b,
  const int base_3b
  )
{
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= N * neigh_num) return;

  int n1 = global_idx / neigh_num;
  int i1 = global_idx % neigh_num;

  int neigh_start_idx = n1 * neigh_num;
  int n2 = g_NL[neigh_start_idx + i1];
  if (n2 < 0) return;

  int tmp_start_idx = n1 * num_types * max_3b * base_3b;
  int dc_idx = n1 * neigh_num * num_types * max_3b * base_3b + i1 * num_types * max_3b * base_3b;

  for (int j = 0; j < num_types; ++j){
    for (int n = 0; n < max_3b; ++n) {
      for (int k = 0; k < base_3b; ++k){
        int dc_id = dc_idx + j * max_3b * base_3b + n*base_3b + k;
        int tmp_dc_id = tmp_start_idx + j * max_3b * base_3b + n*base_3b + k;
        atomicAdd(&tmp_dfeat_c3[tmp_dc_id], dfeat_c3[dc_id]);
      }
    }
  }
}
