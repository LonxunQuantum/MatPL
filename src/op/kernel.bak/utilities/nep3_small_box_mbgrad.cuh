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
#include "nep_utilities_mb_secondc.cuh"

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
        double f12k[TYPES * MAX_NUM_N * 4] = {0.0};// max type is 20
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
            int k_id = j * base_3b * 4 + k * 4;
            dfeat_c3[dc_id] += (f12k[k_id] + f12k[k_id+1] + f12k[k_id+2] + f12k[k_id+3]);
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
      double f12k[TYPES * MAX_NUM_N * 4] = {0.0};// max type is 20
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
          int k_id = j * base_3b * 4 + k * 4;
          dfeat_c3[dc_id] += (f12k[k_id] + f12k[k_id+1] + f12k[k_id+2] + f12k[k_id+3]);
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