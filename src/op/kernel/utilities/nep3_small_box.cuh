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

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __device__ __inline__ double atomicAdd(double* address, double val)
{
  unsigned long long* address_as_ull = (unsigned long long*)address;
  unsigned long long old = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));

  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

//后面删除了nepmb的代码之后，将这部分移动到mbnepgrad下面
static __global__ void aggregate_features(
    double* dfeat_c2,     // 输入张量，维度 [N, n_types, n_max_2b, n_base_2b]
    const int64_t* atom_map,       // 原子类型映射，维度 [N]
    double* output,             // 输出张量，维度 [n_types, n_types, n_max_2b, n_base_2b]
    int N,                     // 中心原子数量
    int n_types,
    int n_max_2b,
    int n_base_2b) {

    // 获取线程索引
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 根据总元素数限制线程数
    int total_elements = n_max_2b * n_base_2b;
    if (tid >= N * total_elements) return;

    // 计算当前线程负责的元素在 dfeat_c2 中的位置
    int atom_idx = tid / total_elements; // 当前中心原子索引
    int base_idx = tid % total_elements; // n_max_2b 和 n_base_2b 的线性索引

    // 获取原子类型
    int center_type = atom_map[atom_idx];

    // 遍历 n_types（对应所有类型）
    for (int neighbor_type = 0; neighbor_type < n_types; ++neighbor_type) {
        // 累加到输出张量的指定位置
        atomicAdd(&output[center_type * n_types * total_elements +
                          neighbor_type * total_elements + base_idx],
                  dfeat_c2[atom_idx * n_types * total_elements +
                           neighbor_type * total_elements + base_idx]);
    }
}

static __global__ void find_mb_descriptor_small_box(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL,
  const double * coeff3,
  double * feats,
  const int64_t* g_type,
  const double* g_d12_radial,
  double* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    // double q[MAX_DIM] = {static_cast<double>(0.0)};
    // get radial descriptors
    double q[MAX_DIM] = {0.0};
    int neigh_start_idx = n1 * neigh_num;
    int r12_start_idx =  n1 * neigh_num * 4;
    int feat_start_idx = n1 * feat_nums; 
    // get angular descriptors
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    int sum_s_start_idx = n1 * n_max_angular * NUM_OF_ABC;
    for (int n = 0; n < n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neigh_num; ++i1) {
        int n2 = g_NL[neigh_start_idx + i1];
        if (n2 < 0) break;
        int t2 = g_type[n2];
        int rij_idx = r12_start_idx + i1*4;
        double d12 = g_d12_radial[rij_idx];
        if (d12 > rc_angular) break;
        double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
        double fc12;
        find_fc(rc_angular, rcinv_angular, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(basis_size_angular, rcinv_angular, d12, fc12, fn12);
        double gn12 = 0.0;
        int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
        for (int k = 0; k < basis_size_angular; ++k) {
          int c_index = c_I_J_idx + n * basis_size_angular + k;
          gn12 += fn12[k] * coeff3[c_index];
        }
        accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
        // if (n1 == 0 and n == 0) {
        //   printf("n1=0 t1=%d n2=%d t2=%d n=0 d12=%f rc=%f rcin=%f gn12=%f\n", 
        //     t1, i1, t2, d12, rc_angular, rcinv_angular, gn12);
        // }
      }
      // if (n1 == 0 and n == 0) {
      //   for (int si = 0; si < 24; si++) {
      //     printf("n1=0 s[%d] = %f\n", si, s[si]);
      //   }
      // }
      if (L_max5 == 1) {
          find_q_with_5body(n_max_angular, n, s, q);
      } else if (L_max4 ==2) {
        find_q_with_4body(n_max_angular, n, s, q);
      } else {
        find_q(n_max_angular, n, s, q);
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[sum_s_start_idx + n * NUM_OF_ABC + abc] = s[abc];
      }
    }
    for (int n1 = 0; n1 < feat_nums; ++n1) {
      feats[feat_start_idx+n1] = q[n1];
    }
  }
}

// find_mb_descriptor_small_box拆分为两个kernel，提高并行度
// 1. 计算 s 的贡献，并行粒度为 (atom, angle, neighbor)
static __global__ void compute_s_optimized(
  const int N,
  const int neigh_num,
  const int n_max_angular,
  const int num_types,
  const int basis_size_angular,
  const int64_t* g_NL,
  const double* g_d12_radial,
  const int64_t* g_type,
  const double* coeff3,
  const double rc_angular,
  const double rcinv_angular,
  double* g_sum_fxyz) // [N, n_max_angular, NUM_OF_ABC]
{
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * n_max_angular * neigh_num;
  if (global_idx >= total) return;
  
  // 解码索引: global_idx = n1 * (n_max_angular * neigh_num) + n * neigh_num + i1
  int n1 = global_idx / (n_max_angular * neigh_num);
  int remainder = global_idx % (n_max_angular * neigh_num);
  int n = remainder / neigh_num;
  int i1 = remainder % neigh_num;
  
  // 检查邻居有效性
  int n2 = g_NL[n1 * neigh_num + i1];
  if (n2 < 0) return;
  
  int rij_idx = n1 * neigh_num * 4 + i1 * 4;
  double d12 = g_d12_radial[rij_idx];
  if (d12 > rc_angular) return;
  
  // 读取相对位置向量
  double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
  
  // 计算截断函数
  double fc12;
  find_fc(rc_angular, rcinv_angular, d12, fc12);
  
  // 计算径向基函数
  double fn12[MAX_NUM_N]; // 20*8=160 Bytes
  find_fn(basis_size_angular, rcinv_angular, d12, fc12, fn12);
  
  // 计算 gn12 = sum_k fn12[k] * coeff3[t1, t2, n, k]
  int t1 = g_type[n1];
  int t2 = g_type[n2];
  int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
  int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
  
  double gn12 = 0.0;
  for (int k = 0; k < basis_size_angular; ++k) {
    int c_index = c_I_J_idx + n * basis_size_angular + k;
    gn12 += fn12[k] * coeff3[c_index];
  }
  
  // 计算局部 s 贡献 (内联 accumulate_s 逻辑)
  const double d12_inv = 1.0 / d12;
  const double x12 = r12[0] * d12_inv;
  const double y12 = r12[1] * d12_inv;
  const double z12 = r12[2] * d12_inv;
  const double x12sq = x12 * x12;
  const double y12sq = y12 * y12;
  const double z12sq = z12 * z12;
  const double x12sq_minus_y12sq = x12sq - y12sq;
  
  double local_s[NUM_OF_ABC];
  local_s[0] = gn12 * z12;
  local_s[1] = gn12 * x12;
  local_s[2] = gn12 * y12;
  local_s[3] = gn12 * (3.0 * z12sq - 1.0);
  local_s[4] = gn12 * x12 * z12;
  local_s[5] = gn12 * y12 * z12;
  local_s[6] = gn12 * x12sq_minus_y12sq;
  local_s[7] = gn12 * 2.0 * x12 * y12;
  local_s[8] = gn12 * (5.0 * z12sq - 3.0) * z12;
  local_s[9] = gn12 * (5.0 * z12sq - 1.0) * x12;
  local_s[10] = gn12 * (5.0 * z12sq - 1.0) * y12;
  local_s[11] = gn12 * x12sq_minus_y12sq * z12;
  local_s[12] = gn12 * 2.0 * x12 * y12 * z12;
  local_s[13] = gn12 * (x12 * x12 - 3.0 * y12 * y12) * x12;
  local_s[14] = gn12 * (3.0 * x12 * x12 - y12 * y12) * y12;
  local_s[15] = gn12 * ((35.0 * z12sq - 30.0) * z12sq + 3.0);
  local_s[16] = gn12 * (7.0 * z12sq - 3.0) * x12 * z12;
  local_s[17] = gn12 * (7.0 * z12sq - 3.0) * y12 * z12;
  local_s[18] = gn12 * (7.0 * z12sq - 1.0) * x12sq_minus_y12sq;
  local_s[19] = gn12 * (7.0 * z12sq - 1.0) * x12 * y12 * 2.0;
  local_s[20] = gn12 * (x12sq - 3.0 * y12sq) * x12 * z12;
  local_s[21] = gn12 * (3.0 * x12sq - y12sq) * y12 * z12;
  local_s[22] = gn12 * (x12sq_minus_y12sq * x12sq_minus_y12sq - 4.0 * x12sq * y12sq);
  local_s[23] = gn12 * (4.0 * x12 * y12 * x12sq_minus_y12sq);
  
  // 原子累加到全局内存
  int s_base = (n1 * n_max_angular + n) * NUM_OF_ABC;
  for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
    atomicAdd(&g_sum_fxyz[s_base + abc], local_s[abc]);
  }
}

// 2. 从 s 计算 q 和 feats，按原子并行
static __global__ void compute_q_and_feats_optimized(
  const int N,
  const int n_max_angular,
  const int feat_nums,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const double* g_sum_fxyz, // [N, n_max_angular, NUM_OF_ABC]
  double* feats) // [N, feat_nums]
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= N) return;
  
  double q[MAX_DIM] = {0.0}; // MAX_DIM*8=(MAX_NUM_N*7)*8=(20*7)*8=1120 Bytes
  int sum_s_start_idx = n1 * n_max_angular * NUM_OF_ABC;
  int feat_start_idx = n1 * feat_nums;
  
  // 对每个 n，从 s 计算 q
  for (int n = 0; n < n_max_angular; ++n) {
    double s[NUM_OF_ABC]; // 24*8=192 Bytes
    int s_base = sum_s_start_idx + n * NUM_OF_ABC;
    for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
      s[abc] = g_sum_fxyz[s_base + abc];
    }
    
    // 根据多体阶数选择对应的 find_q 函数
    if (L_max5 == 1) {
      find_q_with_5body(n_max_angular, n, s, q);
    } else if (L_max4 == 2) {
      find_q_with_4body(n_max_angular, n, s, q);
    } else {
      find_q(n_max_angular, n, s, q);
    }
  }
  
  // 写入 feats
  for (int f = 0; f < feat_nums; ++f) {
    feats[feat_start_idx + f] = q[f];
  }
}

static __global__ void find_angular_gard_small_box(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums, // 3b + 4b + 5b
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL_radial,
  const double* g_d12_radial,
  const double * coeff3,
  const int64_t* g_type,
  const double * grad_output,
  const double* g_sum_fxyz,
  double* dsnlm_dc,
  double* dfeat_c3,
  double* dfeat_drij,//[batch*atom, neighbornum, 3b_feat_num, 4]
  double* grad_d12_angular
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
    int r12_start_idx =  n1 * neigh_num * 4;
    int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
    int dsnlm_dc_start_idx = n1 * num_types * basis_size_angular * NUM_OF_ABC;
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
    int neigh_start_idx = n1 * neigh_num;
    int dfeat_dr_start = n1 * neigh_num * feat_3b_nums * 4;
    double Fp[MAX_DIM_ANGULAR] = {0.0};
    double sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    int b3_nums = n_max_angular * L_max3;
    int dd = 0;
    // if (n1 == 0) {
    //   for (int nn=0; nn < 108; nn++) {//all
    //     printf("grad_out_angluar[b0][%d][:] = ", nn);
    //     // printf("grad[%d + %d]=%f\n", de_start, feat_2b_nums + nn, grad_output[de_start + feat_2b_nums + nn]);
    //     for (int jj = 0; jj < 25; jj++) {
    //       printf("%f  ", grad_output[nn*25 + jj]);
    //     }
    //     printf("\n");
    //   }
    // }
    for (int nn=0; nn < n_max_angular; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        Fp[dd] = grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];// i -> nmax_3b*l_max+2?
        // 0 5 10 15
        // 1 6 11 16
        // 2 7 12 17
        // 3 8 13 18
        // 4 9 14 19 the feature order is L*n_max
        // if (n1==0){
        //   printf("3b Fp[%d] = %f from grad_output[%d + %d] = %f\n", dd, Fp[dd], de_start,  feat_2b_nums + ll * n_max_angular + nn, grad_output[de_start +  feat_2b_nums + ll * n_max_angular + nn]);
        // }
        dd++;
      }
    }
    if (L_max4 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        Fp[b3_nums + ll] = grad_output[de_start + feat_2b_nums + b3_nums + ll];
        // if (n1==0){
        //   printf("4b Fp[%d + %d] = %f from grad_output[%d + %d] = %f\n", 
        //   b3_nums, ll, Fp[b3_nums + ll], de_start,  feat_2b_nums + b3_nums + ll, grad_output[de_start + feat_2b_nums + b3_nums + ll]);
        // }
      }
    }
    if (L_max5 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        Fp[b3_nums + n_max_angular + ll] = grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + ll];
        // if (n1==0){
        //   printf("5b Fp[%d + %d] = %f from grad_output[%d + %d] = %f\n", 
        //   b3_nums, n_max_angular + ll, Fp[b3_nums + n_max_angular + ll], de_start, feat_2b_nums + b3_nums + n_max_angular + ll, grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + ll]);
        // }
      }
    }

    for (int d = 0; d < n_max_angular * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[g_sum_start + d]; // g_sum is [N, n_max, 24]
    }

    int t1 = g_type[n1];
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL_radial[neigh_start_idx + i1];
      if (n2 < 0) break;
      int t2 = g_type[n2];
      int rij_idx = r12_start_idx + i1*4;
      int dsnlm_dc_idx = dsnlm_dc_start_idx + t2 * basis_size_angular * NUM_OF_ABC;
      double d12 = g_d12_radial[rij_idx];
      if (d12 > rc_angular) break;
      int drij_idx = dfeat_dr_start + i1 * feat_3b_nums * 4;
      double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
      double f12[4] = {0.0};

      double fc12, fcp12;
      find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

      double fn12[MAX_NUM_N];
      double fnp12[MAX_NUM_N];
      find_fn_and_fnp(
        basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      
      int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
      double s[NUM_OF_ABC] = {0.0};
      accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);// blm * 1/(r_ij^L) for dfeat/dC_NK^IJ = fk * blm * 1/(r_ij^L)
      for (int n = 0; n < n_max_angular; ++n) {
        double gn12 = 0.0;
        double gnp12 = 0.0;
        for (int k = 0; k < basis_size_angular; ++k) {
          int c_index = c_I_J_idx + n * basis_size_angular + k;
          gn12 += fn12[k] * coeff3[c_index];
          gnp12 += fnp12[k] * coeff3[c_index];
        }
        double f12d[MAX_LMAX * 4] = {0.0}; // 
        if (L_max5 > 0) {
          accumulate_f12_with_5body(
            n, d12, r12, gn12, gnp12, Fp, sum_fxyz,
              s, f12, f12d, dfeat_c3, fn12, fnp12, 
              t2, num_types, L_max3, 
              n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
          // copy 3b
          for (int l_idx = 0; l_idx < L_max3; ++l_idx){
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
           }
          // copy 4b
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
          // copy 5b
          dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 0] = f12d[(L_max3+1) * 4 + 3];
          dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 1] = f12d[(L_max3+1) * 4 + 0];
          dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 2] = f12d[(L_max3+1) * 4 + 1];
          dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 3] = f12d[(L_max3+1) * 4 + 2];
        } else if (L_max4 > 0) {
          accumulate_f12_with_4body(
            n, d12, r12, gn12, gnp12, Fp, sum_fxyz,
              s, f12, f12d, dfeat_c3, fn12, fnp12, 
              t2, num_types, L_max3, 
              n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
          // copy 3b
          for (int l_idx = 0; l_idx < L_max3; ++l_idx){
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
           }
          // copy 4b
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
          dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
        } else {
          accumulate_f12(
            n, d12, r12, gn12, gnp12, Fp, sum_fxyz,
              s, f12, f12d, dfeat_c3, fn12, fnp12, 
              t2, num_types, L_max3, 
              n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
          // copy 3b
          for (int l_idx = 0; l_idx < L_max3; ++l_idx){
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
            dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
           }
        }
        if (n == 0) {
          for(int kk = 0; kk < basis_size_angular;kk++){
            int dsnlm_id = dsnlm_dc_idx + kk * NUM_OF_ABC;
            dsnlm_dc[dsnlm_id + 0] += s[0] * fn12[kk];
            dsnlm_dc[dsnlm_id + 1] += s[1] * fn12[kk];
            dsnlm_dc[dsnlm_id + 2] += s[2] * fn12[kk];

            dsnlm_dc[dsnlm_id + 3] += s[3] * fn12[kk];
            dsnlm_dc[dsnlm_id + 4] += s[4] * fn12[kk];
            dsnlm_dc[dsnlm_id + 5] += s[5] * fn12[kk];
            dsnlm_dc[dsnlm_id + 6] += s[6] * fn12[kk];
            dsnlm_dc[dsnlm_id + 7] += s[7] * fn12[kk];

            dsnlm_dc[dsnlm_id + 8] += s[8] * fn12[kk];
            dsnlm_dc[dsnlm_id + 9] += s[9] * fn12[kk];
            dsnlm_dc[dsnlm_id + 10] += s[10] * fn12[kk];
            dsnlm_dc[dsnlm_id + 11] += s[11] * fn12[kk];
            dsnlm_dc[dsnlm_id + 12] += s[12] * fn12[kk];
            dsnlm_dc[dsnlm_id + 13] += s[13] * fn12[kk];
            dsnlm_dc[dsnlm_id + 14] += s[14] * fn12[kk];

            dsnlm_dc[dsnlm_id + 15] += s[15] * fn12[kk];
            dsnlm_dc[dsnlm_id + 16] += s[16] * fn12[kk];
            dsnlm_dc[dsnlm_id + 17] += s[17] * fn12[kk];
            dsnlm_dc[dsnlm_id + 18] += s[18] * fn12[kk];
            dsnlm_dc[dsnlm_id + 19] += s[19] * fn12[kk];
            dsnlm_dc[dsnlm_id + 20] += s[20] * fn12[kk];
            dsnlm_dc[dsnlm_id + 21] += s[21] * fn12[kk];
            dsnlm_dc[dsnlm_id + 22] += s[22] * fn12[kk];
            dsnlm_dc[dsnlm_id + 23] += s[23] * fn12[kk];    
          }
        }
        
      }

      // copy f12 to dfeat_3rij
      grad_d12_angular[rij_idx]  += f12[3];
      grad_d12_angular[rij_idx+1]+= f12[0];
      grad_d12_angular[rij_idx+2]+= f12[1];
      grad_d12_angular[rij_idx+3]+= f12[2];
    }
  }
}

/**
 * 每个 block 处理一个 atom，每个 thread 处理一个 neighbor，kernel 启动参数为 <<< N, BLOCK_SIZE >>>。
 * 计算其对特征的梯度，并累加到local_dfeat_c3中，最后将local_dfeat_c3中的梯度累加到dfeat_drij中。
 * Shared memory：
 * - 用 shmem 存储 Fp，共 (MAX_NUM_N*6)*8=(20*6)*8=960 Bytes
 * - 用 shmem 存储 sum_fxyz，共 24*20*8=3840 Bytes。
 */
static __global__ void find_angular_gard_small_box_optimized(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums,
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL_radial,
  const double* g_d12_radial,
  const double * coeff3,
  const int64_t* g_type,
  const double * grad_output,
  const double* g_sum_fxyz,
  double* dsnlm_dc,
  double* local_dfeat_c3,  // 临时数组 [N, neigh_num, num_types, n_max, n_base]
  double* dfeat_drij,
  double* grad_d12_angular
  )
{
  int n1 = blockIdx.x; // atom index
  
  if (n1 >= N) return;
  
  // Block 共享存储
  __shared__ double shm_sum_fxyz[NUM_OF_ABC * MAX_NUM_N];  // 24*20*8=3840 Bytes
  __shared__ double shm_Fp[MAX_DIM_ANGULAR];               // (MAX_NUM_N*6)*8=(20*6)*8=960 Bytes
  
  // 加载 sum_fxyz 到 shared memory
  int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
  int total_s_elements = n_max_angular * NUM_OF_ABC;

  for (int k = threadIdx.x; k < total_s_elements; k += blockDim.x) {
    shm_sum_fxyz[k] = g_sum_fxyz[g_sum_start + k];
  }

  __syncthreads();
  
  int neigh_start_idx = n1 * neigh_num;
  
  int r12_start_idx = n1 * neigh_num * 4;
  int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
  int dsnlm_dc_start_idx = n1 * num_types * basis_size_angular * NUM_OF_ABC;
  int de_start = n1 * (feat_3b_nums + feat_2b_nums);
  int dfeat_dr_start = n1 * neigh_num * feat_3b_nums * 4;
  
  int b3_nums = n_max_angular * L_max3;
  int total_Fp_elements = b3_nums + (L_max4 > 0 ? n_max_angular : 0) + (L_max5 > 0 ? n_max_angular : 0);
  
  for (int k = threadIdx.x; k < total_Fp_elements; k += blockDim.x) {
    shm_Fp[k] = 0.0;
  }
  
  // k = nn * L_max3 + ll
  for (int k = threadIdx.x; k < b3_nums; k += blockDim.x) {
    int nn = k / L_max3;
    int ll = k % L_max3;
    shm_Fp[k] = grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];
  }
  if (L_max4 > 0) {
    for (int k = threadIdx.x; k < n_max_angular; k += blockDim.x) {
      shm_Fp[b3_nums + k] = grad_output[de_start + feat_2b_nums + b3_nums + k];
    }
  }
  if (L_max5 > 0) {
    for (int k = threadIdx.x; k < n_max_angular; k += blockDim.x) {
      shm_Fp[b3_nums + n_max_angular + k] = grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + k];
    }
  }
  
  __syncthreads();
  
  int t1 = g_type[n1];
  int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
  for (int i1 = threadIdx.x; i1 < neigh_num; i1 += blockDim.x) {
    int global_idx = n1 * neigh_num + i1;
    int local_dfeat_start = global_idx * num_types * n_max_angular * basis_size_angular;
    int n2 = g_NL_radial[neigh_start_idx + i1];
    if (n2 < 0) continue;

    int t2 = g_type[n2];
    int rij_idx = r12_start_idx + i1*4;
    int dsnlm_dc_idx = dsnlm_dc_start_idx + t2 * basis_size_angular * NUM_OF_ABC;
    double d12 = g_d12_radial[rij_idx];
    if (d12 > rc_angular) continue;
    
    int drij_idx = dfeat_dr_start + i1 * feat_3b_nums * 4;
    double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
    double f12[4] = {0.0};
    
    double fc12, fcp12;
    find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);
    
    double fn12[MAX_NUM_N]; // 20*8=160 Bytes
    double fnp12[MAX_NUM_N]; // 20*8=160 Bytes
    find_fn_and_fnp(basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
    
    int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
    double s[NUM_OF_ABC] = {0.0}; // 24*8=192 Bytes
    accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);
    
    for (int n = 0; n < n_max_angular; ++n) {
      double gn12 = 0.0;
      double gnp12 = 0.0;
      for (int k = 0; k < basis_size_angular; ++k) {
        int c_index = c_I_J_idx + n * basis_size_angular + k;
        gn12 += fn12[k] * coeff3[c_index];
        gnp12 += fnp12[k] * coeff3[c_index];
      }
      double f12d[MAX_LMAX * 4] = {0.0}; // (6*4)*8=192 Bytes
      
      // 计算梯度，输出到 local_dfeat_c3（无竞争）
      if (L_max5 > 0) {
        accumulate_f12_with_5body(
          n, d12, r12, gn12, gnp12, shm_Fp, shm_sum_fxyz,
            s, f12, f12d, local_dfeat_c3 + local_dfeat_start, fn12, fnp12, 
            t2, num_types, L_max3, 
            n_max_angular, basis_size_angular, 0, n1, i1);
        for (int l_idx = 0; l_idx < L_max3; ++l_idx){
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
        }
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
        dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 0] = f12d[(L_max3+1) * 4 + 3];
        dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 1] = f12d[(L_max3+1) * 4 + 0];
        dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 2] = f12d[(L_max3+1) * 4 + 1];
        dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 3] = f12d[(L_max3+1) * 4 + 2];
      } else if (L_max4 > 0) {
        accumulate_f12_with_4body(
          n, d12, r12, gn12, gnp12, shm_Fp, shm_sum_fxyz,
            s, f12, f12d, local_dfeat_c3 + local_dfeat_start, fn12, fnp12, 
            t2, num_types, L_max3, 
            n_max_angular, basis_size_angular, 0, n1, i1);
        for (int l_idx = 0; l_idx < L_max3; ++l_idx){
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
        }
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
        dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
      } else {
        accumulate_f12(
          n, d12, r12, gn12, gnp12, shm_Fp, shm_sum_fxyz,
            s, f12, f12d, local_dfeat_c3 + local_dfeat_start, fn12, fnp12, 
            t2, num_types, L_max3, 
            n_max_angular, basis_size_angular, 0, n1, i1);
        for (int l_idx = 0; l_idx < L_max3; ++l_idx){
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
          dfeat_drij[drij_idx + l_idx *n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
        }
      }
      if (n == 0) {
        for(int kk = 0; kk < basis_size_angular; kk++){
          int dsnlm_id = dsnlm_dc_idx + kk * NUM_OF_ABC;
          atomicAdd(&dsnlm_dc[dsnlm_id + 0], s[0] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 1], s[1] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 2], s[2] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 3], s[3] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 4], s[4] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 5], s[5] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 6], s[6] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 7], s[7] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 8], s[8] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 9], s[9] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 10], s[10] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 11], s[11] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 12], s[12] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 13], s[13] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 14], s[14] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 15], s[15] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 16], s[16] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 17], s[17] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 18], s[18] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 19], s[19] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 20], s[20] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 21], s[21] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 22], s[22] * fn12[kk]);
          atomicAdd(&dsnlm_dc[dsnlm_id + 23], s[23] * fn12[kk]);
        }
      }
    }
    
    atomicAdd(&grad_d12_angular[rij_idx], f12[3]);
    atomicAdd(&grad_d12_angular[rij_idx+1], f12[0]);
    atomicAdd(&grad_d12_angular[rij_idx+2], f12[1]);
    atomicAdd(&grad_d12_angular[rij_idx+3], f12[2]);
  }
}

// dfeat_c3规约
static __global__ void reduce_local_dfeat_c3(
  const int64_t* g_NL,
  const double* local_dfeat_c3,  // [N * neigh_num, num_types, n_max, n_base]
  double* dfeat_c3,               // [N, num_types, n_max, n_base]
  const int N,
  const int neigh_num,
  const int num_types,
  const int n_max_angular,
  const int basis_size_angular
  )
{
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= N * neigh_num) return;
  
  int n1 = global_idx / neigh_num;
  int i1 = global_idx % neigh_num;
  
  int neigh_start_idx = n1 * neigh_num;
  int n2 = g_NL[neigh_start_idx + i1];
  if (n2 < 0) return;
  
  int local_start = global_idx * num_types * n_max_angular * basis_size_angular;
  int output_start = n1 * num_types * n_max_angular * basis_size_angular;
  
  for (int j = 0; j < num_types; ++j) {
    for (int n = 0; n < n_max_angular; ++n) {
      for (int k = 0; k < basis_size_angular; ++k) {
        int local_id = local_start + j * n_max_angular * basis_size_angular + n * basis_size_angular + k;
        int output_id = output_start + j * n_max_angular * basis_size_angular + n * basis_size_angular + k;
        atomicAdd(&dfeat_c3[output_id], local_dfeat_c3[local_id]);
      }
    }
  }
}


static __global__ void find_descriptor(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const double rc_radial,
  const double rcinv_radial,
  const double rc_angular,
  const double rcinv_angular,
  const int n_max_radial,
  const int basis_size_radial,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL_radial,
  const double * coeff2,
  const double * coeff3,
  double * feats,
  const int64_t* g_type,
  const double* g_d12_radial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    // double q[MAX_DIM] = {static_cast<double>(0.0)};
    // get radial descriptors
    double q[MAX_DIM] = {0.0};
    int neigh_start_idx = n1 * neigh_num;
    int r12_start_idx =  n1 * neigh_num * 3;
    int feat_start_idx = n1 * feat_nums; 
    int c2_start_idx = t1 * num_types * n_max_radial * basis_size_radial;
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL_radial[neigh_start_idx + i1]; //the data from cuda find_neighbor 
      if (n2 < 0) break;
      int t2 = g_type[n2];
      int c_I_J_idx = c2_start_idx + t2 * n_max_radial * basis_size_radial;
      int rij_idx = r12_start_idx + i1*3;
      double r12[3] = {g_d12_radial[rij_idx], g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2]};
      double d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double fc12;
      find_fc(rc_radial, rcinv_radial, d12, fc12);
      
      double fn12[MAX_NUM_N];

      find_fn(basis_size_radial, rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n < n_max_radial; ++n) {
        double gn12 = 0.0;
        for (int k = 0; k < basis_size_radial; ++k) {
          int c_index = c_I_J_idx + n * basis_size_radial + k;
          gn12 += fn12[k] * coeff2[c_index];
        }
        // 2b feats
        q[n] += gn12;
      }
    }

    // get angular descriptors
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    for (int n = 0; n < n_max_angular; ++n) {
      double s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neigh_num; ++i1) {
        int n2 = g_NL_radial[neigh_start_idx + i1];
        if (n2 < 0) continue;
        int t2 = g_type[n2];
        int rij_idx = r12_start_idx + i1*3;
        double r12[3] = {g_d12_radial[rij_idx], g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2]};
        double d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        if (d12 > rc_angular) continue;
        double fc12;
        find_fc(rc_angular, rcinv_angular, d12, fc12);
        double fn12[MAX_NUM_N];
        find_fn(basis_size_angular, rcinv_angular, d12, fc12, fn12);
        double gn12 = 0.0;
        int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
        for (int k = 0; k < basis_size_angular; ++k) {
          int c_index = c_I_J_idx + n * basis_size_angular + k;
          gn12 += fn12[k] * coeff3[c_index];
        }
        accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
      }
      if (L_max5 == 1) {
          find_q_with_5body(n_max_angular, n, s, q + n_max_radial);
      } else if (L_max4 ==2) {
        find_q_with_4body(n_max_angular, n, s, q + n_max_radial);
      } else {
        find_q(n_max_angular, n, s, q + n_max_radial);
      }
    }
    for (int n1 = 0; n1 < feat_nums; ++n1) {
      feats[feat_start_idx+n1] = q[n1];
    }
  }
}
