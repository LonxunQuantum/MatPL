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
template <typename T>
static __global__ void aggregate_features(
    T* dfeat_c2,     // 输入张量，维度 [N, n_types, n_max_2b, n_base_2b]
    const int64_t* atom_map,       // 原子类型映射，维度 [N]
    T* output,             // 输出张量，维度 [n_types, n_types, n_max_2b, n_base_2b]
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

template <typename T>
static __global__ void find_mb_descriptor(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const T rc_angular,
  const T rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL,
  const T * coeff3,
  T * feats,
  const int64_t* g_type,
  const T* g_d12_radial,
  T* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    // double q[MAX_DIM] = {static_cast<double>(0.0)};
    // get radial descriptors
    T q[MAX_DIM] = {0.0};
    int neigh_start_idx = n1 * neigh_num;
    int r12_start_idx =  n1 * neigh_num * 4;
    int feat_start_idx = n1 * feat_nums; 
    // get angular descriptors
    int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
    int sum_s_start_idx = n1 * n_max_angular * NUM_OF_ABC;
    for (int n = 0; n < n_max_angular; ++n) {
      T s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neigh_num; ++i1) {
        int n2 = g_NL[neigh_start_idx + i1];
        if (n2 < 0) break;
        int t2 = g_type[n2];
        int rij_idx = r12_start_idx + i1*4;
        T d12 = g_d12_radial[rij_idx];
        if (d12 > rc_angular) break;
        T r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
        T fc12;
        find_fc(rc_angular, rcinv_angular, d12, fc12);
        T fn12[MAX_NUM_N];
        find_fn(basis_size_angular, rcinv_angular, d12, fc12, fn12);
        T gn12 = 0.0;
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


template <typename T>
static __global__ void find_descriptor(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_nums,
  const T rc_radial,
  const T rcinv_radial,
  const T rc_angular,
  const T rcinv_angular,
  const int n_max_radial,
  const int basis_size_radial,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL_radial,
  const T * coeff2,
  const T * coeff3,
  T * feats,
  const int64_t* g_type,
  const T* g_d12_radial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int t1 = g_type[n1];
    // double q[MAX_DIM] = {static_cast<double>(0.0)};
    // get radial descriptors
    T q[MAX_DIM] = {0.0};
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
      T r12[3] = {g_d12_radial[rij_idx], g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2]};
      T d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      T fc12;
      find_fc(rc_radial, rcinv_radial, d12, fc12);
      
      T fn12[MAX_NUM_N];

      find_fn(basis_size_radial, rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n < n_max_radial; ++n) {
        T gn12 = 0.0;
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
      T s[NUM_OF_ABC] = {0.0};
      for (int i1 = 0; i1 < neigh_num; ++i1) {
        int n2 = g_NL_radial[neigh_start_idx + i1];
        if (n2 < 0) continue;
        int t2 = g_type[n2];
        int rij_idx = r12_start_idx + i1*3;
        T r12[3] = {g_d12_radial[rij_idx], g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2]};
        T d12    = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
        if (d12 > rc_angular) continue;
        T fc12;
        find_fc(rc_angular, rcinv_angular, d12, fc12);
        T fn12[MAX_NUM_N];
        find_fn(basis_size_angular, rcinv_angular, d12, fc12, fn12);
        T gn12 = 0.0;
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


// 线程维度为中心原子，Fp、Sumxyz在共享内存区
// static __global__ void find_angular_gard_bk(
//   const int N,
//   const int num_types,
//   const int num_types_sq,
//   const int neigh_num,
//   const int L_max3,
//   const int L_max4,
//   const int L_max5,
//   const int feat_2b_nums,
//   const int feat_3b_nums,
//   const double rc_angular,
//   const double rcinv_angular,
//   const int n_max_angular,
//   const int basis_size_angular,
//   const int64_t* g_NL_radial,
//   const double* g_d12_radial,
//   const double * coeff3,
//   const int64_t* g_type,
//   const double * grad_output,
//   const double* g_sum_fxyz,
//   double* dsnlm_dc,
//   double* dfeat_c3,
//   double* dfeat_drij,
//   double* grad_d12_angular
//   )
// {
//   // 使用共享内存存储Fp和sum_fxyz
//   __shared__ double s_Fp[MAX_DIM_ANGULAR];
//   __shared__ double s_sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
  
//   int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  
//   if (n1 >= N) return;
  
//   int tid = threadIdx.x;
//   int de_start = n1 * (feat_3b_nums + feat_2b_nums);
  
//   // 线程加载Fp和sum_fxyz到共享内存（每个线程加载自己的部分）
//   int dd = 0;
//   for (int nn = 0; nn < n_max_angular; ++nn) {
//     for (int ll = 0; ll < L_max3; ++ll) {
//       s_Fp[tid * MAX_DIM_ANGULAR / blockDim.x + dd] = 
//           grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];
//       dd++;
//     }
//   }
  
//   int b3_nums = n_max_angular * L_max3;
//   if (L_max4 > 0) {
//     for (int ll = 0; ll < n_max_angular; ++ll) {
//       s_Fp[tid * MAX_DIM_ANGULAR / blockDim.x + b3_nums + ll] = 
//           grad_output[de_start + feat_2b_nums + b3_nums + ll];
//     }
//   }
  
//   if (L_max5 > 0) {
//     for (int ll = 0; ll < n_max_angular; ++ll) {
//       s_Fp[tid * MAX_DIM_ANGULAR / blockDim.x + b3_nums + n_max_angular + ll] = 
//           grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + ll];
//     }
//   }
  
//   // 加载sum_fxyz到共享内存
//   int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
//   for (int d = 0; d < n_max_angular * NUM_OF_ABC; ++d) {
//     s_sum_fxyz[tid * (n_max_angular * NUM_OF_ABC) / blockDim.x + d] = 
//         g_sum_fxyz[g_sum_start + d];
//   }
  
//   __syncthreads();
  
//   // 现在每个线程可以访问整个共享内存数组
//   // 线程的Fp在s_Fp + tid * Fp_per_thread
//   // sum_fxyz在s_sum_fxyz + tid * sum_fxyz_per_thread
  
//   // 计算各种起始索引
//   int r12_start_idx = n1 * neigh_num * 4;
//   int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
//   int dsnlm_dc_start_idx = n1 * num_types * basis_size_angular * NUM_OF_ABC;
//   // int de_start_local = n1 * (feat_3b_nums + feat_2b_nums);
//   int neigh_start_idx = n1 * neigh_num;
//   int dfeat_dr_start = n1 * neigh_num * feat_3b_nums * 4;
  
//   double Fp_local[MAX_DIM_ANGULAR] = {0.0};
//   double sum_fxyz_local[NUM_OF_ABC * MAX_NUM_N];
  
//   // 从共享内存复制到寄存器（避免重复访问共享内存）
//   int Fp_offset = tid * MAX_DIM_ANGULAR / blockDim.x;
//   int sum_fxyz_offset = tid * (n_max_angular * NUM_OF_ABC) / blockDim.x;
  
//   for (int i = 0; i < MAX_DIM_ANGULAR / blockDim.x; i++) {
//     Fp_local[i] = s_Fp[Fp_offset + i];
//   }
  
//   for (int i = 0; i < (n_max_angular * NUM_OF_ABC) / blockDim.x; i++) {
//     sum_fxyz_local[i] = s_sum_fxyz[sum_fxyz_offset + i];
//   }
  
//   int t1 = g_type[n1];
  
//   // 循环处理所有近邻
//   for (int i1 = 0; i1 < neigh_num; ++i1) {
//     int n2 = g_NL_radial[neigh_start_idx + i1];
//     if (n2 < 0) break;
    
//     int t2 = g_type[n2];
//     int rij_idx = r12_start_idx + i1 * 4;
//     int dsnlm_dc_idx = dsnlm_dc_start_idx + t2 * basis_size_angular * NUM_OF_ABC;
//     double d12 = g_d12_radial[rij_idx];
    
//     if (d12 > rc_angular) break;
    
//     int drij_idx = dfeat_dr_start + i1 * feat_3b_nums * 4;
//     double r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
//     double f12[4] = {0.0};

//     double fc12, fcp12;
//     find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

//     double fn12[MAX_NUM_N];
//     double fnp12[MAX_NUM_N];
//     find_fn_and_fnp(
//       basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
    
//     int c_I_J_idx = t1 * num_types * n_max_angular * basis_size_angular + 
//                    t2 * n_max_angular * basis_size_angular;
//     double s[NUM_OF_ABC] = {0.0};
//     accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);
    
//     for (int n = 0; n < n_max_angular; ++n) {
//       double gn12 = 0.0;
//       double gnp12 = 0.0;
      
//       for (int k = 0; k < basis_size_angular; ++k) {
//         int c_index = c_I_J_idx + n * basis_size_angular + k;
//         gn12 += fn12[k] * coeff3[c_index];
//         gnp12 += fnp12[k] * coeff3[c_index];
//       }
      
//       double f12d[MAX_LMAX * 4] = {0.0};
      
//       if (L_max5 > 0) {
//         accumulate_f12_with_5body(
//           n, d12, r12, gn12, gnp12, Fp_local, sum_fxyz_local,
//           s, f12, f12d, dfeat_c3, fn12, fnp12, 
//           t2, num_types, L_max3, 
//           n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
        
//         // 复制3b
//         for (int l_idx = 0; l_idx < L_max3; ++l_idx){
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
//         }
        
//         // 复制4b
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
        
//         // 复制5b
//         dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 0] = f12d[(L_max3+1) * 4 + 3];
//         dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 1] = f12d[(L_max3+1) * 4 + 0];
//         dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 2] = f12d[(L_max3+1) * 4 + 1];
//         dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 3] = f12d[(L_max3+1) * 4 + 2];
//       } 
//       else if (L_max4 > 0) {
//         accumulate_f12_with_4body(
//           n, d12, r12, gn12, gnp12, Fp_local, sum_fxyz_local,
//           s, f12, f12d, dfeat_c3, fn12, fnp12, 
//           t2, num_types, L_max3, 
//           n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
        
//         // 复制3b
//         for (int l_idx = 0; l_idx < L_max3; ++l_idx){
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
//         }
        
//         // 复制4b
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
//         dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
//       } 
//       else {
//         accumulate_f12(
//           n, d12, r12, gn12, gnp12, Fp_local, sum_fxyz_local,
//           s, f12, f12d, dfeat_c3, fn12, fnp12, 
//           t2, num_types, L_max3, 
//           n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
        
//         // 复制3b
//         for (int l_idx = 0; l_idx < L_max3; ++l_idx){
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
//           dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
//         }
//       }
      
//       if (n == 0) {
//         for(int kk = 0; kk < basis_size_angular; kk++){
//           int dsnlm_id = dsnlm_dc_idx + kk * NUM_OF_ABC;
//           dsnlm_dc[dsnlm_id + 0] += s[0] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 1] += s[1] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 2] += s[2] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 3] += s[3] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 4] += s[4] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 5] += s[5] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 6] += s[6] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 7] += s[7] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 8] += s[8] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 9] += s[9] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 10] += s[10] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 11] += s[11] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 12] += s[12] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 13] += s[13] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 14] += s[14] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 15] += s[15] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 16] += s[16] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 17] += s[17] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 18] += s[18] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 19] += s[19] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 20] += s[20] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 21] += s[21] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 22] += s[22] * fn12[kk];
//           dsnlm_dc[dsnlm_id + 23] += s[23] * fn12[kk];
//         }
//       }
//     }
    
//     // 累加f12到grad_d12_angular
//     grad_d12_angular[rij_idx]  += f12[3];
//     grad_d12_angular[rij_idx+1] += f12[0];
//     grad_d12_angular[rij_idx+2] += f12[1];
//     grad_d12_angular[rij_idx+3] += f12[2];
//   }
// }

// 线程块对应中心原子，每个线程处理相应的近邻，多了atomicadd操作，但是整体速度有提升，并行度更高了
template <typename T>
static __global__ void find_angular_gard(
  const int N,
  const int num_types,
  const int num_types_sq,
  const int neigh_num,
  const int L_max3,
  const int L_max4,
  const int L_max5,
  const int feat_2b_nums,
  const int feat_3b_nums,
  const T rc_angular,
  const T rcinv_angular,
  const int n_max_angular,
  const int basis_size_angular,
  const int64_t* g_NL_radial,
  const T* g_d12_radial,
  const T * coeff3,
  const int64_t* g_type,
  const T * grad_output,
  const T* g_sum_fxyz,
  T* dsnlm_dc,
  T* dfeat_c3,
  T* dfeat_drij,
  T* grad_d12_angular
  )
{
  // 共享内存：每个线程块（一个中心原子）共享Fp和sum_fxyz
  const int Fp_size = MAX_DIM_ANGULAR;
  const int sum_fxyz_size = n_max_angular * NUM_OF_ABC;
  // 共享内存布局
  extern __shared__ char shared_mem_raw[];
  T* s_Fp = (T*)shared_mem_raw;
  T* s_sum_fxyz = s_Fp + Fp_size;

  int tid = threadIdx.x; // 线程ID
  int n1 = blockIdx.x;  // 每个block处理一个中心原子
  if (n1 >= N) return;

  int b3_nums = n_max_angular * L_max3;
  int g_sum_start = n1 * n_max_angular * NUM_OF_ABC;
  // 线程0加载共享数据（Fp和sum_fxyz对于同一中心原子的所有近邻是相同的）
  if (tid == 0) {
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);
    int dd = 0;
    
    // 加载Fp到共享内存
    for (int nn = 0; nn < n_max_angular; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        s_Fp[dd] = grad_output[de_start + feat_2b_nums + ll * n_max_angular + nn];
        dd++;
      }
    }
    
    if (L_max4 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        s_Fp[b3_nums + ll] = grad_output[de_start + feat_2b_nums + b3_nums + ll];
      }
    }
    
    if (L_max5 > 0) {
      for (int ll = 0; ll < n_max_angular; ++ll) {
        s_Fp[b3_nums + n_max_angular + ll] = grad_output[de_start + feat_2b_nums + b3_nums + n_max_angular + ll];
      }
    }
  }
    
  // 所有线程协作加载sum_fxyz
  for (int d = tid; d < sum_fxyz_size; d += blockDim.x) {
    s_sum_fxyz[d] = g_sum_fxyz[g_sum_start + d];
  }
  
  __syncthreads(); // 确保共享内存加载完成
  
  // 每个线程处理一个近邻
  if (tid >= neigh_num) return;
  
  int i1 = tid;  // 近邻索引
  int neigh_start_idx = n1 * neigh_num;
  int n2 = g_NL_radial[neigh_start_idx + i1];
  if (n2 < 0) return;
  // if (n1 == 0 or n1 == 10) {
  //   for (int ll = 0; ll < b3_nums + 2*n_max_angular; ++ll) {
  //       printf("Fp n1 %d j %d [%d]= %f\n", n1, i1, ll, s_Fp[ll]);
  //     }
  //   for (int d = 0; d < sum_fxyz_size; ++d) {
  //     printf("sum_fxyz n1 %d j %d [%d]= %f\n",n1, i1, d, s_sum_fxyz[d]);
  //   }
  // }

  int t1 = g_type[n1];
  int t2 = g_type[n2];
  
  // 计算各个起始索引
  int r12_start_idx = n1 * neigh_num * 4;
  int rij_idx = r12_start_idx + i1 * 4;
  T d12 = g_d12_radial[rij_idx];
  int dc_start_idx = n1 * num_types * n_max_angular * basis_size_angular;
  if (d12 > rc_angular) return;
  
  T r12[3] = {g_d12_radial[rij_idx+1], g_d12_radial[rij_idx+2], g_d12_radial[rij_idx+3]};
  T f12[4] = {0.0};
  
  // 计算径向函数
  T fc12, fcp12;
  find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);
  
  T fn12[MAX_NUM_N];
  T fnp12[MAX_NUM_N];
  find_fn_and_fnp(basis_size_angular, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
  
  // 计算球谐函数相关项
  T s[NUM_OF_ABC] = {0.0};
  accumulate_blm_rij(d12, r12[0], r12[1], r12[2], s);
  
  int c3_start_idx = t1 * num_types * n_max_angular * basis_size_angular;
  int c_I_J_idx = c3_start_idx + t2 * n_max_angular * basis_size_angular;
  int drij_idx = n1 * neigh_num * feat_3b_nums * 4 + i1 * feat_3b_nums * 4;
  
  // 遍历所有n
  for (int n = 0; n < n_max_angular; ++n) {
    T gn12 = 0.0;
    T gnp12 = 0.0;
    
    // 计算gn12和gnp12
    for (int k = 0; k < basis_size_angular; ++k) {
      int c_index = c_I_J_idx + n * basis_size_angular + k;
      gn12 += fn12[k] * coeff3[c_index];
      gnp12 += fnp12[k] * coeff3[c_index];
    }
    
    T f12d[MAX_LMAX * 4] = {0.0};
    T dfeat_c3_base[MAX_NUM_N] = {0.0}; //临时存储cfeat_c3 对应nbase下的梯度
    // 根据L_max选择不同的计算函数
    if (L_max5 > 0) {
      accumulate_f12_with_5body(
        n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
        s, f12, f12d, dfeat_c3_base, fn12, fnp12, 
        t2, num_types, L_max3, 
        n_max_angular, basis_size_angular, dc_start_idx, n1, i1);

      // 将f12d复制到dfeat_drij
      // 3-body部分
      for (int l_idx = 0; l_idx < L_max3; ++l_idx) {
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
      }
      
      // 4-body部分
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
      
      // 5-body部分
      dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 0] = f12d[(L_max3+1) * 4 + 3];
      dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 1] = f12d[(L_max3+1) * 4 + 0];
      dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 2] = f12d[(L_max3+1) * 4 + 1];
      dfeat_drij[drij_idx + (L_max3+1) * n_max_angular * 4 + n * 4 + 3] = f12d[(L_max3+1) * 4 + 2];
    } 
    else if (L_max4 > 0) {
      accumulate_f12_with_4body(
        n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
        s, f12, f12d, dfeat_c3_base, fn12, fnp12, 
        t2, num_types, L_max3, 
        n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
      
      // 3-body部分
      for (int l_idx = 0; l_idx < L_max3; ++l_idx) {
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
      }
      
      // 4-body部分
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 0] = f12d[L_max3 * 4 + 3];
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 1] = f12d[L_max3 * 4 + 0];
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 2] = f12d[L_max3 * 4 + 1];
      dfeat_drij[drij_idx + L_max3 * n_max_angular * 4 + n * 4 + 3] = f12d[L_max3 * 4 + 2];
    } 
    else {
      accumulate_f12(
        n, d12, r12, gn12, gnp12, s_Fp, s_sum_fxyz,
        s, f12, f12d, dfeat_c3_base, fn12, fnp12, 
        t2, num_types, L_max3, 
        n_max_angular, basis_size_angular, dc_start_idx, n1, i1);
      
      // 3-body部分
      for (int l_idx = 0; l_idx < L_max3; ++l_idx) {
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 0] = f12d[l_idx * 4 + 3];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 1] = f12d[l_idx * 4 + 0];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 2] = f12d[l_idx * 4 + 1];
        dfeat_drij[drij_idx + l_idx * n_max_angular * 4 + n * 4 + 3] = f12d[l_idx * 4 + 2];
      }
    }
    // copy dfeat_c3_base to dfeat_c3 with atomicadd
    for (int baseid = 0; baseid < basis_size_angular; baseid++){
      int dc_id = dc_start_idx + t2 * n_max_angular * basis_size_angular + n*basis_size_angular + baseid;
      atomicAdd(&dfeat_c3[dc_id], dfeat_c3_base[baseid]);
    }
    // 对dsnlm_dc进行累加（只需要在n==0时）
    if (n == 0) {
      int dsnlm_dc_idx = n1 * num_types * basis_size_angular * NUM_OF_ABC + 
                        t2 * basis_size_angular * NUM_OF_ABC;
      
      // 使用原子操作确保线程安全（多个相同类型的近邻会更新同一位置）
      for (int kk = 0; kk < basis_size_angular; kk++) {
        int dsnlm_id = dsnlm_dc_idx + kk * NUM_OF_ABC;
        T fn_val = fn12[kk];
        
        // 使用原子加法
        atomicAdd(&dsnlm_dc[dsnlm_id + 0], s[0] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 1], s[1] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 2], s[2] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 3], s[3] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 4], s[4] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 5], s[5] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 6], s[6] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 7], s[7] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 8], s[8] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 9], s[9] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 10], s[10] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 11], s[11] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 12], s[12] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 13], s[13] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 14], s[14] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 15], s[15] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 16], s[16] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 17], s[17] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 18], s[18] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 19], s[19] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 20], s[20] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 21], s[21] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 22], s[22] * fn_val);
        atomicAdd(&dsnlm_dc[dsnlm_id + 23], s[23] * fn_val);
      }
    }
  }
  
  // 将f12写入grad_d12_angular（不需要原子操作，每个近邻位置唯一）
  grad_d12_angular[rij_idx]   = f12[3];
  grad_d12_angular[rij_idx+1] = f12[0];
  grad_d12_angular[rij_idx+2] = f12[1];
  grad_d12_angular[rij_idx+3] = f12[2];
}