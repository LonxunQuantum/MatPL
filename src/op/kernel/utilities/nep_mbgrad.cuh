#include "nep_utilities.cuh"
#include "nep_utilities_mb_secondc.cuh"
__global__ void buildTypeMapKernel(
    const int64_t* atom_type_map, 
    int* type_index_map, 
    int* unique_types,
    int N,
    int* unique_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        int64_t atom_type = atom_type_map[idx];
        
        int old_val = atomicCAS(&type_index_map[atom_type], -1, INT_MAX);
        if (old_val == -1) {
            int new_index = atomicAdd(unique_count, 1);
            type_index_map[atom_type] = new_index;
            unique_types[new_index] = atom_type;
        }
    }
}

template <typename T>
static __global__ void find_angular_gardc_neigh(
  const int N,
  const T* grad_second,
  const T* g_d12,
  const int64_t* g_NL,
  const T* de_dfeat,
  const T* dsnlm_dc, //[i, J, nbase, 24]
  const T* g_sum_fxyz,
  const int64_t* g_type,
  const int* __restrict__ uniq_map,  // the map of atom type(unique) type-> index
  const int* __restrict__ uniq_type, // index -> type
  const int len_map,    // the len of the atom type map(unique)
  const T * coeff3,
  T * dfeat_c3,
  const T rc_angular,
  const T rcinv_angular,
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
  // 计算共享内存大小
  const int uniq_map_size = 100;
  const int uniq_type_size = 100;
  const int Fp_size = MAX_DIM_ANGULAR;
  const int sum_fxyz_size = max_3b * NUM_OF_ABC;
  
  // 共享内存布局
  extern __shared__ char shared_mem_raw[];
  
  int* s_uniq_map = (int*)shared_mem_raw;
  int* s_uniq_type = s_uniq_map + uniq_map_size;
  T* s_Fp = (T*)(s_uniq_type + uniq_type_size);
  T* s_sum_fxyz = s_Fp + Fp_size;
  
  int tid = threadIdx.x;
  int n1 = blockIdx.x;
  
  // 每个线程块处理一个中心原子
  if (n1 >= atom_nums) return;
  
  // 所有线程协作加载共享数据
  // 加载uniq_map和uniq_type
  for (int i = tid; i < uniq_map_size; i += blockDim.x) {
    s_uniq_map[i] = uniq_map[i];
  }
  for (int i = tid; i < uniq_type_size; i += blockDim.x) {
    s_uniq_type[i] = uniq_type[i];
  }
  
  // 线程0负责加载Fp
  if (tid == 0) {
    int de_start = n1 * (feat_3b_nums + feat_2b_nums);
    int b3_nums = max_3b * L_max3;
    int dd = 0;
    
    // 加载3-body Fp
    for (int nn = 0; nn < max_3b; ++nn) {
      for (int ll = 0; ll < L_max3; ++ll) {
        s_Fp[dd] = de_dfeat[de_start + feat_2b_nums + ll * max_3b + nn];
        dd++;
      }
    }
    
    // 加载4-body Fp
    if (L_max4 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        s_Fp[b3_nums + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + ll];
      }
    }
    
    // 加载5-body Fp
    if (L_max5 > 0) {
      for (int ll = 0; ll < max_3b; ++ll) {
        s_Fp[b3_nums + max_3b + ll] = de_dfeat[de_start + feat_2b_nums + b3_nums + max_3b + ll];
      }
    }
  }
  
  // 所有线程协作加载sum_fxyz
  int g_sum_start = n1 * max_3b * NUM_OF_ABC;
  for (int d = tid; d < sum_fxyz_size; d += blockDim.x) {
    s_sum_fxyz[d] = g_sum_fxyz[g_sum_start + d];
  }
  
  __syncthreads();
  
  // 每个线程处理一个近邻
  if (tid >= neigh_num) return;
  
  int i1 = tid;  // 近邻索引
  int neigh_start_idx = n1 * neigh_num;
  int n2 = g_NL[neigh_start_idx + i1];
  if (n2 < 0) return;
  
  int t1 = g_type[n1];
  int t2 = g_type[n2];
  
  // 计算各个起始索引
  int r12_start_idx = n1 * neigh_num * 4;
  int dc_start_idx = n1 * neigh_num * num_types * max_3b * base_3b + i1 * num_types * max_3b * base_3b;
  int dc_c3_start = n1 * neigh_num * max_3b * len_map * base_3b + i1 * max_3b * len_map * base_3b;
  int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;
  
  // 线程私有变量
  T fn12[MAX_NUM_N];
  T fnp12[MAX_NUM_N];
  
  int rij_idx = r12_start_idx + i1 * 4;
  T d12 = g_d12[rij_idx];
  if (d12 > rc_angular) return;
  
  T r12[3] = {g_d12[rij_idx+1], g_d12[rij_idx+2], g_d12[rij_idx+3]};
  T scd_r12[4] = {grad_second[rij_idx], grad_second[rij_idx+1], 
                       grad_second[rij_idx+2], grad_second[rij_idx+3]};
  T f12[4] = {0.0};
  T fc12, fcp12;
  find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);
  find_fn_and_fnp(base_3b, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
  
  int c_I_J_idx = t1 * num_types * max_3b * base_3b + t2 * max_3b * base_3b;
  
  T blm[NUM_OF_ABC] = {0.0};
  T rij_blm[NUM_OF_ABC] = {0.0};
  T dblm_x[NUM_OF_ABC] = {0.0};
  T dblm_y[NUM_OF_ABC] = {0.0};
  T dblm_z[NUM_OF_ABC] = {0.0};
  T dblm_r[NUM_OF_ABC] = {0.0};
  
  scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2], 
      blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
  
  // 为每个n计算并处理
  for (int n = 0; n < max_3b; ++n) {
    T gn12 = 0.0;
    T gnp12 = 0.0;
    
    for (int k = 0; k < base_3b; ++k) {
      int c_index = c_I_J_idx + n * base_3b + k;
      gn12 += fn12[k] * coeff3[c_index];
      gnp12 += fnp12[k] * coeff3[c_index];
    }
    
    T* f12k = dfeat_c3 + (dc_c3_start + n * len_map * base_3b);
    
    // 使用共享内存中的Fp和sum_fxyz
    if (L_max5 > 0) {
      scd_accumulate_f12_with_5body(
        n, d12, r12, gn12, gnp12, s_Fp, dsnlm_dc, s_sum_fxyz,
        blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
        f12, f12k, scd_r12, fn12, fnp12, 
        t2, num_types, L_max3, 
        max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, s_uniq_map, s_uniq_type, len_map);
    } else if (L_max4 > 0) {
      scd_accumulate_f12_with_4body(
        n, d12, r12, gn12, gnp12, s_Fp, dsnlm_dc, s_sum_fxyz,
        blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
        f12, f12k, scd_r12, fn12, fnp12, 
        t2, num_types, L_max3, 
        max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, s_uniq_map, s_uniq_type, len_map);
    } else {
      scd_accumulate_f12(
        n, d12, r12, gn12, gnp12, s_Fp, dsnlm_dc, s_sum_fxyz,
        blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
        f12, f12k, scd_r12, fn12, fnp12, 
        t2, num_types, L_max3, 
        max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, s_uniq_map, s_uniq_type, len_map);
    }
  }
}

template <typename T>
static __global__ void find_angular_gardc_neigh_bk(
  const int N,
  const T* grad_second,
  const T* g_d12,
  const int64_t* g_NL,
  const T* de_dfeat,
  const T* dsnlm_dc, //[i, J, nbase, 24]
  const T* g_sum_fxyz,
  const int64_t* g_type,
  const int* __restrict__ uniq_map,  // the map of atom type(unique) type-> index
  const int* __restrict__ uniq_type, // index -> type
  const int len_map,    // the len of the atom type map(unique)
  const T * coeff3,
  T * dfeat_c3,
  const T rc_angular,
  const T rcinv_angular,
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
  __shared__ int s_uniq_map[100];
  __shared__ int s_uniq_type[100];
  int tid = threadIdx.x;
  int threads_per_block = blockDim.x;
  // 每个线程处理多个元素，确保覆盖100个位置
  for (int i = tid; i < 100; i += threads_per_block) {
    s_uniq_map[i] = uniq_map[i];
    s_uniq_type[i] = uniq_type[i];
  }
  __syncthreads();

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
  // printf("right %d c3 n1 %d t1 %d neigh_num %d i1 %d n2 %d t2 %d len_map %d max_3b %d base_3b %d\n", \
    N * neigh_num * num_types * max_3b * base_3b > (n1 * neigh_num * num_types * max_3b * base_3b + i1 * num_types * max_3b * base_3b), \
    n1, t1, neigh_num, i1, n2, t2, len_map, max_3b, base_3b);
  int dc_c3_start = n1 * neigh_num * max_3b * len_map * base_3b + i1 * max_3b * len_map * base_3b;
  int de_start = n1 * (feat_3b_nums + feat_2b_nums);// dE/dq
  int dsnlm_start_idx = n1 * num_types * base_3b * NUM_OF_ABC;
  
  T Fp[MAX_DIM_ANGULAR] = {0.0};
  T sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
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
  T d12 = g_d12[rij_idx];
  if (d12 > rc_angular) return;
  T r12[3] = {g_d12[rij_idx+1], g_d12[rij_idx+2], g_d12[rij_idx+3]};
  T scd_r12[4] = {grad_second[rij_idx],grad_second[rij_idx+1],grad_second[rij_idx+2],grad_second[rij_idx+3]};// [r x y z]
  T f12[4] = {0.0};
  T fc12, fcp12;
  find_fc_and_fcp(rc_angular, rcinv_angular, d12, fc12, fcp12);

  T fn12[MAX_NUM_N];
  T fnp12[MAX_NUM_N];
  find_fn_and_fnp(
    base_3b, rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
  
  int c_I_J_idx = c3_start_idx + t2 * max_3b * base_3b;
  // double s[NUM_OF_ABC*6] = {0.0}; 
  T blm[NUM_OF_ABC] = {0.0};
  T rij_blm[NUM_OF_ABC]= {0.0};
  T dblm_x[NUM_OF_ABC] = {0.0};
  T dblm_y[NUM_OF_ABC] = {0.0};
  T dblm_z[NUM_OF_ABC] = {0.0};
  T dblm_r[NUM_OF_ABC] = {0.0};
  scd_accumulate_blm_rij(d12, r12[0], r12[1], r12[2], 
      blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r);
  for (int n = 0; n < max_3b; ++n) {
    T gn12 = 0.0;
    T gnp12 = 0.0;
    for (int k = 0; k < base_3b; ++k) {
      int c_index = c_I_J_idx + n * base_3b + k;
      gn12 += fn12[k] * coeff3[c_index];
      gnp12 += fnp12[k] * coeff3[c_index];
    }

    T* f12k = dfeat_c3 + (dc_c3_start + n * len_map * base_3b);
    if (L_max5 > 0) {
      scd_accumulate_f12_with_5body(
        n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
          blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
          f12, f12k, scd_r12, fn12, fnp12, 
          t2, num_types, L_max3, 
          max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, s_uniq_map, s_uniq_type, len_map);
    } else if (L_max4 > 0) {
      scd_accumulate_f12_with_4body(
        n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
          blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
          f12, f12k, scd_r12, fn12, fnp12, 
          t2, num_types, L_max3, 
          max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, s_uniq_map, s_uniq_type, len_map);
    } else {
      scd_accumulate_f12(
        n, d12, r12, gn12, gnp12, Fp, dsnlm_dc, sum_fxyz,
          blm, rij_blm, dblm_x, dblm_y, dblm_z, dblm_r,
          f12, f12k, scd_r12, fn12, fnp12, 
          t2, num_types, L_max3, 
          max_3b, base_3b, dc_start_idx, dsnlm_start_idx, n1, i1, s_uniq_map, s_uniq_type, len_map);
    }

    // for (int j = 0; j < num_types; ++j){
    //   if (uniq_map[j] == -1) {
    //     continue;
    //     
    //     
    //     
    //     
    //     
    //     
    //   } else {
    //     for (int k = 0; k < base_3b; ++k){
    //       
    //       int dc_id = dc_c3_start + uniq_map[j] * max_3b * base_3b + n*base_3b + k;
    //       int k_id = j * base_3b * 4 + k * 4;
    //       dfeat_c3[dc_id] += f12k[k_id]; 
    //     
    //     
    //     
    //     
    //     }
    //   }
    // }
    //add f12k [k, 4] -> c[atomI, J_type, nmax, k, 4] -> c[atomI, J_type, nmax, k]
    // 是否把4 在scd时候直接给累加起来？ 还是单独加？
  }
}


template <typename T>
static __global__ void aggregate_dfeat_c3(
  const int64_t* g_NL,
  const int64_t* g_type,
  const T* dfeat_c3,
  T* tmp_dfeat_c3,
  const int* unique_types,
  const int len_map,  // the len of the atom type map(unique)
  const int N,
  const int neigh_num,
  const int num_types,
  const int max_3b,
  const int base_3b
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int tmp_start_idx = n1 * num_types * max_3b * base_3b;
    int dc_start_idx = n1 * neigh_num * max_3b * len_map * base_3b;
    int neigh_start_idx = n1 * neigh_num;
    // int t1 = g_type[n1];
    for (int i1 = 0; i1 < neigh_num; ++i1) {
      int n2 = g_NL[neigh_start_idx + i1];
      if (n2 < 0) break;
      // int t2 = g_type[n2];
      int dc_idx = dc_start_idx + i1 * max_3b * len_map * base_3b;
      
      for (int j = 0; j < len_map; ++j){
        for (int n = 0; n < max_3b; ++n){
          for (int k = 0; k < base_3b; ++k){
            // if(n1<1) printf("neigh_num %d n1 %d n2 %d t2 %d len_map %d j %d unique_types[j%d] %d max3b %d n%d base3b %d k %d\n",\
                    neigh_num,   n1,   n2,   t2,   len_map,   j,   j,   unique_types[j], max_3b, n,  base_3b,  k);
            //int dc_id = dc_idx + j * max_3b * base_3b + n * base_3b + k;
            int dc_id = dc_idx + n * len_map * base_3b + j * base_3b + k;
            //这里j需要映射回它对应的元素类型
            int tmp_dc_id = tmp_start_idx + unique_types[j] * max_3b * base_3b + n*base_3b + k;
            tmp_dfeat_c3[tmp_dc_id] += dfeat_c3[dc_id];
          }
        }
      }
    }
  }
}