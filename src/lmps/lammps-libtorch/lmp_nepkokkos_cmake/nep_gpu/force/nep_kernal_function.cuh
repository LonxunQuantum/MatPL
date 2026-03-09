/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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

In the compute() function, the following major additions and modifications have been made compared to GPUMD's compute_large_box(https://github.com/brucefan1983/GPUMD/blob/master/src/force/nep.cu):
 1. Modified the data structures to adapt to the neighbor list index format from LAMMPS KOKKOS.

 2. Added handling for ghost atoms, which are unique to LAMMPS.

 3. Split the descriptor calculation into two independent parts: two-body and many-body, computed separately.

 4. Reworked the kernels for two-body and many-body force calculations so that threads within a block collaborate on computing forces for a central atom.

 5. Optimized register spilling issues by utilizing shared memory.

  wuxingxing@pwmat.com and MatPL development team. 2026. Beijing Lonxun Quantum Co.,Ltd.

*/
#include "nepkk.cuh"
#include "../utilities/common.cuh"
#include "../utilities/nep_utilities.cuh"

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

static __global__ void calc_2b_descriptor_sharemem(
  NEPKK::ParaMB paramb,
  const float* param_c2,
  const int N,//inum
  const int nlocal,
  int device,
  const int num_neigh, // the shape[0] of Neighbor List
  const int* g_numneigh, // 大的近邻表
  const int* g_firstneigh,
  int* g_NN_angular,
  int* g_NL_angular,// 顺带构建出小的多体近邻表
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  float* g_Fp)
{
  extern __shared__ float s_c[];
  const int total_elements = paramb.num_types * paramb.num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
  for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
    s_c[i] = param_c2[i];
  }
  __syncthreads();
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int count_angular = 0;    
    int atomi = g_ilist[n1];
    int t1 = g_type[atomi];
    // float q[MAX_DIM] = {0.0f};
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    int c_start = paramb.num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // get radial descriptors
    for (int i1 = 0; i1 < g_numneigh[atomi]; ++i1) {
      int n2 = g_firstneigh[i1 * num_neigh + atomi] & NEIGHMASK; //这里num_neigh 是总的原子数，与三体近邻维度是一致的。
      int t2 = g_type[n2];
      // if (atomi == 1) printf("calc n1 %d nn %d n2 %d\n", atomi, g_numneigh[atomi], n2);

      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      float x12 = g_pos[n2*3  ] - x1;
      float y12 = g_pos[n2*3+1] - y1;
      float z12 = g_pos[n2*3+2] - z1;
      float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
      // if(n1%10==0) printf("ALL NEIGH n1 %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f n1xyz %f %f %f n2xyz %f %f %f\n",\
        n1, atomi, t1, g_numneigh[atomi], n2, t2, sqrt(d12_square), x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);
      
      if (d12_square > paramb.rc_radial_square) continue;
      if (d12_square < paramb.rc_angular_square) {
        // 构造三体近邻 注意三体近邻维度为 Nlocal*MN_angular， 两体使用大近邻，维度刚好相反
        // if(n1%10==0) printf("3B NEIGH n1 %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f n1xyz %f %f %f n2xyz %f %f %f\n",\
          n1, atomi, t1, g_numneigh[atomi], n2, t2, sqrt(d12_square), x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);
        g_NL_angular[count_angular * nlocal + atomi] = n2; //count_angular * nlocal + atomi
        count_angular++;
      }
      // if(n1%10==0) printf("2B NEIGH n1 %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f n1xyz %f %f %f n2xyz %f %f %f\n",\
          n1, atomi, t1, g_numneigh[atomi], n2, t2, sqrt(d12_square), x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);

      float d12 = sqrt(d12_square);
      // 2b->qn
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      float fn12[MAX_NUM_N];//n_base

      find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          gn12 += fn12[k] * s_c[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
        }
        g_Fp[n * nlocal + atomi] += gn12;
      } // nmax
    }// neigh
    // printf("i-%d-%d-feat-2b: %.15f %.15f %.15f %.15f %.15f \n", n1, atomi, \
    g_Fp[q_idx_start], g_Fp[q_idx_start+1], g_Fp[q_idx_start+2], g_Fp[q_idx_start+3], g_Fp[q_idx_start+4]);
    g_NN_angular[atomi] = count_angular;
    // if(atomi < 10) printf("n1 %d n2b %d n3b %d\n", atomi, g_numneigh[atomi], count_angular);
  }
}

// 2b -> q[n]; 3b NN/L 
static __global__ void calc_2b_descriptor(
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int N,//inum
  const int nlocal,
  int device,
  const int num_neigh, // the shape[0] of Neighbor List
  const int* g_numneigh, // 大的近邻表
  const int* g_firstneigh,
  int* g_NN_angular,
  int* g_NL_angular,// 顺带构建出小的多体近邻表
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  float* g_Fp)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int count_angular = 0;
    int atomi = g_ilist[n1];
    int t1 = g_type[atomi];
    // float q[MAX_DIM] = {0.0f};
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    int c_start = paramb.num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // get radial descriptors
    for (int i1 = 0; i1 < g_numneigh[atomi]; ++i1) {
      int n2 = g_firstneigh[i1 * num_neigh + atomi] & NEIGHMASK; 
      int t2 = g_type[n2];
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      float x12 = g_pos[n2*3  ] - x1;
      float y12 = g_pos[n2*3+1] - y1;
      float z12 = g_pos[n2*3+2] - z1;
      float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
      // if(n1%10==0) printf("ALL NEIGH n1 %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f n1xyz %f %f %f n2xyz %f %f %f\n",\
        n1, atomi, t1, g_numneigh[atomi], n2, t2, sqrt(d12_square), x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);
      
      if (d12_square > paramb.rc_radial_square) continue;
      if (d12_square < paramb.rc_angular_square) {
        // 构造三体近邻 注意三体近邻维度为 Nlocal*MN_angular， 两体使用大近邻，维度刚好相反
        // if(n1%10==0) printf("3B NEIGH n1 %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f n1xyz %f %f %f n2xyz %f %f %f\n",\
          n1, atomi, t1, g_numneigh[atomi], n2, t2, sqrt(d12_square), x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);
        g_NL_angular[count_angular * nlocal + atomi] = n2; //count_angular * nlocal + atomi
        count_angular++;
      }
      // if(n1==1) printf("2B NEIGH n1 %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f n1xyz %f %f %f n2xyz %f %f %f\n",\
          n1, atomi, t1, g_numneigh[atomi], n2, t2, sqrt(d12_square), x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);

      float d12 = sqrt(d12_square);
      // 2b->qn
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      float fn12[MAX_NUM_N];//n_base

      find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        float gn12 = 0.0f;
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          gn12 += fn12[k] * annmb.c[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
          // if (n1==1 and n2 % 5 == 0) printf("2B C n1 %d t1 %d n2 %d t2 %d n %d k %d f12k=%f C=%f\n", n1, t1, n2, t2, n, k, fn12[k], annmb.c[c_idx_I + n * paramb.basis_size_radial_plus1 + k]);
        }
        g_Fp[n * nlocal + atomi] += gn12;
      } // nmax
    }// neigh
    // printf("i-%d-%d-feat-2b: %.15f %.15f %.15f %.15f %.15f \n", n1, atomi, \
    g_Fp[q_idx_start], g_Fp[q_idx_start+1], g_Fp[q_idx_start+2], g_Fp[q_idx_start+3], g_Fp[q_idx_start+4]);
    g_NN_angular[atomi] = count_angular;
  }
}


static __global__ void calc_3b_descriptor_sharemem(
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const float* param_c3,
  const int N,
  const int nlocal,
  int device,
  const int num_feats, // 2b + 3b + 4b +5b features
  const int* g_NN_angular,
  const int* g_NL_angular,// 顺带构建出小的多体近邻表
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  float* g_Fp,
  double* g_pe,
  float* g_sum_fxyz)
{
  extern __shared__ float s_c[];
  const int total_elements = paramb.num_types * paramb.num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
  for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
    s_c[i] = param_c3[i];
  }
  __syncthreads();
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = g_type[atomi];
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < paramb.n_max_radial_plus1; ++d) {
      q[d] = g_Fp[d * nlocal + atomi];
    }

    int c_start = paramb.num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
        int n2 = g_NL_angular[atomi + nlocal * i1];
        int t2 = g_type[n2];
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        float x12 = g_pos[n2*3  ] - x1;
        float y12 = g_pos[n2*3+1] - y1;
        float z12 = g_pos[n2*3+2] - z1;
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);

        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
          gn12 += fn12[k] * s_c[c_idx_I + n * paramb.basis_size_angular_plus1 + k];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular_plus1, n, s, q + paramb.n_max_radial_plus1);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + atomi] = s[abc];
      }
    } // neigh

    for (int d = 0; d < annmb.dim; ++d) {
      // if (device == 0) printf("scale atomi %d->%d q[%d]=%f scaled[%d]=%f\n", n1, atomi, d, q[d], d, paramb.q_scaler[d]);
      q[d] = q[d] * Q_SCALER[d];
    }

    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (paramb.version == 4) {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp, t1);

    } else {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp, t1);    
    }

    // if (device == 0) printf("energyEi1 %d->%d = %f\n", n1, atomi, F);
    g_pe[atomi] += F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * nlocal + atomi] = Fp[d] * Q_SCALER[d]; //paramb.q_scaler
    }
  } // if
} // function

static __global__ void calc_3b_descriptor(
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int N,
  const int nlocal,
  int device,
  const int num_feats, // 2b + 3b + 4b +5b features
  const int* g_NN_angular,
  const int* g_NL_angular,// 顺带构建出小的多体近邻表
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  float* g_Fp,
  double* g_pe,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = g_type[atomi];
    // float q[MAX_DIM] = {0.0f};
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    float q[MAX_DIM] = {0.0f};
    for (int d = 0; d < paramb.n_max_radial_plus1; ++d) {
      q[d] = g_Fp[d * nlocal + atomi];
    }
    int c_start = paramb.num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
        int n2 = g_NL_angular[atomi + nlocal * i1];
        int t2 = g_type[n2];
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        float x12 = g_pos[n2*3  ] - x1;
        float y12 = g_pos[n2*3+1] - y1;
        float z12 = g_pos[n2*3+2] - z1;
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);

        float fn12[MAX_NUM_N];
        find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
        float gn12 = 0.0f;
        for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
          gn12 += fn12[k] * annmb.c[c_idx_I + n * paramb.basis_size_angular_plus1 + k + paramb.num_c_radial];
        }
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(paramb.L_max, paramb.num_L, paramb.n_max_angular_plus1, n, s, q + paramb.n_max_radial_plus1);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + atomi] = s[abc];
        // if (device == 0) printf("g_sum_fxyz atomi%d->%d g_sum_fxyz[%d]=%f\n",n1, atomi, (n * NUM_OF_ABC + abc) * nlocal + n1, g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1]);
      }
    } // neigh
    for (int d = 0; d < annmb.dim; ++d) {
      // if (device == 0) printf("scale atomi %d->%d q[%d]=%f scaled[%d]=%f\n", n1, atomi, d, q[d], d, paramb.q_scaler[d]);
      q[d] = q[d] * Q_SCALER[d];
    }

    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    if (paramb.version == 4) {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp, t1);

    } else {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp, t1);    
    }

    // if (device == 0) printf("energyEi1 %d->%d = %f\n", n1, atomi, F);
    g_pe[atomi] += F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * nlocal + atomi] = Fp[d] * Q_SCALER[d]; //paramb.q_scaler
    }
  } // if
} // function

static __global__ void backward_force_2b_perneigh(
    int vflag_either,
    NEPKK::ParaMB paramb,
    NEPKK::ANN annmb,
    const int nall,
    const int N,               // inum
    const int nlocal,
    const int num_neigh,
    const int* g_NN,
    const int* g_NL,
    const int* __restrict__ g_ilist,
    const int* __restrict__ g_type,
    const float* __restrict__ g_pos,
    const float* __restrict__ g_Fp,
    double* g_f,
    double* g_virial)
{
    // 动态共享内存指针
    extern __shared__ float shared_all[];
    float* s_g_Fp = shared_all;
    float* shared = shared_all + paramb.n_max_radial_plus1;

    // 为每个数组分配偏移量
    float* s_fx = &shared[0];
    float* s_fy = &shared[blockDim.x];
    float* s_fz = &shared[2 * blockDim.x];
    
    float* s_sxx = nullptr;
    float* s_syy = nullptr;
    float* s_szz = nullptr;
    float* s_sxy = nullptr;
    float* s_sxz = nullptr;
    float* s_syz = nullptr;

    int offset = 3 * blockDim.x;  // 已占用 3 个 float 数组
    if (vflag_either) {
        s_sxx = &shared[offset];
        s_syy = &shared[offset + blockDim.x];
        s_szz = &shared[offset + 2 * blockDim.x];
        s_sxy = &shared[offset + 3 * blockDim.x];
        s_sxz = &shared[offset + 4 * blockDim.x];
        s_syz = &shared[offset + 5 * blockDim.x];
    }
    int tid = threadIdx.x;
    int atomi = g_ilist[blockIdx.x];   // 每个 block 处理一个中心原子
    int t1 = g_type[atomi];

    float x1 = g_pos[atomi*3];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];

    float fxi = 0.0f, fyi = 0.0f, fzi = 0.0f;
    float sxxi = 0.0f, syyi = 0.0f, szzi = 0.0f;
    float sxyi = 0.0f, sxzi = 0.0f, syzi = 0.0f;

    int c_start = paramb.num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    int num_neigh_i = g_NN[atomi];
    if (tid < paramb.n_max_radial_plus1) {// 线程块大小一定是两体feature大的
      s_g_Fp[tid] = g_Fp[atomi + tid * nlocal];
    }
    __syncthreads();
    // 循环处理所有邻居，步进使用 blockDim.x
    for (int off = tid; off < num_neigh_i; off += blockDim.x) {
        int n2_idx = off * num_neigh + atomi;
        int n2 = g_NL[n2_idx] & NEIGHMASK;
        int t2 = g_type[n2];
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
        int c_idx_J = t2 * c_start + t1 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

        float r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
        float d12_sq = r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2];
        if (d12_sq > paramb.rc_radial_square) continue;

        float d12 = sqrtf(d12_sq);
        float d12inv = 1.0f / d12;
        float fc12, fcp12;
        float fn12[MAX_NUM_N], fnp12[MAX_NUM_N];
        find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
        find_fn_and_fnp(paramb.basis_size_radial, paramb.rcinv_radial,
                        d12, fc12, fcp12, fn12, fnp12);

        float f12[3] = {0.0f}, f21[3] = {0.0f};

        for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
            float gnp12 = 0.0f, gnp21 = 0.0f;
            for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
                gnp12 += fnp12[k] * annmb.c[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
                gnp21 += fnp12[k] * annmb.c[c_idx_J + n * paramb.basis_size_radial_plus1 + k];
            }
            float tmp12 = s_g_Fp[n] * gnp12 * d12inv; //g_Fp[atomi + n * nlocal]
            float tmp21 = 0.0f;
            if (n2 >= nlocal) {
                // 邻居是 ghost 原子
                for (int d = 0; d < 3; ++d)
                    f12[d] += tmp12 * r12[d];
            } else {
                tmp21 = g_Fp[n2 + n * nlocal] * gnp21 * d12inv;
                for (int d = 0; d < 3; ++d) {
                    f12[d] += tmp12 * r12[d];
                    f21[d] -= tmp21 * r12[d];   // -Rji = Rij
                }
            }
        }
        // 更新邻居原子的力
        if (n2 >= nlocal) {
          fxi += f12[0]; 
          fyi += f12[1]; 
          fzi += f12[2];
          atomicAdd(&g_f[n2*3],   double(-f12[0]));
          atomicAdd(&g_f[n2*3+1], double(-f12[1]));
          atomicAdd(&g_f[n2*3+2], double(-f12[2]));
          if (vflag_either) {
              atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12[0]);
              atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12[1]);
              atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12[2]);
              atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12[1]);
              atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12[2]);
              atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12[2]);
          }
        } else {
        // 累加到中心原子的寄存器变量
        fxi += f12[0] - f21[0];
        fyi += f12[1] - f21[1];
        fzi += f12[2] - f21[2];
        if (vflag_either) {
            sxxi += r12[0] * f21[0];
            syyi += r12[1] * f21[1];
            szzi += r12[2] * f21[2];
            sxyi += r12[0] * f21[1];
            sxzi += r12[0] * f21[2];
            syzi += r12[1] * f21[2];
        }
      }
    }

    // 将每个线程的累加结果写入共享内存
    s_fx[tid] = fxi;
    s_fy[tid] = fyi;
    s_fz[tid] = fzi;
    if (vflag_either) {
        s_sxx[tid] = sxxi;
        s_syy[tid] = syyi;
        s_szz[tid] = szzi;
        s_sxy[tid] = sxyi;
        s_sxz[tid] = sxzi;
        s_syz[tid] = syzi;
    }
    __syncthreads();

    // 树状归约，使用 blockDim.x 动态确定范围
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_fx[tid] += s_fx[tid + s];
            s_fy[tid] += s_fy[tid + s];
            s_fz[tid] += s_fz[tid + s];
            if (vflag_either) {
                s_sxx[tid] += s_sxx[tid + s];
                s_syy[tid] += s_syy[tid + s];
                s_szz[tid] += s_szz[tid + s];
                s_sxy[tid] += s_sxy[tid + s];
                s_sxz[tid] += s_sxz[tid + s];
                s_syz[tid] += s_syz[tid + s];
            }
        }
        __syncthreads();
    }

    // 线程 0 将归约结果原子累加到全局
    if (tid == 0) {
        atomicAdd(&g_f[atomi*3],   double(s_fx[0]));
        atomicAdd(&g_f[atomi*3+1], double(s_fy[0]));
        atomicAdd(&g_f[atomi*3+2], double(s_fz[0]));

        if (vflag_either) {
            atomicAdd(&g_virial[atomi + 0 * nall], s_sxx[0]);
            atomicAdd(&g_virial[atomi + 1 * nall], s_syy[0]);
            atomicAdd(&g_virial[atomi + 2 * nall], s_szz[0]);
            atomicAdd(&g_virial[atomi + 3 * nall], s_sxy[0]);
            atomicAdd(&g_virial[atomi + 4 * nall], s_sxz[0]);
            atomicAdd(&g_virial[atomi + 5 * nall], s_syz[0]);
        }
    }
}

static __global__ void backward_force_2b(
  int vflag_either,
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int num_neigh, // the shape[0] of Neighbor List
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  const float* __restrict__ g_Fp,
  double* g_f,
  double* g_virial
  // double* g_total_virial
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = g_type[atomi];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_syy = 0.0f;
    float s_szz = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syz = 0.0f;

    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    int c_start = paramb.num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // int Fp_idx_start = atomi * annmb.dim;

    for (int i1 = 0; i1 < g_NN[atomi]; ++i1) {
      int n2_idx = i1 * num_neigh + atomi;
      int n2 = g_NL[n2_idx] & NEIGHMASK;
      int t2 = g_type[n2];
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      int c_idx_J = t2 * c_start + t1 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

      float r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      float d12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12_square > paramb.rc_radial_square) continue;
      float d12 = sqrt(d12_square);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};
      // if (0) printf("2b idx %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f\n", n1, atomi, t1, g_NN[atomi], n2, t2, d12);
      float fc12, fcp12;
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      find_fn_and_fnp(
        paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          gnp12 += fnp12[k] * annmb.c[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
          gnp21 += fnp12[k] * annmb.c[c_idx_J + n * paramb.basis_size_radial_plus1 + k];// shape of c [N_max+1, N_base+1, I, J]
        }
        float tmp12 = g_Fp[atomi + n * nlocal] * gnp12 * d12inv; //atomi + n * nlocal (dUi/diqn)*(diqn/drij) Fp 提前放到寄存器速度变慢
        float tmp21 = 0;
        if (n2 >= nlocal) {
          for (int d = 0; d < 3; ++d) {//编译器自动展开
            f12[d] += tmp12 * r12[d];
          }
        } else {
          tmp21 = g_Fp[n2 + n * nlocal] * gnp21 * d12inv; // (dUj/diqn)*(diqn/drij)
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
            f21[d] -= tmp21 * r12[d]; // -Rji = Rij
          }
        }
      }
      if (n2 >= nlocal) {
        s_fx += f12[0];
        s_fy += f12[1];
        s_fz += f12[2];

        atomicAdd(&g_f[n2*3], double(-f12[0]));// ghost atom
        atomicAdd(&g_f[n2*3+1], double(-f12[1]));
        atomicAdd(&g_f[n2*3+2], double(-f12[2]));
        
        if(vflag_either) {
          atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12[0]);
          atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12[1]);
          atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12[2]);
          atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12[1]);
          atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12[2]);
          atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12[2]);
          // atomicAdd(&g_virial[n2 + 6 * nall], -r12[1] * f12[0]);
          // atomicAdd(&g_virial[n2 + 7 * nall], -r12[2] * f12[0]);
          // atomicAdd(&g_virial[n2 + 8 * nall], -r12[2] * f12[1]);
        }
      } else {
        s_fx += f12[0] - f21[0];
        s_fy += f12[1] - f21[1];
        s_fz += f12[2] - f21[2];
        if(vflag_either) {
          s_sxx += r12[0] * f21[0];
          s_syy += r12[1] * f21[1];
          s_szz += r12[2] * f21[2];

          s_sxy += r12[0] * f21[1];
          s_sxz += r12[0] * f21[2];
          s_syz += r12[1] * f21[2];

          // s_syx += r12[1] * f21[0];
          // s_szx += r12[2] * f21[0];
          // s_szy += r12[2] * f21[1];
        }
      }
    }

    //对于ghost atom，需要加上ghost atom 对应的force；并且保存ghost atom 对应的force
    g_f[atomi*3] += s_fx;
    g_f[atomi*3+1] += s_fy;
    g_f[atomi*3+2] += s_fz;

    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    if(vflag_either) {
      g_virial[atomi + 0 * nall] += s_sxx;
      g_virial[atomi + 1 * nall] += s_syy;
      g_virial[atomi + 2 * nall] += s_szz;
      g_virial[atomi + 3 * nall] += s_sxy;
      g_virial[atomi + 4 * nall] += s_sxz;
      g_virial[atomi + 5 * nall] += s_syz;
    }
    // g_virial[atomi + 6 * nall] += s_syx;
    // g_virial[atomi + 7 * nall] += s_szx;
    // g_virial[atomi + 8 * nall] += s_szy;
  }
} 

__global__ void backward_force_3b_per_atom_sharemem(
    NEPKK::ParaMB paramb,
    NEPKK::ANN annmb,
    int N,
    int nlocal,
    const int* g_NN_angular,
    const int* g_NL_angular,
    const int* __restrict__ g_ilist,
    const int* __restrict__ g_type,
    const float* __restrict__ g_pos,
    const float* __restrict__ g_Fp,
    const float* __restrict__ g_sum_fxyz,
    float* g_f12x,
    float* g_f12y,
    float* g_f12z
) {
    extern __shared__ float shmem[];
    int i = blockIdx.x;  // 每个 block 负责一个中心原子
    if (i >= N) return;

    int atomi = g_ilist[i];
    int t1    = g_type[atomi];

    float* s_x1       = shmem + 0;                           // 1
    float* s_y1       = shmem + 1;                           // 1
    float* s_z1       = shmem + 2;                           // 1
    float* s_Fp       = shmem + 3;                           // dim_angular
    float* s_sum_fxyz = shmem + 3 + paramb.dim_angular;      // n_max_angular_plus1 * NUM_OF_ABC
  // 每个线程独占的 fn12 和 fnp12 空间
    const int per_thread_fn_size = MAX_NUM_N * 2;            // fn12 + fnp12 = 20 + 20 = 40
    float* s_fn_base = s_sum_fxyz + (paramb.n_max_angular_plus1 * NUM_OF_ABC);
    float* s_fn  = s_fn_base + threadIdx.x * per_thread_fn_size;
    float* s_fnp = s_fn + MAX_NUM_N;

    // ==============================
    //  由 warp 0 负责加载中心原子公共数据
    // ==============================
    if (threadIdx.x < blockDim.x) {
        if (threadIdx.x == 0) {
            s_x1[0] = g_pos[atomi * 3 + 0];
            s_y1[0] = g_pos[atomi * 3 + 1];
            s_z1[0] = g_pos[atomi * 3 + 2];
        }

        // 加载 Fp
        for (int d = threadIdx.x; d < paramb.dim_angular; d += blockDim.x) {
            s_Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * nlocal + atomi];
        }
        int sum_size = paramb.n_max_angular_plus1 * NUM_OF_ABC;
        // 加载 sum_fxyz
        for (int d = threadIdx.x; d < sum_size; d += blockDim.x) {
            s_sum_fxyz[d] = g_sum_fxyz[d * nlocal + atomi];
        }
    }
    __syncthreads();

    // ==============================
    //  每个线程处理部分近邻
    // ==============================
    int nn = g_NN_angular[atomi];

    for (int i1 = threadIdx.x; i1 < nn; i1 += blockDim.x) {
        int index = i1 * nlocal + atomi;
        int n2    = g_NL_angular[index];
        int t2    = g_type[n2];

        float r12[3] = {
            g_pos[n2 * 3 + 0] - s_x1[0],
            g_pos[n2 * 3 + 1] - s_y1[0],
            g_pos[n2 * 3 + 2] - s_z1[0]
        };

        float d12 = sqrtf(r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2]);
        if (d12 > paramb.rc_angular) continue;  // 可选：加 cutoff 检查

        float fc12, fcp12;
        find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
        // 直接使用共享内存中的 fn12 和 fnp12
        find_fn_and_fnp(
            paramb.basis_size_angular,
            paramb.rcinv_angular,
            d12, fc12, fcp12,
            s_fn, s_fnp
        );

        int c_start = paramb.num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        float f12_local[3] = {0.0f, 0.0f, 0.0f};
        for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
            float gn12  = 0.0f;
            float gnp12 = 0.0f;

            for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
                int idx = c_idx_I + n * paramb.basis_size_angular_plus1 + k + paramb.num_c_radial;
                gn12  += s_fn[k]  * annmb.c[idx];
                gnp12 += s_fnp[k] * annmb.c[idx];
            }

            accumulate_f12(
                paramb.L_max,
                paramb.num_L,
                n,
                paramb.n_max_angular + 1,
                d12,
                r12,
                gn12,
                gnp12,
                s_Fp,           // 使用共享内存
                s_sum_fxyz,     // 使用共享内存
                f12_local       // 累加到本地
            );
        }
        g_f12x[index] = f12_local[0];
        g_f12y[index] = f12_local[1];
        g_f12z[index] = f12_local[2];
    }
}

static __global__ void backward_force_3b_dqnl(
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int N,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
  float* g_f12x,
  float* g_f12y,
  float* g_f12z
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];

    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial_plus1 + d) * nlocal + atomi];
    }
    for (int d = 0; d < paramb.n_max_angular_plus1 * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * nlocal + atomi];
    }

    int   t1 = g_type[atomi];
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    int c_start = paramb.num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
      int index = i1 * nlocal + atomi;
      int n2 = g_NL_angular[index];
      int t2 = g_type[n2];
      
      float f12[3] = {0.0f};
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
      float r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      // if (0) printf("3bhalf idx %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f fixyz %f %f %f fjxyz %f %f %f\n", n1, atomi, t1, g_NN_angular[atomi], n2, t2, d12, x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(
        paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
          gn12  += fn12[k]  * annmb.c[c_idx_I + n * paramb.basis_size_angular_plus1 + k + paramb.num_c_radial];
          gnp12 += fnp12[k] * annmb.c[c_idx_I + n * paramb.basis_size_angular_plus1 + k + paramb.num_c_radial];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular + 1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
    }
  }
}


static __global__ void backward_force_3b_merge(
  int vflag_either,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const int* __restrict__ g_ilist,
  const float* __restrict__ g_pos,
  double* g_f,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f;
  float s_syy = 0.0f;
  float s_szz = 0.0f;
  float s_sxy = 0.0f;
  float s_sxz = 0.0f;
  float s_syz = 0.0f;

  // float s_syx = 0.0f;
  // float s_szx = 0.0f;
  // float s_szy = 0.0f;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];

    for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
      int index = i1 * nlocal + atomi;
      int n2 = g_NL_angular[index];
      float x12 = g_pos[n2*3]   - x1;
      float y12 = g_pos[n2*3+1] - y1;
      float z12 = g_pos[n2*3+2] - z1;

      float r12[3] = {x12, y12, z12};
      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];

      if (n2 < nlocal) {
        float f21x = 0;
        float f21y = 0;
        float f21z = 0;
        int offset = 0;
        int neighbor_number_2 = 0;
        neighbor_number_2 = g_NN_angular[n2];

        for (int k = 0; k < neighbor_number_2; ++k) {
          if (atomi == g_NL_angular[n2 + nlocal * k]) {
            offset = k;
            break;
          }
        }// n2 作为中心原子，对应的邻居为n1
        index = offset * nlocal + n2;
        f21x = g_f12x[index];
        f21y = g_f12y[index];
        f21z = g_f12z[index];

        s_fx += f12x - f21x;
        s_fy += f12y - f21y;
        s_fz += f12z - f21z;

        // per-atom virial
        if(vflag_either) {
          s_sxx += x12 * f21x;
          s_syy += y12 * f21y;
          s_szz += z12 * f21z;

          s_sxy += x12 * f21y;
          s_sxz += x12 * f21z;
          s_syz += y12 * f21z;

          // s_syx += y12 * f21x;
          // s_szx += z12 * f21x;
          // s_szy += z12 * f21y;
        }
      } else {
        s_fx += f12x;
        s_fy += f12y;
        s_fz += f12z;
        atomicAdd(&g_f[n2*3], double(-f12x));// ghost atom
        atomicAdd(&g_f[n2*3+1], double(-f12y));
        atomicAdd(&g_f[n2*3+2], double(-f12z));
        if(vflag_either) {
          atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12x);
          atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12y);
          atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12z);
          atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12y);
          atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12z);
          atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12z);
          // atomicAdd(&g_virial[n2 + 6 * nall], -r12[1] * f12x);
          // atomicAdd(&g_virial[n2 + 7 * nall], -r12[2] * f12x);
          // atomicAdd(&g_virial[n2 + 8 * nall], -r12[2] * f12y);
        }
      }
    }

    // save force
    g_f[atomi*3] += s_fx;
    g_f[atomi*3+1] += s_fy;
    g_f[atomi*3+2] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    if(vflag_either) {
      g_virial[atomi + 0 * nall] += s_sxx;
      g_virial[atomi + 1 * nall] += s_syy;
      g_virial[atomi + 2 * nall] += s_szz;
      g_virial[atomi + 3 * nall] += s_sxy;
      g_virial[atomi + 4 * nall] += s_sxz;
      g_virial[atomi + 5 * nall] += s_syz;
      // g_virial[atomi + 6 * nall] += s_syx;
      // g_virial[atomi + 7 * nall] += s_szx;
      // g_virial[atomi + 8 * nall] += s_szy;
    }
  }
}


__global__ void calculate_total_virial(const double* virial, double* total_virial, int N) {
    __shared__ double shared_virial[6 * 64]; // 使用共享内存存储部分和
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

    for (int i = 0; i < 6; ++i) {
        shared_virial[i * blockDim.x + tid] = 0.0;
    }
    __syncthreads();

    // 累加每个原子的virial值
    if (index < N) {
        atomicAdd(&shared_virial[0 * blockDim.x + tid], virial[0 * N + index]);
        atomicAdd(&shared_virial[1 * blockDim.x + tid], virial[1 * N + index]);
        atomicAdd(&shared_virial[2 * blockDim.x + tid], virial[2 * N + index]);
        atomicAdd(&shared_virial[3 * blockDim.x + tid], virial[3 * N + index]);
        atomicAdd(&shared_virial[4 * blockDim.x + tid], virial[4 * N + index]);
        atomicAdd(&shared_virial[5 * blockDim.x + tid], virial[5 * N + index]);
    }
    __syncthreads();

    // 归约每个块内的部分和
    if (tid < 6) {
        for (int i = 1; i < blockDim.x; ++i) {
            shared_virial[tid * blockDim.x] += shared_virial[tid * blockDim.x + i];
        }
    }
    __syncthreads();

    // 将每个块的部分和累加到全局内存
    if (tid < 6) {
        atomicAdd(&total_virial[tid], shared_virial[tid * blockDim.x]);
    }
}

__global__ void block_reduce(const double* data, double* partial_sums, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = (i < n) ? data[i] : 0.0;

    __syncthreads();

    // 树状 reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

__global__ void global_reduce(const double* partial_sums, double* result, int num_blocks) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    // 初始化共享内存
    sdata[tid] = 0.0;
    
    // 每个线程处理多个partial sums（跨步循环）
    for (int idx = tid; idx < num_blocks; idx += blockDim.x) {
        sdata[tid] += partial_sums[idx];
    }
    __syncthreads();
    
    // 树状reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // 原子加
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void copyArrayKernel(double* dest, double* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        dest[idx] = src[idx];
    }
}

__global__ void doubleTofloat(float* dest, double* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] = src[idx];
    }
}

// [Nmax, Nbase, TYPE, TYPE] to [TYPEi, TYPEj, Nmax, Nbase]
__global__ void convert_c_dim(float* c, float* temp, int NtypeI, int Nmax, int Nbase) {
    int total_elements = (Nmax + 1) * (Nbase + 1) * NtypeI * NtypeI;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        // 从线性索引反算出原始四维索引 [i][j][k][l]
        int l = idx % NtypeI;
        int k = (idx / NtypeI) % NtypeI;
        int j = (idx / (NtypeI * NtypeI)) % (Nbase + 1);
        int i = (idx / (NtypeI * NtypeI * (Nbase + 1))) % (Nmax + 1);
        // 原始索引
        int old_idx = ((i * (Nbase + 1) + j) * NtypeI + k) * NtypeI + l;
        // 新索引 [k][l][i][j]
        int new_idx = ((k * NtypeI + l) * (Nmax + 1) + i) * (Nbase + 1) + j;
        temp[new_idx] = c[old_idx];
        // printf("cvt[%d %d %d %d]: [%d][%d][%d][%d]=[%d]%f to [%d][%d][%d][%d]=[%d]%f\n",NtypeI, NtypeI, Nmax, Nbase, i, j, k, l, old_idx, c[old_idx], k, l, i, j, new_idx, temp[new_idx]);
    }
}

__global__ void convert_atom_types(
    const int nall, 
    const int inum,
    const int nlocal,//对于部分原子受力，ilist长度为inum <= nlocal，itype长度为 nlocal+ghost
    const int* __restrict__ g_ilist,
    const int* __restrict__ g_type,
    const int* __restrict__ type_map, //力场元素类型和结构类型的关系映射
    int* copy_ilist,
    int* cvt_type_map) 
{ 
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 >= nall) return;
    int t1 = type_map[g_type[n1]-1];//问题这里g_type 长度为nall？
    cvt_type_map[n1] = t1;
}

static __global__ void gpu_sort_neighbor_list(const int N, const int* NN, int* NL)
{
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int neighbor_number = NN[bid];
  int atom_index;
  extern __shared__ int atom_index_copy[];

  if (tid < neighbor_number) {
    atom_index = NL[bid + tid * N];
    atom_index_copy[tid] = atom_index;
  }
  int count = 0;
  __syncthreads();

  for (int j = 0; j < neighbor_number; ++j) {
    if (atom_index > atom_index_copy[j]) {
      count++;
    }
  }

  if (tid < neighbor_number) {
    NL[bid + count * N] = atom_index;
  }
}

// 奇偶排序，行存储，适合于lammps近邻的下标部分有序
__global__ void sort_neighbor_simple_fast_kernel(
    int nlocal,
    int MN_angular,
    const int* __restrict__ NN_angular,
    int* __restrict__ NL_angular)
{
    int center_idx = blockIdx.x;
    if (center_idx >= nlocal) return;
    
    int nn = NN_angular[center_idx];
    if (nn <= 1) return;
    
    int* row_data = &NL_angular[center_idx * MN_angular];
    
    // 共享内存
    extern __shared__ int shared_data[];
    int* sdata = shared_data;
    
    int tid = threadIdx.x;
    int threads = blockDim.x;
    
    // 加载数据
    for (int idx = tid; idx < nn; idx += threads) {
        sdata[idx] = row_data[idx];
    }
    __syncthreads();
    
    if (nn >= 32 && nn <= 128) {
        // 优化范围：使用早期退出和更高效的内存访问
        // 尝试检测是否已经有序（常见于部分有序数据）
        __shared__ bool needs_sorting;
        if (tid == 0) {
            needs_sorting = false;
            // 快速检查是否已经有序
            for (int i = 0; i < nn - 1; i++) {
                if (sdata[i] > sdata[i + 1]) {
                    needs_sorting = true;
                    break;
                }
            }
        }
        __syncthreads();
        
        if (!needs_sorting) {
            // 已经有序，直接跳过排序
            goto write_back;
        }
        
        bool sorted = false;
        int max_phases = nn;  // 最大轮次
        
        for (int phase = 0; phase < max_phases && !sorted; phase++) {
            __shared__ bool phase_sorted;
            
            if (tid == 0) {
                phase_sorted = true;
            }
            __syncthreads();
            
            bool local_sorted = true;
            
            if (phase % 2 == 0) {
                // 偶数阶段
                for (int i = tid * 2; i < nn - 1; i += threads * 2) {
                    if (sdata[i] > sdata[i + 1]) {
                        // 使用XOR交换技巧（可能更快）
                        sdata[i] ^= sdata[i + 1];
                        sdata[i + 1] ^= sdata[i];
                        sdata[i] ^= sdata[i + 1];
                        local_sorted = false;
                    }
                }
            } else {
                // 奇数阶段
                for (int i = tid * 2 + 1; i < nn - 1; i += threads * 2) {
                    if (sdata[i] > sdata[i + 1]) {
                        sdata[i] ^= sdata[i + 1];
                        sdata[i + 1] ^= sdata[i];
                        sdata[i] ^= sdata[i + 1];
                        local_sorted = false;
                    }
                }
            }
            
            __syncthreads();
            
            // 更新全局状态
            if (!local_sorted) {
                atomicAnd((unsigned int*)&phase_sorted, 0);
            }
            __syncthreads();
            
            sorted = phase_sorted;
        }
    }
    else {
        // 非优化范围：使用基本奇偶排序保证正确性
        for (int phase = 0; phase < nn; phase++) {
            if (phase % 2 == 0) {
                for (int i = tid * 2; i < nn - 1; i += threads * 2) {
                    if (sdata[i] > sdata[i + 1]) {
                        int temp = sdata[i];
                        sdata[i] = sdata[i + 1];
                        sdata[i + 1] = temp;
                    }
                }
            } else {
                for (int i = tid * 2 + 1; i < nn - 1; i += threads * 2) {
                    if (sdata[i] > sdata[i + 1]) {
                        int temp = sdata[i];
                        sdata[i] = sdata[i + 1];
                        sdata[i + 1] = temp;
                    }
                }
            }
            __syncthreads();
        }
    }
    
write_back:
    // 写回数据
    __syncthreads();
    for (int idx = tid; idx < nn; idx += threads) {
        row_data[idx] = sdata[idx];
    }
}

static __global__ void backward_force_ZBL(
  int vflag_either,
  const NEPKK::ZBL zbl,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int MN_angular,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const float* __restrict__ g_pos,
  double* g_f,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    float s_pe = 0.0f;
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;

    float s_sxx = 0.0f;
    float s_syy = 0.0f;
    float s_szz = 0.0f;

    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syz = 0.0f;

    // float s_syx = 0.0f;
    // float s_szx = 0.0f;
    // float s_szy = 0.0f;
    int atomi = g_ilist[n1];
    float x1 = g_pos[atomi*3  ];
    float y1 = g_pos[atomi*3+1];
    float z1 = g_pos[atomi*3+2];
    int type1 = g_type[atomi];
    float zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[atomi]; ++i1) {
      int n2 = g_NL[atomi + nlocal * i1];
      int type2 = g_type[n2];
      float r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f, fp;
      float zj = zbl.atomic_numbers[type2];
      float a_inv = (pow_zi + pow(zj, 0.23f)) * 2.134563f;
      float zizj = K_C_SP * zi * zj;
      if (zbl.flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl.num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        float ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = zbl.para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        find_f_and_fp_zbl(zizj, a_inv, zbl.rc_inner, zbl.rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      // if (0) printf("zbl n1 %d n2 %d d12 %f e_c_half %f\n", n1, n2, d12, f);
      if (n2 >= nlocal) {
        s_fx += f12[0];
        s_fy += f12[1];
        s_fz += f12[2];
        atomicAdd(&g_f[n2*3], double(f21[0]));// ghost atom
        atomicAdd(&g_f[n2*3+1], double(f21[1]));
        atomicAdd(&g_f[n2*3+2], double(f21[2]));
        if (vflag_either) {
          atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12[0]);
          atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12[1]);
          atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12[2]);
          atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12[1]);
          atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12[2]);
          atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12[2]);
          // atomicAdd(&g_virial[n2 + 6 * nall], -r12[1] * f12[0]);
          // atomicAdd(&g_virial[n2 + 7 * nall], -r12[2] * f12[0]);
          // atomicAdd(&g_virial[n2 + 8 * nall], -r12[2] * f12[1]);
        }
      } else {
        s_fx += f12[0] - f21[0];
        s_fy += f12[1] - f21[1];
        s_fz += f12[2] - f21[2];
        if (vflag_either) {
          s_sxx += r12[0] * f21[0];
          s_syy += r12[1] * f21[1];
          s_szz += r12[2] * f21[2];

          s_sxy += r12[0] * f21[1];
          s_sxz += r12[0] * f21[2];
          s_syz += r12[1] * f21[2];

          // s_syx += r12[1] * f21[0];
          // s_szx += r12[2] * f21[0];
          // s_szy += r12[2] * f21[1];
        }
      }
      s_pe += f * 0.5f;
    }
    g_f[atomi*3] += s_fx;
    g_f[atomi*3+1] += s_fy;
    g_f[atomi*3+2] += s_fz;
    if (vflag_either) {
      g_virial[atomi + 0 * nall] += s_sxx;
      g_virial[atomi + 1 * nall] += s_syy;
      g_virial[atomi + 2 * nall] += s_szz;
      g_virial[atomi + 3 * nall] += s_sxy;
      g_virial[atomi + 4 * nall] += s_sxz;
      g_virial[atomi + 5 * nall] += s_syz;
      // g_virial[atomi + 6 * nall] += s_syx;
      // g_virial[atomi + 7 * nall] += s_szx;
      // g_virial[atomi + 8 * nall] += s_szy;
    }
    g_pe[atomi] += s_pe;
  }
}
