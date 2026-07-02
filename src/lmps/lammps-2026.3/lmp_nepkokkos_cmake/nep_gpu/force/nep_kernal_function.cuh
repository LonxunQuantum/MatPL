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
  const NEP_FLOAT* param_c2,
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
  const NEP_FLOAT* __restrict__ g_pos,
  NEP_FLOAT* g_Fp)
{
  extern __shared__ NEP_FLOAT s_c[];
  const int total_elements = paramb.sim_num_types * paramb.sim_num_types *
    paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
  for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
    s_c[i] = param_c2[i];
  }
  __syncthreads();
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int count_angular = 0;    
    int atomi = g_ilist[n1];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // get radial descriptors
    for (int i1 = 0; i1 < g_numneigh[atomi]; ++i1) {
      int n2 = g_firstneigh[i1 * num_neigh + atomi] & NEIGHMASK; //这里num_neigh 是总的原子数，与三体近邻维度是一致的。
      int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
      // if (atomi == 1) printf("calc n1 %d nn %d n2 %d\n", atomi, g_numneigh[atomi], n2);

      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      NEP_FLOAT x12 = g_pos[n2*3  ] - x1;
      NEP_FLOAT y12 = g_pos[n2*3+1] - y1;
      NEP_FLOAT z12 = g_pos[n2*3+2] - z1;
      NEP_FLOAT d12_square = x12 * x12 + y12 * y12 + z12 * z12;
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

      NEP_FLOAT d12 = sqrt(d12_square);
      // 2b->qn
      NEP_FLOAT fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      NEP_FLOAT gn12[MAX_NUM_N];
#pragma unroll
      for (int n = 0; n < MAX_NUM_N; ++n) {
        gn12[n] = FLOAT_LIT(0.0);
      }

      const NEP_FLOAT radial_x =
        FLOAT_LIT(2.0) * (d12 * paramb.rcinv_radial - FLOAT_LIT(1.0)) *
        (d12 * paramb.rcinv_radial - FLOAT_LIT(1.0)) - FLOAT_LIT(1.0);
      const NEP_FLOAT half_fc12 = FLOAT_LIT(0.5) * fc12;
      NEP_FLOAT chebyshev_k_minus_2 = FLOAT_LIT(1.0);
      NEP_FLOAT chebyshev_k_minus_1 = radial_x;
      for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
        NEP_FLOAT fn12_k;
        if (k == 0) {
          fn12_k = fc12;
        } else if (k == 1) {
          fn12_k = (radial_x + FLOAT_LIT(1.0)) * half_fc12;
        } else {
          const NEP_FLOAT chebyshev_k =
            FLOAT_LIT(2.0) * radial_x * chebyshev_k_minus_1 - chebyshev_k_minus_2;
          chebyshev_k_minus_2 = chebyshev_k_minus_1;
          chebyshev_k_minus_1 = chebyshev_k;
          fn12_k = (chebyshev_k + FLOAT_LIT(1.0)) * half_fc12;
        }
#pragma unroll
        for (int n = 0; n < MAX_NUM_N; ++n) {
          if (n < paramb.n_max_radial_plus1) {
            gn12[n] +=
              fn12_k * s_c[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
          }
        }
      }
#pragma unroll
      for (int n = 0; n < MAX_NUM_N; ++n) {
        if (n < paramb.n_max_radial_plus1) {
          g_Fp[n * nlocal + atomi] += gn12[n];
        }
      }
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
  const NEP_FLOAT* param_c2,
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
  const NEP_FLOAT* __restrict__ g_pos,
  NEP_FLOAT* g_Fp)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int count_angular = 0;
    int atomi = g_ilist[n1];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // get radial descriptors
    for (int i1 = 0; i1 < g_numneigh[atomi]; ++i1) {
      int n2 = g_firstneigh[i1 * num_neigh + atomi] & NEIGHMASK; 
      int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      NEP_FLOAT x12 = g_pos[n2*3  ] - x1;
      NEP_FLOAT y12 = g_pos[n2*3+1] - y1;
      NEP_FLOAT z12 = g_pos[n2*3+2] - z1;
      NEP_FLOAT d12_square = x12 * x12 + y12 * y12 + z12 * z12;
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

      NEP_FLOAT d12 = sqrt(d12_square);
      // 2b->qn
      NEP_FLOAT fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      NEP_FLOAT fn12[MAX_NUM_N];//n_base

      find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        NEP_FLOAT gn12 = FLOAT_LIT(0.0);
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          gn12 += fn12[k] * param_c2[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
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
  const NEP_FLOAT* param_c3,
  const int N,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,// 顺带构建出小的多体近邻表
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  NEP_FLOAT* g_Fp,
  NEP_FLOAT* g_sum_fxyz)
{
  extern __shared__ NEP_FLOAT s_c[];
  const int total_elements = paramb.sim_num_types * paramb.sim_num_types *
    paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
  for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
    s_c[i] = param_c3[i];
  }
  __syncthreads();
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
      NEP_FLOAT s[NUM_OF_ABC] = {FLOAT_LIT(0.0)};
      for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
        int n2 = g_NL_angular[atomi + nlocal * i1];
        int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        NEP_FLOAT x12 = g_pos[n2*3  ] - x1;
        NEP_FLOAT y12 = g_pos[n2*3+1] - y1;
        NEP_FLOAT z12 = g_pos[n2*3+2] - z1;
        NEP_FLOAT d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        NEP_FLOAT fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);

        const NEP_FLOAT gn12 = find_gn(
          paramb.basis_size_angular,
          paramb.rcinv_angular,
          d12,
          fc12,
          s_c + c_idx_I + n * paramb.basis_size_angular_plus1);
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max,
        paramb.num_L,
        paramb.n_max_angular_plus1,
        n,
        s,
        g_Fp + paramb.n_max_radial_plus1 * nlocal + atomi,
        nlocal);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + atomi] = s[abc];
      }
    } // neigh

  } // if
} // function

static __global__ void calc_3b_descriptor(
  NEPKK::ParaMB paramb,
  const NEP_FLOAT* param_c3,
  const int N,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,// 顺带构建出小的多体近邻表
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  NEP_FLOAT* g_Fp,
  NEP_FLOAT* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
      NEP_FLOAT s[NUM_OF_ABC] = {FLOAT_LIT(0.0)};
      for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
        int n2 = g_NL_angular[atomi + nlocal * i1];
        int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        NEP_FLOAT x12 = g_pos[n2*3  ] - x1;
        NEP_FLOAT y12 = g_pos[n2*3+1] - y1;
        NEP_FLOAT z12 = g_pos[n2*3+2] - z1;
        NEP_FLOAT d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        NEP_FLOAT fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);

        const NEP_FLOAT gn12 = find_gn(
          paramb.basis_size_angular,
          paramb.rcinv_angular,
          d12,
          fc12,
          param_c3 + c_idx_I + n * paramb.basis_size_angular_plus1);
        accumulate_s(paramb.L_max, d12, x12, y12, z12, gn12, s);
      }
      find_q(
        paramb.L_max,
        paramb.num_L,
        paramb.n_max_angular_plus1,
        n,
        s,
        g_Fp + paramb.n_max_radial_plus1 * nlocal + atomi,
        nlocal);
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + atomi] = s[abc];
        // if (device == 0) printf("g_sum_fxyz atomi%d->%d g_sum_fxyz[%d]=%f\n",n1, atomi, (n * NUM_OF_ABC + abc) * nlocal + n1, g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1]);
      }
    } // neigh
  } // if
} // function

static __global__ void apply_ann_forward(
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int N,
  const int nlocal,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_Fp,
  double* g_pe,
  NEP_FLOAT* g_charge,
  NEP_FLOAT* g_ann_alpha)
{
  extern __shared__ NEP_FLOAT s_q[];
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  const bool active = n1 < N;
  const int atomi = active ? g_ilist[n1] : 0;
  for (int d = 0; d < annmb.dim; ++d) {
    s_q[d * blockDim.x + threadIdx.x] = active ?
      g_Fp[d * nlocal + atomi] * Q_SCALER[d] : FLOAT_LIT(0.0);
  }
  __syncthreads();

  if (active) {
    const int nep_t1 = g_type[atomi];
    const NEP_FLOAT* w0 = annmb.w0[nep_t1];
    const NEP_FLOAT* b0 = annmb.b0[nep_t1];
    const NEP_FLOAT* w1 = annmb.w1[nep_t1];
    NEP_FLOAT F = FLOAT_LIT(0.0);
    NEP_FLOAT charge = FLOAT_LIT(0.0);
    for (int n = 0; n < annmb.num_neurons1; ++n) {
      NEP_FLOAT w0_times_q = FLOAT_LIT(0.0);
      for (int d = 0; d < annmb.dim; ++d) {
        const NEP_FLOAT q = s_q[d * blockDim.x + threadIdx.x];
        w0_times_q += w0[n * annmb.dim + d] * q;
      }
      const NEP_FLOAT x1 = tanh(w0_times_q - b0[n]);
      const NEP_FLOAT tanh_der = FLOAT_LIT(1.0) - x1 * x1;
      F += w1[n] * x1;
      g_ann_alpha[n * N + n1] = w1[n] * tanh_der;
      if (paramb.charge_mode == 2) {
        const int charge_alpha_offset = annmb.num_neurons1 * N;
        charge += w1[n + annmb.num_neurons1] * x1;
        g_ann_alpha[charge_alpha_offset + n * N + n1] =
          w1[n + annmb.num_neurons1] * tanh_der;
      }
    }
    if (paramb.charge_mode == 2) {
      F -= (paramb.version == 5 ? w1[annmb.num_neurons1 * 2] + annmb.b1[0] : annmb.b1[0]);
      g_charge[atomi] = charge;
    } else if (paramb.version == 4) {
      F -= annmb.b1[nep_t1];
    } else {
      F -= w1[annmb.num_neurons1] + annmb.b1[0];
    }
    g_pe[atomi] += F;
  }
}

static __global__ void apply_ann_derivative(
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int N,
  const int nlocal,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_ann_alpha,
  NEP_FLOAT* g_charge_derivative,
  NEP_FLOAT* g_Fp)
{
  const int atoms_per_block = 8;
  extern __shared__ NEP_FLOAT s_alpha[];
  const int atom_begin = blockIdx.x * atoms_per_block;
  const int alpha_outputs = paramb.charge_mode == 2 ? 2 : 1;
  const int alpha_tile_size = annmb.num_neurons1 * atoms_per_block * alpha_outputs;
  for (int index = threadIdx.x; index < alpha_tile_size; index += blockDim.x) {
    const int atom_offset = index % atoms_per_block;
    const int packed = index / atoms_per_block;
    const int n = packed % annmb.num_neurons1;
    const int output = packed / annmb.num_neurons1;
    const int n1 = atom_begin + atom_offset;
    const int global_offset = output * annmb.num_neurons1 * N;
    s_alpha[index] = n1 < N ? g_ann_alpha[global_offset + n * N + n1] : FLOAT_LIT(0.0);
  }
  __syncthreads();

  const int derivative_tile_size = annmb.dim * atoms_per_block;
  for (int index = threadIdx.x; index < derivative_tile_size; index += blockDim.x) {
    const int atom_offset = index % atoms_per_block;
    const int d = index / atoms_per_block;
    const int n1 = atom_begin + atom_offset;
    if (n1 < N) {
      const int atomi = g_ilist[n1];
      const int nep_t1 = g_type[atomi];
      const NEP_FLOAT* w0 = annmb.w0[nep_t1];
      NEP_FLOAT derivative = FLOAT_LIT(0.0);
      NEP_FLOAT charge_derivative = FLOAT_LIT(0.0);
      const int charge_alpha_offset = annmb.num_neurons1 * atoms_per_block;
      for (int n = 0; n < annmb.num_neurons1; ++n) {
        const NEP_FLOAT w0_nd = w0[n * annmb.dim + d];
        derivative += s_alpha[n * atoms_per_block + atom_offset] * w0_nd;
        if (paramb.charge_mode == 2) {
          charge_derivative += s_alpha[charge_alpha_offset + n * atoms_per_block + atom_offset] * w0_nd;
        }
      }
      g_Fp[d * nlocal + atomi] = derivative * Q_SCALER[d];
      if (paramb.charge_mode == 2) {
        g_charge_derivative[d * nlocal + atomi] = charge_derivative * Q_SCALER[d];
      }
    }
  }
}

static __global__ void add_charge_energy_charge2(
  const int N,
  const NEP_FLOAT* __restrict__ g_charge,
  const NEP_FLOAT* __restrict__ g_D_real,
  double* g_pe)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    g_pe[n] += 0.5 * double(g_charge[n]) * double(g_D_real[n]);
  }
}

static __global__ void backward_force_2b_perneigh(
    int vflag_either,
    int cvflag_atom,
    int vatom_num,
    NEPKK::ParaMB paramb,
    const NEP_FLOAT* param_c2,
    const int nall,
    const int N,               // inum
    const int nlocal,
    const int num_neigh,
    const int* g_NN,
    const int* g_NL,
    const int* __restrict__ g_ilist,
    const int* __restrict__ g_type,
    const NEP_FLOAT* __restrict__ g_pos,
    const NEP_FLOAT* __restrict__ g_Fp,
    double* g_f,
    double* g_virial)
{
    // 动态共享内存指针
    extern __shared__ NEP_FLOAT shared_all[];
    NEP_FLOAT* s_g_Fp = shared_all;
    NEP_FLOAT* shared = shared_all + paramb.n_max_radial_plus1;

    // 为每个数组分配偏移量
    NEP_FLOAT* s_fx = &shared[0];
    NEP_FLOAT* s_fy = &shared[blockDim.x];
    NEP_FLOAT* s_fz = &shared[2 * blockDim.x];
    
    NEP_FLOAT* s_sxx = nullptr;
    NEP_FLOAT* s_syy = nullptr;
    NEP_FLOAT* s_szz = nullptr;
    NEP_FLOAT* s_sxy = nullptr;
    NEP_FLOAT* s_sxz = nullptr;
    NEP_FLOAT* s_syz = nullptr;
    NEP_FLOAT* s_syx = nullptr;
    NEP_FLOAT* s_szx = nullptr;
    NEP_FLOAT* s_szy = nullptr;

    int offset = 3 * blockDim.x;  // 已占用 3 个 float 数组
    if (vflag_either) {
        s_sxx = &shared[offset];
        s_syy = &shared[offset + blockDim.x];
        s_szz = &shared[offset + 2 * blockDim.x];
        s_sxy = &shared[offset + 3 * blockDim.x];
        s_sxz = &shared[offset + 4 * blockDim.x];
        s_syz = &shared[offset + 5 * blockDim.x];
        if (cvflag_atom) {
            s_syx = &shared[offset + 6 * blockDim.x];
            s_szx = &shared[offset + 7 * blockDim.x];
            s_szy = &shared[offset + 8 * blockDim.x];
        }
    }
    int tid = threadIdx.x;
    int atomi = g_ilist[blockIdx.x];   // 每个 block 处理一个中心原子
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);

    NEP_FLOAT x1 = g_pos[atomi*3];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];

    NEP_FLOAT fxi = FLOAT_LIT(0.0), fyi = FLOAT_LIT(0.0), fzi = FLOAT_LIT(0.0);
    NEP_FLOAT sxxi = FLOAT_LIT(0.0), syyi = FLOAT_LIT(0.0), szzi = FLOAT_LIT(0.0);
    NEP_FLOAT sxyi = FLOAT_LIT(0.0), sxzi = FLOAT_LIT(0.0), syzi = FLOAT_LIT(0.0);
    NEP_FLOAT syxi = FLOAT_LIT(0.0), szxi = FLOAT_LIT(0.0), szyi = FLOAT_LIT(0.0);

    int c_start = paramb.sim_num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    int num_neigh_i = g_NN[atomi];
    if (tid < paramb.n_max_radial_plus1) {// 线程块大小一定是两体feature大的
      s_g_Fp[tid] = g_Fp[atomi + tid * nlocal];
    }
    __syncthreads();
    // 循环处理所有邻居，步进使用 blockDim.x
    for (int off = tid; off < num_neigh_i; off += blockDim.x) {
        int n2_idx = off * num_neigh + atomi;
        int n2 = g_NL[n2_idx] & NEIGHMASK;
        int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
        int c_idx_J = t2 * c_start + t1 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

        NEP_FLOAT r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
        NEP_FLOAT d12_sq = r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2];
        if (d12_sq > paramb.rc_radial_square) continue;

        NEP_FLOAT d12 = sqrt(d12_sq);
        NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
        NEP_FLOAT fc12, fcp12;
        NEP_FLOAT fnp12[MAX_NUM_N];
        find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
        find_fnp(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fnp12);

        NEP_FLOAT f12[3] = {FLOAT_LIT(0.0)}, f21[3] = {FLOAT_LIT(0.0)};

        for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
            NEP_FLOAT gnp12 = FLOAT_LIT(0.0), gnp21 = FLOAT_LIT(0.0);
            for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
                gnp12 += fnp12[k] * param_c2[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
                gnp21 += fnp12[k] * param_c2[c_idx_J + n * paramb.basis_size_radial_plus1 + k];
            }
            NEP_FLOAT tmp12 = s_g_Fp[n] * gnp12 * d12inv; //g_Fp[atomi + n * nlocal]
            NEP_FLOAT tmp21 = FLOAT_LIT(0.0);
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
              atomicAdd(&g_virial[n2 * vatom_num + 0], -r12[0] * f12[0]);
              atomicAdd(&g_virial[n2 * vatom_num + 1], -r12[1] * f12[1]);
              atomicAdd(&g_virial[n2 * vatom_num + 2], -r12[2] * f12[2]);
              atomicAdd(&g_virial[n2 * vatom_num + 3], -r12[0] * f12[1]);
              atomicAdd(&g_virial[n2 * vatom_num + 4], -r12[0] * f12[2]);
              atomicAdd(&g_virial[n2 * vatom_num + 5], -r12[1] * f12[2]);
              if (cvflag_atom) {
                  atomicAdd(&g_virial[n2 * vatom_num + 6], -r12[1] * f12[0]);
                  atomicAdd(&g_virial[n2 * vatom_num + 7], -r12[2] * f12[0]);
                  atomicAdd(&g_virial[n2 * vatom_num + 8], -r12[2] * f12[1]);
              }
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
            if (cvflag_atom) {
                syxi += r12[1] * f21[0];
                szxi += r12[2] * f21[0];
                szyi += r12[2] * f21[1];
            }
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
        if (cvflag_atom) {
            s_syx[tid] = syxi;
            s_szx[tid] = szxi;
            s_szy[tid] = szyi;
        }
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
                if (cvflag_atom) {
                    s_syx[tid] += s_syx[tid + s];
                    s_szx[tid] += s_szx[tid + s];
                    s_szy[tid] += s_szy[tid + s];
                }
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
            atomicAdd(&g_virial[atomi * vatom_num + 0], s_sxx[0]);
            atomicAdd(&g_virial[atomi * vatom_num + 1], s_syy[0]);
            atomicAdd(&g_virial[atomi * vatom_num + 2], s_szz[0]);
            atomicAdd(&g_virial[atomi * vatom_num + 3], s_sxy[0]);
            atomicAdd(&g_virial[atomi * vatom_num + 4], s_sxz[0]);
            atomicAdd(&g_virial[atomi * vatom_num + 5], s_syz[0]);
            if (cvflag_atom) {
                atomicAdd(&g_virial[atomi * vatom_num + 6], s_syx[0]);
                atomicAdd(&g_virial[atomi * vatom_num + 7], s_szx[0]);
                atomicAdd(&g_virial[atomi * vatom_num + 8], s_szy[0]);
            }
        }
    }
}

static __global__ void find_bec_diagonal_lmp(
  const int N,
  const int nall,
  const int* __restrict__ g_ilist,
  const NEP_FLOAT* __restrict__ g_charge,
  NEP_FLOAT* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int atomi = g_ilist[n1];
    const NEP_FLOAT q = g_charge[atomi];
    g_bec[atomi + nall * 0] = q;
    g_bec[atomi + nall * 1] = FLOAT_LIT(0.0);
    g_bec[atomi + nall * 2] = FLOAT_LIT(0.0);
    g_bec[atomi + nall * 3] = FLOAT_LIT(0.0);
    g_bec[atomi + nall * 4] = q;
    g_bec[atomi + nall * 5] = FLOAT_LIT(0.0);
    g_bec[atomi + nall * 6] = FLOAT_LIT(0.0);
    g_bec[atomi + nall * 7] = FLOAT_LIT(0.0);
    g_bec[atomi + nall * 8] = q;
  }
}

static __global__ void scale_bec_lmp(
  const int nall,
  const NEP_FLOAT* __restrict__ sqrt_epsilon_inf,
  NEP_FLOAT* g_bec)
{
  const int atomi = blockIdx.x * blockDim.x + threadIdx.x;
  if (atomi < nall) {
    const NEP_FLOAT scale = sqrt_epsilon_inf[0];
    for (int d = 0; d < 9; ++d) {
      g_bec[atomi + nall * d] *= scale;
    }
  }
}

static __global__ void find_bec_radial_lmp(
  const int N,
  const int nlocal,
  const int nall,
  const int num_neigh,
  const int* g_numneigh,
  const int* g_firstneigh,
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  const NEP_FLOAT* __restrict__ g_charge_derivative,
  NEP_FLOAT* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int atomi = g_ilist[n1];
    const int t1 = g_type[atomi];
    const NEP_FLOAT x1 = g_pos[atomi * 3];
    const NEP_FLOAT y1 = g_pos[atomi * 3 + 1];
    const NEP_FLOAT z1 = g_pos[atomi * 3 + 2];
    const int c_start = paramb.num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

    for (int i1 = 0; i1 < g_numneigh[atomi]; ++i1) {
      const int n2 = g_firstneigh[i1 * num_neigh + atomi] & NEIGHMASK;
      const int t2 = g_type[n2];
      const int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

      NEP_FLOAT r12[3] = {g_pos[n2 * 3] - x1, g_pos[n2 * 3 + 1] - y1, g_pos[n2 * 3 + 2] - z1};
      const NEP_FLOAT d12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12_square > paramb.rc_radial_square) continue;
      const NEP_FLOAT d12 = sqrt(d12_square);
      const NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
      NEP_FLOAT fc12, fcp12;
      NEP_FLOAT fn12[MAX_NUM_N];
      NEP_FLOAT fnp12[MAX_NUM_N];
      NEP_FLOAT f12[3] = {FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)};

      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      find_fn_and_fnp(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        NEP_FLOAT gnp12 = FLOAT_LIT(0.0);
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          gnp12 += fnp12[k] * annmb.c[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
        }
        const NEP_FLOAT tmp12 = g_charge_derivative[atomi + n * nlocal] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      const NEP_FLOAT bec[9] = {
        FLOAT_LIT(0.5) * r12[0] * f12[0],
        FLOAT_LIT(0.5) * r12[0] * f12[1],
        FLOAT_LIT(0.5) * r12[0] * f12[2],
        FLOAT_LIT(0.5) * r12[1] * f12[0],
        FLOAT_LIT(0.5) * r12[1] * f12[1],
        FLOAT_LIT(0.5) * r12[1] * f12[2],
        FLOAT_LIT(0.5) * r12[2] * f12[0],
        FLOAT_LIT(0.5) * r12[2] * f12[1],
        FLOAT_LIT(0.5) * r12[2] * f12[2]};
      for (int d = 0; d < 9; ++d) {
        atomicAdd(&g_bec[atomi + nall * d], bec[d]);
        atomicAdd(&g_bec[n2 + nall * d], -bec[d]);
      }
    }
  }
}

static __global__ void find_bec_angular_lmp(
  const int N,
  const int nlocal,
  const int nall,
  const int* g_NN_angular,
  const int* g_NL_angular,
  NEPKK::ParaMB paramb,
  NEPKK::ANN annmb,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  const NEP_FLOAT* __restrict__ g_charge_derivative,
  const NEP_FLOAT* __restrict__ g_sum_fxyz,
  NEP_FLOAT* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const int atomi = g_ilist[n1];
    NEP_FLOAT Fp[MAX_DIM_ANGULAR] = {FLOAT_LIT(0.0)};
    NEP_FLOAT sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial_plus1 + d) * nlocal + atomi];
    }
    for (int d = 0; d < paramb.n_max_angular_plus1 * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * nlocal + atomi];
    }

    const int t1 = g_type[atomi];
    const NEP_FLOAT x1 = g_pos[atomi * 3];
    const NEP_FLOAT y1 = g_pos[atomi * 3 + 1];
    const NEP_FLOAT z1 = g_pos[atomi * 3 + 2];
    const int c_start = paramb.num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
      const int index = i1 * nlocal + atomi;
      const int n2 = g_NL_angular[index];
      const int t2 = g_type[n2];
      const int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
      NEP_FLOAT r12[3] = {g_pos[n2 * 3] - x1, g_pos[n2 * 3 + 1] - y1, g_pos[n2 * 3 + 2] - z1};
      const NEP_FLOAT d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      NEP_FLOAT fc12, fcp12;
      NEP_FLOAT fn12[MAX_NUM_N];
      NEP_FLOAT fnp12[MAX_NUM_N];
      NEP_FLOAT f12[3] = {FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)};

      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      find_fn_and_fnp(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
        NEP_FLOAT gn12 = FLOAT_LIT(0.0);
        NEP_FLOAT gnp12 = FLOAT_LIT(0.0);
        for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
          const int c_index = c_idx_I + n * paramb.basis_size_angular_plus1 + k + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        accumulate_f12(
          paramb.L_max,
          paramb.num_L,
          n,
          paramb.n_max_angular_plus1,
          d12,
          r12,
          gn12,
          gnp12,
          Fp,
          sum_fxyz,
          f12);
      }

      const NEP_FLOAT bec[9] = {
        FLOAT_LIT(0.5) * r12[0] * f12[0],
        FLOAT_LIT(0.5) * r12[0] * f12[1],
        FLOAT_LIT(0.5) * r12[0] * f12[2],
        FLOAT_LIT(0.5) * r12[1] * f12[0],
        FLOAT_LIT(0.5) * r12[1] * f12[1],
        FLOAT_LIT(0.5) * r12[1] * f12[2],
        FLOAT_LIT(0.5) * r12[2] * f12[0],
        FLOAT_LIT(0.5) * r12[2] * f12[1],
        FLOAT_LIT(0.5) * r12[2] * f12[2]};
      for (int d = 0; d < 9; ++d) {
        atomicAdd(&g_bec[atomi + nall * d], bec[d]);
        atomicAdd(&g_bec[n2 + nall * d], -bec[d]);
      }
    }
  }
}

static __global__ void backward_force_2b_nep(
  int vflag_either,
  int cvflag_atom,
  int vatom_num,
  NEPKK::ParaMB paramb,
  const NEP_FLOAT* param_c2,
  const int use_shared_c2,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int num_neigh, // the shape[0] of Neighbor List
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  const NEP_FLOAT* __restrict__ g_Fp,
  double* g_f,
  double* g_virial
  // double* g_total_virial
  )
{
  extern __shared__ NEP_FLOAT s_c2[];
  int c2_size = paramb.sim_num_types * paramb.sim_num_types *
    paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
  NEP_FLOAT* s_fnp12 = s_c2 + (use_shared_c2 ? c2_size : 0);
  if (use_shared_c2) {
    for (int i = threadIdx.x; i < c2_size; i += blockDim.x) {
      s_c2[i] = param_c2[i];
    }
  }
  __syncthreads();
  const NEP_FLOAT* c2 = use_shared_c2 ? s_c2 : param_c2;

  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT s_fx = FLOAT_LIT(0.0);
    NEP_FLOAT s_fy = FLOAT_LIT(0.0);
    NEP_FLOAT s_fz = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxx = FLOAT_LIT(0.0);
    NEP_FLOAT s_syy = FLOAT_LIT(0.0);
    NEP_FLOAT s_szz = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxy = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxz = FLOAT_LIT(0.0);
    NEP_FLOAT s_syz = FLOAT_LIT(0.0);
    NEP_FLOAT s_syx = FLOAT_LIT(0.0);
    NEP_FLOAT s_szx = FLOAT_LIT(0.0);
    NEP_FLOAT s_szy = FLOAT_LIT(0.0);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // int Fp_idx_start = atomi * annmb.dim;
    NEP_FLOAT Fp_atomi[MAX_NUM_N];
    for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
      Fp_atomi[n] = g_Fp[atomi + n * nlocal];
    }

    for (int i1 = 0; i1 < g_NN[atomi]; ++i1) {
      int n2_idx = i1 * num_neigh + atomi;
      int n2 = g_NL[n2_idx] & NEIGHMASK;
      int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      int c_idx_J = t2 * c_start + t1 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

      NEP_FLOAT r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      NEP_FLOAT d12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12_square > paramb.rc_radial_square) continue;
      NEP_FLOAT d12 = sqrt(d12_square);
      NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
      NEP_FLOAT f12[3] = {FLOAT_LIT(0.0)};
      NEP_FLOAT f21[3] = {FLOAT_LIT(0.0)};
      // if (0) printf("2b idx %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f\n", n1, atomi, t1, g_NN[atomi], n2, t2, d12);
      NEP_FLOAT fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      NEP_FLOAT* fnp12 = s_fnp12 + threadIdx.x;
      find_fnp_strided(
        paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, blockDim.x, fnp12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        NEP_FLOAT gnp12 = FLOAT_LIT(0.0);
        NEP_FLOAT gnp21 = FLOAT_LIT(0.0);
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          NEP_FLOAT fnp12_k = fnp12[k * blockDim.x];
          gnp12 += fnp12_k * c2[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
          gnp21 += fnp12_k * c2[c_idx_J + n * paramb.basis_size_radial_plus1 + k];// shape of c [N_max+1, N_base+1, I, J]
        }
        NEP_FLOAT descriptor_derivative12 = Fp_atomi[n];
        NEP_FLOAT tmp12 = descriptor_derivative12 * gnp12 * d12inv; //atomi + n * nlocal (dUi/diqn)*(diqn/drij)
        NEP_FLOAT tmp21 = FLOAT_LIT(0.0);
        if (n2 >= nlocal) {
          for (int d = 0; d < 3; ++d) {//编译器自动展开
            f12[d] += tmp12 * r12[d];
          }
        } else {
          NEP_FLOAT descriptor_derivative21 = g_Fp[n2 + n * nlocal];
          tmp21 = descriptor_derivative21 * gnp21 * d12inv; // (dUj/diqn)*(diqn/drij)
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
          atomicAdd(&g_virial[n2 * vatom_num + 0], -r12[0] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 1], -r12[1] * f12[1]);
          atomicAdd(&g_virial[n2 * vatom_num + 2], -r12[2] * f12[2]);
          atomicAdd(&g_virial[n2 * vatom_num + 3], -r12[0] * f12[1]);
          atomicAdd(&g_virial[n2 * vatom_num + 4], -r12[0] * f12[2]);
          atomicAdd(&g_virial[n2 * vatom_num + 5], -r12[1] * f12[2]);
          if(cvflag_atom) {
          atomicAdd(&g_virial[n2 * vatom_num + 6], -r12[1] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 7], -r12[2] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 8], -r12[2] * f12[1]);
          }
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
          if(cvflag_atom) {
          s_syx += r12[1] * f21[0];
          s_szx += r12[2] * f21[0];
          s_szy += r12[2] * f21[1];
          }
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
      g_virial[atomi * vatom_num + 0] += s_sxx;
      g_virial[atomi * vatom_num + 1] += s_syy;
      g_virial[atomi * vatom_num + 2] += s_szz;
      g_virial[atomi * vatom_num + 3] += s_sxy;
      g_virial[atomi * vatom_num + 4] += s_sxz;
      g_virial[atomi * vatom_num + 5] += s_syz;
      if(cvflag_atom) {
      g_virial[atomi * vatom_num + 6] += s_syx;
      g_virial[atomi * vatom_num + 7] += s_szx;
      g_virial[atomi * vatom_num + 8] += s_szy;
      }
    }
  }
} 

static __global__ void backward_force_2b_charge(
  int vflag_either,
  int cvflag_atom,
  int vatom_num,
  NEPKK::ParaMB paramb,
  const NEP_FLOAT* param_c2,
  const int use_shared_c2,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int num_neigh, // the shape[0] of Neighbor List
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  const NEP_FLOAT* __restrict__ g_Fp,
  const NEP_FLOAT* __restrict__ g_charge_derivative,
  const NEP_FLOAT* __restrict__ g_D_real,
  double* g_f,
  double* g_virial
  // double* g_total_virial
  )
{
  extern __shared__ NEP_FLOAT s_c2[];
  int c2_size = paramb.sim_num_types * paramb.sim_num_types *
    paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
  NEP_FLOAT* s_fnp12 = s_c2 + (use_shared_c2 ? c2_size : 0);
  if (use_shared_c2) {
    for (int i = threadIdx.x; i < c2_size; i += blockDim.x) {
      s_c2[i] = param_c2[i];
    }
  }
  __syncthreads();
  const NEP_FLOAT* c2 = use_shared_c2 ? s_c2 : param_c2;

  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT s_fx = FLOAT_LIT(0.0);
    NEP_FLOAT s_fy = FLOAT_LIT(0.0);
    NEP_FLOAT s_fz = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxx = FLOAT_LIT(0.0);
    NEP_FLOAT s_syy = FLOAT_LIT(0.0);
    NEP_FLOAT s_szz = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxy = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxz = FLOAT_LIT(0.0);
    NEP_FLOAT s_syz = FLOAT_LIT(0.0);
    NEP_FLOAT s_syx = FLOAT_LIT(0.0);
    NEP_FLOAT s_szx = FLOAT_LIT(0.0);
    NEP_FLOAT s_szy = FLOAT_LIT(0.0);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
    // int Fp_idx_start = atomi * annmb.dim;
    NEP_FLOAT Fp_atomi[MAX_NUM_N];
    for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
      Fp_atomi[n] = g_Fp[atomi + n * nlocal];
    }

    for (int i1 = 0; i1 < g_NN[atomi]; ++i1) {
      int n2_idx = i1 * num_neigh + atomi;
      int n2 = g_NL[n2_idx] & NEIGHMASK;
      int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;
      int c_idx_J = t2 * c_start + t1 * paramb.n_max_radial_plus1 * paramb.basis_size_radial_plus1;

      NEP_FLOAT r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      NEP_FLOAT d12_square = r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2];
      if (d12_square > paramb.rc_radial_square) continue;
      NEP_FLOAT d12 = sqrt(d12_square);
      NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
      NEP_FLOAT f12[3] = {FLOAT_LIT(0.0)};
      NEP_FLOAT f21[3] = {FLOAT_LIT(0.0)};
      // if (0) printf("2b idx %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f\n", n1, atomi, t1, g_NN[atomi], n2, t2, d12);
      NEP_FLOAT fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      NEP_FLOAT* fnp12 = s_fnp12 + threadIdx.x;
      find_fnp_strided(
        paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, blockDim.x, fnp12);
      for (int n = 0; n < paramb.n_max_radial_plus1; ++n) {
        NEP_FLOAT gnp12 = FLOAT_LIT(0.0);
        NEP_FLOAT gnp21 = FLOAT_LIT(0.0);
        for (int k = 0; k < paramb.basis_size_radial_plus1; ++k) {
          NEP_FLOAT fnp12_k = fnp12[k * blockDim.x];
          gnp12 += fnp12_k * c2[c_idx_I + n * paramb.basis_size_radial_plus1 + k];
          gnp21 += fnp12_k * c2[c_idx_J + n * paramb.basis_size_radial_plus1 + k];// shape of c [N_max+1, N_base+1, I, J]
        }
        NEP_FLOAT descriptor_derivative12 = Fp_atomi[n];
        if (paramb.charge_mode == 2) {
          descriptor_derivative12 += g_charge_derivative[atomi + n * nlocal] * g_D_real[atomi];
        }
        NEP_FLOAT tmp12 = descriptor_derivative12 * gnp12 * d12inv; //atomi + n * nlocal (dUi/diqn)*(diqn/drij)
        NEP_FLOAT tmp21 = FLOAT_LIT(0.0);
        if (n2 >= nlocal) {
          for (int d = 0; d < 3; ++d) {//编译器自动展开
            f12[d] += tmp12 * r12[d];
          }
        } else {
          NEP_FLOAT descriptor_derivative21 = g_Fp[n2 + n * nlocal];
          if (paramb.charge_mode == 2) {
            descriptor_derivative21 += g_charge_derivative[n2 + n * nlocal] * g_D_real[n2];
          }
          tmp21 = descriptor_derivative21 * gnp21 * d12inv; // (dUj/diqn)*(diqn/drij)
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
          atomicAdd(&g_virial[n2 * vatom_num + 0], -r12[0] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 1], -r12[1] * f12[1]);
          atomicAdd(&g_virial[n2 * vatom_num + 2], -r12[2] * f12[2]);
          atomicAdd(&g_virial[n2 * vatom_num + 3], -r12[0] * f12[1]);
          atomicAdd(&g_virial[n2 * vatom_num + 4], -r12[0] * f12[2]);
          atomicAdd(&g_virial[n2 * vatom_num + 5], -r12[1] * f12[2]);
          if(cvflag_atom) {
          atomicAdd(&g_virial[n2 * vatom_num + 6], -r12[1] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 7], -r12[2] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 8], -r12[2] * f12[1]);
          }
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
          if(cvflag_atom) {
          s_syx += r12[1] * f21[0];
          s_szx += r12[2] * f21[0];
          s_szy += r12[2] * f21[1];
          }
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
      g_virial[atomi * vatom_num + 0] += s_sxx;
      g_virial[atomi * vatom_num + 1] += s_syy;
      g_virial[atomi * vatom_num + 2] += s_szz;
      g_virial[atomi * vatom_num + 3] += s_sxy;
      g_virial[atomi * vatom_num + 4] += s_sxz;
      g_virial[atomi * vatom_num + 5] += s_syz;
      if(cvflag_atom) {
      g_virial[atomi * vatom_num + 6] += s_syx;
      g_virial[atomi * vatom_num + 7] += s_szx;
      g_virial[atomi * vatom_num + 8] += s_szy;
      }
    }
  }
} 

__global__ void backward_force_3b_per_atom_sharemem(
    NEPKK::ParaMB paramb,
    const NEP_FLOAT* param_c3,
    int N,
    int nlocal,
    const int* g_NN_angular,
    const int* g_NL_angular,
    const int* __restrict__ g_ilist,
    const int* __restrict__ g_type,
    const NEP_FLOAT* __restrict__ g_pos,
    const NEP_FLOAT* __restrict__ g_Fp,
    const NEP_FLOAT* __restrict__ g_sum_fxyz,
    NEP_FLOAT* g_f12x,
    NEP_FLOAT* g_f12y,
    NEP_FLOAT* g_f12z
) {
    extern __shared__ NEP_FLOAT shmem[];
    int i = blockIdx.x;  // 每个 block 负责一个中心原子
    if (i >= N) return;

    int atomi = g_ilist[i];
    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);

    NEP_FLOAT* s_x1       = shmem + 0;                           // 1
    NEP_FLOAT* s_y1       = shmem + 1;                           // 1
    NEP_FLOAT* s_z1       = shmem + 2;                           // 1
    NEP_FLOAT* s_Fp       = shmem + 3;                           // dim_angular
    NEP_FLOAT* s_sum_fxyz = shmem + 3 + paramb.dim_angular;      // n_max_angular_plus1 * NUM_OF_ABC
  // 每个线程独占的 fn12 和 fnp12 空间
    const int per_thread_fn_size = MAX_NUM_N * 2;            // fn12 + fnp12 = 20 + 20 = 40
    NEP_FLOAT* s_fn_base = s_sum_fxyz + (paramb.n_max_angular_plus1 * NUM_OF_ABC);
    NEP_FLOAT* s_fn  = s_fn_base + threadIdx.x * per_thread_fn_size;
    NEP_FLOAT* s_fnp = s_fn + MAX_NUM_N;
    NEP_FLOAT* s_c3 = s_fn_base + blockDim.x * per_thread_fn_size;
    const int c3_size = paramb.sim_num_types * paramb.sim_num_types *
      paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;

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
            s_Fp[d] = g_Fp[(paramb.n_max_radial_plus1 + d) * nlocal + atomi];
        }
        int sum_size = paramb.n_max_angular_plus1 * NUM_OF_ABC;
        // 加载 sum_fxyz
        for (int d = threadIdx.x; d < sum_size; d += blockDim.x) {
            s_sum_fxyz[d] = g_sum_fxyz[d * nlocal + atomi];
        }
        for (int d = threadIdx.x; d < c3_size; d += blockDim.x) {
            s_c3[d] = param_c3[d];
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
        int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);

        NEP_FLOAT r12[3] = {
            g_pos[n2 * 3 + 0] - s_x1[0],
            g_pos[n2 * 3 + 1] - s_y1[0],
            g_pos[n2 * 3 + 2] - s_z1[0]
        };

        NEP_FLOAT d12 = sqrt(r12[0]*r12[0] + r12[1]*r12[1] + r12[2]*r12[2]);
        if (d12 > paramb.rc_angular) continue;  // 可选：加 cutoff 检查

        NEP_FLOAT fc12, fcp12;
        find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
        // 直接使用共享内存中的 fn12 和 fnp12
        find_fn_and_fnp(
            paramb.basis_size_angular,
            paramb.rcinv_angular,
            d12, fc12, fcp12,
            s_fn, s_fnp
        );

        int c_start = paramb.sim_num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
        NEP_FLOAT f12_local[3] = {FLOAT_LIT(0.0), FLOAT_LIT(0.0), FLOAT_LIT(0.0)};
        for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
            NEP_FLOAT gn12  = FLOAT_LIT(0.0);
            NEP_FLOAT gnp12 = FLOAT_LIT(0.0);

            for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
                int idx = c_idx_I + n * paramb.basis_size_angular_plus1 + k;
                gn12  += s_fn[k]  * s_c3[idx];
                gnp12 += s_fnp[k] * s_c3[idx];
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
  const NEP_FLOAT* param_c3,
  const int use_shared_c3,
  const int N,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  const NEP_FLOAT* __restrict__ g_Fp,
  const NEP_FLOAT* __restrict__ g_charge_derivative,
  const NEP_FLOAT* __restrict__ g_D_real,
  const NEP_FLOAT* __restrict__ g_sum_fxyz,
  NEP_FLOAT* g_f12x,
  NEP_FLOAT* g_f12y,
  NEP_FLOAT* g_f12z
  )
{
  extern __shared__ NEP_FLOAT s_c3[];
  const int c3_size = paramb.sim_num_types * paramb.sim_num_types *
    paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
  const int fn_size = blockDim.x * paramb.basis_size_angular_plus1;
  NEP_FLOAT* s_fn = s_c3 + (use_shared_c3 ? c3_size : 0);
  NEP_FLOAT* s_fnp = s_fn + fn_size;
  if (use_shared_c3) {
    for (int i = threadIdx.x; i < c3_size; i += blockDim.x) {
      s_c3[i] = param_c3[i];
    }
  }
  __syncthreads();
  const NEP_FLOAT* c3 = use_shared_c3 ? s_c3 : param_c3;

  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    int atomi = g_ilist[n1];

    NEP_FLOAT Fp[MAX_DIM_ANGULAR] = {FLOAT_LIT(0.0)};
    NEP_FLOAT sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      NEP_FLOAT descriptor_derivative = g_Fp[(paramb.n_max_radial_plus1 + d) * nlocal + atomi];
      if (paramb.charge_mode == 2) {
        descriptor_derivative +=
          g_charge_derivative[(paramb.n_max_radial_plus1 + d) * nlocal + atomi] * g_D_real[atomi];
      }
      Fp[d] = descriptor_derivative;
    }
    for (int d = 0; d < paramb.n_max_angular_plus1 * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * nlocal + atomi];
    }

    int t1 = static_cast<int>(paramb.nep_to_sim[g_type[atomi]]);
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int c_start = paramb.sim_num_types * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
    for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
      int index = i1 * nlocal + atomi;
      int n2 = g_NL_angular[index];
      int t2 = static_cast<int>(paramb.nep_to_sim[g_type[n2]]);
      
      NEP_FLOAT f12[3] = {FLOAT_LIT(0.0)};
      int c_idx_I = t1 * c_start + t2 * paramb.n_max_angular_plus1 * paramb.basis_size_angular_plus1;
      NEP_FLOAT r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      NEP_FLOAT d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      // if (0) printf("3bhalf idx %d atomi %d t1 %d jnums %d n2 %d t2 %d r12 %f fixyz %f %f %f fjxyz %f %f %f\n", n1, atomi, t1, g_NN_angular[atomi], n2, t2, d12, x1, y1, z1, g_pos[n2*3], g_pos[n2*3+1], g_pos[n2*3+2]);
      NEP_FLOAT fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      NEP_FLOAT* fn12 = s_fn + threadIdx.x;
      NEP_FLOAT* fnp12 = s_fnp + threadIdx.x;
      find_fn_and_fnp_strided(
        paramb.basis_size_angular,
        paramb.rcinv_angular,
        d12,
        fc12,
        fcp12,
        blockDim.x,
        fn12,
        fnp12);
      for (int n = 0; n < paramb.n_max_angular_plus1; ++n) {
        NEP_FLOAT gn12 = FLOAT_LIT(0.0);
        NEP_FLOAT gnp12 = FLOAT_LIT(0.0);
        for (int k = 0; k < paramb.basis_size_angular_plus1; ++k) {
          const int basis_index = k * blockDim.x;
          const NEP_FLOAT coefficient = c3[c_idx_I + n * paramb.basis_size_angular_plus1 + k];
          gn12 += fn12[basis_index] * coefficient;
          gnp12 += fnp12[basis_index] * coefficient;
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
  int cvflag_atom,
  int vatom_num,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const NEP_FLOAT* __restrict__ g_f12x,
  const NEP_FLOAT* __restrict__ g_f12y,
  const NEP_FLOAT* __restrict__ g_f12z,
  const int* __restrict__ g_ilist,
  const NEP_FLOAT* __restrict__ g_pos,
  double* g_f,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  NEP_FLOAT s_fx = FLOAT_LIT(0.0);  // force_x
  NEP_FLOAT s_fy = FLOAT_LIT(0.0);  // force_y
  NEP_FLOAT s_fz = FLOAT_LIT(0.0);  // force_z
  NEP_FLOAT s_sxx = FLOAT_LIT(0.0);
  NEP_FLOAT s_syy = FLOAT_LIT(0.0);
  NEP_FLOAT s_szz = FLOAT_LIT(0.0);
  NEP_FLOAT s_sxy = FLOAT_LIT(0.0);
  NEP_FLOAT s_sxz = FLOAT_LIT(0.0);
  NEP_FLOAT s_syz = FLOAT_LIT(0.0);
  NEP_FLOAT s_syx = FLOAT_LIT(0.0);
  NEP_FLOAT s_szx = FLOAT_LIT(0.0);
  NEP_FLOAT s_szy = FLOAT_LIT(0.0);
  if (n1 < N) {
    int atomi = g_ilist[n1];
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];

    for (int i1 = 0; i1 < g_NN_angular[atomi]; ++i1) {
      int index = i1 * nlocal + atomi;
      int n2 = g_NL_angular[index];
      NEP_FLOAT x12 = g_pos[n2*3]   - x1;
      NEP_FLOAT y12 = g_pos[n2*3+1] - y1;
      NEP_FLOAT z12 = g_pos[n2*3+2] - z1;

      NEP_FLOAT r12[3] = {x12, y12, z12};
      NEP_FLOAT f12x = g_f12x[index];
      NEP_FLOAT f12y = g_f12y[index];
      NEP_FLOAT f12z = g_f12z[index];

      if (n2 < nlocal) {
        NEP_FLOAT f21x = FLOAT_LIT(0.0);
        NEP_FLOAT f21y = FLOAT_LIT(0.0);
        NEP_FLOAT f21z = FLOAT_LIT(0.0);
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
          if(cvflag_atom) {
          s_syx += y12 * f21x;
          s_szx += z12 * f21x;
          s_szy += z12 * f21y;
          }
        }
      } else {
        s_fx += f12x;
        s_fy += f12y;
        s_fz += f12z;
        atomicAdd(&g_f[n2*3], double(-f12x));// ghost atom
        atomicAdd(&g_f[n2*3+1], double(-f12y));
        atomicAdd(&g_f[n2*3+2], double(-f12z));
        if(vflag_either) {
          atomicAdd(&g_virial[n2 * vatom_num + 0], -r12[0] * f12x);
          atomicAdd(&g_virial[n2 * vatom_num + 1], -r12[1] * f12y);
          atomicAdd(&g_virial[n2 * vatom_num + 2], -r12[2] * f12z);
          atomicAdd(&g_virial[n2 * vatom_num + 3], -r12[0] * f12y);
          atomicAdd(&g_virial[n2 * vatom_num + 4], -r12[0] * f12z);
          atomicAdd(&g_virial[n2 * vatom_num + 5], -r12[1] * f12z);
          if(cvflag_atom){
          atomicAdd(&g_virial[n2 * vatom_num + 6], -r12[1] * f12x);
          atomicAdd(&g_virial[n2 * vatom_num + 7], -r12[2] * f12x);
          atomicAdd(&g_virial[n2 * vatom_num + 8], -r12[2] * f12y);
          }
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
      g_virial[atomi * vatom_num + 0] += s_sxx;
      g_virial[atomi * vatom_num + 1] += s_syy;
      g_virial[atomi * vatom_num + 2] += s_szz;
      g_virial[atomi * vatom_num + 3] += s_sxy;
      g_virial[atomi * vatom_num + 4] += s_sxz;
      g_virial[atomi * vatom_num + 5] += s_syz;
      if(cvflag_atom) {
      g_virial[atomi * vatom_num + 6] += s_syx;
      g_virial[atomi * vatom_num + 7] += s_szx;
      g_virial[atomi * vatom_num + 8] += s_szy;
      }
    }
  }
}
template<int NUM_COMP>
__global__ void calculate_partial_virial(const double* virial, double* partial_virial, int N) {
    extern __shared__ double shared_virial[];
    const int tid = threadIdx.x;
    const int index = blockIdx.x * blockDim.x + tid;

    for (int comp = 0; comp < NUM_COMP; ++comp) {
        shared_virial[comp * blockDim.x + tid] = (index < N) ? virial[index * NUM_COMP + comp] : 0.0;
    }
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            for (int comp = 0; comp < NUM_COMP; ++comp) {
                shared_virial[comp * blockDim.x + tid] += shared_virial[comp * blockDim.x + tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < NUM_COMP) {
        partial_virial[blockIdx.x * NUM_COMP + tid] = shared_virial[tid * blockDim.x];
    }
}

template<int NUM_COMP>
__global__ void finalize_total_virial(const double* partial_virial, double* total_virial, int num_blocks) {
    extern __shared__ double shared_sum[];
    const int tid = threadIdx.x;
    const int comp = blockIdx.x;
    double sum = 0.0;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_virial[i * NUM_COMP + comp];
    }
    shared_sum[tid] = sum;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        total_virial[comp] = shared_sum[0];
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

__global__ void doubleTofloat(NEP_FLOAT* dest, double* src, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] = static_cast<NEP_FLOAT>(src[idx]);
    }
}

// [Nmax, Nbase, TYPE, TYPE] to [TYPEi, TYPEj, Nmax, Nbase]
__global__ void convert_c_dim(NEP_FLOAT* c, NEP_FLOAT* temp, int NtypeI, int Nmax, int Nbase) {
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
    const int* __restrict__ g_ilist, //[0, 1, 2, ..., n] len=inum
    const int* __restrict__ g_type,  //[1, 1, 1, 2, 2, 1,...] such as HfO2, type order same as lmp.config type1 = O, type2=Hf
    const int* __restrict__ type_map, //atom type order of nep.txt,such as 'nep4 2 Hf O'  in nep.txt -> type_map = [1, 0]
    int* copy_ilist,
    int* cvt_type_map) 
{ 
    int n1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (n1 >= nall) return;
    int t1 = type_map[g_type[n1]-1];
    // if(n1 < inum) printf("atomi %d g_type[%d]=%d, t1=%d\n", n1, n1, g_type[n1], t1);
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
  int cvflag_atom,
  int vatom_num,
  const int nall, //all atoms
  const int N,
  const int nlocal,
  const int MN_angular,
  const bool zbl_flexibled,
  const NEP_FLOAT zbl_rc_inner,
  const NEP_FLOAT zbl_rc_outer,
  const bool use_typewise_cutoff_zbl,
  const NEP_FLOAT typewise_cutoff_zbl_factor,
  const int zbl_num_types,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_type,
  const NEP_FLOAT* __restrict__ g_pos,
  const int* __restrict__ g_atomic_numbers,
  const NEP_FLOAT* __restrict__ g_para,
  double* g_f,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    NEP_FLOAT s_pe = FLOAT_LIT(0.0);
    NEP_FLOAT s_fx = FLOAT_LIT(0.0);
    NEP_FLOAT s_fy = FLOAT_LIT(0.0);
    NEP_FLOAT s_fz = FLOAT_LIT(0.0);

    NEP_FLOAT s_sxx = FLOAT_LIT(0.0);
    NEP_FLOAT s_syy = FLOAT_LIT(0.0);
    NEP_FLOAT s_szz = FLOAT_LIT(0.0);

    NEP_FLOAT s_sxy = FLOAT_LIT(0.0);
    NEP_FLOAT s_sxz = FLOAT_LIT(0.0);
    NEP_FLOAT s_syz = FLOAT_LIT(0.0);

    NEP_FLOAT s_syx = FLOAT_LIT(0.0);
    NEP_FLOAT s_szx = FLOAT_LIT(0.0);
    NEP_FLOAT s_szy = FLOAT_LIT(0.0);
    int atomi = g_ilist[n1];
    NEP_FLOAT x1 = g_pos[atomi*3  ];
    NEP_FLOAT y1 = g_pos[atomi*3+1];
    NEP_FLOAT z1 = g_pos[atomi*3+2];
    int type1 = g_type[atomi];
    int zi = g_atomic_numbers[type1];
    NEP_FLOAT zi_fp = static_cast<NEP_FLOAT>(zi);
    NEP_FLOAT pow_zi = pow(zi_fp, FLOAT_LIT(0.23));
    for (int i1 = 0; i1 < g_NN[atomi]; ++i1) {
      int n2 = g_NL[atomi + nlocal * i1];
      int type2 = g_type[n2];
      NEP_FLOAT r12[3] = {g_pos[n2*3] - x1, g_pos[n2*3+1] - y1, g_pos[n2*3+2] - z1};
      NEP_FLOAT d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      NEP_FLOAT d12inv = FLOAT_LIT(1.0) / d12;
      NEP_FLOAT f, fp;
      int zj = g_atomic_numbers[type2];
      NEP_FLOAT zj_fp = static_cast<NEP_FLOAT>(zj);
      NEP_FLOAT a_inv = (pow_zi + pow(zj_fp, FLOAT_LIT(0.23))) * FLOAT_LIT(2.134563);
      NEP_FLOAT zizj = K_C_SP * zi * zj;
      if (zbl_flexibled) {
        int t1, t2;
        if (type1 < type2) {
          t1 = type1;
          t2 = type2;
        } else {
          t1 = type2;
          t2 = type1;
        }
        int zbl_index = t1 * zbl_num_types - (t1 * (t1 - 1)) / 2 + (t2 - t1);
        NEP_FLOAT ZBL_para[10];
        for (int i = 0; i < 10; ++i) {
          ZBL_para[i] = g_para[10 * zbl_index + i];
        }
        find_f_and_fp_zbl(ZBL_para, zizj, a_inv, d12, d12inv, f, fp);
      } else {
        NEP_FLOAT rc_outer = zbl_rc_outer;
        if (use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min((COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * typewise_cutoff_zbl_factor, rc_outer);
        }
        find_f_and_fp_zbl(zizj, a_inv, zbl_rc_inner, rc_outer, d12, d12inv, f, fp); // if use typewise_cutoff_zbl_factor, the rc_inner=0.0 when read the nep.txt
      }
      NEP_FLOAT f2 = fp * d12inv * FLOAT_LIT(0.5);
      NEP_FLOAT f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      NEP_FLOAT f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      // if (0) printf("zbl n1 %d n2 %d d12 %f e_c_half %f\n", n1, n2, d12, f);
      if (n2 >= nlocal) {
        s_fx += f12[0];
        s_fy += f12[1];
        s_fz += f12[2];
        atomicAdd(&g_f[n2*3], double(f21[0]));// ghost atom
        atomicAdd(&g_f[n2*3+1], double(f21[1]));
        atomicAdd(&g_f[n2*3+2], double(f21[2]));
        if (vflag_either) {
          atomicAdd(&g_virial[n2 * vatom_num + 0], -r12[0] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 1], -r12[1] * f12[1]);
          atomicAdd(&g_virial[n2 * vatom_num + 2], -r12[2] * f12[2]);
          atomicAdd(&g_virial[n2 * vatom_num + 3], -r12[0] * f12[1]);
          atomicAdd(&g_virial[n2 * vatom_num + 4], -r12[0] * f12[2]);
          atomicAdd(&g_virial[n2 * vatom_num + 5], -r12[1] * f12[2]);
          if(cvflag_atom){
          atomicAdd(&g_virial[n2 * vatom_num + 6], -r12[1] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 7], -r12[2] * f12[0]);
          atomicAdd(&g_virial[n2 * vatom_num + 8], -r12[2] * f12[1]);
          }
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
          if(cvflag_atom){
          s_syx += r12[1] * f21[0];
          s_szx += r12[2] * f21[0];
          s_szy += r12[2] * f21[1];
          }
        }
      }
      s_pe += f * 0.5f;
    }
    g_f[atomi*3] += s_fx;
    g_f[atomi*3+1] += s_fy;
    g_f[atomi*3+2] += s_fz;
    if (vflag_either) {
      g_virial[atomi * vatom_num + 0] += s_sxx;
      g_virial[atomi * vatom_num + 1] += s_syy;
      g_virial[atomi * vatom_num + 2] += s_szz;
      g_virial[atomi * vatom_num + 3] += s_sxy;
      g_virial[atomi * vatom_num + 4] += s_sxz;
      g_virial[atomi * vatom_num + 5] += s_syz;
      if(cvflag_atom){
      g_virial[atomi * vatom_num + 6] += s_syx;
      g_virial[atomi * vatom_num + 7] += s_szx;
      g_virial[atomi * vatom_num + 8] += s_szy;
      }
    }
    g_pe[atomi] += s_pe;
  }
}

// calculate_total_CVirial uses the same two-stage reduction kernels as calculate_total_virial.

__global__ void virial9To6Kernel(
    const double* __restrict__ virial9,  // 输入：N*9 的9分量virial数组（顺序：xx,yy,zz,xy,xz,yz,yx,zx,zy）
    double* __restrict__ virial6,        // 输出：N*6 的6分量virial数组（顺序：xx,yy,zz,xy,xz,yz）
    size_t N)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    size_t base9 = i * 9;
    size_t base6 = i * 6;
    // =============== 精确映射===============
    // 9分量顺序： [0]=xx, [1]=yy, [2]=zz, [3]=xy, [4]=xz, [5]=yz, [6]=yx, [7]=zx, [8]=zy
    // 6分量顺序： [0]=xx, [1]=yy, [2]=zz, [3]=xy, [4]=xz, [5]=yz
    // =======================================
    virial6[base6 + 0] = virial9[base9 + 0];  // xx
    virial6[base6 + 1] = virial9[base9 + 1];  // yy
    virial6[base6 + 2] = virial9[base9 + 2];  // zz
    virial6[base6 + 3] = virial9[base9 + 3];  // xy
    virial6[base6 + 4] = virial9[base9 + 4];  // xz
    virial6[base6 + 5] = virial9[base9 + 5];  // yz
}
