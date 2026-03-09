/*
This code is developed based on the GPUMD source code and adds ghost atom processing in LAMMPS. 
  Support multi GPUs.
  Support GPUMD NEP shared bias and MATPL NEP independent bias forcefield.

We have made the following improvements based on NEP4
http://doc.lonxun.com/PWMLFF/models/nep/NEP%20model/
*/

/*
    the open source code from https://github.com/brucefan1983/GPUMD
    the licnese of NEP_CPU is as follows:

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

#include "nep3.cuh"
#include "utilities/common.cuh"
#include "utilities/nep_utilities.cuh"
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

static __global__ void find_neighbor_list_large_box(
  NEP3::ParaMB paramb,
  const int n_all,
  const int N, // inum
  const int N1, // 0
  const int NM,
  const int* __restrict__ g_type,
  const int* __restrict__ g_ilist,
  const int* __restrict__ g_numneigh,
  const int* __restrict__ g_firstneigh,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int count_radial = 0;
    int count_angular = 0;
    int i = g_ilist[n1];
    int jnum = g_numneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      int n2 = g_firstneigh[n1*NM + jj];
      double x12 = g_x[n2] - g_x[n1];
      double y12 = g_y[n2] - g_y[n1];
      double z12 = g_z[n2] - g_z[n1];
      float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);

      if (distance_square < paramb.rc_radial * paramb.rc_radial) {
        g_NL_radial[ count_radial * N + n1] = n2;
                count_radial++;
      }
      if (distance_square < paramb.rc_angular * paramb.rc_angular) {
        g_NL_angular[count_angular * N + n1] = n2;
                count_angular++;
      }
    } //for
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }// if
}

static __global__ void find_descriptor_large_box(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int N1,
  const int nlocal,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  // const bool is_polarizability,
#ifdef USE_TABLE
  const float* __restrict__ g_gn_radial,
  const float* __restrict__ g_gn_angular,
#endif
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    // get radial descriptors
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int n2 = g_NL_radial[i1 * N + n1];
      int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);
      float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        q[n] +=
          g_gn_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gn_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
      }
#else
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      float fn12[MAX_NUM_N];
      if (paramb.version == 2) {
        find_fn(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float c = (paramb.num_types == 1)
                      ? 1.0f
                      : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          q[n] += fn12[n] * c;
        }
      } else {
        find_fn(paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fn12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          q[n] += gn12;
        }
      }
#endif
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int n2 = g_NL_angular[n1 + N * i1];
        double x12double = g_x[n2] - x1;
        double y12double = g_y[n2] - y1;
        double z12double = g_z[n2] - z1;
        float x12 = float(x12double);
        float y12 = float(y12double);
        float z12 = float(z12double);
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
#ifdef USE_TABLE
        int index_left, index_right;
        float weight_left, weight_right;
        find_index_and_weight(
          d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
        int t12 = t1 * paramb.num_types + g_type[n2];
        float gn12 =
          g_gn_angular[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_left +
          g_gn_angular[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n] *
            weight_right;
        accumulate_s(d12, r12[0], r12[1], r12[2], gn12, s);
#else
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
        int t2 = g_type[n2];
        if (paramb.version == 2) {
          float fn;
          find_fn(n, paramb.rcinv_angular, d12, fc12, fn);
          fn *=
            (paramb.num_types == 1)
              ? 1.0f
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          accumulate_s(d12, x12, y12, z12, fn, s);
        } else {
          float fn12[MAX_NUM_N];
          find_fn(paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fn12);
          float gn12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_angular; ++k) {
            int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
            gn12 += fn12[k] * annmb.c[c_index];
          }
          accumulate_s(d12, x12, y12, z12, gn12, s);
        }
#endif
      }
      if (paramb.num_L == paramb.L_max) {
        find_q(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else if (paramb.num_L == paramb.L_max + 1) {
        find_q_with_4body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else {
        find_q_with_5body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1] = s[abc];
        // printf("g_sum_fxyz n1=%d g_sum_fxyz[%d]=%f\n",n1, (n * NUM_OF_ABC + abc) * nlocal + n1, g_sum_fxyz[(n * NUM_OF_ABC + abc) * nlocal + n1]);
      } 
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }
    // get energy and energy gradient
    float F = 0.0f, Fp[MAX_DIM] = {0.0f};

    // if (is_polarizability) {
    //   apply_ann_one_layer(
    //     annmb.dim,
    //     annmb.num_neurons1,
    //     annmb.w0_pol[t1],
    //     annmb.b0_pol[t1],
    //     annmb.w1_pol[t1],
    //     annmb.b1_pol,
    //     q,
    //     F,
    //     Fp,
    //     t1);
    //   // Add the potential values to the diagonal of the virial
    //   g_virial[n1] = F;
    //   g_virial[n1 + N * 1] = F;
    //   g_virial[n1 + N * 2] = F;

    //   F = 0.0f;
    //   for (int d = 0; d < annmb.dim; ++d) {
    //     Fp[d] = 0.0f;
    //   }
    // }
    if (paramb.version == 4) {
      apply_ann_one_layer(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp, t1);
    }else if(paramb.version == 5) {
      apply_ann_one_layer_nep5(
        annmb.dim, annmb.num_neurons1, annmb.w0[t1], annmb.b0[t1], annmb.w1[t1], annmb.b1, q, F, Fp, t1);    
    }

    g_pe[n1] += F;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * nlocal + n1] = Fp[d] * paramb.q_scaler[d];
      // printf("g_Fp n1=%d g_Fp[%d]=%f\n",n1, d * nlocal + n1, g_Fp[d * nlocal + n1]);
    }
  }
}

static __global__ void find_force_radial_large_box(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int nall, //all atoms
  const int N,
  const int N1,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  // const bool is_dipole,
#ifdef USE_TABLE
  const float* __restrict__ g_gnp_radial,
#endif
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial
  // double* g_total_virial
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int t1 = g_type[n1];
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float f21[3] = {0.0f};

#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_radial, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + t2;
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 =
          g_gnp_radial[(index_left * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_left +
          g_gnp_radial[(index_right * paramb.num_types_sq + t12) * (paramb.n_max_radial + 1) + n] *
            weight_right;
        float tmp12 = g_Fp[n1 + n * nlocal] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }
#else
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];

      find_fn_and_fnp(
        paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        float gnp21 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          gnp12 += fnp12[k] * annmb.c[c_index + t1 * paramb.num_types + t2];
          gnp21 += fnp12[k] * annmb.c[c_index + t2 * paramb.num_types + t1];// shape of c [N_max+1, N_base+1, I, J]
        }
        float tmp12 = g_Fp[n1 + n * nlocal] * gnp12 * d12inv;
        float tmp21 = 0;
        if (n2 >= N) {
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        } else {
          tmp21 = g_Fp[n2 + n * nlocal] * gnp21 * d12inv;
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
            f21[d] -= tmp21 * r12[d];
          }
        }
      }
      // printf("n1=%d, n2=%d, r12=%f, f12[0]=%f, f12[1]=%f, f12[2]=%f\n",n1, n2, d12, f12[0],f12[1],f12[2]);
#endif
      if (n2 >= N) {
        s_fx += f12[0];
        s_fy += f12[1];
        s_fz += f12[2];
        atomicAdd(&g_fx[n2], double(-f12[0]));// ghost atom
        atomicAdd(&g_fy[n2], double(-f12[1]));
        atomicAdd(&g_fz[n2], double(-f12[2]));
        
        atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12[0]);
        atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12[1]);
        atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12[2]);
        atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12[1]);
        atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12[2]);
        atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12[2]);
        atomicAdd(&g_virial[n2 + 6 * nall], -r12[1] * f12[0]);
        atomicAdd(&g_virial[n2 + 7 * nall], -r12[2] * f12[0]);
        atomicAdd(&g_virial[n2 + 8 * nall], -r12[2] * f12[1]);
      } else {
        // 对于i 的 ghost 部分应该加上哪些东西？
        s_fx += f12[0] - f21[0];
        s_fy += f12[1] - f21[1];
        s_fz += f12[2] - f21[2];

        s_sxx += r12[0] * f21[0];
        s_syy += r12[1] * f21[1];
        s_szz += r12[2] * f21[2];

        s_sxy += r12[0] * f21[1];
        s_sxz += r12[0] * f21[2];
        s_syx += r12[1] * f21[0];
        s_syz += r12[1] * f21[2];
        s_szx += r12[2] * f21[0];
        s_szy += r12[2] * f21[1];
      }
    }

    //对于ghost atom，需要加上ghost atom 对应的force；并且保存ghost atom 对应的force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * nall] += s_sxx;
    g_virial[n1 + 1 * nall] += s_syy;
    g_virial[n1 + 2 * nall] += s_szz;
    g_virial[n1 + 3 * nall] += s_sxy;
    g_virial[n1 + 4 * nall] += s_sxz;
    g_virial[n1 + 5 * nall] += s_syz;
    g_virial[n1 + 6 * nall] += s_syx;
    g_virial[n1 + 7 * nall] += s_szx;
    g_virial[n1 + 8 * nall] += s_szy;
  }
}

static __global__ void find_partial_force_angular_large_box(
  NEP3::ParaMB paramb,
  NEP3::ANN annmb,
  const int N,
  const int N1,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_sum_fxyz,
#ifdef USE_TABLE
  const float* __restrict__ g_gn_angular,
  const float* __restrict__ g_gnp_angular,
#endif
  float* g_f12x,
  float* g_f12y,
  float* g_f12z
  // double* g_virial,
  // double* g_total_virial
  )
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    // if (n1 == 0){
    //   printf("find_force_angular_large_box n1 = 0\n");
    // }
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_Fp[(paramb.n_max_radial + 1 + d) * nlocal + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * nlocal + n1];
    }

    int t1 = g_type[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int t2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};

#ifdef USE_TABLE
      int index_left, index_right;
      float weight_left, weight_right;
      find_index_and_weight(
        d12 * paramb.rcinv_angular, index_left, index_right, weight_left, weight_right);
      int t12 = t1 * paramb.num_types + g_type[n2];
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        int index_left_all =
          (index_left * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        int index_right_all =
          (index_right * paramb.num_types_sq + t12) * (paramb.n_max_angular + 1) + n;
        float gn12 =
          g_gn_angular[index_left_all] * weight_left + g_gn_angular[index_right_all] * weight_right;
        float gnp12 = g_gnp_angular[index_left_all] * weight_left +
                      g_gnp_angular[index_right_all] * weight_right;
        if (paramb.num_L == paramb.L_max) {
          accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
        } else if (paramb.num_L == paramb.L_max + 1) {
          accumulate_f12_with_4body(
            n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
        } else {
          accumulate_f12_with_5body(
            n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
        }
      }
#else
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      find_fn_and_fnp(
        paramb.basis_size_angular, paramb.rcinv_angular, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_angular; ++n) {
        float gn12 = 0.0f;
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_angular; ++k) {
          int c_index = (n * (paramb.basis_size_angular + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2 + paramb.num_c_radial;
          gn12 += fn12[k] * annmb.c[c_index];
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        if (paramb.num_L == paramb.L_max) {
          accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
        } else if (paramb.num_L == paramb.L_max + 1) {
          accumulate_f12_with_4body(
            n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
        } else {
          accumulate_f12_with_5body(
            n, paramb.n_max_angular + 1, d12, r12, gn12, gnp12, Fp, sum_fxyz, f12);
        }
      }
      // printf("angular n1=%d, n2=%d, d12=%f, f12=[%f, %f, %f]\n", n1, n2, d12, f12[0], f12[1], f12[2]);
#endif
      g_f12x[index] = f12[0];
      g_f12y[index] = f12[1];
      g_f12z[index] = f12[2];
      // printf("angular n1=%d, n2=%d, r12=%f, f12[0]=%f, f12[1]=%f, f12[2]=%f\n",n1, n2, d12, f12[0],f12[1],f12[2]);
    }
  }
}

static __global__ void gpu_find_force_many_body(
  const int nall, //all atoms
  const int N,
  const int N1,
  const int nlocal,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const float* __restrict__ g_f12x,
  const float* __restrict__ g_f12y,
  const float* __restrict__ g_f12z,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  float s_fx = 0.0f;  // force_x
  float s_fy = 0.0f;  // force_y
  float s_fz = 0.0f;  // force_z
  float s_sxx = 0.0f; // virial_stress_xx
  float s_sxy = 0.0f; // virial_stress_xy
  float s_sxz = 0.0f; // virial_stress_xz
  float s_syx = 0.0f; // virial_stress_yx
  float s_syy = 0.0f; // virial_stress_yy
  float s_syz = 0.0f; // virial_stress_yz
  float s_szx = 0.0f; // virial_stress_zx
  float s_szy = 0.0f; // virial_stress_zy
  float s_szz = 0.0f; // virial_stress_zz

  if (n1 >= N1 && n1 < N) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      float x12 = float(x12double);
      float y12 = float(y12double);
      float z12 = float(z12double);

      float r12[3] = {x12, y12, z12};
      // float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);

      float f12x = g_f12x[index];
      float f12y = g_f12y[index];
      float f12z = g_f12z[index];

      if (n2 < N) {
        float f21x = 0;
        float f21y = 0;
        float f21z = 0;
        int offset = 0;
        int neighbor_number_2 = 0;
        neighbor_number_2 = g_NN_angular[n2];

        for (int k = 0; k < neighbor_number_2; ++k) {
          if (n1 == g_NL_angular[n2 + N * k]) {
            offset = k;
            break;
          }
        }
        index = offset * N + n2;

        f21x = g_f12x[index];
        f21y = g_f12y[index];
        f21z = g_f12z[index];

        s_fx += f12x - f21x;
        s_fy += f12y - f21y;
        s_fz += f12z - f21z;

        // per-atom virial
        s_sxx += x12 * f21x;
        s_syy += y12 * f21y;
        s_szz += z12 * f21z;

        s_sxy += x12 * f21y;
        s_sxz += x12 * f21z;
        s_syx += y12 * f21x;
        s_syz += y12 * f21z;
        s_szx += z12 * f21x;
        s_szy += z12 * f21y;
      } else {
        s_fx += f12x;
        s_fy += f12y;
        s_fz += f12z;
        atomicAdd(&g_fx[n2], double(-f12x));// ghost atom
        atomicAdd(&g_fy[n2], double(-f12y));
        atomicAdd(&g_fz[n2], double(-f12z));
        atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12x);
        atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12y);
        atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12z);
        atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12y);
        atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12z);
        atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12z);
        atomicAdd(&g_virial[n2 + 6 * nall], -r12[1] * f12x);
        atomicAdd(&g_virial[n2 + 7 * nall], -r12[2] * f12x);
        atomicAdd(&g_virial[n2 + 8 * nall], -r12[2] * f12y);
      }
    }

    // save force
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * nall] += s_sxx;
    g_virial[n1 + 1 * nall] += s_syy;
    g_virial[n1 + 2 * nall] += s_szz;
    g_virial[n1 + 3 * nall] += s_sxy;
    g_virial[n1 + 4 * nall] += s_sxz;
    g_virial[n1 + 5 * nall] += s_syz;
    g_virial[n1 + 6 * nall] += s_syx;
    g_virial[n1 + 7 * nall] += s_szx;
    g_virial[n1 + 8 * nall] += s_szy;
    // printf("many body n1=%d, g_fx[%d]=%f, g_fy[%d]=%f, g_fz[%d]=%f\n", n1, n1, g_fx[n1], n1, g_fy[n1], n1, g_fz[n1]);
  }
}

__global__ void calculate_total_virial(const double* virial, double* total_virial, int N) {
    __shared__ double shared_virial[6 * 64]; // 使用共享内存存储部分和
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int index = bid * blockDim.x + tid;

    // 初始化共享内存
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

// __global__ void compare_arrays(int *array1, int *array2, int *result, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     __shared__ bool mismatch;

//     if (threadIdx.x == 0) {
//         mismatch = false;
//     }
//     __syncthreads();

//     if (idx < N) {
//         if (array1[idx] != array2[idx]) {
//             mismatch = true;
//         }
//     }
//     __syncthreads();

//     if (threadIdx.x == 0) {
//         result[0] = mismatch;
//     }
// }

static __global__ void find_force_ZBL(
  const NEP3::ZBL zbl,
  const int nall, //all atoms
  const int N,
  const int N1,
  const int nlocal,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    float s_pe = 0.0f;
    float s_fx = 0.0f;
    float s_fy = 0.0f;
    float s_fz = 0.0f;
    float s_sxx = 0.0f;
    float s_sxy = 0.0f;
    float s_sxz = 0.0f;
    float s_syx = 0.0f;
    float s_syy = 0.0f;
    float s_syz = 0.0f;
    float s_szx = 0.0f;
    float s_szy = 0.0f;
    float s_szz = 0.0f;
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int type1 = g_type[n1];
    float zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int n2 = g_NL[n1 + N * i1];
      int type2 = g_type[n2];
      double x12double = g_x[n2] - x1;
      double y12double = g_y[n2] - y1;
      double z12double = g_z[n2] - z1;
      float r12[3] = {float(x12double), float(y12double), float(z12double)};
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
      // printf("zbl n1 %d n2 %d d12 %f e_c_half %f\n", n1, n2, d12, f);
      if (n2 >= N) {
        s_fx += f12[0];
        s_fy += f12[1];
        s_fz += f12[2];
        atomicAdd(&g_fx[n2], double(f21[0]));// ghost atom
        atomicAdd(&g_fy[n2], double(f21[1]));
        atomicAdd(&g_fz[n2], double(f21[2]));

        atomicAdd(&g_virial[n2 + 0 * nall], -r12[0] * f12[0]);
        atomicAdd(&g_virial[n2 + 1 * nall], -r12[1] * f12[1]);
        atomicAdd(&g_virial[n2 + 2 * nall], -r12[2] * f12[2]);
        atomicAdd(&g_virial[n2 + 3 * nall], -r12[0] * f12[1]);
        atomicAdd(&g_virial[n2 + 4 * nall], -r12[0] * f12[2]);
        atomicAdd(&g_virial[n2 + 5 * nall], -r12[1] * f12[2]);
        atomicAdd(&g_virial[n2 + 6 * nall], -r12[1] * f12[0]);
        atomicAdd(&g_virial[n2 + 7 * nall], -r12[2] * f12[0]);
        atomicAdd(&g_virial[n2 + 8 * nall], -r12[2] * f12[1]);

      } else {
        s_fx += f12[0] - f21[0];
        s_fy += f12[1] - f21[1];
        s_fz += f12[2] - f21[2];

        s_sxx += r12[0] * f21[0];
        s_syy += r12[1] * f21[1];
        s_szz += r12[2] * f21[2];

        s_sxy += r12[0] * f21[1];
        s_sxz += r12[0] * f21[2];
        s_syx += r12[1] * f21[0];
        s_syz += r12[1] * f21[2];
        s_szx += r12[2] * f21[0];
        s_szy += r12[2] * f21[1];
      }
      s_pe += f * 0.5f;
    }
    g_fx[n1] += s_fx;
    g_fy[n1] += s_fy;
    g_fz[n1] += s_fz;
    g_virial[n1 + 0 * nall] += s_sxx;
    g_virial[n1 + 1 * nall] += s_syy;
    g_virial[n1 + 2 * nall] += s_szz;
    g_virial[n1 + 3 * nall] += s_sxy;
    g_virial[n1 + 4 * nall] += s_sxz;
    g_virial[n1 + 5 * nall] += s_syz;
    g_virial[n1 + 6 * nall] += s_syx;
    g_virial[n1 + 7 * nall] += s_szx;
    g_virial[n1 + 8 * nall] += s_szy;
    g_pe[n1] += s_pe;
  }
}
