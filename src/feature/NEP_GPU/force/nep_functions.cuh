/*
This code is developed based on the GPUMD source code and adds ghost atom processing in LAMMPS. 
  Support multi GPUs.
  Support GPUMD NEP shared bias and PWMLFF NEP independent bias forcefield.

We have made the following improvements based on NEP4
http://doc.lonxun.com/MatPL/models/nep/
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

#include "nep.cuh"
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

static __device__ void apply_mic_small_box(
  const Box& box, const NEP::ExpandedBox& ebox, double& x12, double& y12, double& z12)
{
    double sx12 = ebox.h[9] * x12 + ebox.h[10] * y12 + ebox.h[11] * z12;
    double sy12 = ebox.h[12] * x12 + ebox.h[13] * y12 + ebox.h[14] * z12;
    double sz12 = ebox.h[15] * x12 + ebox.h[16] * y12 + ebox.h[17] * z12;
    sx12 -= nearbyint(sx12);
    sy12 -= nearbyint(sy12);
    sz12 -= nearbyint(sz12);
    x12 = ebox.h[0] * sx12 + ebox.h[1] * sy12 + ebox.h[2] * sz12;
    y12 = ebox.h[3] * sx12 + ebox.h[4] * sy12 + ebox.h[5] * sz12;
    z12 = ebox.h[6] * sx12 + ebox.h[7] * sy12 + ebox.h[8] * sz12;
}

static __global__ void find_neighbor(
  NEP::ParaMB paramb,
  const int N,
  const int N1,
  const Box box,
  const NEP::ExpandedBox ebox,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial,
  int* g_NN_angular,
  int* g_NL_angular,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  float* g_x12_angular,
  float* g_y12_angular,
  float* g_z12_angular)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];
    int count_radial = 0;
    int count_angular = 0;
    for (int n2 = N1; n2 < N; ++n2) {
      for (int ia = 0; ia < ebox.num_cells[0]; ++ia) {
        for (int ib = 0; ib < ebox.num_cells[1]; ++ib) {
          for (int ic = 0; ic < ebox.num_cells[2]; ++ic) {
            if (ia == 0 && ib == 0 && ic == 0 && n1 == n2) {
              continue; // exclude self
            }
            double delta[3];
            delta[0] = box.cpu_h[0] * ia + box.cpu_h[1] * ib + box.cpu_h[2] * ic;
            delta[1] = box.cpu_h[3] * ia + box.cpu_h[4] * ib + box.cpu_h[5] * ic;
            delta[2] = box.cpu_h[6] * ia + box.cpu_h[7] * ib + box.cpu_h[8] * ic;

            double x12 = g_x[n2] + delta[0] - x1;
            double y12 = g_y[n2] + delta[1] - y1;
            double z12 = g_z[n2] + delta[2] - z1;
            

            apply_mic_small_box(box, ebox, x12, y12, z12);

            float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
            if (distance_square < paramb.rc_radial * paramb.rc_radial) {
              // if (n1 == 0) {
              //   printf("radial n1 = %d, n2 = %d, r12 = %f %f\n", n1, n2, distance_square, sqrt(distance_square));
              // }
              g_NL_radial[count_radial * N + n1] = n2;
              g_x12_radial[count_radial * N + n1] = float(x12);
              g_y12_radial[count_radial * N + n1] = float(y12);
              g_z12_radial[count_radial * N + n1] = float(z12);
              count_radial++;
            }
            if (distance_square < paramb.rc_angular * paramb.rc_angular) {
              // if (n1 == 0) {
              //   printf("angular n1 = %d, n2 = %d, r12 = %f %f\n", n1, n2, distance_square, sqrt(distance_square));
              // }
              g_NL_angular[count_angular * N + n1] = n2;
              g_x12_angular[count_angular * N + n1] = float(x12);
              g_y12_angular[count_angular * N + n1] = float(y12);
              g_z12_angular[count_angular * N + n1] = float(z12);
              count_angular++;
            }
          }
        }
      }
    }
    g_NN_radial[n1] = count_radial;
    g_NN_angular[n1] = count_angular;
  }
}

static __global__ void find_descriptor(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const Box box,
  const NEP::ExpandedBox ebox,
  const int N,
  const int N1,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  double* g_pe,
  float* g_Fp,
  double* g_virial,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};
    // get radial descriptors
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int t2 = g_type[n2];
      
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      // printf("feat n1 %d t1 %d n2 %d t2 %d d12 %f = %f %f %f\n",n1, t1, n2, t2, d12, r12[0], r12[1], r12[2]);
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      // if (n1 ==0){
      //   printf("n1 %d t1 %d n2 %d t2 %d r12 %f fc %f\n", n1, t1, n2, t2, d12, fc12);
      // }
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
    }

    // get angular descriptors
    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        float x12 = g_x12_angular[index];
        float y12 = g_y12_angular[index];
        float z12 = g_z12_angular[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
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
      }
      if (paramb.num_L == paramb.L_max) {
        find_q(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else if (paramb.num_L == paramb.L_max + 1) {
        find_q_with_4body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else {
        find_q_with_5body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
        // printf("g_sum_fxyz n1=%d g_sum_fxyz[%d]=%f\n",n1, (n * NUM_OF_ABC + abc) * N + n1, g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1]);
      } 
    }

    // nomalize descriptor
    for (int d = 0; d < annmb.dim; ++d) {
      // printf("atom %d dim %d q %f scale %f\n", n1, d, q[d], paramb.q_scaler[d]);
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
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
      // printf("g_Fp n1=%d g_Fp[%d]=%f\n",n1, d * N + n1, g_Fp[d * N + n1]);
    }
  }
}

static __global__ void find_descriptor_charge2(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_radial,
  const float* __restrict__ g_y12_radial,
  const float* __restrict__ g_z12_radial,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  double* g_pe,
  float* g_Fp,
  float* g_charge,
  float* g_charge_derivative,
  float* g_sum_fxyz)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int t1 = g_type[n1];
    float q[MAX_DIM] = {0.0f};

    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12_radial[index], g_y12_radial[index], g_z12_radial[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float fc12;
      find_fc(paramb.rc_radial, paramb.rcinv_radial, d12, fc12);
      float fn12[MAX_NUM_N];
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

    for (int n = 0; n <= paramb.n_max_angular; ++n) {
      float s[NUM_OF_ABC] = {0.0f};
      for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
        int index = i1 * N + n1;
        int n2 = g_NL_angular[index];
        int t2 = g_type[n2];
        float x12 = g_x12_angular[index];
        float y12 = g_y12_angular[index];
        float z12 = g_z12_angular[index];
        float d12 = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
        float fc12;
        find_fc(paramb.rc_angular, paramb.rcinv_angular, d12, fc12);
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
      if (paramb.num_L == paramb.L_max) {
        find_q(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else if (paramb.num_L == paramb.L_max + 1) {
        find_q_with_4body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      } else {
        find_q_with_5body(paramb.n_max_angular + 1, n, s, q + (paramb.n_max_radial + 1));
      }
      for (int abc = 0; abc < NUM_OF_ABC; ++abc) {
        g_sum_fxyz[(n * NUM_OF_ABC + abc) * N + n1] = s[abc];
      }
    }

    for (int d = 0; d < annmb.dim; ++d) {
      q[d] = q[d] * paramb.q_scaler[d];
    }

    float F = 0.0f;
    float Fp[MAX_DIM] = {0.0f};
    float charge = 0.0f;
    float charge_derivative[MAX_DIM] = {0.0f};
    apply_ann_one_layer_charge(
      annmb.dim,
      annmb.num_neurons1,
      annmb.w0[t1],
      annmb.b0[t1],
      annmb.w1[t1],
      annmb.b1,
      q,
      F,
      Fp,
      charge,
      charge_derivative);

    g_pe[n1] += F;
    g_charge[n1] = charge;
    for (int d = 0; d < annmb.dim; ++d) {
      g_Fp[d * N + n1] = Fp[d] * paramb.q_scaler[d];
      g_charge_derivative[d * N + n1] = charge_derivative[d] * paramb.q_scaler[d];
    }
  }
}

static __global__ void zero_total_charge(const int N, float* g_charge)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ float s_charge[1024];
  float charge = 0.0f;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      charge += g_charge[n];
    }
  }
  s_charge[tid] = charge;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_charge[tid] += s_charge[tid + offset];
    }
    __syncthreads();
  }

  float mean_charge = s_charge[0] / N;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_charge[n] -= mean_charge;
    }
  }
}

static __device__ void ewald_cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

static __device__ float ewald_get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

static __global__ void find_k_and_G_charge2(
  const int num_kpoints_max,
  const float alpha,
  const float alpha_factor,
  const Box box,
  int* g_num_kpoints,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G)
{
  if (threadIdx.x + blockIdx.x * blockDim.x == 0) {
    const float det = float(box.cpu_h[0] * (box.cpu_h[4] * box.cpu_h[8] - box.cpu_h[5] * box.cpu_h[7]) +
                            box.cpu_h[1] * (box.cpu_h[5] * box.cpu_h[6] - box.cpu_h[3] * box.cpu_h[8]) +
                            box.cpu_h[2] * (box.cpu_h[3] * box.cpu_h[7] - box.cpu_h[4] * box.cpu_h[6]));
    const float a1[3] = {float(box.cpu_h[0]), float(box.cpu_h[3]), float(box.cpu_h[6])};
    const float a2[3] = {float(box.cpu_h[1]), float(box.cpu_h[4]), float(box.cpu_h[7])};
    const float a3[3] = {float(box.cpu_h[2]), float(box.cpu_h[5]), float(box.cpu_h[8])};
    float b1[3] = {0.0f};
    float b2[3] = {0.0f};
    float b3[3] = {0.0f};
    ewald_cross_product(a2, a3, b1);
    ewald_cross_product(a3, a1, b2);
    ewald_cross_product(a1, a2, b3);

    const float two_pi = 6.2831853f;
    const float two_pi_over_det = two_pi / det;
    for (int d = 0; d < 3; ++d) {
      b1[d] *= two_pi_over_det;
      b2[d] *= two_pi_over_det;
      b3[d] *= two_pi_over_det;
    }

    const float volume_k = two_pi * two_pi * two_pi / abs(det);
    int n1_max = int(alpha * two_pi * ewald_get_area(b2, b3) / volume_k);
    int n2_max = int(alpha * two_pi * ewald_get_area(b3, b1) / volume_k);
    int n3_max = int(alpha * two_pi * ewald_get_area(b1, b2) / volume_k);
    float ksq_max = two_pi * two_pi * alpha * alpha;

    int nk = 0;
    for (int n1 = 0; n1 <= n1_max; ++n1) {
      for (int n2 = -n2_max; n2 <= n2_max; ++n2) {
        for (int n3 = -n3_max; n3 <= n3_max; ++n3) {
          const int nsq = n1 * n1 + n2 * n2 + n3 * n3;
          if (nsq == 0 || (n1 == 0 && n2 < 0) || (n1 == 0 && n2 == 0 && n3 < 0)) {
            continue;
          }
          const float kx = n1 * b1[0] + n2 * b2[0] + n3 * b3[0];
          const float ky = n1 * b1[1] + n2 * b2[1] + n3 * b3[1];
          const float kz = n1 * b1[2] + n2 * b2[2] + n3 * b3[2];
          const float ksq = kx * kx + ky * ky + kz * kz;
          if (ksq < ksq_max) {
            if (nk < num_kpoints_max) {
              g_kx[nk] = kx;
              g_ky[nk] = ky;
              g_kz[nk] = kz;
              g_G[nk] = 2.0f * abs(two_pi_over_det) / ksq * exp(-ksq * alpha_factor);
            }
            ++nk;
          }
        }
      }
    }
    g_num_kpoints[0] = nk < num_kpoints_max ? nk : num_kpoints_max;
  }
}

static __global__ void find_structure_factor_charge2(
  const int N,
  const int num_kpoints,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  float* g_S_real,
  float* g_S_imag)
{
  int nk = blockIdx.x * blockDim.x + threadIdx.x;
  if (nk < num_kpoints) {
    float S_real = 0.0f;
    float S_imag = 0.0f;
    for (int n = 0; n < N; ++n) {
      const float kr = g_kx[nk] * float(g_x[n]) + g_ky[nk] * float(g_y[n]) + g_kz[nk] * float(g_z[n]);
      const float q = g_charge[n];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      S_real += q * cos_kr;
      S_imag -= q * sin_kr;
    }
    g_S_real[nk] = S_real;
    g_S_imag[nk] = S_imag;
  }
}

static __global__ void find_force_charge_reciprocal_space_charge2(
  const int N,
  const int num_kpoints,
  const float alpha_factor,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const float* g_S_real,
  const float* g_S_imag,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_total_virial,
  double* g_pe)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    const float q = g_charge[n];
    float temp_energy_sum = 0.0f;
    float temp_virial_sum[6] = {0.0f};
    float temp_force_sum[3] = {0.0f};
    float temp_D_real_sum = 0.0f;
    for (int nk = 0; nk < num_kpoints; ++nk) {
      const float kx = g_kx[nk];
      const float ky = g_ky[nk];
      const float kz = g_kz[nk];
      const float kr = kx * float(g_x[n]) + ky * float(g_y[n]) + kz * float(g_z[n]);
      const float G = g_G[nk];
      const float S_real = g_S_real[nk];
      const float S_imag = g_S_imag[nk];
      float sin_kr = sin(kr);
      float cos_kr = cos(kr);
      const float imag_term = G * (S_real * sin_kr + S_imag * cos_kr);
      const float GSE = G * (S_real * cos_kr - S_imag * sin_kr);
      const float qGSE = q * GSE;
      temp_energy_sum += qGSE;
      const float alpha_k_factor = 2.0f * alpha_factor + 2.0f / (kx * kx + ky * ky + kz * kz);
      temp_virial_sum[0] += qGSE * (1.0f - alpha_k_factor * kx * kx);
      temp_virial_sum[1] += qGSE * (1.0f - alpha_k_factor * ky * ky);
      temp_virial_sum[2] += qGSE * (1.0f - alpha_k_factor * kz * kz);
      temp_virial_sum[3] -= qGSE * (alpha_k_factor * kx * ky);
      temp_virial_sum[4] -= qGSE * (alpha_k_factor * ky * kz);
      temp_virial_sum[5] -= qGSE * (alpha_k_factor * kz * kx);
      temp_D_real_sum += GSE;
      temp_force_sum[0] += kx * imag_term;
      temp_force_sum[1] += ky * imag_term;
      temp_force_sum[2] += kz * imag_term;
    }
    const double virial_xx = double(K_C_SP * temp_virial_sum[0]);
    const double virial_yy = double(K_C_SP * temp_virial_sum[1]);
    const double virial_zz = double(K_C_SP * temp_virial_sum[2]);
    const double virial_xy = double(K_C_SP * temp_virial_sum[3]);
    const double virial_yz = double(K_C_SP * temp_virial_sum[4]);
    const double virial_zx = double(K_C_SP * temp_virial_sum[5]);
    // Keep potential_per_atom as neural-network Ei; charge energy contributes force and virial only.
    g_virial[n + 0 * N] += virial_xx;
    g_virial[n + 1 * N] += virial_yy;
    g_virial[n + 2 * N] += virial_zz;
    g_virial[n + 3 * N] += virial_xy;
    g_virial[n + 4 * N] += virial_zx;
    g_virial[n + 5 * N] += virial_yz;
    g_virial[n + 6 * N] += virial_xy;
    g_virial[n + 7 * N] += virial_zx;
    g_virial[n + 8 * N] += virial_yz;
    atomicAdd(&g_total_virial[0], virial_xx);
    atomicAdd(&g_total_virial[1], virial_yy);
    atomicAdd(&g_total_virial[2], virial_zz);
    atomicAdd(&g_total_virial[3], virial_xy);
    atomicAdd(&g_total_virial[4], virial_zx);
    atomicAdd(&g_total_virial[5], virial_yz);
    g_D_real[n] = 2.0f * K_C_SP * temp_D_real_sum;
    const float charge_factor = K_C_SP * 2.0f * q;
    g_fx[n] += double(charge_factor * temp_force_sum[0]);
    g_fy[n] += double(charge_factor * temp_force_sum[1]);
    g_fz[n] += double(charge_factor * temp_force_sum[2]);
  }
}

static __global__ void zero_mean_D_real_charge2(const int N, float* g_D_real)
{
  int tid = threadIdx.x;
  int number_of_batches = (N - 1) / 1024 + 1;
  __shared__ double s_sum[1024];
  double sum = 0.0;
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      sum += double(g_D_real[n]);
    }
  }
  s_sum[tid] = sum;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_sum[tid] += s_sum[tid + offset];
    }
    __syncthreads();
  }

  const float mean_D = float(s_sum[0] / N);
  for (int batch = 0; batch < number_of_batches; ++batch) {
    int n = tid + batch * 1024;
    if (n < N) {
      g_D_real[n] -= mean_D;
    }
  }
}

static __global__ void find_bec_diagonal(const int N, const float* g_charge, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    g_bec[n1 + N * 0] = g_charge[n1];
    g_bec[n1 + N * 1] = 0.0f;
    g_bec[n1 + N * 2] = 0.0f;
    g_bec[n1 + N * 3] = 0.0f;
    g_bec[n1 + N * 4] = g_charge[n1];
    g_bec[n1 + N * 5] = 0.0f;
    g_bec[n1 + N * 6] = 0.0f;
    g_bec[n1 + N * 7] = 0.0f;
    g_bec[n1 + N * 8] = g_charge[n1];
  }
}

static __global__ void scale_bec(const int N, const float* sqrt_epsilon_inf, float* g_bec)
{
  int n1 = threadIdx.x + blockIdx.x * blockDim.x;
  if (n1 < N) {
    for (int d = 0; d < 9; ++d) {
      g_bec[n1 + N * d] *= sqrt_epsilon_inf[0];
    }
  }
}

static __global__ void find_bec_radial(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int* g_NN_radial,
  const int* g_NL_radial,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_charge_derivative,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_radial[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_radial[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      float f12[3] = {0.0f};

      find_fn_and_fnp(
        paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
      for (int n = 0; n <= paramb.n_max_radial; ++n) {
        float gnp12 = 0.0f;
        for (int k = 0; k <= paramb.basis_size_radial; ++k) {
          int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
          c_index += t1 * paramb.num_types + t2;
          gnp12 += fnp12[k] * annmb.c[c_index];
        }
        const float tmp12 = g_charge_derivative[n1 + n * N] * gnp12 * d12inv;
        for (int d = 0; d < 3; ++d) {
          f12[d] += tmp12 * r12[d];
        }
      }

      float bec_xx = 0.5f * r12[0] * f12[0];
      float bec_xy = 0.5f * r12[0] * f12[1];
      float bec_xz = 0.5f * r12[0] * f12[2];
      float bec_yx = 0.5f * r12[1] * f12[0];
      float bec_yy = 0.5f * r12[1] * f12[1];
      float bec_yz = 0.5f * r12[1] * f12[2];
      float bec_zx = 0.5f * r12[2] * f12[0];
      float bec_zy = 0.5f * r12[2] * f12[1];
      float bec_zz = 0.5f * r12[2] * f12[2];

      atomicAdd(&g_bec[n1 + N * 0], bec_xx);
      atomicAdd(&g_bec[n1 + N * 1], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2 + N * 0], -bec_xx);
      atomicAdd(&g_bec[n2 + N * 1], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_bec_angular(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_charge_derivative,
  const float* __restrict__ g_sum_fxyz,
  float* g_bec)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      Fp[d] = g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1];
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
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

      float bec_xx = 0.5f * r12[0] * f12[0];
      float bec_xy = 0.5f * r12[0] * f12[1];
      float bec_xz = 0.5f * r12[0] * f12[2];
      float bec_yx = 0.5f * r12[1] * f12[0];
      float bec_yy = 0.5f * r12[1] * f12[1];
      float bec_yz = 0.5f * r12[1] * f12[2];
      float bec_zx = 0.5f * r12[2] * f12[0];
      float bec_zy = 0.5f * r12[2] * f12[1];
      float bec_zz = 0.5f * r12[2] * f12[2];

      atomicAdd(&g_bec[n1 + N * 0], bec_xx);
      atomicAdd(&g_bec[n1 + N * 1], bec_xy);
      atomicAdd(&g_bec[n1 + N * 2], bec_xz);
      atomicAdd(&g_bec[n1 + N * 3], bec_yx);
      atomicAdd(&g_bec[n1 + N * 4], bec_yy);
      atomicAdd(&g_bec[n1 + N * 5], bec_yz);
      atomicAdd(&g_bec[n1 + N * 6], bec_zx);
      atomicAdd(&g_bec[n1 + N * 7], bec_zy);
      atomicAdd(&g_bec[n1 + N * 8], bec_zz);

      atomicAdd(&g_bec[n2 + N * 0], -bec_xx);
      atomicAdd(&g_bec[n2 + N * 1], -bec_xy);
      atomicAdd(&g_bec[n2 + N * 2], -bec_xz);
      atomicAdd(&g_bec[n2 + N * 3], -bec_yx);
      atomicAdd(&g_bec[n2 + N * 4], -bec_yy);
      atomicAdd(&g_bec[n2 + N * 5], -bec_yz);
      atomicAdd(&g_bec[n2 + N * 6], -bec_zx);
      atomicAdd(&g_bec[n2 + N * 7], -bec_zy);
      atomicAdd(&g_bec[n2 + N * 8], -bec_zz);
    }
  }
}

static __global__ void find_force_radial(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_charge_derivative,
  const float* __restrict__ g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_total_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    // if (n1 == 0){
    //   printf("find_force_radial n1 = 0\n");
    // }
    int t1 = g_type[n1];
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL[index];
      int t2 = g_type[n2];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float d12inv = 1.0f / d12;
      float f12[3] = {0.0f};
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_radial, paramb.rcinv_radial, d12, fc12, fcp12);
      float fn12[MAX_NUM_N];
      float fnp12[MAX_NUM_N];
      if (paramb.version == 2) {
        find_fn_and_fnp(paramb.n_max_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float tmp12 = g_Fp[n1 + n * N] * fnp12[n] * d12inv;
          tmp12 *= (paramb.num_types == 1)
                     ? 1.0f
                     : annmb.c[(n * paramb.num_types + t1) * paramb.num_types + t2];
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        }
      } else {
        find_fn_and_fnp(
          paramb.basis_size_radial, paramb.rcinv_radial, d12, fc12, fcp12, fn12, fnp12);
        for (int n = 0; n <= paramb.n_max_radial; ++n) {
          float gnp12 = 0.0f;
          for (int k = 0; k <= paramb.basis_size_radial; ++k) {
            int c_index = (n * (paramb.basis_size_radial + 1) + k) * paramb.num_types_sq;
            c_index += t1 * paramb.num_types + t2;
            gnp12 += fnp12[k] * annmb.c[c_index];
          }
          float descriptor_derivative = g_Fp[n1 + n * N];
          if (paramb.charge_mode == 2) {
            descriptor_derivative += g_charge_derivative[n1 + n * N] * g_D_real[n1];
          }
          float tmp12 = descriptor_derivative * gnp12 * d12inv;
          for (int d = 0; d < 3; ++d) {
            f12[d] += tmp12 * r12[d];
          }
        }
      }
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;
      
      s_sxx -= r12[0] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syz -= r12[1] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];

      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);

      atomicAdd(&g_total_virial[0], s_sxx);// xx
      atomicAdd(&g_total_virial[1], s_syy);// yy
      atomicAdd(&g_total_virial[2], s_szz);// zz
      atomicAdd(&g_total_virial[3], s_sxy);// xy
      atomicAdd(&g_total_virial[4], s_sxz);// xz
      atomicAdd(&g_total_virial[5], s_syz);// yz
    }
  }
}



static __global__ void find_force_angular(
  NEP::ParaMB paramb,
  NEP::ANN annmb,
  const int N,
  const int N1,
  const int* g_NN_angular,
  const int* g_NL_angular,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12,
  const float* __restrict__ g_y12,
  const float* __restrict__ g_z12,
  const float* __restrict__ g_Fp,
  const float* __restrict__ g_charge_derivative,
  const float* __restrict__ g_D_real,
  const float* __restrict__ g_sum_fxyz,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_total_virial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    // if (n1 == 0){
    //   printf("find_force_angular_small_box n1 = 0\n");
    // }
    float Fp[MAX_DIM_ANGULAR] = {0.0f};
    float sum_fxyz[NUM_OF_ABC * MAX_NUM_N];
    for (int d = 0; d < paramb.dim_angular; ++d) {
      float descriptor_derivative = g_Fp[(paramb.n_max_radial + 1 + d) * N + n1];
      if (paramb.charge_mode == 2) {
        descriptor_derivative +=
          g_charge_derivative[(paramb.n_max_radial + 1 + d) * N + n1] * g_D_real[n1];
      }
      Fp[d] = descriptor_derivative;
    }
    for (int d = 0; d < (paramb.n_max_angular + 1) * NUM_OF_ABC; ++d) {
      sum_fxyz[d] = g_sum_fxyz[d * N + n1];
    }

    int t1 = g_type[n1];

    for (int i1 = 0; i1 < g_NN_angular[n1]; ++i1) {
      int index = i1 * N + n1;
      int n2 = g_NL_angular[n1 + N * i1];
      float r12[3] = {g_x12[index], g_y12[index], g_z12[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      float f12[3] = {0.0f};
      float fc12, fcp12;
      find_fc_and_fcp(paramb.rc_angular, paramb.rcinv_angular, d12, fc12, fcp12);
      int t2 = g_type[n2];
      if (paramb.version == 2) {
        for (int n = 0; n <= paramb.n_max_angular; ++n) {
          float fn;
          float fnp;
          find_fn_and_fnp(n, paramb.rcinv_angular, d12, fc12, fcp12, fn, fnp);
          const float c =
            (paramb.num_types == 1)
              ? 1.0f
              : annmb.c
                  [((paramb.n_max_radial + 1 + n) * paramb.num_types + t1) * paramb.num_types + t2];
          fn *= c;
          fnp *= c;
          accumulate_f12(n, paramb.n_max_angular + 1, d12, r12, fn, fnp, Fp, sum_fxyz, f12);
        }
      } else {
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
      }
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;
      
      s_sxx -= r12[0] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syz -= r12[1] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];

      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);

      atomicAdd(&g_total_virial[0], s_sxx);// xx
      atomicAdd(&g_total_virial[1], s_syy);// yy
      atomicAdd(&g_total_virial[2], s_szz);// zz
      atomicAdd(&g_total_virial[3], s_sxy);// xy
      atomicAdd(&g_total_virial[4], s_sxz);// xz
      atomicAdd(&g_total_virial[5], s_syz);// yz
    }
  }
}

static __global__ void find_force_ZBL(
  NEP::ParaMB paramb,
  const NEP::ZBL zbl,
  const int N,
  const int N1,
  const int* g_NN,
  const int* g_NL,
  const int* __restrict__ g_type,
  const float* __restrict__ g_x12_angular,
  const float* __restrict__ g_y12_angular,
  const float* __restrict__ g_z12_angular,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_total_virial,
  double* g_pe)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 < N) {
    int type1 = g_type[n1];
    int zi = zbl.atomic_numbers[type1];
    float pow_zi = pow(zi, 0.23f);
    for (int i1 = 0; i1 < g_NN[n1]; ++i1) {
      int index = n1 + N * i1;
      int n2 = g_NL[index];
      int type2 = g_type[n2];

      float r12[3] = {g_x12_angular[index], g_y12_angular[index], g_z12_angular[index]};
      float d12 = sqrt(r12[0] * r12[0] + r12[1] * r12[1] + r12[2] * r12[2]);
      double max_rc_outer = 2.5;
      if (d12 >= max_rc_outer) {
        continue;
      }
      float d12inv = 1.0f / d12;
      float f, fp;
      int zj = zbl.atomic_numbers[type2];
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
        float rc_inner = zbl.rc_inner;
        float rc_outer = zbl.rc_outer;
        if (paramb.use_typewise_cutoff_zbl) {
          // zi and zj start from 1, so need to minus 1 here
          rc_outer = min(
            (COVALENT_RADIUS[zi - 1] + COVALENT_RADIUS[zj - 1]) * paramb.typewise_cutoff_zbl_factor,
            rc_outer);
          rc_inner = 0.0f;
        }
        find_f_and_fp_zbl(zizj, a_inv, rc_inner, rc_outer, d12, d12inv, f, fp);
      }
      float f2 = fp * d12inv * 0.5f;
      float f12[3] = {r12[0] * f2, r12[1] * f2, r12[2] * f2};
      // float f21[3] = {-r12[0] * f2, -r12[1] * f2, -r12[2] * f2};
      // printf("zbl n1 %d n2 %d d12 %f e_c_half %f\n", n1, n2, d12, f);
      
      double s_sxx = 0.0;
      double s_sxy = 0.0;
      double s_sxz = 0.0;
      double s_syx = 0.0;
      double s_syy = 0.0;
      double s_syz = 0.0;
      double s_szx = 0.0;
      double s_szy = 0.0;
      double s_szz = 0.0;

      s_sxx -= r12[0] * f12[0];
      s_syy -= r12[1] * f12[1];
      s_szz -= r12[2] * f12[2];
      s_sxy -= r12[0] * f12[1];
      s_sxz -= r12[0] * f12[2];
      s_syz -= r12[1] * f12[2];
      s_syx -= r12[1] * f12[0];
      s_szx -= r12[2] * f12[0];
      s_szy -= r12[2] * f12[1];

      atomicAdd(&g_fx[n1], double(f12[0]));
      atomicAdd(&g_fy[n1], double(f12[1]));
      atomicAdd(&g_fz[n1], double(f12[2]));
      atomicAdd(&g_fx[n2], double(-f12[0]));
      atomicAdd(&g_fy[n2], double(-f12[1]));
      atomicAdd(&g_fz[n2], double(-f12[2]));
      
      // save virial
      // xx xy xz    0 3 4
      // yx yy yz    6 1 5
      // zx zy zz    7 8 2
      atomicAdd(&g_virial[n2 + 0 * N], s_sxx);
      atomicAdd(&g_virial[n2 + 1 * N], s_syy);
      atomicAdd(&g_virial[n2 + 2 * N], s_szz);
      atomicAdd(&g_virial[n2 + 3 * N], s_sxy);
      atomicAdd(&g_virial[n2 + 4 * N], s_sxz);
      atomicAdd(&g_virial[n2 + 5 * N], s_syz);
      atomicAdd(&g_virial[n2 + 6 * N], s_syx);
      atomicAdd(&g_virial[n2 + 7 * N], s_szx);
      atomicAdd(&g_virial[n2 + 8 * N], s_szy);

      atomicAdd(&g_total_virial[0], s_sxx);// xx
      atomicAdd(&g_total_virial[1], s_syy);// yy
      atomicAdd(&g_total_virial[2], s_szz);// zz
      atomicAdd(&g_total_virial[3], s_sxy);// xy
      atomicAdd(&g_total_virial[4], s_sxz);// xz
      atomicAdd(&g_total_virial[5], s_syz);// yz

      g_pe[n1] += f * 0.5;
    }
  }
}
