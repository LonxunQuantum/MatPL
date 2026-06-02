/*
    Ewald reciprocal-space force helper for NEP charge inference.
    The implementation follows GPUMD src/force/ewald.cu and is kept header-only
    because the NEP GPU module currently builds a single CUDA translation unit.
*/

#pragma once

#include "../utilities/common.cuh"
#include "../utilities/error.cuh"
#include "../utilities/gpu_vector.cuh"
#include "box.cuh"
#include <cmath>

__device__ void ewald_cross_product(const float a[3], const float b[3], float c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

__device__ float ewald_get_area(const float* a, const float* b)
{
  const float s1 = a[1] * b[2] - a[2] * b[1];
  const float s2 = a[2] * b[0] - a[0] * b[2];
  const float s3 = a[0] * b[1] - a[1] * b[0];
  return sqrt(s1 * s1 + s2 * s2 + s3 * s3);
}

__global__ void find_k_and_G_charge2(
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

__global__ void find_structure_factor_charge2(
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

__global__ void find_force_charge_reciprocal_space_charge2(
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

__global__ void zero_mean_D_real_charge2(const int N, float* g_D_real)
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


inline void ewald_find_force_charge2(
  const int N,
  const int block_size,
  const int grid_size,
  const int num_kpoints_max,
  const float alpha,
  const float alpha_factor,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& num_kpoints,
  GPU_Vector<float>& kx,
  GPU_Vector<float>& ky,
  GPU_Vector<float>& kz,
  GPU_Vector<float>& G,
  GPU_Vector<float>& S_real,
  GPU_Vector<float>& S_imag,
  GPU_Vector<float>& D_real,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& total_virial,
  GPU_Vector<double>& potential_per_atom)
{
  find_k_and_G_charge2<<<1, 1>>>(
    num_kpoints_max,
    alpha,
    alpha_factor,
    box,
    num_kpoints.data(),
    kx.data(),
    ky.data(),
    kz.data(),
    G.data());
  CUDA_CHECK_KERNEL

  int cpu_num_kpoints = 0;
  num_kpoints.copy_to_host(&cpu_num_kpoints);
  const int k_grid_size = (cpu_num_kpoints - 1) / block_size + 1;
  find_structure_factor_charge2<<<k_grid_size, block_size>>>(
    N,
    cpu_num_kpoints,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    kx.data(),
    ky.data(),
    kz.data(),
    S_real.data(),
    S_imag.data());
  CUDA_CHECK_KERNEL

  find_force_charge_reciprocal_space_charge2<<<grid_size, block_size>>>(
    N,
    cpu_num_kpoints,
    alpha_factor,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    kx.data(),
    ky.data(),
    kz.data(),
    G.data(),
    S_real.data(),
    S_imag.data(),
    D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data(),
    total_virial.data(),
    potential_per_atom.data());
  CUDA_CHECK_KERNEL
}
