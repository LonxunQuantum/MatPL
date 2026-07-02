/*
    PPPM reciprocal-space force helper for NEP charge inference.
    The implementation follows GPUMD src/force/pppm.cu and is kept header-only
    because the NEP GPU module currently builds a single CUDA translation unit.
*/

#pragma once

#include "../utilities/common.cuh"
#include "../utilities/error.cuh"
#include "../utilities/gpu_vector.cuh"
#include "box.cuh"
#include <cmath>
#include <cstdlib>
#include <cufft.h>
#include <iostream>

struct PPPM_Para {
  int K0K1K2 = 0;
  int K0K1 = 0;
  int K[3] = {0, 0, 0};
  int K_half[3] = {0, 0, 0};
  float alpha = 0.0f;
  float alpha_factor = 0.0f;
  float two_pi_over_V = 0.0f;
  float potential_factor = 0.0f;
  float b[3][3];
  float two_pi_over_K[3];
};

struct PPPM_Data {
  PPPM_Para para;
  GPU_Vector<float> kx;
  GPU_Vector<float> ky;
  GPU_Vector<float> kz;
  GPU_Vector<float> G;
  GPU_Vector<cufftComplex> mesh;
  GPU_Vector<cufftComplex> mesh_G;
  GPU_Vector<cufftComplex> mesh_x;
  GPU_Vector<cufftComplex> mesh_y;
  GPU_Vector<cufftComplex> mesh_z;
  GPU_Vector<cufftComplex> mesh_virial;
  cufftHandle plan = 0;
  cufftHandle plan_virial = 0;
  bool plan_initialized = false;
  bool plan_virial_initialized = false;
};

#ifdef __CUDACC__
namespace {

int get_best_pppm_K(const int m)
{
  int n = 16;
  while (n < m) {
    n *= 2;
  }
  return n;
}

__constant__ float pppm_sinc_coeff[6] = {
  1.0f, -1.6666667e-1f, 8.3333333e-3f, -1.9841270e-4f, 2.7557319e-6f, -2.5052108e-8f};
__constant__ float pppm_G_coeff[5] = {
  1.0000000e+00f, -1.6666667e+00f, 7.7777778e-01f, -8.9947090e-02f, 7.0546737e-04f};
__constant__ float pppm_W_coeff[5][5] = {
  {2.6041667e-03f, -2.0833333e-02f, 6.2500000e-02f, -8.3333333e-02f, 4.1666667e-02f},
  {1.9791667e-01f, -4.5833333e-01f, 2.5000000e-01f, 1.6666667e-01f, -1.6666667e-01f},
  {5.9895833e-01f, 0.0000000e+00f, -6.2500000e-01f, 0.0000000e+00f, 2.5000000e-01f},
  {1.9791667e-01f, 4.5833333e-01f, 2.5000000e-01f, -1.6666667e-01f, -1.6666667e-01f},
  {2.6041667e-03f, 2.0833333e-02f, 6.2500000e-02f, 8.3333333e-02f, 4.1666667e-02f}};

__device__ inline float pppm_sinc(const float x)
{
  float y = 0.0f;
  if (x * x <= 1.0f) {
    float term = 1.0f;
    for (int i = 0; i < 6; ++i) {
      y += pppm_sinc_coeff[i] * term;
      term *= x * x;
    }
  } else {
    y = sin(x) / x;
  }
  return y;
}

__device__ inline int pppm_mesh_index(const int K, const int n)
{
  int y = n;
  if (n >= K) {
    y = n - K;
  } else if (n < 0) {
    y = n + K;
  }
  return y;
}

__global__ void pppm_find_k_and_G(
  const PPPM_Para para,
  float* g_kx,
  float* g_ky,
  float* g_kz,
  float* g_G)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    int nk[3];
    nk[2] = n / para.K0K1;
    nk[1] = (n - nk[2] * para.K0K1) / para.K[0];
    nk[0] = n % para.K[0];

    float denominator[3] = {0.0f};
    for (int d = 0; d < 3; ++d) {
      if (nk[d] >= para.K_half[d]) {
        nk[d] -= para.K[d];
      }
      float t = sin(0.5f * para.two_pi_over_K[d] * nk[d]);
      t *= t;
      t = (((pppm_G_coeff[4] * t + pppm_G_coeff[3]) * t + pppm_G_coeff[2]) * t
           + pppm_G_coeff[1]) * t + pppm_G_coeff[0];
      denominator[d] = t * t;
    }

    const float kx = nk[0] * para.b[0][0] + nk[1] * para.b[1][0] + nk[2] * para.b[2][0];
    const float ky = nk[0] * para.b[0][1] + nk[1] * para.b[1][1] + nk[2] * para.b[2][1];
    const float kz = nk[0] * para.b[0][2] + nk[1] * para.b[1][2] + nk[2] * para.b[2][2];
    g_kx[n] = kx;
    g_ky[n] = ky;
    g_kz[n] = kz;
    const float ksq = kx * kx + ky * ky + kz * kz;

    float numerator = pppm_sinc(0.5f * para.two_pi_over_K[0] * nk[0]);
    numerator *= pppm_sinc(0.5f * para.two_pi_over_K[1] * nk[1]);
    numerator *= pppm_sinc(0.5f * para.two_pi_over_K[2] * nk[2]);
    numerator = numerator * numerator * numerator * numerator * numerator;
    numerator *= numerator;

    if (ksq == 0.0f) {
      g_G[n] = 0.0f;
    } else {
      float G_opt = numerator * para.two_pi_over_V / ksq * exp(-ksq * para.alpha_factor);
      G_opt /= denominator[0] * denominator[1] * denominator[2];
      g_G[n] = G_opt;
    }
  }
}

__global__ void pppm_set_mesh_to_zero(const PPPM_Para para, cufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    g_mesh[n].x = 0.0f;
    g_mesh[n].y = 0.0f;
  }
}

__global__ void pppm_find_mesh(
  const int N1,
  const int N2,
  const PPPM_Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  cufftComplex* g_mesh)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = g_charge[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f);
    const int iy = int(sy + 0.5f);
    const int iz = int(sz + 0.5f);
    const float dx = sx - ix;
    const float dy = sy - iy;
    const float dz = sz - iz;
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((pppm_W_coeff[d][4] * dx + pppm_W_coeff[d][3]) * dx + pppm_W_coeff[d][2]) * dx
               + pppm_W_coeff[d][1]) * dx + pppm_W_coeff[d][0];
      Wy[d] = (((pppm_W_coeff[d][4] * dy + pppm_W_coeff[d][3]) * dy + pppm_W_coeff[d][2]) * dy
               + pppm_W_coeff[d][1]) * dy + pppm_W_coeff[d][0];
      Wz[d] = (((pppm_W_coeff[d][4] * dz + pppm_W_coeff[d][3]) * dz + pppm_W_coeff[d][2]) * dz
               + pppm_W_coeff[d][1]) * dz + pppm_W_coeff[d][0];
    }
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = pppm_mesh_index(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = pppm_mesh_index(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = pppm_mesh_index(para.K[2], iz + n2);
          const int idx = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          atomicAdd(&g_mesh[idx].x, q * W);
        }
      }
    }
  }
}

__global__ void pppm_ik_times_mesh_times_G(
  const PPPM_Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const cufftComplex* g_mesh_fft,
  cufftComplex* g_mesh_fft_x,
  cufftComplex* g_mesh_fft_y,
  cufftComplex* g_mesh_fft_z)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float G = g_G[n];
    const cufftComplex mesh_fft = g_mesh_fft[n];
    g_mesh_fft_x[n] = {mesh_fft.y * kx * G, -mesh_fft.x * kx * G};
    g_mesh_fft_y[n] = {mesh_fft.y * ky * G, -mesh_fft.x * ky * G};
    g_mesh_fft_z[n] = {mesh_fft.y * kz * G, -mesh_fft.x * kz * G};
  }
}

__global__ void pppm_find_mesh_G(
  const PPPM_Para para,
  const float* g_G,
  const cufftComplex* g_mesh,
  cufftComplex* g_mesh_G)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float G = g_G[n];
    const cufftComplex mesh = g_mesh[n];
    g_mesh_G[n] = {mesh.x * G, mesh.y * G};
  }
}

__global__ void pppm_find_mesh_virial(
  const PPPM_Para para,
  const float* g_kx,
  const float* g_ky,
  const float* g_kz,
  const float* g_G,
  const cufftComplex* g_S,
  cufftComplex* g_mesh_virial_xx,
  cufftComplex* g_mesh_virial_yy,
  cufftComplex* g_mesh_virial_zz,
  cufftComplex* g_mesh_virial_xy,
  cufftComplex* g_mesh_virial_yz,
  cufftComplex* g_mesh_virial_zx)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < para.K0K1K2) {
    const float kx = g_kx[n];
    const float ky = g_ky[n];
    const float kz = g_kz[n];
    const float ksq = kx * kx + ky * ky + kz * kz;
    if (ksq != 0.0f) {
      const float alpha_k_factor = 2.0f * para.alpha_factor + 2.0f / ksq;
      const float G = g_G[n];
      const cufftComplex S = g_S[n];
      const float GSx = G * S.x;
      const float GSy = G * S.y;
      float B = 1.0f - alpha_k_factor * kx * kx;
      g_mesh_virial_xx[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * ky * ky;
      g_mesh_virial_yy[n] = {B * GSx, B * GSy};
      B = 1.0f - alpha_k_factor * kz * kz;
      g_mesh_virial_zz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kx * ky;
      g_mesh_virial_xy[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * ky * kz;
      g_mesh_virial_yz[n] = {B * GSx, B * GSy};
      B = -alpha_k_factor * kz * kx;
      g_mesh_virial_zx[n] = {B * GSx, B * GSy};
    }
  }
}

__global__ void pppm_find_force_virial_from_field(
  const int N,
  const int N1,
  const int N2,
  const PPPM_Para para,
  const Box box,
  const float* g_charge,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const cufftComplex* g_mesh_G,
  const cufftComplex* g_mesh_fft_x_ifft,
  const cufftComplex* g_mesh_fft_y_ifft,
  const cufftComplex* g_mesh_fft_z_ifft,
  const cufftComplex* g_mesh_virial_xx,
  const cufftComplex* g_mesh_virial_yy,
  const cufftComplex* g_mesh_virial_zz,
  const cufftComplex* g_mesh_virial_xy,
  const cufftComplex* g_mesh_virial_yz,
  const cufftComplex* g_mesh_virial_zx,
  float* g_D_real,
  double* g_fx,
  double* g_fy,
  double* g_fz,
  double* g_virial,
  double* g_total_virial,
  double* g_pe)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n < N2) {
    const double x = g_x[n];
    const double y = g_y[n];
    const double z = g_z[n];
    const float q = K_C_SP * g_charge[n];
    const float sx = (box.cpu_h[9] * x + box.cpu_h[10] * y + box.cpu_h[11] * z) * para.K[0];
    const float sy = (box.cpu_h[12] * x + box.cpu_h[13] * y + box.cpu_h[14] * z) * para.K[1];
    const float sz = (box.cpu_h[15] * x + box.cpu_h[16] * y + box.cpu_h[17] * z) * para.K[2];
    const int ix = int(sx + 0.5f);
    const int iy = int(sy + 0.5f);
    const int iz = int(sz + 0.5f);
    const float dx = sx - ix;
    const float dy = sy - iy;
    const float dz = sz - iz;
    float Wx[5] = {0.0f};
    float Wy[5] = {0.0f};
    float Wz[5] = {0.0f};
    for (int d = 0; d < 5; ++d) {
      Wx[d] = (((pppm_W_coeff[d][4] * dx + pppm_W_coeff[d][3]) * dx + pppm_W_coeff[d][2]) * dx
               + pppm_W_coeff[d][1]) * dx + pppm_W_coeff[d][0];
      Wy[d] = (((pppm_W_coeff[d][4] * dy + pppm_W_coeff[d][3]) * dy + pppm_W_coeff[d][2]) * dy
               + pppm_W_coeff[d][1]) * dy + pppm_W_coeff[d][0];
      Wz[d] = (((pppm_W_coeff[d][4] * dz + pppm_W_coeff[d][3]) * dz + pppm_W_coeff[d][2]) * dz
               + pppm_W_coeff[d][1]) * dz + pppm_W_coeff[d][0];
    }
    float D_real = 0.0f;
    float E[3] = {0.0f};
    float V[6] = {0.0f};
    for (int n0 = -2; n0 <= 2; ++n0) {
      const int neighbor0 = pppm_mesh_index(para.K[0], ix + n0);
      for (int n1 = -2; n1 <= 2; ++n1) {
        const int neighbor1 = pppm_mesh_index(para.K[1], iy + n1);
        for (int n2 = -2; n2 <= 2; ++n2) {
          const int neighbor2 = pppm_mesh_index(para.K[2], iz + n2);
          const int idx = neighbor0 + para.K[0] * (neighbor1 + para.K[1] * neighbor2);
          const float W = Wx[n0 + 2] * Wy[n1 + 2] * Wz[n2 + 2];
          D_real += W * g_mesh_G[idx].x;
          E[0] += W * g_mesh_fft_x_ifft[idx].x;
          E[1] += W * g_mesh_fft_y_ifft[idx].x;
          E[2] += W * g_mesh_fft_z_ifft[idx].x;
          V[0] += W * g_mesh_virial_xx[idx].x;
          V[1] += W * g_mesh_virial_yy[idx].x;
          V[2] += W * g_mesh_virial_zz[idx].x;
          V[3] += W * g_mesh_virial_xy[idx].x;
          V[4] += W * g_mesh_virial_yz[idx].x;
          V[5] += W * g_mesh_virial_zx[idx].x;
        }
      }
    }

    g_D_real[n] = 2.0f * K_C_SP * D_real;
    g_pe[n] += double(0.5f * g_charge[n] * g_D_real[n]);
    g_fx[n] += double(2.0f * q * E[0]);
    g_fy[n] += double(2.0f * q * E[1]);
    g_fz[n] += double(2.0f * q * E[2]);

    const double virial_xx = double(q * V[0]);
    const double virial_yy = double(q * V[1]);
    const double virial_zz = double(q * V[2]);
    const double virial_xy = double(q * V[3]);
    const double virial_yz = double(q * V[4]);
    const double virial_zx = double(q * V[5]);
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
  }
}

inline void pppm_destroy(PPPM_Data& pppm)
{
  if (pppm.plan_initialized) {
    cufftDestroy(pppm.plan);
    pppm.plan_initialized = false;
  }
  if (pppm.plan_virial_initialized) {
    cufftDestroy(pppm.plan_virial);
    pppm.plan_virial_initialized = false;
  }
}

inline void pppm_allocate_memory(PPPM_Data& pppm)
{
  pppm_destroy(pppm);
  const int K0K1K2 = pppm.para.K0K1K2;
  pppm.kx.resize(K0K1K2);
  pppm.ky.resize(K0K1K2);
  pppm.kz.resize(K0K1K2);
  pppm.G.resize(K0K1K2);
  pppm.mesh.resize(K0K1K2);
  pppm.mesh_G.resize(K0K1K2);
  pppm.mesh_x.resize(K0K1K2);
  pppm.mesh_y.resize(K0K1K2);
  pppm.mesh_z.resize(K0K1K2);
  pppm.mesh_virial.resize(K0K1K2 * 6);
  if (cufftPlan3d(&pppm.plan, pppm.para.K[2], pppm.para.K[1], pppm.para.K[0], CUFFT_C2C)
      != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: PPPM plan creation failed" << std::endl;
    exit(1);
  }
  pppm.plan_initialized = true;
  int n[3] = {pppm.para.K[2], pppm.para.K[1], pppm.para.K[0]};
  if (cufftPlanMany(
        &pppm.plan_virial, 3, n, NULL, 1, K0K1K2, NULL, 1, K0K1K2, CUFFT_C2C, 6)
      != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: PPPM virial plan creation failed" << std::endl;
    exit(1);
  }
  pppm.plan_virial_initialized = true;
}

inline void pppm_find_para(PPPM_Data& pppm, const int N, const float alpha, const float alpha_factor, const Box& box)
{
  PPPM_Para& para = pppm.para;
  const float two_pi = 6.2831853f;
  const double mesh_spacing = 1.0;
  const double volume = box.get_volume();
  para.alpha = alpha;
  para.alpha_factor = alpha_factor;
  para.two_pi_over_V = two_pi / volume;
  int K[3] = {0};
  for (int d = 0; d < 3; ++d) {
    const double box_thickness = volume / box.get_area(d);
    K[d] = get_best_pppm_K(int(box_thickness / mesh_spacing));
    para.K_half[d] = K[d] / 2;
    para.two_pi_over_K[d] = two_pi / K[d];
  }
  para.K0K1 = K[0] * K[1];
  para.K0K1K2 = para.K0K1 * K[2];
  const bool need_allocate =
    !pppm.plan_initialized || K[0] != para.K[0] || K[1] != para.K[1] || K[2] != para.K[2];
  para.K[0] = K[0];
  para.K[1] = K[1];
  para.K[2] = K[2];
  para.potential_factor = K_C_SP / N;
  for (int d = 0; d < 3; ++d) {
    para.b[0][d] = two_pi * float(box.cpu_h[9 + d]);
    para.b[1][d] = two_pi * float(box.cpu_h[12 + d]);
    para.b[2][d] = two_pi * float(box.cpu_h[15 + d]);
  }
  if (need_allocate) {
    pppm_allocate_memory(pppm);
  }
}

inline void pppm_find_force_charge2(
  PPPM_Data& pppm,
  const int N,
  const int N1,
  const int N2,
  const float alpha,
  const float alpha_factor,
  const Box& box,
  const GPU_Vector<float>& charge,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<float>& D_real,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  GPU_Vector<double>& total_virial,
  GPU_Vector<double>& potential_per_atom)
{
  pppm_find_para(pppm, N, alpha, alpha_factor, box);
  const PPPM_Para para = pppm.para;
  const int mesh_grid_size = (para.K0K1K2 - 1) / 64 + 1;
  const int atom_grid_size = (N2 - N1 - 1) / 64 + 1;

  pppm_find_k_and_G<<<mesh_grid_size, 64>>>(
    para,
    pppm.kx.data(),
    pppm.ky.data(),
    pppm.kz.data(),
    pppm.G.data());
  CUDA_CHECK_KERNEL

  pppm_set_mesh_to_zero<<<mesh_grid_size, 64>>>(para, pppm.mesh.data());
  CUDA_CHECK_KERNEL

  pppm_find_mesh<<<atom_grid_size, 64>>>(
    N1,
    N2,
    para,
    box,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    pppm.mesh.data());
  CUDA_CHECK_KERNEL

  if (cufftExecC2C(pppm.plan, pppm.mesh.data(), pppm.mesh.data(), CUFFT_FORWARD)
      != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: PPPM forward transform failed" << std::endl;
    exit(1);
  }

  pppm_ik_times_mesh_times_G<<<mesh_grid_size, 64>>>(
    para,
    pppm.kx.data(),
    pppm.ky.data(),
    pppm.kz.data(),
    pppm.G.data(),
    pppm.mesh.data(),
    pppm.mesh_x.data(),
    pppm.mesh_y.data(),
    pppm.mesh_z.data());
  CUDA_CHECK_KERNEL

  pppm_find_mesh_G<<<mesh_grid_size, 64>>>(
    para,
    pppm.G.data(),
    pppm.mesh.data(),
    pppm.mesh_G.data());
  CUDA_CHECK_KERNEL

  for (int d = 0; d < 6; ++d) {
    pppm_set_mesh_to_zero<<<mesh_grid_size, 64>>>(
      para, pppm.mesh_virial.data() + para.K0K1K2 * d);
    CUDA_CHECK_KERNEL
  }
  pppm_find_mesh_virial<<<mesh_grid_size, 64>>>(
    para,
    pppm.kx.data(),
    pppm.ky.data(),
    pppm.kz.data(),
    pppm.G.data(),
    pppm.mesh.data(),
    pppm.mesh_virial.data() + para.K0K1K2 * 0,
    pppm.mesh_virial.data() + para.K0K1K2 * 1,
    pppm.mesh_virial.data() + para.K0K1K2 * 2,
    pppm.mesh_virial.data() + para.K0K1K2 * 3,
    pppm.mesh_virial.data() + para.K0K1K2 * 4,
    pppm.mesh_virial.data() + para.K0K1K2 * 5);
  CUDA_CHECK_KERNEL

  if (cufftExecC2C(pppm.plan, pppm.mesh_G.data(), pppm.mesh_G.data(), CUFFT_INVERSE)
      != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan, pppm.mesh_x.data(), pppm.mesh_x.data(), CUFFT_INVERSE)
      != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan, pppm.mesh_y.data(), pppm.mesh_y.data(), CUFFT_INVERSE)
      != CUFFT_SUCCESS ||
      cufftExecC2C(pppm.plan, pppm.mesh_z.data(), pppm.mesh_z.data(), CUFFT_INVERSE)
      != CUFFT_SUCCESS ||
      cufftExecC2C(
        pppm.plan_virial, pppm.mesh_virial.data(), pppm.mesh_virial.data(), CUFFT_INVERSE)
      != CUFFT_SUCCESS) {
    std::cout << "CUFFT error: PPPM inverse transform failed" << std::endl;
    exit(1);
  }

  pppm_find_force_virial_from_field<<<atom_grid_size, 64>>>(
    N,
    N1,
    N2,
    para,
    box,
    charge.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    pppm.mesh_G.data(),
    pppm.mesh_x.data(),
    pppm.mesh_y.data(),
    pppm.mesh_z.data(),
    pppm.mesh_virial.data() + para.K0K1K2 * 0,
    pppm.mesh_virial.data() + para.K0K1K2 * 1,
    pppm.mesh_virial.data() + para.K0K1K2 * 2,
    pppm.mesh_virial.data() + para.K0K1K2 * 3,
    pppm.mesh_virial.data() + para.K0K1K2 * 4,
    pppm.mesh_virial.data() + para.K0K1K2 * 5,
    D_real.data(),
    force_per_atom.data(),
    force_per_atom.data() + N,
    force_per_atom.data() + N * 2,
    virial_per_atom.data(),
    total_virial.data(),
    potential_per_atom.data());
  CUDA_CHECK_KERNEL
}

} // namespace

#endif // __CUDACC__
