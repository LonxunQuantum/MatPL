#pragma once

static constexpr const char* kNepFittingJitSource = R"MATPL_JIT(
extern "C" __global__ void fitting_atoms_forward(
    const double* x, const double* w, const double* b, const double* v,
    const double* c, const long long* ids, const long long* offsets,
    double* y, double* out, int n) {
  constexpr int TILE = 16;
  constexpr int ATOMS = 8;
  const int type = blockIdx.y;
  const long long first = offsets[type], end = offsets[type + 1];
  const long long start = first + static_cast<long long>(blockIdx.x) * ATOMS;
  if (start >= end) return;
  const int tid = threadIdx.x, row = tid / TILE, lane = tid % TILE;
  const long long atom = start + row < end ? ids[start + row] : -1;
  __shared__ double sw[JIT_D * TILE];
  __shared__ double sx[ATOMS * JIT_D];
  __shared__ double seed[JIT_Q * ATOMS * TILE];
  for (int f = lane; f < JIT_D; f += TILE)
    sx[row * JIT_D + f] = atom >= 0 ? x[atom * JIT_D + f] : 0.0;
  double result[JIT_Q][(JIT_D + TILE - 1) / TILE] = {};
  double energy[JIT_Q] = {};
  __syncthreads();
  for (int tile = 0; tile < JIT_H; tile += TILE) {
    for (int p = tid; p < JIT_D * TILE; p += TILE * ATOMS) {
      const int j = tile + p % TILE;
      sw[p] = j < JIT_H
          ? w[(static_cast<long long>(type) * JIT_D + p / TILE) * JIT_H + j]
          : 0.0;
    }
    __syncthreads();
    const int j = tile + lane;
    double z = j < JIT_H ? b[type * JIT_H + j] : 0.0;
    #pragma unroll 4
    for (int f = 0; f < JIT_D; ++f)
      z += sx[row * JIT_D + f] * sw[f * TILE + lane];
    const double hidden = tanh(z), slope = 1.0 - hidden * hidden;
    #pragma unroll
    for (int q = 0; q < JIT_Q; ++q) {
      const double head = j < JIT_H
          ? v[(type * JIT_H + j) * JIT_Q + q] : 0.0;
      energy[q] += hidden * head;
      seed[(q * ATOMS + row) * TILE + lane] = head * slope;
    }
    __syncthreads();
    #pragma unroll
    for (int q = 0; q < JIT_Q; ++q) {
      for (int f = lane, k = 0; f < JIT_D; f += TILE, ++k) {
        double value = 0.0;
        #pragma unroll
        for (int j1 = 0; j1 < TILE; ++j1)
          value += sw[f * TILE + j1] * seed[(q * ATOMS + row) * TILE + j1];
        result[q][k] += value;
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int q = 0; q < JIT_Q; ++q)
    for (int f = lane, k = 0; f < JIT_D; f += TILE, ++k)
      if (atom >= 0) out[(q * n + atom) * JIT_D + f] = result[q][k];
  #pragma unroll
  for (int q = 0; q < JIT_Q; ++q) {
    #pragma unroll
    for (int delta = TILE / 2; delta; delta /= 2)
      energy[q] += __shfl_down_sync(0xffffffff, energy[q], delta, TILE);
    if (lane == 0 && atom >= 0)
      y[q * n + atom] = energy[q] + c[type * JIT_Q + q];
  }
}
)MATPL_JIT";
