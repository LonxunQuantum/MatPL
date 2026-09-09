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

extern "C" __global__ void fitting_atoms_backward(
    const double* x, const double* w, const double* b, const double* v,
    const long long* ids, const long long* offsets, const double* a,
    const double* u, double* dx, int n) {
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
  __shared__ double su[JIT_Q * ATOMS * JIT_D];
  __shared__ double seed[ATOMS * TILE];
  for (int f = lane; f < JIT_D; f += TILE) {
    sx[row * JIT_D + f] = atom >= 0 ? x[atom * JIT_D + f] : 0.0;
    #pragma unroll
    for (int q = 0; q < JIT_Q; ++q)
      su[(q * ATOMS + row) * JIT_D + f] =
          atom >= 0 ? u[(q * n + atom) * JIT_D + f] : 0.0;
  }
  double result[(JIT_D + TILE - 1) / TILE] = {};
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
    double t[JIT_Q] = {};
    #pragma unroll 4
    for (int f = 0; f < JIT_D; ++f) {
      const double weight = sw[f * TILE + lane];
      z += sx[row * JIT_D + f] * weight;
      #pragma unroll
      for (int q = 0; q < JIT_Q; ++q)
        t[q] += su[(q * ATOMS + row) * JIT_D + f] * weight;
    }
    const double hidden = tanh(z), slope = 1.0 - hidden * hidden;
    double r = 0.0;
    #pragma unroll
    for (int q = 0; q < JIT_Q; ++q) {
      const double head = j < JIT_H
          ? v[(type * JIT_H + j) * JIT_Q + q] : 0.0;
      const double aq = atom >= 0 ? a[q * n + atom] : 0.0;
      r += slope * head * (aq - 2.0 * hidden * t[q]);
    }
    seed[row * TILE + lane] = r;
    __syncthreads();
    for (int f = lane, k = 0; f < JIT_D; f += TILE, ++k) {
      double value = 0.0;
      #pragma unroll
      for (int j1 = 0; j1 < TILE; ++j1)
        value += sw[f * TILE + j1] * seed[row * TILE + j1];
      result[k] += value;
    }
    __syncthreads();
  }
  for (int f = lane, k = 0; f < JIT_D; f += TILE, ++k)
    if (atom >= 0) dx[atom * JIT_D + f] = result[k];
}

extern "C" __global__ void fitting_parameter_partials(
    const double* x, const double* w, const double* b, const double* v,
    const long long* ids, const long long* offsets, const double* a,
    const double* u, double* partial, int n, int type_start, int slots) {
  constexpr int TILE = 16;
  constexpr int ATOMS = 8;
  constexpr int THREADS = TILE * ATOMS;
  constexpr int CHUNK = 32;
  constexpr int TILES = (JIT_H + TILE - 1) / TILE;
  constexpr int STRIDE = (JIT_D + 1 + JIT_Q) * TILE + JIT_Q;
  constexpr int ACCUM = (STRIDE + THREADS - 1) / THREADS;
  const int tid = threadIdx.x, row = tid / TILE, lane = tid % TILE;
  const int type = type_start + blockIdx.z, tile = blockIdx.y * TILE;
  const long long first = offsets[type], end = offsets[type + 1];
  __shared__ double sw[JIT_D * TILE];
  __shared__ double sx[ATOMS * JIT_D];
  __shared__ double su[JIT_Q * ATOMS * JIT_D];
  __shared__ double sa[JIT_Q * ATOMS];
  __shared__ double sr[ATOMS * TILE];
  __shared__ double ss[ATOMS * TILE];
  __shared__ double sv[JIT_Q * ATOMS * TILE];
  for (int p = tid; p < JIT_D * TILE; p += THREADS) {
    const int j = tile + p % TILE;
    sw[p] = j < JIT_H
        ? w[(static_cast<long long>(type) * JIT_D + p / TILE) * JIT_H + j]
        : 0.0;
  }
  double acc[ACCUM] = {};
  __syncthreads();
  for (long long chunk = first + static_cast<long long>(blockIdx.x) * CHUNK;
       chunk < end; chunk += static_cast<long long>(slots) * CHUNK) {
    for (int sub = 0; sub < CHUNK; sub += ATOMS) {
      const long long pos = chunk + sub + row;
      const long long atom = pos < end ? ids[pos] : -1;
      for (int f = lane; f < JIT_D; f += TILE) {
        sx[row * JIT_D + f] = atom >= 0 ? x[atom * JIT_D + f] : 0.0;
        #pragma unroll
        for (int q = 0; q < JIT_Q; ++q)
          su[(q * ATOMS + row) * JIT_D + f] =
              atom >= 0 ? u[(q * n + atom) * JIT_D + f] : 0.0;
      }
      if (lane == 0) {
        #pragma unroll
        for (int q = 0; q < JIT_Q; ++q)
          sa[q * ATOMS + row] = atom >= 0 ? a[q * n + atom] : 0.0;
      }
      __syncthreads();
      const int j = tile + lane;
      double z = j < JIT_H ? b[type * JIT_H + j] : 0.0;
      double t[JIT_Q] = {};
      #pragma unroll 4
      for (int f = 0; f < JIT_D; ++f) {
        const double weight = sw[f * TILE + lane];
        z += sx[row * JIT_D + f] * weight;
        #pragma unroll
        for (int q = 0; q < JIT_Q; ++q)
          t[q] += su[(q * ATOMS + row) * JIT_D + f] * weight;
      }
      const double hidden = tanh(z), slope = 1.0 - hidden * hidden;
      double r = 0.0;
      #pragma unroll
      for (int q = 0; q < JIT_Q; ++q) {
        const double head = j < JIT_H
            ? v[(type * JIT_H + j) * JIT_Q + q] : 0.0;
        r += slope * head *
            (sa[q * ATOMS + row] - 2.0 * hidden * t[q]);
        sv[(q * ATOMS + row) * TILE + lane] =
            sa[q * ATOMS + row] * hidden + t[q] * slope;
      }
      sr[row * TILE + lane] = r;
      ss[row * TILE + lane] = slope;
      __syncthreads();
      #pragma unroll
      for (int k = 0; k < ACCUM; ++k) {
        const int p = tid + k * THREADS;
        if (p >= STRIDE) continue;
        double value = 0.0;
        if (p < JIT_D * TILE) {
          const int f = p / TILE, j1 = p % TILE;
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1) {
            value += sx[row1 * JIT_D + f] * sr[row1 * TILE + j1];
            #pragma unroll
            for (int q = 0; q < JIT_Q; ++q) {
              const double head = tile + j1 < JIT_H
                  ? v[(type * JIT_H + tile + j1) * JIT_Q + q] : 0.0;
              value += su[(q * ATOMS + row1) * JIT_D + f] * head *
                       ss[row1 * TILE + j1];
            }
          }
        } else if (p < (JIT_D + 1) * TILE) {
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1)
            value += sr[row1 * TILE + p % TILE];
        } else if (p < (JIT_D + 1 + JIT_Q) * TILE) {
          const int q = (p - (JIT_D + 1) * TILE) / TILE;
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1)
            value += sv[(q * ATOMS + row1) * TILE + p % TILE];
        } else if (tile == 0) {
          const int q = p - (JIT_D + 1 + JIT_Q) * TILE;
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1)
            value += sa[q * ATOMS + row1];
        }
        acc[k] += value;
      }
      __syncthreads();
    }
  }
  const long long base =
      ((static_cast<long long>(blockIdx.z) * slots + blockIdx.x) * TILES +
       blockIdx.y) * STRIDE;
  #pragma unroll
  for (int k = 0; k < ACCUM; ++k) {
    const int p = tid + k * THREADS;
    if (p < STRIDE) partial[base + p] = acc[k];
  }
}

extern "C" __global__ void fitting_parameter_reduce(
    const double* partial, double* dw, double* db, double* dv, double* dc,
    int type_start, int slots) {
  constexpr int TILE = 16;
  constexpr int TILES = (JIT_H + TILE - 1) / TILE;
  constexpr int STRIDE = (JIT_D + 1 + JIT_Q) * TILE + JIT_Q;
  constexpr int WEIGHT_SIZE = JIT_D * JIT_H;
  constexpr int PARAM_SIZE = WEIGHT_SIZE + JIT_H + JIT_H * JIT_Q + JIT_Q;
  const int local_type = blockIdx.y;
  const int type = type_start + local_type;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= PARAM_SIZE) return;
  int tile, part;
  if (p < WEIGHT_SIZE) {
    tile = (p % JIT_H) / TILE;
    part = (p / JIT_H) * TILE + p % JIT_H % TILE;
  } else if (p < WEIGHT_SIZE + JIT_H) {
    const int j = p - WEIGHT_SIZE;
    tile = j / TILE;
    part = JIT_D * TILE + j % TILE;
  } else if (p < WEIGHT_SIZE + JIT_H + JIT_H * JIT_Q) {
    const int j = (p - WEIGHT_SIZE - JIT_H) / JIT_Q;
    const int q = (p - WEIGHT_SIZE - JIT_H) % JIT_Q;
    tile = j / TILE;
    part = (JIT_D + 1 + q) * TILE + j % TILE;
  } else {
    tile = 0;
    part = (JIT_D + 1 + JIT_Q) * TILE +
           p - WEIGHT_SIZE - JIT_H - JIT_H * JIT_Q;
  }
  double value = 0.0;
  for (int s = 0; s < slots; ++s)
    value += partial[((static_cast<long long>(local_type) * slots + s) *
                      TILES + tile) * STRIDE + part];
  if (p < WEIGHT_SIZE)
    dw[static_cast<long long>(type) * WEIGHT_SIZE + p] = value;
  else if (p < WEIGHT_SIZE + JIT_H)
    db[type * JIT_H + p - WEIGHT_SIZE] = value;
  else if (p < WEIGHT_SIZE + JIT_H + JIT_H * JIT_Q)
    dv[type * JIT_H * JIT_Q + p - WEIGHT_SIZE - JIT_H] = value;
  else
    dc[type * JIT_Q + p - WEIGHT_SIZE - JIT_H - JIT_H * JIT_Q] = value;
}
)MATPL_JIT";
