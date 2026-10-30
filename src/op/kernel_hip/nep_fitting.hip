#include <ATen/ATen.h>
#include <algorithm>
#include <cstdlib>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include "../include/nep_fitting_hip.h"
#include "../include/nep_fitting_limits.h"

namespace {

constexpr int kTile = 16;
constexpr int kAtoms = 8;
constexpr int kThreads = kTile * kAtoms;
constexpr int kMaxD = nep_fitting_limits::kMaxFeatures;
constexpr int kMaxQ = 2;
constexpr int kMaxStride = (kMaxD + 1 + kMaxQ) * kTile + kMaxQ;
constexpr int kMaxAccum = (kMaxStride + kThreads - 1) / kThreads;

__global__ void fitting_atoms_forward_hip(
    const double* x, const double* w, const double* b, const double* v,
    const double* c, const int64_t* ids, const int64_t* offsets,
    double* y, double* g, int n, int d, int h, int q) {
  const int type = blockIdx.y;
  const int64_t first = offsets[type], end = offsets[type + 1];
  const int64_t start = first + static_cast<int64_t>(blockIdx.x) * kAtoms;
  if (start >= end) return;
  const int tid = threadIdx.x, row = tid / kTile, lane = tid % kTile;
  const int64_t atom = start + row < end ? ids[start + row] : -1;

  __shared__ double sw[kMaxD * kTile];
  __shared__ double sx[kAtoms * kMaxD];
  __shared__ double seed[kMaxQ * kAtoms * kTile];
  __shared__ double result[kMaxQ * kAtoms * kMaxD];
  __shared__ double energy_reduce[kMaxQ * kAtoms * kTile];

  for (int f = lane; f < d; f += kTile)
    sx[row * kMaxD + f] = atom >= 0 ? x[atom * d + f] : 0.0;
  for (int qi = 0; qi < q; ++qi)
    for (int f = lane; f < d; f += kTile)
      result[(qi * kAtoms + row) * kMaxD + f] = 0.0;
  double energy[kMaxQ] = {0.0, 0.0};
  __syncthreads();

  for (int tile = 0; tile < h; tile += kTile) {
    for (int p = tid; p < d * kTile; p += kThreads) {
      const int j = tile + p % kTile;
      sw[p] = j < h
          ? w[(static_cast<int64_t>(type) * d + p / kTile) * h + j]
          : 0.0;
    }
    __syncthreads();
    const int j = tile + lane;
    double z = j < h ? b[type * h + j] : 0.0;
    for (int f = 0; f < d; ++f)
      z += sx[row * kMaxD + f] * sw[f * kTile + lane];
    const double hidden = tanh(z);
    const double slope = 1.0 - hidden * hidden;
    for (int qi = 0; qi < q; ++qi) {
      const double head = j < h ? v[(type * h + j) * q + qi] : 0.0;
      energy[qi] += hidden * head;
      seed[(qi * kAtoms + row) * kTile + lane] = head * slope;
    }
    __syncthreads();
    for (int qi = 0; qi < q; ++qi) {
      for (int f = lane; f < d; f += kTile) {
        double value = 0.0;
        for (int j1 = 0; j1 < kTile; ++j1)
          value += sw[f * kTile + j1] *
              seed[(qi * kAtoms + row) * kTile + j1];
        result[(qi * kAtoms + row) * kMaxD + f] += value;
      }
    }
    __syncthreads();
  }

  for (int qi = 0; qi < q; ++qi) {
    for (int f = lane; f < d; f += kTile)
      if (atom >= 0)
        g[(static_cast<int64_t>(qi) * n + atom) * d + f] =
            result[(qi * kAtoms + row) * kMaxD + f];
    energy_reduce[(qi * kAtoms + row) * kTile + lane] = energy[qi];
  }
  __syncthreads();
  if (lane == 0 && atom >= 0) {
    for (int qi = 0; qi < q; ++qi) {
      double total = 0.0;
      for (int j1 = 0; j1 < kTile; ++j1)
        total += energy_reduce[(qi * kAtoms + row) * kTile + j1];
      y[static_cast<int64_t>(qi) * n + atom] = total + c[type * q + qi];
    }
  }
}

__global__ void fitting_atoms_backward_hip(
    const double* x, const double* w, const double* b, const double* v,
    const int64_t* ids, const int64_t* offsets, const double* a,
    const double* u, double* dx, int n, int d, int h, int q) {
  const int type = blockIdx.y;
  const int64_t first = offsets[type], end = offsets[type + 1];
  const int64_t start = first + static_cast<int64_t>(blockIdx.x) * kAtoms;
  if (start >= end) return;
  const int tid = threadIdx.x, row = tid / kTile, lane = tid % kTile;
  const int64_t atom = start + row < end ? ids[start + row] : -1;

  __shared__ double sw[kMaxD * kTile];
  __shared__ double sx[kAtoms * kMaxD];
  __shared__ double su[kMaxQ * kAtoms * kMaxD];
  __shared__ double seed[kAtoms * kTile];
  __shared__ double result[kAtoms * kMaxD];

  for (int f = lane; f < d; f += kTile) {
    sx[row * kMaxD + f] = atom >= 0 ? x[atom * d + f] : 0.0;
    for (int qi = 0; qi < q; ++qi)
      su[(qi * kAtoms + row) * kMaxD + f] =
          atom >= 0 ? u[(static_cast<int64_t>(qi) * n + atom) * d + f] : 0.0;
    result[row * kMaxD + f] = 0.0;
  }
  __syncthreads();

  for (int tile = 0; tile < h; tile += kTile) {
    for (int p = tid; p < d * kTile; p += kThreads) {
      const int j = tile + p % kTile;
      sw[p] = j < h
          ? w[(static_cast<int64_t>(type) * d + p / kTile) * h + j]
          : 0.0;
    }
    __syncthreads();
    const int j = tile + lane;
    double z = j < h ? b[type * h + j] : 0.0;
    double t[kMaxQ] = {0.0, 0.0};
    for (int f = 0; f < d; ++f) {
      const double weight = sw[f * kTile + lane];
      z += sx[row * kMaxD + f] * weight;
      for (int qi = 0; qi < q; ++qi)
        t[qi] += su[(qi * kAtoms + row) * kMaxD + f] * weight;
    }
    const double hidden = tanh(z);
    const double slope = 1.0 - hidden * hidden;
    double r = 0.0;
    for (int qi = 0; qi < q; ++qi) {
      const double head = j < h ? v[(type * h + j) * q + qi] : 0.0;
      const double aq = atom >= 0 ? a[static_cast<int64_t>(qi) * n + atom] : 0.0;
      r += slope * head * (aq - 2.0 * hidden * t[qi]);
    }
    seed[row * kTile + lane] = r;
    __syncthreads();
    for (int f = lane; f < d; f += kTile) {
      double value = 0.0;
      for (int j1 = 0; j1 < kTile; ++j1)
        value += sw[f * kTile + j1] * seed[row * kTile + j1];
      result[row * kMaxD + f] += value;
    }
    __syncthreads();
  }
  for (int f = lane; f < d; f += kTile)
    if (atom >= 0) dx[atom * d + f] = result[row * kMaxD + f];
}

__global__ void fitting_parameters_backward_hip(
    const double* x, const double* w, const double* b, const double* v,
    const int64_t* ids, const int64_t* offsets, const double* a,
    const double* u, double* dw, double* db, double* dv, double* dc,
    int n, int d, int h, int q) {
  const int type = blockIdx.y;
  const int tile = blockIdx.x * kTile;
  const int tid = threadIdx.x, row = tid / kTile, lane = tid % kTile;
  const int stride = (d + 1 + q) * kTile + q;
  const int64_t first = offsets[type], end = offsets[type + 1];

  __shared__ double sw[kMaxD * kTile];
  __shared__ double sx[kAtoms * kMaxD];
  __shared__ double su[kMaxQ * kAtoms * kMaxD];
  __shared__ double sa[kMaxQ * kAtoms];
  __shared__ double sr[kAtoms * kTile];
  __shared__ double ss[kAtoms * kTile];
  __shared__ double sv[kMaxQ * kAtoms * kTile];

  for (int p = tid; p < d * kTile; p += kThreads) {
    const int j = tile + p % kTile;
    sw[p] = j < h
        ? w[(static_cast<int64_t>(type) * d + p / kTile) * h + j]
        : 0.0;
  }
  double acc[kMaxAccum] = {};
  __syncthreads();

  for (int64_t start = first; start < end; start += kAtoms) {
    const int64_t atom = start + row < end ? ids[start + row] : -1;
    for (int f = lane; f < d; f += kTile) {
      sx[row * kMaxD + f] = atom >= 0 ? x[atom * d + f] : 0.0;
      for (int qi = 0; qi < q; ++qi)
        su[(qi * kAtoms + row) * kMaxD + f] =
            atom >= 0 ? u[(static_cast<int64_t>(qi) * n + atom) * d + f] : 0.0;
    }
    if (lane == 0)
      for (int qi = 0; qi < q; ++qi)
        sa[qi * kAtoms + row] =
            atom >= 0 ? a[static_cast<int64_t>(qi) * n + atom] : 0.0;
    __syncthreads();

    const int j = tile + lane;
    double z = j < h ? b[type * h + j] : 0.0;
    double t[kMaxQ] = {0.0, 0.0};
    for (int f = 0; f < d; ++f) {
      const double weight = sw[f * kTile + lane];
      z += sx[row * kMaxD + f] * weight;
      for (int qi = 0; qi < q; ++qi)
        t[qi] += su[(qi * kAtoms + row) * kMaxD + f] * weight;
    }
    const double hidden = tanh(z);
    const double slope = 1.0 - hidden * hidden;
    double r = 0.0;
    for (int qi = 0; qi < q; ++qi) {
      const double head = j < h ? v[(type * h + j) * q + qi] : 0.0;
      r += slope * head *
          (sa[qi * kAtoms + row] - 2.0 * hidden * t[qi]);
      sv[(qi * kAtoms + row) * kTile + lane] =
          sa[qi * kAtoms + row] * hidden + t[qi] * slope;
    }
    sr[row * kTile + lane] = r;
    ss[row * kTile + lane] = slope;
    __syncthreads();

    for (int k = 0; k < kMaxAccum; ++k) {
      const int p = tid + k * kThreads;
      if (p >= stride) continue;
      double value = 0.0;
      if (p < d * kTile) {
        const int f = p / kTile, j1 = p % kTile;
        for (int row1 = 0; row1 < kAtoms; ++row1) {
          value += sx[row1 * kMaxD + f] * sr[row1 * kTile + j1];
          for (int qi = 0; qi < q; ++qi) {
            const double head = tile + j1 < h
                ? v[(type * h + tile + j1) * q + qi] : 0.0;
            value += su[(qi * kAtoms + row1) * kMaxD + f] * head *
                ss[row1 * kTile + j1];
          }
        }
      } else if (p < (d + 1) * kTile) {
        for (int row1 = 0; row1 < kAtoms; ++row1)
          value += sr[row1 * kTile + p % kTile];
      } else if (p < (d + 1 + q) * kTile) {
        const int qi = (p - (d + 1) * kTile) / kTile;
        for (int row1 = 0; row1 < kAtoms; ++row1)
          value += sv[(qi * kAtoms + row1) * kTile + p % kTile];
      } else if (tile == 0) {
        const int qi = p - (d + 1 + q) * kTile;
        for (int row1 = 0; row1 < kAtoms; ++row1)
          value += sa[qi * kAtoms + row1];
      }
      acc[k] += value;
    }
    __syncthreads();
  }

  for (int k = 0; k < kMaxAccum; ++k) {
    const int p = tid + k * kThreads;
    if (p >= stride) continue;
    if (p < d * kTile) {
      const int f = p / kTile, j1 = p % kTile, j = tile + j1;
      if (j < h) dw[(static_cast<int64_t>(type) * d + f) * h + j] = acc[k];
    } else if (p < (d + 1) * kTile) {
      const int j = tile + p % kTile;
      if (j < h) db[type * h + j] = acc[k];
    } else if (p < (d + 1 + q) * kTile) {
      const int qi = (p - (d + 1) * kTile) / kTile;
      const int j = tile + p % kTile;
      if (j < h) dv[(type * h + j) * q + qi] = acc[k];
    } else if (tile == 0) {
      const int qi = p - (d + 1 + q) * kTile;
      if (qi < q) dc[type * q + qi] = acc[k];
    }
  }
}

__global__ void fitting_parameter_partials_hip(
    const double* x, const double* w, const double* b, const double* v,
    const int64_t* ids, const int64_t* offsets, const double* a,
    const double* u, double* partial, int slots,
    int n, int d, int h, int q) {
  const int type = blockIdx.z;
  const int tile = blockIdx.y * kTile;
  const int tid = threadIdx.x, row = tid / kTile, lane = tid % kTile;
  const int stride = (d + 1 + q) * kTile + q;
  const int64_t first = offsets[type], end = offsets[type + 1];

  __shared__ double sw[kMaxD * kTile];
  __shared__ double sx[kAtoms * kMaxD];
  __shared__ double su[kMaxQ * kAtoms * kMaxD];
  __shared__ double sa[kMaxQ * kAtoms];
  __shared__ double sr[kAtoms * kTile];
  __shared__ double ss[kAtoms * kTile];
  __shared__ double sv[kMaxQ * kAtoms * kTile];

  for (int p = tid; p < d * kTile; p += kThreads) {
    const int j = tile + p % kTile;
    sw[p] = j < h
        ? w[(static_cast<int64_t>(type) * d + p / kTile) * h + j]
        : 0.0;
  }
  double acc[kMaxAccum] = {};
  __syncthreads();

  for (int64_t chunk = first + static_cast<int64_t>(blockIdx.x) * 32;
       chunk < end; chunk += static_cast<int64_t>(slots) * 32) {
    for (int sub = 0; sub < 32; sub += kAtoms) {
    const int64_t start = chunk + sub;
    const int64_t atom = start + row < end ? ids[start + row] : -1;
    for (int f = lane; f < d; f += kTile) {
      sx[row * kMaxD + f] = atom >= 0 ? x[atom * d + f] : 0.0;
      for (int qi = 0; qi < q; ++qi)
        su[(qi * kAtoms + row) * kMaxD + f] =
            atom >= 0 ? u[(static_cast<int64_t>(qi) * n + atom) * d + f] : 0.0;
    }
    if (lane == 0)
      for (int qi = 0; qi < q; ++qi)
        sa[qi * kAtoms + row] =
            atom >= 0 ? a[static_cast<int64_t>(qi) * n + atom] : 0.0;
    __syncthreads();

    const int j = tile + lane;
    double z = j < h ? b[type * h + j] : 0.0;
    double t[kMaxQ] = {0.0, 0.0};
    for (int f = 0; f < d; ++f) {
      const double weight = sw[f * kTile + lane];
      z += sx[row * kMaxD + f] * weight;
      for (int qi = 0; qi < q; ++qi)
        t[qi] += su[(qi * kAtoms + row) * kMaxD + f] * weight;
    }
    const double hidden = tanh(z);
    const double slope = 1.0 - hidden * hidden;
    double r = 0.0;
    for (int qi = 0; qi < q; ++qi) {
      const double head = j < h ? v[(type * h + j) * q + qi] : 0.0;
      r += slope * head *
          (sa[qi * kAtoms + row] - 2.0 * hidden * t[qi]);
      sv[(qi * kAtoms + row) * kTile + lane] =
          sa[qi * kAtoms + row] * hidden + t[qi] * slope;
    }
    sr[row * kTile + lane] = r;
    ss[row * kTile + lane] = slope;
    __syncthreads();

    for (int k = 0; k < kMaxAccum; ++k) {
      const int p = tid + k * kThreads;
      if (p >= stride) continue;
      double value = 0.0;
      if (p < d * kTile) {
        const int f = p / kTile, j1 = p % kTile;
        for (int row1 = 0; row1 < kAtoms; ++row1) {
          value += sx[row1 * kMaxD + f] * sr[row1 * kTile + j1];
          for (int qi = 0; qi < q; ++qi) {
            const double head = tile + j1 < h
                ? v[(type * h + tile + j1) * q + qi] : 0.0;
            value += su[(qi * kAtoms + row1) * kMaxD + f] * head *
                ss[row1 * kTile + j1];
          }
        }
      } else if (p < (d + 1) * kTile) {
        for (int row1 = 0; row1 < kAtoms; ++row1)
          value += sr[row1 * kTile + p % kTile];
      } else if (p < (d + 1 + q) * kTile) {
        const int qi = (p - (d + 1) * kTile) / kTile;
        for (int row1 = 0; row1 < kAtoms; ++row1)
          value += sv[(qi * kAtoms + row1) * kTile + p % kTile];
      } else if (tile == 0) {
        const int qi = p - (d + 1 + q) * kTile;
        for (int row1 = 0; row1 < kAtoms; ++row1)
          value += sa[qi * kAtoms + row1];
      }
      acc[k] += value;
    }
    __syncthreads();
  }

  } // chunks
  const int tiles = (h + kTile - 1) / kTile;
  const int64_t base = ((static_cast<int64_t>(type) * slots + blockIdx.x) * tiles + blockIdx.y) * stride;
  for (int k = 0; k < kMaxAccum; ++k) {
    const int p = tid + k * kThreads;
    if (p < stride) partial[base + p] = acc[k];
  }
}

__global__ void fitting_parameter_reduce_hip(
    const double* partial, double* dw, double* db, double* dv, double* dc,
    int d, int h, int q, int slots) {
  const int type = blockIdx.y;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int weight_size = d * h;
  const int param_size = weight_size + h + h * q + q;
  if (p >= param_size) return;
  const int tiles = (h + kTile - 1) / kTile;
  const int stride = (d + 1 + q) * kTile + q;
  int tile, part;
  if (p < weight_size) {
    tile = (p % h) / kTile;
    part = (p / h) * kTile + p % h % kTile;
  } else if (p < weight_size + h) {
    const int j = p - weight_size;
    tile = j / kTile;
    part = d * kTile + j % kTile;
  } else if (p < weight_size + h + h * q) {
    const int j = (p - weight_size - h) / q;
    const int qi = (p - weight_size - h) % q;
    tile = j / kTile;
    part = (d + 1 + qi) * kTile + j % kTile;
  } else {
    tile = 0;
    part = (d + 1 + q) * kTile + p - weight_size - h - h * q;
  }
  double value = 0.0;
  for (int slot = 0; slot < slots; ++slot)
    value += partial[((static_cast<int64_t>(type) * slots + slot) * tiles + tile) * stride + part];
  if (p < weight_size) dw[static_cast<int64_t>(type) * weight_size + p] = value;
  else if (p < weight_size + h) db[type * h + p - weight_size] = value;
  else if (p < weight_size + h + h * q) dv[type * h * q + p - weight_size - h] = value;
  else dc[type * q + p - weight_size - h - h * q] = value;
}

template<int D, int H, int Q>
__global__ void fitting_parameter_partials_specialized_hip(
    const double* x, const double* w, const double* b, const double* v,
    const int64_t* ids, const int64_t* offsets, const double* a,
    const double* u, double* partial, int n, int type_start, int slots) {
  constexpr int TILE = D > 96 ? 8 : 16;
  constexpr int ATOMS = 8;
  constexpr int THREADS = TILE * ATOMS;
  constexpr int CHUNK = 32;
  constexpr int TILES = (H + TILE - 1) / TILE;
  constexpr int STRIDE = (D + 1 + Q) * TILE + Q;
  constexpr int ACCUM = (STRIDE + THREADS - 1) / THREADS;
  const int tid = threadIdx.x, row = tid / TILE, lane = tid % TILE;
  const int type = type_start + blockIdx.z, tile = blockIdx.y * TILE;
  const int64_t first = offsets[type], end = offsets[type + 1];
  __shared__ double sw[D * TILE];
  __shared__ double sx[ATOMS * D];
  __shared__ double su[Q * ATOMS * D];
  __shared__ double sa[Q * ATOMS];
  __shared__ double sr[ATOMS * TILE];
  __shared__ double ss[ATOMS * TILE];
  __shared__ double sv[Q * ATOMS * TILE];
  for (int p = tid; p < D * TILE; p += THREADS) {
    const int j = tile + p % TILE;
    sw[p] = j < H
        ? w[(static_cast<int64_t>(type) * D + p / TILE) * H + j]
        : 0.0;
  }
  double acc[ACCUM] = {};
  __syncthreads();
  for (int64_t chunk = first + static_cast<int64_t>(blockIdx.x) * CHUNK;
       chunk < end; chunk += static_cast<int64_t>(slots) * CHUNK) {
    for (int sub = 0; sub < CHUNK; sub += ATOMS) {
      const int64_t pos = chunk + sub + row;
      const int64_t atom = pos < end ? ids[pos] : -1;
      for (int f = lane; f < D; f += TILE) {
        sx[row * D + f] = atom >= 0 ? x[atom * D + f] : 0.0;
        #pragma unroll
        for (int q = 0; q < Q; ++q)
          su[(q * ATOMS + row) * D + f] =
              atom >= 0 ? u[(q * n + atom) * D + f] : 0.0;
      }
      if (lane == 0) {
        #pragma unroll
        for (int q = 0; q < Q; ++q)
          sa[q * ATOMS + row] = atom >= 0 ? a[q * n + atom] : 0.0;
      }
      __syncthreads();
      const int j = tile + lane;
      double z = j < H ? b[type * H + j] : 0.0;
      double t[Q] = {};
      #pragma unroll 4
      for (int f = 0; f < D; ++f) {
        const double weight = sw[f * TILE + lane];
        z += sx[row * D + f] * weight;
        #pragma unroll
        for (int q = 0; q < Q; ++q)
          t[q] += su[(q * ATOMS + row) * D + f] * weight;
      }
      const double hidden = tanh(z), slope = 1.0 - hidden * hidden;
      double r = 0.0;
      #pragma unroll
      for (int q = 0; q < Q; ++q) {
        const double head = j < H
            ? v[(type * H + j) * Q + q] : 0.0;
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
        if (p < D * TILE) {
          const int f = p / TILE, j1 = p % TILE;
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1) {
            value += sx[row1 * D + f] * sr[row1 * TILE + j1];
            #pragma unroll
            for (int q = 0; q < Q; ++q) {
              const double head = tile + j1 < H
                  ? v[(type * H + tile + j1) * Q + q] : 0.0;
              value += su[(q * ATOMS + row1) * D + f] * head *
                       ss[row1 * TILE + j1];
            }
          }
        } else if (p < (D + 1) * TILE) {
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1)
            value += sr[row1 * TILE + p % TILE];
        } else if (p < (D + 1 + Q) * TILE) {
          const int q = (p - (D + 1) * TILE) / TILE;
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1)
            value += sv[(q * ATOMS + row1) * TILE + p % TILE];
        } else if (tile == 0) {
          const int q = p - (D + 1 + Q) * TILE;
          #pragma unroll
          for (int row1 = 0; row1 < ATOMS; ++row1)
            value += sa[q * ATOMS + row1];
        }
        acc[k] += value;
      }
      __syncthreads();
    }
  }
  const int64_t base =
      ((static_cast<int64_t>(blockIdx.z) * slots + blockIdx.x) * TILES +
       blockIdx.y) * STRIDE;
  #pragma unroll
  for (int k = 0; k < ACCUM; ++k) {
    const int p = tid + k * THREADS;
    if (p < STRIDE) partial[base + p] = acc[k];
  }
}



// CUDA specializes by runtime D/H/Q. These AOT instances cover the seven cases.
// Other bounded shapes retain the generic HIP implementation.
#define MATPL_FITTING_SHAPES(APPLY) \
  APPLY(32, 80, 1) \
  APPLY(35, 40, 1) \
  APPLY(35, 100, 1) \
  APPLY(82, 40, 1) \
  APPLY(95, 60, 1) \
  APPLY(95, 100, 1)

bool fitting_specialization_enabled() {
  const char* value = std::getenv("MATPL_FITTING_GENERIC");
  return !(value && value[0] == '1');
}

template<int D, int H, int Q>
__global__ void fitting_atoms_forward_specialized_hip(
    const double* x, const double* w, const double* b, const double* v,
    const double* c, const int64_t* ids, const int64_t* offsets,
    double* y, double* g, int n) {
  const int type = blockIdx.y;
  const int64_t first = offsets[type], end = offsets[type + 1];
  const int64_t start = first + static_cast<int64_t>(blockIdx.x) * kAtoms;
  if (start >= end) return;
  const int tid = threadIdx.x, row = tid / kTile, lane = tid % kTile;
  const int64_t atom = start + row < end ? ids[start + row] : -1;

  __shared__ double sw[D * kTile];
  __shared__ double sx[kAtoms * D];
  __shared__ double seed[Q * kAtoms * kTile];
  __shared__ double result[Q * kAtoms * D];
  __shared__ double energy_reduce[Q * kAtoms * kTile];

  for (int f = lane; f < D; f += kTile)
    sx[row * D + f] = atom >= 0 ? x[atom * D + f] : 0.0;
  for (int qi = 0; qi < Q; ++qi)
    for (int f = lane; f < D; f += kTile)
      result[(qi * kAtoms + row) * D + f] = 0.0;
  double energy[Q] = {};
  __syncthreads();

  for (int tile = 0; tile < H; tile += kTile) {
    for (int p = tid; p < D * kTile; p += kThreads) {
      const int j = tile + p % kTile;
      sw[p] = j < H
          ? w[(static_cast<int64_t>(type) * D + p / kTile) * H + j]
          : 0.0;
    }
    __syncthreads();
    const int j = tile + lane;
    double z = j < H ? b[type * H + j] : 0.0;
    for (int f = 0; f < D; ++f)
      z += sx[row * D + f] * sw[f * kTile + lane];
    const double hidden = tanh(z);
    const double slope = 1.0 - hidden * hidden;
    for (int qi = 0; qi < Q; ++qi) {
      const double head = j < H ? v[(type * H + j) * Q + qi] : 0.0;
      energy[qi] += hidden * head;
      seed[(qi * kAtoms + row) * kTile + lane] = head * slope;
    }
    __syncthreads();
    for (int qi = 0; qi < Q; ++qi) {
      for (int f = lane; f < D; f += kTile) {
        double value = 0.0;
        for (int j1 = 0; j1 < kTile; ++j1)
          value += sw[f * kTile + j1] *
              seed[(qi * kAtoms + row) * kTile + j1];
        result[(qi * kAtoms + row) * D + f] += value;
      }
    }
    __syncthreads();
  }

  for (int qi = 0; qi < Q; ++qi) {
    for (int f = lane; f < D; f += kTile)
      if (atom >= 0)
        g[(static_cast<int64_t>(qi) * n + atom) * D + f] =
            result[(qi * kAtoms + row) * D + f];
    energy_reduce[(qi * kAtoms + row) * kTile + lane] = energy[qi];
  }
  __syncthreads();
  if (lane == 0 && atom >= 0) {
    for (int qi = 0; qi < Q; ++qi) {
      double total = 0.0;
      for (int j1 = 0; j1 < kTile; ++j1)
        total += energy_reduce[(qi * kAtoms + row) * kTile + j1];
      y[static_cast<int64_t>(qi) * n + atom] = total + c[type * Q + qi];
    }
  }
}

template<int D, int H, int Q>
__global__ void fitting_atoms_backward_specialized_hip(
    const double* x, const double* w, const double* b, const double* v,
    const int64_t* ids, const int64_t* offsets, const double* a,
    const double* u, double* dx, int n) {
  const int type = blockIdx.y;
  const int64_t first = offsets[type], end = offsets[type + 1];
  const int64_t start = first + static_cast<int64_t>(blockIdx.x) * kAtoms;
  if (start >= end) return;
  const int tid = threadIdx.x, row = tid / kTile, lane = tid % kTile;
  const int64_t atom = start + row < end ? ids[start + row] : -1;

  __shared__ double sw[D * kTile];
  __shared__ double sx[kAtoms * D];
  __shared__ double su[Q * kAtoms * D];
  __shared__ double seed[kAtoms * kTile];
  __shared__ double result[kAtoms * D];

  for (int f = lane; f < D; f += kTile) {
    sx[row * D + f] = atom >= 0 ? x[atom * D + f] : 0.0;
    for (int qi = 0; qi < Q; ++qi)
      su[(qi * kAtoms + row) * D + f] =
          atom >= 0 ? u[(static_cast<int64_t>(qi) * n + atom) * D + f] : 0.0;
    result[row * D + f] = 0.0;
  }
  __syncthreads();

  for (int tile = 0; tile < H; tile += kTile) {
    for (int p = tid; p < D * kTile; p += kThreads) {
      const int j = tile + p % kTile;
      sw[p] = j < H
          ? w[(static_cast<int64_t>(type) * D + p / kTile) * H + j]
          : 0.0;
    }
    __syncthreads();
    const int j = tile + lane;
    double z = j < H ? b[type * H + j] : 0.0;
    double t[Q] = {};
    for (int f = 0; f < D; ++f) {
      const double weight = sw[f * kTile + lane];
      z += sx[row * D + f] * weight;
      for (int qi = 0; qi < Q; ++qi)
        t[qi] += su[(qi * kAtoms + row) * D + f] * weight;
    }
    const double hidden = tanh(z);
    const double slope = 1.0 - hidden * hidden;
    double r = 0.0;
    for (int qi = 0; qi < Q; ++qi) {
      const double head = j < H ? v[(type * H + j) * Q + qi] : 0.0;
      const double aq = atom >= 0 ? a[static_cast<int64_t>(qi) * n + atom] : 0.0;
      r += slope * head * (aq - 2.0 * hidden * t[qi]);
    }
    seed[row * kTile + lane] = r;
    __syncthreads();
    for (int f = lane; f < D; f += kTile) {
      double value = 0.0;
      for (int j1 = 0; j1 < kTile; ++j1)
        value += sw[f * kTile + j1] * seed[row * kTile + j1];
      result[row * D + f] += value;
    }
    __syncthreads();
  }
  for (int f = lane; f < D; f += kTile)
    if (atom >= 0) dx[atom * D + f] = result[row * D + f];
}

template<int D, int H, int Q>
__global__ void fitting_parameter_reduce_specialized_hip(
    const double* partial, double* dw, double* db, double* dv, double* dc,
    int slots) {
  const int type = blockIdx.y;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  const int weight_size = D * H;
  const int param_size = weight_size + H + H * Q + Q;
  if (p >= param_size) return;
  const int tiles = (H + kTile - 1) / kTile;
  const int stride = (D + 1 + Q) * kTile + Q;
  int tile, part;
  if (p < weight_size) {
    tile = (p % H) / kTile;
    part = (p / H) * kTile + p % H % kTile;
  } else if (p < weight_size + H) {
    const int j = p - weight_size;
    tile = j / kTile;
    part = D * kTile + j % kTile;
  } else if (p < weight_size + H + H * Q) {
    const int j = (p - weight_size - H) / Q;
    const int qi = (p - weight_size - H) % Q;
    tile = j / kTile;
    part = (D + 1 + qi) * kTile + j % kTile;
  } else {
    tile = 0;
    part = (D + 1 + Q) * kTile + p - weight_size - H - H * Q;
  }
  double value = 0.0;
  for (int slot = 0; slot < slots; ++slot)
    value += partial[((static_cast<int64_t>(type) * slots + slot) * tiles + tile) * stride + part];
  if (p < weight_size) dw[static_cast<int64_t>(type) * weight_size + p] = value;
  else if (p < weight_size + H) db[type * H + p - weight_size] = value;
  else if (p < weight_size + H + H * Q) dv[type * H * Q + p - weight_size - H] = value;
  else dc[type * Q + p - weight_size - H - H * Q] = value;
}


}  // namespace

void launch_nep_fitting_hip_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y,
    at::Tensor& g) {
  const int n = static_cast<int>(x.size(0));
  const int d = static_cast<int>(x.size(1));
  const int h = static_cast<int>(w.size(2));
  const int q = static_cast<int>(v.size(2));
  const auto specialized_stream = c10::hip::getCurrentHIPStream(x.get_device()).stream();
#define MATPL_FITTING_FORWARD(D, H, Q) \
  if (fitting_specialization_enabled() && d == D && h == H && q == Q) { \
    hipLaunchKernelGGL((fitting_atoms_forward_specialized_hip<D,H,Q>), \
        dim3((max_count + kAtoms - 1) / kAtoms, w.size(0)), dim3(kThreads), 0, specialized_stream, \
        x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(), \
        v.data_ptr<double>(), c.data_ptr<double>(), atom_ids.data_ptr<int64_t>(), \
        offsets.data_ptr<int64_t>(), y.data_ptr<double>(), g.data_ptr<double>(), n); \
    C10_HIP_KERNEL_LAUNCH_CHECK(); \
    return; \
  }
  MATPL_FITTING_SHAPES(MATPL_FITTING_FORWARD)
#undef MATPL_FITTING_FORWARD
  const dim3 grid((max_count + kAtoms - 1) / kAtoms, w.size(0));
  const auto stream = c10::hip::getCurrentHIPStream(x.get_device()).stream();
  hipLaunchKernelGGL(fitting_atoms_forward_hip, grid, dim3(kThreads), 0, stream,
      x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(),
      v.data_ptr<double>(), c.data_ptr<double>(), atom_ids.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(), y.data_ptr<double>(), g.data_ptr<double>(),
      n, d, h, q);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

void launch_nep_fitting_hip_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count,
    const at::Tensor& grad_y, const at::Tensor& grad_g,
    std::vector<at::Tensor>& grads) {
  const int n = static_cast<int>(x.size(0));
  const int d = static_cast<int>(x.size(1));
  const int h = static_cast<int>(w.size(2));
  const int q = static_cast<int>(v.size(2));
  const auto stream = c10::hip::getCurrentHIPStream(x.get_device()).stream();
  const dim3 atom_grid((max_count + kAtoms - 1) / kAtoms, w.size(0));
  bool specialized_dx = false;
#define MATPL_FITTING_DX(D, H, Q) \
  if (fitting_specialization_enabled() && d == D && h == H && q == Q) { \
    hipLaunchKernelGGL((fitting_atoms_backward_specialized_hip<D,H,Q>), atom_grid, dim3(kThreads), 0, stream, \
        x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(), \
        v.data_ptr<double>(), atom_ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), \
        grad_y.data_ptr<double>(), grad_g.data_ptr<double>(), grads[0].data_ptr<double>(), n); \
    specialized_dx = true; \
  }
  MATPL_FITTING_SHAPES(MATPL_FITTING_DX)
#undef MATPL_FITTING_DX
  if (!specialized_dx) {
  hipLaunchKernelGGL(fitting_atoms_backward_hip, atom_grid, dim3(kThreads), 0, stream,
      x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(),
      v.data_ptr<double>(), atom_ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      grad_y.data_ptr<double>(), grad_g.data_ptr<double>(), grads[0].data_ptr<double>(),
      n, d, h, q);
  C10_HIP_KERNEL_LAUNCH_CHECK();
  }
  // Set MATPL_FITTING_LEGACY=1 for an in-library numerical reference.
  const char* legacy = std::getenv("MATPL_FITTING_LEGACY");
  if (!(legacy && legacy[0] == '1')) {
    const int types = w.size(0);
    const int tiles = (h + kTile - 1) / kTile;
    const int stride = (d + 1 + q) * kTile + q;
    const int64_t per_slot = static_cast<int64_t>(types) * tiles * stride * sizeof(double);
    const int max_slots = std::max<int64_t>(1, (32LL * 1024 * 1024) / per_slot);
    const int slots = std::min<int64_t>((max_count + 31) / 32, max_slots);
    auto partial = at::empty({types, slots, tiles, stride}, x.options());
#define MATPL_FITTING_PARAMETERS(D, H, Q) \
    if (fitting_specialization_enabled() && d == D && h == H && q == Q) { \
      hipLaunchKernelGGL((fitting_parameter_partials_specialized_hip<D,H,Q>), \
          dim3(slots, tiles, types), dim3(kThreads), 0, stream, \
          x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(), \
          v.data_ptr<double>(), atom_ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), \
          grad_y.data_ptr<double>(), grad_g.data_ptr<double>(), partial.data_ptr<double>(), n, 0, slots); \
      C10_HIP_KERNEL_LAUNCH_CHECK(); \
      constexpr int parameter_size = D * H + H + H * Q + Q; \
      hipLaunchKernelGGL((fitting_parameter_reduce_specialized_hip<D,H,Q>), \
          dim3((parameter_size + 255) / 256, types), dim3(256), 0, stream, \
          partial.data_ptr<double>(), grads[1].data_ptr<double>(), \
          grads[2].data_ptr<double>(), grads[3].data_ptr<double>(), grads[4].data_ptr<double>(), slots); \
      C10_HIP_KERNEL_LAUNCH_CHECK(); \
      return; \
    }
    MATPL_FITTING_SHAPES(MATPL_FITTING_PARAMETERS)
#undef MATPL_FITTING_PARAMETERS
    hipLaunchKernelGGL(fitting_parameter_partials_hip,
        dim3(slots, tiles, types), dim3(kThreads), 0, stream,
        x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(),
        v.data_ptr<double>(), atom_ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        grad_y.data_ptr<double>(), grad_g.data_ptr<double>(), partial.data_ptr<double>(),
        slots, n, d, h, q);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    const int parameter_size = d * h + h + h * q + q;
    hipLaunchKernelGGL(fitting_parameter_reduce_hip,
        dim3((parameter_size + 255) / 256, types), dim3(256), 0, stream,
        partial.data_ptr<double>(), grads[1].data_ptr<double>(),
        grads[2].data_ptr<double>(), grads[3].data_ptr<double>(), grads[4].data_ptr<double>(),
        d, h, q, slots);
    C10_HIP_KERNEL_LAUNCH_CHECK();
    return;
  }
  const dim3 parameter_grid((h + kTile - 1) / kTile, w.size(0));
  hipLaunchKernelGGL(fitting_parameters_backward_hip, parameter_grid,
      dim3(kThreads), 0, stream, x.data_ptr<double>(), w.data_ptr<double>(),
      b.data_ptr<double>(), v.data_ptr<double>(), atom_ids.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(), grad_y.data_ptr<double>(),
      grad_g.data_ptr<double>(), grads[1].data_ptr<double>(),
      grads[2].data_ptr<double>(), grads[3].data_ptr<double>(),
      grads[4].data_ptr<double>(), n, d, h, q);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}
