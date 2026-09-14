#include <ATen/ATen.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>

#include "../include/nep_fitting_hip.h"

namespace {

constexpr int kTile = 16;
constexpr int kAtoms = 8;
constexpr int kThreads = kTile * kAtoms;
constexpr int kMaxD = 96;
constexpr int kMaxH = 100;
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
  hipLaunchKernelGGL(fitting_atoms_backward_hip, atom_grid, dim3(kThreads), 0, stream,
      x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(),
      v.data_ptr<double>(), atom_ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
      grad_y.data_ptr<double>(), grad_g.data_ptr<double>(), grads[0].data_ptr<double>(),
      n, d, h, q);
  C10_HIP_KERNEL_LAUNCH_CHECK();
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
