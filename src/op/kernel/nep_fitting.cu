#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <c10/macros/Macros.h>
#include "../include/nep_fitting_jit.h"

namespace {
constexpr int kHiddenTile = 16;
constexpr int kAtoms = 8;
constexpr int kThreads = kHiddenTile * kAtoms;
constexpr int kMaxFeatures = 96;
constexpr int kAtomChunk = 32;
constexpr int64_t kWorkspaceBytes = 16 * 1024 * 1024;

// A half warp handles one atom. Shared weights are only D x 16, including
// for D=96/H=100; all kernels stay below Pascal's 48 KiB per-block limit.
template<int Q, bool Backward>
__global__ void fitting_atoms(
    const double* x, const double* w, const double* b, const double* v,
    const double* c, const int64_t* ids, const int64_t* offsets,
    const double* a, const double* u, double* y, double* out,
    int n, int d, int h) {
    const int type = blockIdx.y;
    const int64_t first = offsets[type], end = offsets[type + 1];
    CUDA_KERNEL_ASSERT(first >= 0 && end >= first && end <= n);
    const int64_t start = first + static_cast<int64_t>(blockIdx.x) * kAtoms;
    if (start >= end) return; // uniform across the block
    const int tid = threadIdx.x, row = tid / kHiddenTile, lane = tid % kHiddenTile;
    const int64_t atom = start + row < end ? ids[start + row] : -1;
    CUDA_KERNEL_ASSERT(atom >= -1 && atom < n);
    __shared__ double sw[kMaxFeatures * kHiddenTile];
    __shared__ double sx[kAtoms * kMaxFeatures];
    __shared__ double su[Backward ? Q * kAtoms * kMaxFeatures : 1];
    __shared__ double seed[(Backward ? 1 : Q) * kAtoms * kHiddenTile];
    for (int f = lane; f < d; f += kHiddenTile) {
        sx[row * d + f] = atom >= 0 ? x[atom * d + f] : 0.0;
        if (Backward) {
            #pragma unroll
            for (int q = 0; q < Q; ++q)
                su[(q * kAtoms + row) * d + f] = atom >= 0 ? u[(q * n + atom) * d + f] : 0.0;
        }
    }
    double result[Backward ? 1 : Q][kMaxFeatures / kHiddenTile] = {};
    double energy[Q] = {};
    __syncthreads();
    for (int tile = 0; tile < h; tile += kHiddenTile) {
        for (int p = tid; p < d * kHiddenTile; p += kThreads) {
            const int j = tile + p % kHiddenTile;
            sw[p] = j < h ? w[(static_cast<int64_t>(type) * d + p / kHiddenTile) * h + j] : 0.0;
        }
        __syncthreads();
        const int j = tile + lane;
        double z = j < h ? b[type * h + j] : 0.0;
        double t[Q] = {};
        for (int f = 0; f < d; ++f) {
            const double weight = sw[f * kHiddenTile + lane];
            z += sx[row * d + f] * weight;
            if (Backward) {
                #pragma unroll
                for (int q = 0; q < Q; ++q) t[q] += su[(q * kAtoms + row) * d + f] * weight;
            }
        }
        const double hidden = tanh(z), s = 1.0 - hidden * hidden;
        double r = 0.0;
        #pragma unroll
        for (int q = 0; q < Q; ++q) {
            const double head = j < h ? v[(type * h + j) * Q + q] : 0.0;
            if (Backward) {
                const double aq = atom >= 0 ? a[q * n + atom] : 0.0;
                r += s * head * (aq - 2.0 * hidden * t[q]);
            } else {
                energy[q] += hidden * head;
                seed[(q * kAtoms + row) * kHiddenTile + lane] = head * s;
            }
        }
        if (Backward) seed[row * kHiddenTile + lane] = r;
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < (Backward ? 1 : Q); ++q) {
            for (int f = lane, k = 0; f < d; f += kHiddenTile, ++k) {
                double value = 0.0;
                #pragma unroll
                for (int j1 = 0; j1 < kHiddenTile; ++j1)
                    value += sw[f * kHiddenTile + j1] * seed[(q * kAtoms + row) * kHiddenTile + j1];
                result[q][k] += value;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < (Backward ? 1 : Q); ++q)
        for (int f = lane, k = 0; f < d; f += kHiddenTile, ++k)
            if (atom >= 0) out[(q * n + atom) * d + f] = result[q][k];
    if (!Backward) {
        #pragma unroll
        for (int q = 0; q < Q; ++q) {
            #pragma unroll
            for (int delta = kHiddenTile / 2; delta; delta /= 2)
                energy[q] += __shfl_down_sync(0xffffffff, energy[q], delta, kHiddenTile);
            if (lane == 0 && atom >= 0) y[q * n + atom] = energy[q] + c[type * Q + q];
        }
    }
}

// Each block owns one hidden tile and a bounded partial sum. For large atom
// counts it visits several chunks, keeping workspace independent of N.
template<int Q>
__global__ void fitting_parameter_partials(
    const double* x, const double* w, const double* b, const double* v,
    const int64_t* ids, const int64_t* offsets, const double* a,
    const double* u, double* partial, int n, int d, int h,
    int type_start, int slots, int tiles, int stride) {
    constexpr int max_stride = (kMaxFeatures + 1 + Q) * kHiddenTile + Q;
    constexpr int accum_size = (max_stride + kThreads - 1) / kThreads;
    const int tid = threadIdx.x, row = tid / kHiddenTile, lane = tid % kHiddenTile;
    const int type = type_start + blockIdx.z, tile = blockIdx.y * kHiddenTile;
    const int64_t first = offsets[type], end = offsets[type + 1];
    CUDA_KERNEL_ASSERT(first >= 0 && end >= first && end <= n);
    __shared__ double sw[kMaxFeatures * kHiddenTile];
    __shared__ double sx[kAtoms * kMaxFeatures];
    __shared__ double su[Q * kAtoms * kMaxFeatures];
    __shared__ double sa[Q * kAtoms];
    __shared__ double sr[kAtoms * kHiddenTile];
    __shared__ double ss[kAtoms * kHiddenTile];
    __shared__ double sv[Q * kAtoms * kHiddenTile];
    for (int p = tid; p < d * kHiddenTile; p += kThreads) {
        const int j = tile + p % kHiddenTile;
        sw[p] = j < h ? w[(static_cast<int64_t>(type) * d + p / kHiddenTile) * h + j] : 0.0;
    }
    double acc[accum_size] = {};
    __syncthreads();
    for (int64_t chunk = first + static_cast<int64_t>(blockIdx.x) * kAtomChunk;
         chunk < end; chunk += static_cast<int64_t>(slots) * kAtomChunk) {
        for (int sub = 0; sub < kAtomChunk; sub += kAtoms) {
            const int64_t pos = chunk + sub + row;
            const int64_t atom = pos < end ? ids[pos] : -1;
            CUDA_KERNEL_ASSERT(atom >= -1 && atom < n);
            for (int f = lane; f < d; f += kHiddenTile) {
                sx[row * d + f] = atom >= 0 ? x[atom * d + f] : 0.0;
                #pragma unroll
                for (int q = 0; q < Q; ++q)
                    su[(q * kAtoms + row) * d + f] = atom >= 0 ? u[(q * n + atom) * d + f] : 0.0;
            }
            if (lane == 0) {
                #pragma unroll
                for (int q = 0; q < Q; ++q) sa[q * kAtoms + row] = atom >= 0 ? a[q * n + atom] : 0.0;
            }
            __syncthreads();
            const int j = tile + lane;
            double z = j < h ? b[type * h + j] : 0.0;
            double t[Q] = {};
            for (int f = 0; f < d; ++f) {
                const double weight = sw[f * kHiddenTile + lane];
                z += sx[row * d + f] * weight;
                #pragma unroll
                for (int q = 0; q < Q; ++q) t[q] += su[(q * kAtoms + row) * d + f] * weight;
            }
            const double hidden = tanh(z), s = 1.0 - hidden * hidden;
            double r = 0.0;
            #pragma unroll
            for (int q = 0; q < Q; ++q) {
                const double head = j < h ? v[(type * h + j) * Q + q] : 0.0;
                r += s * head * (sa[q * kAtoms + row] - 2.0 * hidden * t[q]);
                sv[(q * kAtoms + row) * kHiddenTile + lane] = sa[q * kAtoms + row] * hidden + t[q] * s;
            }
            sr[row * kHiddenTile + lane] = r;
            ss[row * kHiddenTile + lane] = s;
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < accum_size; ++k) {
                const int p = tid + k * kThreads;
                if (p >= stride) continue;
                double value = 0.0;
                if (p < d * kHiddenTile) {
                    const int f = p / kHiddenTile, j1 = p % kHiddenTile;
                    #pragma unroll
                    for (int row1 = 0; row1 < kAtoms; ++row1) {
                        value += sx[row1 * d + f] * sr[row1 * kHiddenTile + j1];
                        #pragma unroll
                        for (int q = 0; q < Q; ++q) {
                            const double head = tile + j1 < h ? v[(type * h + tile + j1) * Q + q] : 0.0;
                            value += su[(q * kAtoms + row1) * d + f] * head * ss[row1 * kHiddenTile + j1];
                        }
                    }
                } else if (p < (d + 1) * kHiddenTile) {
                    #pragma unroll
                    for (int row1 = 0; row1 < kAtoms; ++row1) value += sr[row1 * kHiddenTile + p % kHiddenTile];
                } else if (p < (d + 1 + Q) * kHiddenTile) {
                    const int q = (p - (d + 1) * kHiddenTile) / kHiddenTile;
                    #pragma unroll
                    for (int row1 = 0; row1 < kAtoms; ++row1) value += sv[(q * kAtoms + row1) * kHiddenTile + p % kHiddenTile];
                } else if (tile == 0) {
                    const int q = p - (d + 1 + Q) * kHiddenTile;
                    #pragma unroll
                    for (int row1 = 0; row1 < kAtoms; ++row1) value += sa[q * kAtoms + row1];
                }
                acc[k] += value;
            }
            __syncthreads();
        }
    }
    const int64_t base = ((static_cast<int64_t>(blockIdx.z) * slots + blockIdx.x) * tiles + blockIdx.y) * stride;
    #pragma unroll
    for (int k = 0; k < accum_size; ++k) {
        const int p = tid + k * kThreads;
        if (p < stride) partial[base + p] = acc[k];
    }
}

template<int Q>
__global__ void fitting_parameter_reduce(
    const double* partial, double* dw, double* db, double* dv, double* dc,
    int d, int h, int type_start, int slots, int tiles, int stride) {
    const int type = type_start + blockIdx.y;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int weight_size = d * h, psize = weight_size + h + h * Q + Q;
    if (p >= psize) return;
    int tile, part;
    if (p < weight_size) {
        tile = (p % h) / kHiddenTile;
        part = (p / h) * kHiddenTile + p % h % kHiddenTile;
    } else if (p < weight_size + h) {
        const int j = p - weight_size;
        tile = j / kHiddenTile; part = d * kHiddenTile + j % kHiddenTile;
    } else if (p < weight_size + h + h * Q) {
        const int j = (p - weight_size - h) / Q, q = (p - weight_size - h) % Q;
        tile = j / kHiddenTile; part = (d + 1 + q) * kHiddenTile + j % kHiddenTile;
    } else {
        tile = 0; part = (d + 1 + Q) * kHiddenTile + p - weight_size - h - h * Q;
    }
    double value = 0.0;
    for (int s = 0; s < slots; ++s)
        value += partial[((static_cast<int64_t>(blockIdx.y) * slots + s) * tiles + tile) * stride + part];
    if (p < weight_size) dw[static_cast<int64_t>(type) * weight_size + p] = value;
    else if (p < weight_size + h) db[type * h + p - weight_size] = value;
    else if (p < weight_size + h + h * Q) dv[type * h * Q + p - weight_size - h] = value;
    else dc[type * Q + p - weight_size - h - h * Q] = value;
}

template<int Q>
void forward_impl(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
                  const at::Tensor& v, const at::Tensor& c, const at::Tensor& ids,
                  const at::Tensor& offsets, int64_t max_count, at::Tensor& y,
                  at::Tensor& g) {
    fitting_atoms<Q, false><<<dim3((max_count + kAtoms - 1) / kAtoms, w.size(0)),
        kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(),
        v.data_ptr<double>(), c.data_ptr<double>(), ids.data_ptr<int64_t>(),
        offsets.data_ptr<int64_t>(), nullptr, nullptr, y.data_ptr<double>(),
        g.data_ptr<double>(), x.size(0), x.size(1), w.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<int Q>
void backward_impl(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
                   const at::Tensor& v, const at::Tensor& ids,
                   const at::Tensor& offsets, int64_t max_count,
                   const at::Tensor& a, const at::Tensor& u,
                   std::vector<at::Tensor>& grads) {
    const int n = x.size(0), d = x.size(1), h = w.size(2), types = w.size(0);
    const auto stream = at::cuda::getCurrentCUDAStream();
    fitting_atoms<Q, true><<<dim3((max_count + kAtoms - 1) / kAtoms, types), kThreads, 0, stream>>>(
        x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(),
        v.data_ptr<double>(), nullptr, ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(),
        a.data_ptr<double>(), u.data_ptr<double>(), nullptr, grads[0].data_ptr<double>(), n, d, h);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const int tiles = (h + kHiddenTile - 1) / kHiddenTile;
    const int stride = (d + 1 + Q) * kHiddenTile + Q;
    const int groups = std::min<int64_t>(types, kWorkspaceBytes / (tiles * stride * sizeof(double)));
    const int slots = std::min<int64_t>((max_count + kAtomChunk - 1) / kAtomChunk,
        kWorkspaceBytes / (groups * tiles * stride * sizeof(double)));
    auto workspace = at::empty({groups, slots, tiles, stride}, x.options());
    for (int start = 0; start < types; start += groups) {
        const int batch = std::min(groups, types - start);
        fitting_parameter_partials<Q><<<dim3(slots, tiles, batch), kThreads, 0, stream>>>(
            x.data_ptr<double>(), w.data_ptr<double>(), b.data_ptr<double>(), v.data_ptr<double>(),
            ids.data_ptr<int64_t>(), offsets.data_ptr<int64_t>(), a.data_ptr<double>(),
            u.data_ptr<double>(), workspace.data_ptr<double>(), n, d, h, start, slots, tiles, stride);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        fitting_parameter_reduce<Q><<<dim3((d * h + h + h * Q + Q + 255) / 256, batch), 256, 0, stream>>>(
            workspace.data_ptr<double>(), grads[1].data_ptr<double>(), grads[2].data_ptr<double>(),
            grads[3].data_ptr<double>(), grads[4].data_ptr<double>(), d, h, start, slots, tiles, stride);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
} // namespace

void launch_nep_fitting_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y, at::Tensor& g) {
    if (try_launch_nep_fitting_jit_forward(
            x, w, b, v, c, atom_ids, offsets, max_count, y, g)) return;
    if (v.size(2) == 1) forward_impl<1>(x, w, b, v, c, atom_ids, offsets, max_count, y, g);
    else forward_impl<2>(x, w, b, v, c, atom_ids, offsets, max_count, y, g);
}

void launch_nep_fitting_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count,
    const at::Tensor& grad_y, const at::Tensor& grad_g,
    std::vector<at::Tensor>& grads) {
    if (v.size(2) == 1) backward_impl<1>(x, w, b, v, atom_ids, offsets, max_count, grad_y, grad_g, grads);
    else backward_impl<2>(x, w, b, v, atom_ids, offsets, max_count, grad_y, grad_g, grads);
}
