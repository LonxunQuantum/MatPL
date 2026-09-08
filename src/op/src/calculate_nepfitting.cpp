// HIP builds share src/*.cpp; this implementation belongs to CUDA only.
#ifdef MATPL_ENABLE_FUSED_FITTING
#include "../include/calculate_nepfitting.h"
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <limits>

namespace {
void check_tensor(const at::Tensor& t, const at::Tensor& x,
                  at::ScalarType dtype, const char* name) {
    TORCH_CHECK(t.device() == x.device(), name, " must be on the feature device");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has an unsupported dtype");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

int64_t check_inputs(const at::Tensor& x, const at::Tensor& w,
                    const at::Tensor& b, const at::Tensor& v,
                    const at::Tensor& c, const at::Tensor& atom_ids,
                    const at::Tensor& offsets, at::IntArrayRef counts) {
    TORCH_CHECK(x.is_cuda(), "fused fitting requires CUDA features");
    check_tensor(x, x, at::kDouble, "features");
    check_tensor(w, x, at::kDouble, "W");
    check_tensor(b, x, at::kDouble, "b");
    check_tensor(v, x, at::kDouble, "V");
    check_tensor(c, x, at::kDouble, "c");
    check_tensor(atom_ids, x, at::kLong, "atom_ids");
    check_tensor(offsets, x, at::kLong, "offsets");
    TORCH_CHECK(x.dim() == 2 && w.dim() == 3 && v.dim() == 3,
                "expected X[N,D], W[T,D,H], V[T,H,Q]");
    const auto n = x.size(0), d = x.size(1), t = w.size(0), h = w.size(2);
    const auto q = v.size(2);
    TORCH_CHECK(d >= 1 && d <= 96 && h >= 1 && h <= 100,
                "fused fitting requires 1 <= D <= 96 and 1 <= H <= 100");
    TORCH_CHECK(q == 1 || q == 2, "fused fitting supports one or two heads");
    TORCH_CHECK(t <= 65535 && n <= std::numeric_limits<int>::max(),
                "fitting launch dimensions exceed CUDA limits");
    TORCH_CHECK(w.size(1) == d && v.size(0) == t && v.size(1) == h,
                "incompatible fitting weight shapes");
    TORCH_CHECK(b.sizes() == at::IntArrayRef({t, h}) &&
                c.sizes() == at::IntArrayRef({t, q}), "incompatible bias shapes");
    TORCH_CHECK(atom_ids.dim() == 1 && atom_ids.numel() == n &&
                offsets.dim() == 1 && offsets.numel() == t + 1 &&
                static_cast<int64_t>(counts.size()) == t,
                "group metadata does not match X and parameters");
    int64_t total = 0, max_count = 0;
    for (auto count : counts) {
        TORCH_CHECK(count >= 0 && count <= n, "invalid atom group count");
        total += count;
        max_count = std::max(max_count, count);
    }
    TORCH_CHECK(total == n, "group counts must cover every atom exactly once");
    // Indices are built/validated on CPU in collate. Do not read them back here.
    return max_count;
}
} // namespace

std::vector<at::Tensor> nep_fitting_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, at::IntArrayRef counts) {
    const auto max_count = check_inputs(x, w, b, v, c, atom_ids, offsets, counts);
    const c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty({v.size(2), x.size(0)}, x.options());
    auto g = at::empty({v.size(2), x.size(0), x.size(1)}, x.options());
    if (x.size(0)) launch_nep_fitting_forward(x, w, b, v, c, atom_ids,
                                            offsets, max_count, y, g);
    return {y, g};
}

std::vector<at::Tensor> nep_fitting_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, at::IntArrayRef counts,
    const at::Tensor& grad_y, const at::Tensor& grad_g) {
    const auto max_count = check_inputs(x, w, b, v, c, atom_ids, offsets, counts);
    check_tensor(grad_y, x, at::kDouble, "grad_Y");
    check_tensor(grad_g, x, at::kDouble, "grad_G");
    TORCH_CHECK(grad_y.sizes() == at::IntArrayRef({v.size(2), x.size(0)}) &&
                grad_g.sizes() == at::IntArrayRef({v.size(2), x.size(0), x.size(1)}),
                "incompatible output adjoint shapes");
    const c10::cuda::CUDAGuard guard(x.device());
    std::vector<at::Tensor> grads = {at::empty_like(x), at::zeros_like(w),
        at::zeros_like(b), at::zeros_like(v), at::zeros_like(c)};
    if (x.size(0)) launch_nep_fitting_backward(x, w, b, v, atom_ids, offsets,
                                              max_count, grad_y, grad_g, grads);
    return grads;
}
#endif
