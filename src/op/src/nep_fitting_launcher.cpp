#include <ATen/ATen.h>

#ifdef MATPL_FUSED_FITTING_HIP
#include "../include/nep_fitting_hip.h"
#else
#include "../include/nep_fitting_jit.h"
#endif

void launch_nep_fitting_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y,
    at::Tensor& g) {
#ifdef MATPL_FUSED_FITTING_HIP
    launch_nep_fitting_hip_forward(
#else
    launch_nep_fitting_jit_forward(
#endif
        x, w, b, v, c, atom_ids, offsets, max_count, y, g);
}

void launch_nep_fitting_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count,
    const at::Tensor& grad_y, const at::Tensor& grad_g,
    std::vector<at::Tensor>& grads) {
#ifdef MATPL_FUSED_FITTING_HIP
    launch_nep_fitting_hip_backward(
#else
    launch_nep_fitting_jit_backward(
#endif
        x, w, b, v, atom_ids, offsets, max_count, grad_y, grad_g, grads);
}
