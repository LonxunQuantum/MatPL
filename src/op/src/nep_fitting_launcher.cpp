#include <ATen/ATen.h>

#include "../include/nep_fitting_jit.h"

void launch_nep_fitting_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y,
    at::Tensor& g) {
    launch_nep_fitting_jit_forward(
        x, w, b, v, c, atom_ids, offsets, max_count, y, g);
}

void launch_nep_fitting_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count,
    const at::Tensor& grad_y, const at::Tensor& grad_g,
    std::vector<at::Tensor>& grads) {
    launch_nep_fitting_jit_backward(
        x, w, b, v, atom_ids, offsets, max_count, grad_y, grad_g, grads);
}