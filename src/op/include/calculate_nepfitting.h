#pragma once

#ifdef MATPL_ENABLE_FUSED_FITTING
#include <ATen/ATen.h>
#include <vector>

std::vector<at::Tensor> nep_fitting_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, at::IntArrayRef counts);

std::vector<at::Tensor> nep_fitting_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, at::IntArrayRef counts,
    const at::Tensor& grad_y, const at::Tensor& grad_g);

bool prepare_nep_fitting_jit(
    const at::Tensor& reference, int64_t d, int64_t h, int64_t q);

void launch_nep_fitting_forward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& c, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count, at::Tensor& y, at::Tensor& g);

void launch_nep_fitting_backward(
    const at::Tensor& x, const at::Tensor& w, const at::Tensor& b,
    const at::Tensor& v, const at::Tensor& atom_ids,
    const at::Tensor& offsets, int64_t max_count,
    const at::Tensor& grad_y, const at::Tensor& grad_g,
    std::vector<at::Tensor>& grads);
#endif
