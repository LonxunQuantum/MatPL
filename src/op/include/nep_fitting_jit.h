#pragma once

#include <ATen/ATen.h>

bool prepare_nep_fitting_jit(
    const at::Tensor& reference, int64_t d, int64_t h, int64_t q);
