#pragma once

#include <cstddef>

struct NepMbSecondGradArgs;

void launch_nep_mb_secondgrad_jit(
    const NepMbSecondGradArgs& args, int nmax, int nbasis,
    bool has_four_body, bool has_five_body,
    int device, size_t shared_bytes);
