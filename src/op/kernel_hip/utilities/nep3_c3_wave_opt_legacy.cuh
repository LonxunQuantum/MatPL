/*
 * Legacy matpl-opt C3 kernel isolated from the runtime-generic helper set.
 *
 * The wave-level kernel and its compact second-gradient helpers must stay in
 * the same namespace: their historical interfaces differ from the generic
 * LDS implementation used for non-whitelisted descriptor shapes.
 */
#pragma once

namespace nep3_opt_legacy {
// matpl-opt defined TYPES as 20 in its nep_utilities.cuh.  The current
// runtime-generic helper removed that legacy constant, so retain it locally
// for the unused generic legacy kernels parsed by this translation unit.
constexpr int TYPES = MAX_NUM_N;
#include "nep3_c3_wave_opt_legacy_impl.cuh"
}  // namespace nep3_opt_legacy
