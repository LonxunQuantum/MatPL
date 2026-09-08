# NEP FP64 Many-Body Second-Gradient Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the OMat24 many-body coefficient second-gradient path with one FP64 fused CUDA kernel that removes the 2.625 GiB intermediate tensor and reduces training-step time.

**Architecture:** Keep the current implementation as the general fallback. For the OMat24 shape, launch one CTA per center atom, discover only the neighbor types present around that center, process those types in tiles of four, reduce in shared memory, and flush directly to `gradsecond_c3`. Compile the fixed radial, basis, and angular dimensions into the optimized kernel so each angular order uses a short-lived register scope.

**Tech Stack:** C++17, CUDA 11.8/12.4, PyTorch C++ extension, pytest, Slurm, compute-sanitizer, Nsight Systems, Nsight Compute.

**Spec:** `docs/superpowers/specs/2026-09-08-nep-mb-secondgrad-kernel-design.md`

## Global Constraints

- All inputs, arithmetic, shared-memory staging, and accumulation remain FP64.
- Preserve the PyTorch API, tensor layouts, trainable-gradient semantics, and original atom order.
- Build for SM60, SM70, and SM86, covering P100, V100, RTX 3080 Ti, and RTX 3090.
- Do not introduce Tensor Core code, mixed precision, atom sorting, or changes to other operators.
- The first optimized specialization is `n_max_3b=5`, `n_base_3b=9`, `lmax_3=4`, with both four-body and five-body terms enabled.
- Unsupported dimensions, insufficient shared memory, and explicit `legacy` mode use the existing implementation.
- `auto` becomes optimized by default only after numerical, sanitizer, memory, and performance gates pass.

## File Map

- Create `src/op/kernel/utilities/nep_mbgrad_opt.cuh`: FP64 template algebra, local-type collection, shared-memory tile layout, fused CTA kernel, and typed launcher.
- Modify `src/op/kernel/calculateNepMbFeat_secondgradout.cu`: preserve the public launch signature, isolate the legacy path, parse the runtime mode, validate support, and dispatch the fused kernel.
- Create `src/test/test_nep_electric/test_mb_secondgrad_kernel.py`: forced legacy/optimized/auto comparisons and fallback/error coverage through the public autograd API.
- Create `tests/check_nep_mb_secondgrad_ptxas.py`: parse the CUDA build log and enforce resource checks for the optimized kernel.
- Do not modify `src/op/kernel/utilities/nep_mbgrad.cuh` or `nep_utilities_mb_secondc.cuh`; they remain the unchanged numerical reference and fallback.
- Do not modify `mini_data_test/run.sh`; select the implementation with an exported environment variable when submitting the existing script.

---

### Task 1: Add deterministic runtime selection around the unchanged legacy path

**Files:**
- Modify: `src/op/kernel/calculateNepMbFeat_secondgradout.cu:225-382`
- Create: `src/test/test_nep_electric/test_mb_secondgrad_kernel.py`

**Interfaces:**
- Produces: `enum class NepMbSecondGradMode { Auto, Optimized, Legacy };`
- Produces: `NepMbSecondGradMode nep_mb_secondgrad_mode();`
- Produces: `void launch_calculate_nepmbfeat_secondgradout_c3_legacy(...)` with the same arguments as the current public launcher.
- Preserves: `void launch_calculate_nepmbfeat_secondgradout_c3(...)` as the only external CUDA entry point.

- [ ] **Step 1: Add a public-path test helper and an invalid-mode test**

  Create a helper that reaches `CalculateNepMbFeatGrad::backward`, because its returned coefficient gradient is `gradsecond_c3`:

  ```python
  @dataclasses.dataclass(frozen=True)
  class CaseSpec:
      atom_types: tuple[int, ...]
      neighbors: tuple[tuple[int, ...], ...]
      n_max: int
      n_base: int
      feat_2b_num: int
      lmax_3: int
      lmax_4: int
      lmax_5: int


  def _make_case(case):
      torch.manual_seed(20260908)
      device, dtype = torch.device("cuda"), torch.float64
      atom_map = torch.tensor(case.atom_types, dtype=torch.int64, device=device)
      nl = torch.tensor(case.neighbors, dtype=torch.int64, device=device)
      natoms, max_neighbors = nl.shape
      coords = torch.randn(natoms, max_neighbors, 3, dtype=dtype, device=device) * 0.25
      distance = coords.square().sum(-1, keepdim=True).sqrt() + 0.8
      d12 = torch.cat((distance, coords), dim=-1).detach().requires_grad_(True)
      ntypes = max(case.atom_types) + 1
      coeff = torch.randn(
          ntypes, ntypes, case.n_max, case.n_base,
          dtype=dtype, device=device, requires_grad=True,
      )
      many_body = case.n_max * (
          case.lmax_3 + int(case.lmax_4 > 0) + int(case.lmax_5 > 0)
      )
      feats = torch.zeros(
          natoms, case.feat_2b_num + many_body, dtype=dtype, device=device,
          requires_grad=True,
      )
      seed = torch.randn_like(feats, requires_grad=True)
      probe = torch.randn_like(d12)
      return coeff, d12, nl, atom_map, feats, seed, probe


  def _coefficient_second_grad(case, mode):
      previous = os.environ.get("MATPL_NEP_MB_SECONDGRAD_MODE")
      os.environ["MATPL_NEP_MB_SECONDGRAD_MODE"] = mode
      try:
          coeff, d12, nl, atom_map, feats, seed, probe = _make_case(case)
          feat, dc3, d3, d3_noc, sums = CalcOps.calculateNepMbFeatWithGradContext(
              coeff, d12, nl, atom_map, feats,
              case.feat_2b_num, case.lmax_3, case.lmax_4, case.lmax_5,
              5.0, 0,
          )
          vjp = CalcOps.calculateNepMbFeatInputGrad(
              seed, coeff, d12, nl, dc3, d3, d3_noc, sums, atom_map,
              case.feat_2b_num, case.lmax_3, case.lmax_4, case.lmax_5,
              5.0, 0,
          )
          loss = (vjp * probe).sum()
          grad_seed, grad_coeff = torch.autograd.grad(loss, (seed, coeff))
          torch.cuda.synchronize()
          return loss.detach(), grad_seed.detach(), grad_coeff.detach()
      finally:
          if previous is None:
              os.environ.pop("MATPL_NEP_MB_SECONDGRAD_MODE", None)
          else:
              os.environ["MATPL_NEP_MB_SECONDGRAD_MODE"] = previous


  def _assert_triplet_close(actual, expected):
      for actual_tensor, expected_tensor in zip(actual, expected):
          absolute = (actual_tensor - expected_tensor).abs()
          relative = absolute / expected_tensor.abs().clamp_min(1e-30)
          print(f"max_abs={absolute.max().item():.6e} "
                f"max_rel={relative.max().item():.6e}")
          torch.testing.assert_close(
              actual_tensor, expected_tensor, rtol=1e-9, atol=1e-11)


  def test_invalid_secondgrad_mode_is_rejected():
      with pytest.raises(RuntimeError, match="MATPL_NEP_MB_SECONDGRAD_MODE"):
          _coefficient_second_grad(_single_type_case(), "invalid")
  ```

- [ ] **Step 2: Run the focused test and verify the missing selector is visible**

  Run on a q4/3090 allocation after loading the documented environment:

  ```bash
  python -m pytest src/test/test_nep_electric/test_mb_secondgrad_kernel.py::test_invalid_secondgrad_mode_is_rejected -q
  ```

  Expected: FAIL because the current launcher ignores the environment value.

- [ ] **Step 3: Extract the current body unchanged and add strict mode parsing**

  Move lines 248-381 into the `_legacy` function without changing launch geometry, allocation sizes, synchronization, or arithmetic. Parse the variable on every call so one Python process can perform A/B comparisons:

  ```cpp
  enum class NepMbSecondGradMode { Auto, Optimized, Legacy };

  static NepMbSecondGradMode nep_mb_secondgrad_mode() {
    const char* value = std::getenv("MATPL_NEP_MB_SECONDGRAD_MODE");
    if (value == nullptr || std::strcmp(value, "auto") == 0) {
      return NepMbSecondGradMode::Auto;
    }
    if (std::strcmp(value, "optimized") == 0) {
      return NepMbSecondGradMode::Optimized;
    }
    if (std::strcmp(value, "legacy") == 0) {
      return NepMbSecondGradMode::Legacy;
    }
    throw std::runtime_error(
        "MATPL_NEP_MB_SECONDGRAD_MODE must be auto, optimized, or legacy");
  }
  ```

  Until Task 2 adds the fused launcher, route `auto` and `legacy` to `_legacy`; make forced `optimized` report that the specialization is unavailable.

- [ ] **Step 4: Rebuild and run the selector and existing descriptor tests**

  ```bash
  cmake -S src/op -B src/op/build/cuda \
    -DMATPL_GPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES='60;70;86'
  cmake --build src/op/build/cuda -j4
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    src/test/test_nep_electric/test_descriptor_vjp.py -q
  ```

  Expected: selector tests PASS; all existing descriptor VJP tests PASS.

- [ ] **Step 5: Commit the selector and test harness**

  ```bash
  git add src/op/kernel/calculateNepMbFeat_secondgradout.cu \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py
  git commit -m "test: add NEP second-gradient runtime selection"
  ```

---

### Task 2: Implement the correctness-first fused FP64 CTA kernel

**Files:**
- Create: `src/op/kernel/utilities/nep_mbgrad_opt.cuh`
- Modify: `src/op/kernel/calculateNepMbFeat_secondgradout.cu:1-8,225-382`
- Modify: `src/test/test_nep_electric/test_mb_secondgrad_kernel.py`

**Interfaces:**
- Consumes: `NepMbSecondGradMode` and the unchanged public launch arguments from Task 1.
- Produces: `struct NepMbSecondGradArgs` containing the nine input pointers, output pointer, cutoffs, and runtime sizes.
- Produces: `template<int NMAX, int NBASIS, int LMAX3, bool HAS4, bool HAS5, int TYPE_TILE, int CTA_THREADS> __global__ void nep_mb_secondgrad_fused(NepMbSecondGradArgs);`
- Produces: `bool launch_nep_mb_secondgrad_omat24(const NepMbSecondGradArgs&, int device);`

- [ ] **Step 1: Add supported-shape A/B cases that fail while optimized mode is unavailable**

  Parameterize fixed-shape cases with deterministic seed `20260908`:

  ```python
  @pytest.mark.parametrize("case", [
      _single_type_case(n_max=5, n_base=9, lmax=(4, 2, 1)),
      _repeated_type_case(n_types=3, n_max=5, n_base=9, lmax=(4, 2, 1)),
      _six_local_type_case(n_max=5, n_base=9, lmax=(4, 2, 1)),
      _wide_neighbor_case(valid_neighbors=39, max_neighbors=43,
                          n_max=5, n_base=9, lmax=(4, 2, 1)),
      _empty_slot_case(max_neighbors=8, n_max=5, n_base=9,
                       lmax=(4, 2, 1)),
  ])
  def test_optimized_matches_legacy(case):
      legacy = _coefficient_second_grad(case, "legacy")
      optimized = _coefficient_second_grad(case, "optimized")
      for actual, expected in zip(optimized, legacy):
          torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-11)
  ```

  Construct `d12[..., 0]` in `[0.8, 3.5]`, keep padded `NL` entries at `-1`, and retain nonzero coordinate and second-gradient components so every algebra branch contributes.

  The case factories return only `CaseSpec` values. `_six_local_type_case` uses a center connected to types `0..5`; `_wide_neighbor_case` uses 40 atoms with 39 valid neighbors and four `-1` slots; `_empty_slot_case` pads every row to eight entries with `-1`. This keeps GPU tensor creation inside `_make_case` after the backend availability check.

- [ ] **Step 2: Run the supported A/B test and verify forced optimized mode fails**

  ```bash
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py::test_optimized_matches_legacy -q
  ```

  Expected: FAIL with the Task 1 “specialization unavailable” error.

- [ ] **Step 3: Define the compact kernel arguments and shared-memory layout**

  Use four 32-bit words for the 118-element bitset and one list entry per possible element. Reuse the tail of dynamic shared memory between tiles:

  ```cpp
  struct NepMbSecondGradArgs {
    const double* grad_second;
    const double* d12;
    const int64_t* neighbor_list;
    const double* de_dfeat;
    const double* dsnlm_dc;
    const double* sum_fxyz;
    const int64_t* atom_type;
    const double* coeff3;
    double* gradsecond_c3;
    double rcut;
    double rcut_inv;
    int atom_count;
    int max_neighbors;
    int atom_types;
    int feat_2b_count;
    int many_body_feat_count;
  };

  template<int NMAX, int NBASIS, int TYPE_TILE>
  struct SharedLayout {
    unsigned int* type_bits;       // 4 words
    int* local_types;              // NEP_MAX_ELEMENT_TYPES entries
    int* local_type_count;         // 1 entry
    double* fp;                    // NMAX * 6 for OMat24: 4 three-body + 1 four-body + 1 five-body
    double* sum_fxyz;              // NMAX * 24
    double* dsnlm_tile;            // TYPE_TILE * NBASIS * 24
    double* output_tile;           // TYPE_TILE * NMAX * NBASIS
  };

  constexpr size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
  }
  ```

  Compute every offset from `align_up(offset, alignof(double))`; add a host/device `shared_bytes<NMAX, NBASIS, TYPE_TILE>()` function and assert every region lies within the returned byte count.

- [ ] **Step 4: Build the local type list once per center atom**

  Threads walk `neighbor = threadIdx.x; neighbor < max_neighbors; neighbor += blockDim.x`. For valid neighbors inside the cutoff, set `type_bits[type >> 5]` with `atomicOr`. After a barrier, thread 0 enumerates element IDs in ascending order into `local_types`; this stable order does not change atom storage or neighbor order.

  ```cpp
  for (int i = threadIdx.x; i < 4; i += blockDim.x) s.type_bits[i] = 0;
  if (threadIdx.x == 0) *s.local_type_count = 0;
  __syncthreads();
  for (int j = threadIdx.x; j < a.max_neighbors; j += blockDim.x) {
    const int64_t neighbor = a.neighbor_list[center * a.max_neighbors + j];
    if (neighbor >= 0 && a.d12[(center * a.max_neighbors + j) * 4] <= a.rcut) {
      const int type = static_cast<int>(a.atom_type[neighbor]);
      atomicOr(&s.type_bits[type >> 5], 1u << (type & 31));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int type = 0; type < a.atom_types; ++type) {
      if (s.type_bits[type >> 5] & (1u << (type & 31))) {
        s.local_types[(*s.local_type_count)++] = type;
      }
    }
  }
  __syncthreads();
  ```

- [ ] **Step 5: Port the existing FP64 algebra into a tiled shared accumulator**

  Keep the operation order inside each per-neighbor formula. Replace the old `f12k[...] += value` destinations with this sink:

  ```cpp
  template<int NMAX, int NBASIS, int TYPE_TILE>
  struct SharedTileSink {
    double* values;

    __device__ __forceinline__ void add(
        int type_slot, int n, int basis, double value) const {
      atomicAdd(&values[(type_slot * NMAX + n) * NBASIS + basis], value);
    }
  };
  ```

  Port the old functions with the following exact mapping:

  | Existing numerical source | Optimized helper | Change |
  |---|---|---|
  | `scd_get_f12_1/2/3/4` | `accumulate_direct<L>` | Write only the physical neighbor type slot through `SharedTileSink`. |
  | `scd_get_f12_1_J/2_J/3_J/4_J` | `accumulate_cross<L>` | Iterate only the current local-type tile and read its staged `dsnlm_dc`. |
  | `scd_get_f12_4body[_J]` | `accumulate_four_body` | Compile under `if constexpr (HAS4)`. |
  | `scd_get_f12_5body[_J]` | `accumulate_five_body` | Compile under `if constexpr (HAS5)`. |
  | `find_fc_and_fcp`, `find_fn_and_fnp` | existing utility calls | Pass compile-time `NBASIS`; store exactly `double fn[NBASIS]` and `double fnp[NBASIS]`. |

  For each tile: cooperatively clear `output_tile`, load `dsnlm_tile`, synchronize, process all valid neighbors and `n=0..NMAX-1`, synchronize, then atomically flush each `[center_type, local_type, n, basis]` value once into `gradsecond_c3`. Synchronize before reusing shared memory for the next tile.

- [ ] **Step 6: Add the OMat24 launcher and connect forced optimized mode**

  ```cpp
  static bool is_omat24_specialization(
      int nmax, int nbasis, int lmax3, int lmax4, int lmax5) {
    return nmax == 5 && nbasis == 9 && lmax3 == 4 && lmax4 > 0 && lmax5 > 0;
  }

  bool launch_nep_mb_secondgrad_omat24(
      const NepMbSecondGradArgs& args, int device) {
    const int threads = args.max_neighbors <= 32 ? 32 : 64;
    const size_t bytes = nep_mb_secondgrad_shared_bytes<5, 9, 4>();
    if (threads == 32) {
      nep_mb_secondgrad_fused<5, 9, 4, true, true, 4, 32>
          <<<args.atom_count, 32, bytes>>>(args);
    } else {
      nep_mb_secondgrad_fused<5, 9, 4, true, true, 4, 64>
          <<<args.atom_count, 64, bytes>>>(args);
    }
    CUDA_CHECK_KERNEL
    return true;
  }
  ```

  Pass the actual runtime dimensions into the support predicate before this typed launcher. Do not allocate `GPU_Vector`, copy counts to the host, or call `cudaDeviceSynchronize` in the optimized branch.

- [ ] **Step 7: Rebuild and run the full A/B matrix**

  ```bash
  cmake --build src/op/build/cuda -j4 2>&1 | tee .validation/mb-secondgrad-build.log
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    src/test/test_nep_electric/test_descriptor_vjp.py -q
  ```

  Expected: every supported optimized result matches legacy at `rtol=1e-9`, `atol=1e-11`; existing VJP tests remain green.

- [ ] **Step 8: Commit the fused correctness path**

  ```bash
  git add src/op/kernel/utilities/nep_mbgrad_opt.cuh \
    src/op/kernel/calculateNepMbFeat_secondgradout.cu \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py
  git commit -m "feat: fuse FP64 NEP many-body second gradient"
  ```

---

### Task 3: Reduce register lifetime with angular-order templates

**Files:**
- Modify: `src/op/kernel/utilities/nep_mbgrad_opt.cuh`
- Create: `tests/check_nep_mb_secondgrad_ptxas.py`

**Interfaces:**
- Consumes: `nep_mb_secondgrad_fused<5,9,4,true,true,4,CTA_THREADS>` from Task 2.
- Produces: `template<int L> struct AngularScratch` with arrays of exactly `2*L+1` doubles.
- Produces: `template<int L, ...> __device__ void accumulate_angular_order(...)`.
- Produces: resource checker CLI `python tests/check_nep_mb_secondgrad_ptxas.py BUILD_LOG`.

- [ ] **Step 1: Capture the correctness-first kernel resource baseline**

  ```bash
  rm -f src/op/build/cuda/cmake/cuda/CMakeFiles/CalcOps_cuda.dir/__/__/kernel/calculateNepMbFeat_secondgradout.cu.o
  cmake --build src/op/build/cuda -j1 2>&1 | tee .validation/mb-secondgrad-ptxas-before.log
  rg -n "nep_mb_secondgrad_fused|register|spill|stack frame" \
    .validation/mb-secondgrad-ptxas-before.log
  ```

  Record registers, stack frame, spill loads, and spill stores for both 32- and 64-thread instantiations.

- [ ] **Step 2: Add a resource checker that fails on spill or regression to the old 253-register kernel**

  ```python
  KERNEL = "nep_mb_secondgrad_fused"

  @dataclasses.dataclass(frozen=True)
  class ResourceRecord:
      name: str
      sm: int
      cta_threads: int
      registers: int
      stack_bytes: int
      spill_load_bytes: int
      spill_store_bytes: int

  def validate(records):
      fused = [record for record in records if KERNEL in record.name]
      required = {(sm, threads) for sm in (60, 70, 86) for threads in (32, 64)}
      present = {(record.sm, record.cta_threads) for record in fused}
      if missing := required - present:
          raise SystemExit(f"missing fused resource records: {sorted(missing)}")
      for record in fused:
          if record.spill_load_bytes or record.spill_store_bytes:
              raise SystemExit(f"{record.name}: spills detected: {record}")
          if record.registers >= 253:
              raise SystemExit(f"{record.name}: register count did not improve: {record.registers}")
  ```

  Parse the `ptxas info` function line followed by its resource line; treat missing records as failure rather than success.

- [ ] **Step 3: Run the checker against the baseline and retain the result as evidence**

  ```bash
  python tests/check_nep_mb_secondgrad_ptxas.py \
    .validation/mb-secondgrad-ptxas-before.log
  ```

  Expected: PASS only if the correctness-first kernel already has zero spill and fewer than 253 registers; otherwise FAIL and identify the offending instantiation.

- [ ] **Step 4: Split L=1,2,3,4 work into non-overlapping template scopes**

  ```cpp
  template<int L>
  struct AngularScratch {
    double blm[2 * L + 1];
    double rij_blm[2 * L + 1];
    double dblm_x[2 * L + 1];
    double dblm_y[2 * L + 1];
    double dblm_z[2 * L + 1];
    double dblm_r[2 * L + 1];
  };

  template<int L, int NMAX, int NBASIS, int TYPE_TILE>
  __device__ __forceinline__ void accumulate_angular_order(
      const NepMbSecondGradArgs& args,
      int center,
      int neighbor_slot,
      int radial_index,
      const double* fp,
      const double* sum_fxyz,
      const double (&fn)[NBASIS],
      const double (&fnp)[NBASIS],
      const SharedTileSink<NMAX, NBASIS, TYPE_TILE>& sink) {
    AngularScratch<L> scratch{};
    build_angular_scratch<L>(args, center, neighbor_slot, scratch);
    accumulate_direct<L>(
        args, center, neighbor_slot, radial_index,
        fp, sum_fxyz, fn, fnp, scratch, sink);
    accumulate_cross<L>(
        args, center, neighbor_slot, radial_index,
        fp, sum_fxyz, fn, fnp, scratch, sink);
  }
  ```

  Invoke orders in separate braces so their arrays cannot overlap in lifetime. Keep `n` and `basis` loops at fixed bounds with `#pragma unroll`; do not apply `--maxrregcount`.

- [ ] **Step 5: Rebuild, enforce resource limits, and rerun numerical tests**

  ```bash
  rm -f src/op/build/cuda/cmake/cuda/CMakeFiles/CalcOps_cuda.dir/__/__/kernel/calculateNepMbFeat_secondgradout.cu.o
  cmake --build src/op/build/cuda -j1 2>&1 | tee .validation/mb-secondgrad-ptxas-after.log
  python tests/check_nep_mb_secondgrad_ptxas.py \
    .validation/mb-secondgrad-ptxas-after.log
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    src/test/test_nep_electric/test_descriptor_vjp.py -q
  ```

  Expected: both instantiations have zero spill and fewer than 253 registers; all FP64 comparisons pass.

- [ ] **Step 6: Commit the register-lifetime specialization**

  ```bash
  git add src/op/kernel/utilities/nep_mbgrad_opt.cuh \
    tests/check_nep_mb_secondgrad_ptxas.py
  git commit -m "perf: specialize NEP angular second-gradient algebra"
  ```

---

### Task 4: Complete fallback and automatic dispatch coverage

**Files:**
- Modify: `src/op/kernel/calculateNepMbFeat_secondgradout.cu:225-382`
- Modify: `src/test/test_nep_electric/test_mb_secondgrad_kernel.py`

**Interfaces:**
- Consumes: legacy launcher from Task 1 and OMat24 launcher from Task 2.
- Produces: `bool nep_mb_secondgrad_optimized_supported(...)` including dimension, architecture, and shared-memory checks.
- Preserves: `auto`, `optimized`, and `legacy` environment values.

- [ ] **Step 1: Add fallback, forced-mode, and three/four/five-body tests**

  ```python
  @pytest.mark.parametrize("lmax", [(4, 0, 0), (4, 2, 0), (4, 2, 1)])
  def test_auto_matches_legacy_for_body_combinations(lmax):
      case = _repeated_type_case(n_types=3, n_max=4, n_base=8, lmax=lmax)
      _assert_triplet_close(
          _coefficient_second_grad(case, "auto"),
          _coefficient_second_grad(case, "legacy"),
      )


  def test_forced_optimized_rejects_unsupported_shape():
      case = _single_type_case(n_max=4, n_base=8, lmax=(4, 2, 1))
      with pytest.raises(RuntimeError, match="unsupported optimized NEP"):
          _coefficient_second_grad(case, "optimized")


  def test_auto_matches_optimized_for_omat24_shape():
      case = _six_local_type_case(n_max=5, n_base=9, lmax=(4, 2, 1))
      _assert_triplet_close(
          _coefficient_second_grad(case, "auto"),
          _coefficient_second_grad(case, "optimized"),
      )
  ```

  The last assertion is numerical coverage only: legacy and optimized must agree, so it passes while `auto` deliberately remains on legacy until Task 6.

- [ ] **Step 2: Implement explicit support checks without a GPU-to-CPU tensor readback**

  Query `cudaDeviceProp` on the host and check:

  ```cpp
  static bool nep_mb_secondgrad_optimized_supported(
      int nmax, int nbasis, int lmax3, int lmax4, int lmax5,
      size_t shared_bytes, const cudaDeviceProp& prop) {
    return prop.major >= 6 &&
           nmax == 5 && nbasis == 9 && lmax3 == 4 &&
           lmax4 > 0 && lmax5 > 0 &&
           shared_bytes <= prop.sharedMemPerBlock;
  }
  ```

  `legacy` always calls the unchanged path. Forced `optimized` throws a message containing all five runtime dimensions and the available/required shared memory when unsupported. For this task, `auto` records the support decision but still calls legacy; Task 5 flips the supported branch after measurement.

- [ ] **Step 3: Run supported and fallback tests**

  ```bash
  cmake --build src/op/build/cuda -j4
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    src/test/test_nep_electric/test_descriptor_vjp.py -q
  ```

  Expected: every test passes. Dispatch selection itself is verified from kernel names in the Task 6 nsys trace.

- [ ] **Step 4: Commit complete dispatch guards**

  ```bash
  git add src/op/kernel/calculateNepMbFeat_secondgradout.cu \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py
  git commit -m "feat: guard optimized NEP second-gradient dispatch"
  ```

---

### Task 5: Validate FP64 correctness and memory safety on GPU nodes

**Files:**
- Modify only if a failure identifies a defect: `src/op/kernel/utilities/nep_mbgrad_opt.cuh`, `src/op/kernel/calculateNepMbFeat_secondgradout.cu`, or `src/test/test_nep_electric/test_mb_secondgrad_kernel.py`
- Do not track: `.validation/*`

**Interfaces:**
- Consumes: forced `legacy` and `optimized` modes from Tasks 1-4.
- Produces: passing test, sanitizer, and architecture-build evidence stored under ignored `.validation/`.

- [ ] **Step 1: Build one binary containing SM60, SM70, and SM86 code**

  ```bash
  source /data/home/wuxingxing/anaconda3/etc/profile.d/conda.sh
  conda activate matpl-2026.3
  module load cuda/11.8-share cmake/3.31.6
  source /opt/rh/devtoolset-8/enable
  cmake -S src/op -B src/op/build/cuda \
    -DMATPL_GPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES='60;70;86'
  cmake --build src/op/build/cuda -j4 2>&1 | tee .validation/mb-secondgrad-sm60-sm70-sm86.log
  ```

  Expected: build succeeds and ptxas reports both optimized instantiations with zero spill.

- [ ] **Step 2: Run the full numerical matrix on a q4/3090 allocation**

  ```bash
  export PYTHONPATH=/data/home/wuxingxing/xcode/MatPL-dcu-dev/.worktrees/nep-fused-fitting
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    src/test/test_nep_electric/test_descriptor_vjp.py -q
  ```

  Expected: all active tests pass with `rtol=1e-9`, `atol=1e-11`; logs include maximum absolute and relative error for each case.

- [ ] **Step 3: Run compute-sanitizer against the wide-neighbor and six-type cases**

  ```bash
  for tool in memcheck racecheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode=99 \
      python -m pytest \
      src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
      -k 'wide_neighbor or six_local_type' -q \
      > ".validation/mb-secondgrad-${tool}.log" 2>&1
  done
  ```

  Expected: exit code 0 for all three tools and zero reported errors.

- [ ] **Step 4: Run the smallest numerical case on P100 and V100 when their nodes are available**

  Use the cluster's P100/V100 partition names without changing compiler options. Run:

  ```bash
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    -k 'single_type and optimized_matches_legacy' -q
  ```

  Expected: PASS on both SM60 and SM70. If those nodes are unavailable, retain the successful fat-binary build as compile coverage and record runtime coverage as unavailable.

- [ ] **Step 5: Fix only diagnosed failures and repeat their smallest reproducer**

  For a numerical mismatch, print the failing `[center_type, neighbor_type, n, basis]` index and compare the corresponding legacy direct/cross contribution. For sanitizer failures, fix the exact shared-memory offset, barrier, or bound named by the report. Rerun the single failing case, then Steps 1-3.

- [ ] **Step 6: Commit any correctness fixes**

  ```bash
  git add src/op/kernel/utilities/nep_mbgrad_opt.cuh \
    src/op/kernel/calculateNepMbFeat_secondgradout.cu \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py
  git diff --cached --quiet || git commit -m "fix: validate fused NEP second gradient"
  ```

---

### Task 6: Profile, tune, and enable the optimized auto path

**Files:**
- Modify: `src/op/kernel/utilities/nep_mbgrad_opt.cuh` only when a measured 32/64-thread or unroll change wins and passes Task 5 again.
- Modify: `src/op/kernel/calculateNepMbFeat_secondgradout.cu` to enable the supported `auto` branch.
- Modify: `src/test/test_nep_electric/test_mb_secondgrad_kernel.py` only if measured tuning adds a new boundary case.
- Do not modify: `/data/home/wuxingxing/datas/training_test/mgpus/omat24/mini_data_test/run.sh`

**Interfaces:**
- Consumes: the established profiler harness in `/data/home/wuxingxing/datas/training_test/debug/nep-fused-profile-20260908/`.
- Produces: `auto` dispatch to the optimized kernel for supported OMat24 shapes.
- Produces: nsys/ncu reports and legacy/optimized epoch timings in a new timestamped debug directory.

- [ ] **Step 1: Capture the same five-warmup/eight-steady-batch window for both modes**

  Submit on the unchanged `3090,q4` resources with CUDA 11.8. For each mode run:

  ```bash
  MATPL_NEP_MB_SECONDGRAD_MODE="$mode" \
  nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
    --cuda-memory-usage=true --force-overwrite=true \
    -o "$out/$mode/timeline" \
    python -u "$profile_root/profile_train.py" \
      --repo "$repo" --case "$case_dir" --out "$out/$mode/train" \
      --warmup 5 --steps 8
  ```

  Use the same GPU allocation for both runs. Export both reports to SQLite and run the existing `analyze_nsys.py` against identical batch indices.

- [ ] **Step 2: Check kernel, step-time, launch-count, and allocation gates**

  Require all of the following before enabling `auto`:

  - Optimized fused-kernel total time is lower than legacy `find_angular_gardc_neigh + aggregate_dfeat_c3 + aggregate_features` over the same eight batches.
  - Mean steady batch wall time is lower in optimized mode.
  - `aggregate_dfeat_c3` and `aggregate_features` disappear from the optimized timeline.
  - The native allocation corresponding to the previous 2.625 GiB `dfeat_c3` tensor disappears.
  - Loss values and input hashes match the legacy capture.

- [ ] **Step 3: Collect hardware counters for one optimized launch**

  First try q4/3090 with CUDA 11.8:

  ```bash
  PROFILE_KERNEL_FILTER='regex:.*nep_mb_secondgrad_fused.*' \
  PROFILE_LABEL=mb-secondgrad-optimized \
  sbatch "$profile_root/profile-ncu.slurm"
  ```

  If the node returns `ERR_NVGPUCTRPERM`, submit the same script to `4090t` with `PROFILE_CUDA_MODULE=cuda/12.4-share`. Record registers/thread, achieved occupancy, FP64 pipe activity, DRAM throughput, and the dominant warp-stall reasons. Confirm zero local-memory spill.

- [ ] **Step 4: Apply only measured register or launch-geometry changes**

  Compare the existing 32-thread path for `max_neighbors <= 32` and 64-thread path for larger lists. If one path loses, change only that threshold or one unroll decision, rebuild, run the A/B matrix, and repeat the same eight-batch capture. Keep a change only when both fused-kernel time and mean batch time improve; otherwise restore the prior version.

- [ ] **Step 5: Run two complete legacy epochs and two complete optimized epochs**

  Submit one wrapper job with the same `nodes=1`, `ntasks-per-node=1`, `cpus-per-task=4`, `gres=gpu:1`, and `partition=3090,q4` settings as `run.sh`. Inside that single allocation, call the existing script with `bash` so all four measurements use one physical GPU; do not edit `run.sh`:

  ```bash
  cd /data/home/wuxingxing/datas/training_test/mgpus/omat24/mini_data_test
  for repeat in 1 2; do
    MATPL_NEP_MB_SECONDGRAD_MODE=legacy \
      bash run.sh > "$out/epoch-legacy-${repeat}.log" 2>&1
    MATPL_NEP_MB_SECONDGRAD_MODE=optimized \
      bash run.sh > "$out/epoch-optimized-${repeat}.log" 2>&1
  done
  ```

  Compare median epoch time, final logged losses, and maximum CUDA memory. Optimized mode must be faster, must not increase peak memory, and must preserve training metrics within the numerical tolerance already established by the A/B tests.

- [ ] **Step 6: Enable optimized dispatch for supported `auto` calls**

  ```cpp
  switch (mode) {
    case NepMbSecondGradMode::Legacy:
      return launch_calculate_nepmbfeat_secondgradout_c3_legacy(args);
    case NepMbSecondGradMode::Optimized:
      if (!supported) throw_unsupported_optimized(args, prop, shared_bytes);
      return launch_calculate_nepmbfeat_secondgradout_c3_optimized(args, device);
    case NepMbSecondGradMode::Auto:
      if (supported) {
        return launch_calculate_nepmbfeat_secondgradout_c3_optimized(args, device);
      }
      return launch_calculate_nepmbfeat_secondgradout_c3_legacy(args);
  }
  ```

  Run `test_auto_matches_optimized_for_omat24_shape` after the switch, then confirm nsys records `nep_mb_secondgrad_fused` under `auto`.

- [ ] **Step 7: Run final verification**

  ```bash
  cmake --build src/op/build/cuda -j4 2>&1 | tee .validation/mb-secondgrad-final-build.log
  python tests/check_nep_mb_secondgrad_ptxas.py \
    .validation/mb-secondgrad-final-build.log
  python -m pytest \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py \
    src/test/test_nep_electric/test_descriptor_vjp.py \
    tests/test_nep_fused_fitting.py \
    tests/test_nep_fused_training.py -q
  ```

  Expected: all tests pass; resource checker passes; `auto`, forced optimized, and legacy behavior match their defined support rules.

- [ ] **Step 8: Commit the measured default and final tuning**

  ```bash
  git add src/op/kernel/utilities/nep_mbgrad_opt.cuh \
    src/op/kernel/calculateNepMbFeat_secondgradout.cu \
    src/test/test_nep_electric/test_mb_secondgrad_kernel.py
  git commit -m "perf: enable fused NEP second gradient for OMat24"
  ```
