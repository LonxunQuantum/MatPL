#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace nep_type_slim {

struct Layout {
  std::vector<int> sim_to_nep;
  std::vector<int> nep_to_sim;
};

inline Layout build_layout(const int num_nep_types, const std::vector<int>& configured_nep_types)
{
  if (num_nep_types <= 0) {
    throw std::invalid_argument("NEP type count must be positive");
  }
  if (configured_nep_types.empty()) {
    throw std::invalid_argument("LAMMPS type map must not be empty");
  }

  std::vector<bool> seen(static_cast<std::size_t>(num_nep_types), false);
  for (std::size_t i = 0; i < configured_nep_types.size(); ++i) {
    const int type = configured_nep_types[i];
    if (type < 0 || type >= num_nep_types) {
      throw std::out_of_range("LAMMPS type maps outside the NEP type range");
    }
    seen[static_cast<std::size_t>(type)] = true;
  }

  Layout layout;
  layout.nep_to_sim.assign(static_cast<std::size_t>(num_nep_types), -1);
  for (int nep_type = 0; nep_type < num_nep_types; ++nep_type) {
    if (!seen[static_cast<std::size_t>(nep_type)]) continue;
    layout.nep_to_sim[static_cast<std::size_t>(nep_type)] =
      static_cast<int>(layout.sim_to_nep.size());
    layout.sim_to_nep.push_back(nep_type);
  }
  return layout;
}

template <typename T>
std::vector<T> extract_coefficients(
  const std::vector<T>& full,
  const int num_nep_types,
  const std::vector<int>& sim_to_nep,
  const int num_n,
  const int num_basis)
{
  if (num_nep_types <= 0 || num_n <= 0 || num_basis <= 0) {
    throw std::invalid_argument("coefficient dimensions must be positive");
  }
  if (sim_to_nep.empty()) {
    throw std::invalid_argument("compact type set must not be empty");
  }

  const std::size_t pair_block =
    static_cast<std::size_t>(num_n) * static_cast<std::size_t>(num_basis);
  const std::size_t expected_full =
    static_cast<std::size_t>(num_nep_types) * static_cast<std::size_t>(num_nep_types) * pair_block;
  if (full.size() != expected_full) {
    throw std::invalid_argument("full coefficient buffer has an unexpected size");
  }

  const int sim_types = static_cast<int>(sim_to_nep.size());
  std::vector<T> compact(
    static_cast<std::size_t>(sim_types) * static_cast<std::size_t>(sim_types) * pair_block);
  for (int sim_i = 0; sim_i < sim_types; ++sim_i) {
    const int nep_i = sim_to_nep[static_cast<std::size_t>(sim_i)];
    if (nep_i < 0 || nep_i >= num_nep_types) {
      throw std::out_of_range("compact type maps outside the NEP type range");
    }
    for (int sim_j = 0; sim_j < sim_types; ++sim_j) {
      const int nep_j = sim_to_nep[static_cast<std::size_t>(sim_j)];
      if (nep_j < 0 || nep_j >= num_nep_types) {
        throw std::out_of_range("compact type maps outside the NEP type range");
      }
      const std::size_t src =
        (static_cast<std::size_t>(nep_i) * num_nep_types + nep_j) * pair_block;
      const std::size_t dst =
        (static_cast<std::size_t>(sim_i) * sim_types + sim_j) * pair_block;
      for (std::size_t offset = 0; offset < pair_block; ++offset) {
        compact[dst + offset] = full[src + offset];
      }
    }
  }
  return compact;
}

} // namespace nep_type_slim
