#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

// Shared by the CPU and GPU text inference readers, independent of training.
namespace nep_inference {
constexpr int kElementCount = 118;
constexpr int kCovalentRadiusCount = 94;
constexpr int kFlexibleZblTypeCount = 10; // 550 parameters = 10 * 10 * 11 / 2.

inline std::vector<int> parse_elements(const std::vector<std::string>& tokens)
{
  static const char* const symbols[kElementCount] = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh",
    "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
    "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db",
    "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};
  if (tokens.size() < 2) {
    throw std::invalid_argument("NEP header requires an element count and atom symbols");
  }
  int count = 0;
  std::size_t consumed = 0;
  try {
    count = std::stoi(tokens[1], &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument("NEP element count must be an integer from 1 to 118");
  }
  if (consumed != tokens[1].size() || count < 1 || count > kElementCount) {
    throw std::invalid_argument("NEP element count must be an integer from 1 to 118");
  }
  if (tokens.size() != static_cast<std::size_t>(count + 2)) {
    throw std::invalid_argument("NEP element count does not match the number of atom symbols");
  }
  bool seen[kElementCount] = {};
  std::vector<int> numbers;
  numbers.reserve(count);
  for (int i = 0; i < count; ++i) {
    int index = 0;
    while (index < kElementCount && tokens[i + 2] != symbols[index]) {
      ++index;
    }
    if (index == kElementCount) {
      throw std::invalid_argument("NEP unknown element symbol: " + tokens[i + 2]);
    }
    if (seen[index]) {
      throw std::invalid_argument("NEP duplicate element symbol: " + tokens[i + 2]);
    }
    seen[index] = true;
    numbers.push_back(index + 1);
  }
  return numbers;
}

inline void validate_types(const int* types, std::size_t count,
                           const std::vector<int>& atomic_numbers,
                           bool model_loaded, bool typewise_zbl)
{
  if (!model_loaded) {
    throw std::invalid_argument("NEP model must be loaded before inference");
  }
  if (count == 0) {
    throw std::invalid_argument("NEP inference requires at least one atom");
  }
  for (std::size_t i = 0; i < count; ++i) {
    const int type = types[i];
    if (type < 0 || static_cast<std::size_t>(type) >= atomic_numbers.size()) {
      throw std::invalid_argument("NEP atom type index is outside the loaded model");
    }
    // Do not invent covalent radii for the elements absent from the table.
    // Unused elements in a 118-element model do not restrict supported inputs.
    if (typewise_zbl && atomic_numbers[type] > kCovalentRadiusCount) {
      throw std::invalid_argument("NEP typewise ZBL requires atomic numbers from 1 to 94");
    }
  }
}

inline void validate_sizes(std::size_t atoms, std::size_t box, std::size_t positions)
{
  if (box != 9) {
    throw std::invalid_argument("NEP box must contain 9 components");
  }
  if (positions != 3 * atoms) {
    throw std::invalid_argument("NEP position must contain 3 components per atom");
  }
}

inline void validate_flexible_zbl(int types)
{
  if (types > kFlexibleZblTypeCount) {
    throw std::invalid_argument("NEP flexible ZBL supports at most 10 element types");
  }
}
} // namespace nep_inference
