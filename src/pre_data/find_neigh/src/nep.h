/*
    Copyright 2022 Zheyong Fan, Junjie Wang, Eric Lindgren
    This file is part of NEP_CPU.
    NEP_CPU is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    NEP_CPU is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with NEP_CPU.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include <string>
#include <vector>

// #define USE_TABLE_FOR_RADIAL_FUNCTIONS

class NEP3
{
public:
  NEP3();

  void find_neigh(
    const double rc_radial,
    const double rc_angular,
    const int MN, // max neighs of config ,which will be num_types * max_neigh set in json file
    const std::vector<int>& atom_type_map,
    const std::vector<double>& box,
    const std::vector<double>& position);

  int num_atoms = 0;
  int num_cells[3];
  double ebox[18];
  // NN nums of neighbors; NL neighbor lists; NLT neigbors' type
  std::vector<int> NN_radial, NL_radial, NLT_radial, NN_angular, NL_angular, NLT_angular;
  // std::vector<double> r12;
  std::vector<double> r12_radial;
  std::vector<double> r12_angular;
  // void allocate_memory(const int N);
};
