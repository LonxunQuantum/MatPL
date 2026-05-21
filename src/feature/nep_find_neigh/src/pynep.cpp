/*
List of modified records by Wu Xingxing (email stars_sparkling@163.com)
1. Added network structure support for NEP4 model independent bias
    Modified force field reading;
    Modified the applyann_one_layer method;
2. Added handling of inconsistency between the atomic order of the input structure of LAMMPS and the atomic order in the force field
3. In order to adapt to multiple model biases, the function has been added with computefor_lamps() and the int model_index parameter has been added  
4. Support GPUMD NEP shared bias and PWMLFF NEP independent bias forcefield

We have made the following improvements based on NEP4
http://doc.lonxun.com/MatPL/models/nep/
*/

/*
the open source code from https://github.com/brucefan1983/NEP_CPU
the licnese of NEP_CPU is as follows:
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

/*----------------------------------------------------------------------------80
A CPU implementation of the neuroevolution potential (NEP)
Ref: Zheyong Fan et al., Neuroevolution machine learning potentials:
Combining high accuracy and low cost in atomistic simulations and application to
heat transport, Phys. Rev. B. 104, 104309 (2021).
------------------------------------------------------------------------------*/

/*
The compute interface is used for inference in the PWMLFF test and inner interfaces, using the Cartesian coordinate system of GPUMD;
The find_neighbor interface is used for model training and adopts the score coordinates of PWMLFF.
*/
#include "nep.h"
#include "dp.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <time.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

class FindNeigh
{
  public:
    FindNeigh();
    std::tuple<std::vector<double>, std::vector<double>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> getNeigh(double, double, int, std::vector<int>, std::vector<double>, std::vector<double>);
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> inference(std::vector<int>, std::vector<double>, std::vector<double>);
    // returns per-atom descriptor as a flat row-major vector of length N*dim
    // ordered as atom0_d0..atom0_d{dim-1}, atom1_d0.., ..., atom{N-1}_d{dim-1}
    std::vector<double> getDescriptor(std::vector<int> atom_type_map,
                                      std::vector<double> box,
                                      std::vector<double> position);
    int getDim() const { return calc.annmb.dim; }
    std::tuple<std::vector<int>, std::vector<double>> getNeighDP( int ntypes,
                                                                  int max_neigh_num,
                                                                  double rcut,
                                                                  std::vector<double>  rcut_type,
                                                                  std::vector<int> atom_type_map,
                                                                  std::vector<double> box,
                                                                  std::vector<double> position);

    void allocate_memory(const int N);
    void init_model(const std::string& potential_filename);
  // private:
    NEP_CPU calc;
    DP_CPU  calc_dp;
    int num_atoms = 0;

    std::vector<double> potential;
    std::vector<double> force;
    std::vector<double> virial;
    std::vector<double> total_virial;
    std::vector<double> descriptor;

};

FindNeigh::FindNeigh(): calc(NEP_CPU()), calc_dp(DP_CPU()) {}

void FindNeigh::init_model(const std::string& potential_filename) {
  calc.init_from_file(potential_filename, true);
}

void FindNeigh::allocate_memory(const int N)
{
  if (num_atoms != N) {
    potential.resize(N);
    force.resize(N * 3);
    virial.resize(N * 9);
    total_virial.resize(9);
    num_atoms = N;
  }
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> FindNeigh::inference(
  std::vector<int> atom_type_map,
  std::vector<double> box,
  std::vector<double> position
)
{
  int N = atom_type_map.size();
  allocate_memory(N);
  calc.compute(atom_type_map, box, position, potential, force, virial, total_virial);
  return std::make_tuple(potential, force, total_virial);
}

std::vector<double> FindNeigh::getDescriptor(
  std::vector<int> atom_type_map,
  std::vector<double> box,
  std::vector<double> position)
{
  const int N = atom_type_map.size();
  const int dim = calc.annmb.dim;
  allocate_memory(N);
  // compute_descriptor writes layout d0[0..N-1], d1[0..N-1], ... (column-major in (N,dim))
  std::vector<double> col_major(static_cast<size_t>(N) * dim, 0.0);
  calc.compute_descriptor(atom_type_map, box, position, col_major);

  // Transpose to row-major so Python sees shape (N, dim) directly.
  if (descriptor.size() != static_cast<size_t>(N) * dim) {
    descriptor.assign(static_cast<size_t>(N) * dim, 0.0);
  }
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < dim; ++d) {
      descriptor[static_cast<size_t>(n) * dim + d] = col_major[static_cast<size_t>(d) * N + n];
    }
  }
  return descriptor;
}

std::tuple<std::vector<double>, std::vector<double>, std::vector<int>, std::vector<int>, std::vector<int>, std::vector<int>> FindNeigh::getNeigh(
  double rc_radial,
  double rc_angular,
  int max_neigh_num,
  std::vector<int> atom_type_map,
  std::vector<double> box,
  std::vector<double> position)
{
  calc.find_neigh(rc_radial, rc_angular, max_neigh_num, atom_type_map, box, position);
  return std::make_tuple(calc.r12_radial, calc.r12_angular, calc.NL_radial, calc.NL_angular, calc.NLT_radial, calc.NLT_angular);
  // return std::make_tuple(calc.r12_radial, calc.NL_radial, calc.NLT_radial);
}

std::tuple<std::vector<int>, std::vector<double>> FindNeigh::getNeighDP(
  int ntypes,
  int max_neigh_num,
  double rcut,
  std::vector<double>  rcut_type,
  std::vector<int> atom_type_map,
  std::vector<double> box,
  std::vector<double> position)
{
  calc_dp.find_neigh(ntypes, max_neigh_num, rcut, rcut_type, atom_type_map, box, position);
  return std::make_tuple(calc_dp.dp_NL, calc_dp.dp_r12);
}

PYBIND11_MODULE(findneigh, m){
    m.doc() = "findneigh";
    py::class_<FindNeigh>(m, "FindNeigh")
    // .def("setAtoms", &FindNeigh::setAtoms)
		.def(py::init())
    .def("getNeigh", &FindNeigh::getNeigh)
    .def("getNeighDP", &FindNeigh::getNeighDP)
    .def("inference", &FindNeigh::inference)
    .def("getDescriptor", &FindNeigh::getDescriptor)
    .def("getDim", &FindNeigh::getDim)
    .def("init_model", &FindNeigh::init_model)
		;
}
