#include "nep.h"
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
  
  private:
    NEP3 calc;
};

FindNeigh::FindNeigh(): calc(NEP3()) {}

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
}

PYBIND11_MODULE(findneigh, m){
    m.doc() = "findneigh";
    py::class_<FindNeigh>(m, "FindNeigh")
    // .def("setAtoms", &FindNeigh::setAtoms)
		.def(py::init())
    .def("getNeigh", &FindNeigh::getNeigh)
		;
}
