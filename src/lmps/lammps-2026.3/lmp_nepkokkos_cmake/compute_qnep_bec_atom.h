/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(qnep/bec/atom,ComputeQNEPBECAtom);
// clang-format on
#else

#ifndef LMP_COMPUTE_QNEP_BEC_ATOM_H
#define LMP_COMPUTE_QNEP_BEC_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeQNEPBECAtom : public Compute {
 public:
  ComputeQNEPBECAtom(class LAMMPS *, int, char **);
  ~ComputeQNEPBECAtom() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;

 private:
  int nmax;
};

}    // namespace LAMMPS_NS

#endif
#endif
