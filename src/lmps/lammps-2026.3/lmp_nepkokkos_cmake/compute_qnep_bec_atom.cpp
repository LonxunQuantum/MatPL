/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
------------------------------------------------------------------------- */

#include "compute_qnep_bec_atom.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair_nep.h"
#include "update.h"

#include <cstring>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ComputeQNEPBECAtom::ComputeQNEPBECAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), nmax(0)
{
  if (narg != 3) error->all(FLERR, "Illegal compute qnep/bec/atom command");

  peratom_flag = 1;
  size_peratom_cols = 9;
  array_atom = nullptr;
}

/* ---------------------------------------------------------------------- */

ComputeQNEPBECAtom::~ComputeQNEPBECAtom()
{
  memory->destroy(array_atom);
}

/* ---------------------------------------------------------------------- */

void ComputeQNEPBECAtom::init()
{
  if (force->pair == nullptr) {
    error->all(FLERR, "compute qnep/bec/atom requires pair_style matpl/nep/kk");
  }
  auto *pair = dynamic_cast<PairNEP *>(force->pair);
  if (pair == nullptr) {
    error->all(FLERR, "compute qnep/bec/atom requires pair_style matpl/nep/kk");
  }
}

/* ---------------------------------------------------------------------- */

void ComputeQNEPBECAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (atom->nmax > nmax) {
    memory->destroy(array_atom);
    nmax = atom->nmax;
    memory->create(array_atom, nmax, 9, "qnep/bec/atom:array");
  }

  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; ++i) {
    for (int d = 0; d < 9; ++d) array_atom[i][d] = 0.0;
  }

  auto *pair = dynamic_cast<PairNEP *>(force->pair);
  if (pair == nullptr) {
    error->all(FLERR, "compute qnep/bec/atom requires pair_style matpl/nep/kk");
  }
  pair->qnep_bec_atom(array_atom, nmax);

  int *mask = atom->mask;
  for (int i = 0; i < nlocal; ++i) {
    if (mask[i] & groupbit) continue;
    for (int d = 0; d < 9; ++d) array_atom[i][d] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

double ComputeQNEPBECAtom::memory_usage()
{
  return static_cast<double>(nmax) * 9 * sizeof(double);
}
