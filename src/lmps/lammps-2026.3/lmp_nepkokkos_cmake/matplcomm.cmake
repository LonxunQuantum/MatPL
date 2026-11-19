# Preset that turns on all existing packages. Using the combination
# of this preset followed by the nolib.cmake preset should configure
# a LAMMPS binary, with as many packages included, that can be compiled
# with just a working C++ compiler and an MPI library.

set(ALL_PACKAGES
  CG-SPICA
  CLASS2
  COLLOID
  COLVARS
  COMPRESS
  CORESHELL
  DIELECTRIC
  DIFFRACTION
  DIPOLE
  DPD-BASIC
  DPD-MESO
  DPD-REACT
  DPD-SMOOTH
  DRUDE
  EFF
  ELECTRODE
  EXTRA-COMMAND
  EXTRA-COMPUTE
  EXTRA-FIX
  EXTRA-DUMP
  EXTRA-MOLECULE
  EXTRA-PAIR
  FEP
  GRANULAR
  INTEL
  INTERLAYER
  KSPACE
  LATBOLTZ
  LEPTON
  MANIFOLD
  MANYBODY
  MC
  MEAM
  MGPT
  MISC
  MOFFF
  MOLECULE
  MOLFILE
  OPENMP
  OPT
  ORIENT
  PERI
  PHONON
  PLUMED
  POEMS
  PTM
  PYTHON
  QEQ
  QMMM
  QTB
  REACTION
  REAXFF
  REPLICA
  RIGID
  SHOCK
  SMTBQ
  SPH
  SPIN
  SRD
  TALLY
  UEF
  VORONOI
  YAFF)

foreach(PKG ${ALL_PACKAGES})
  set(PKG_${PKG} ON CACHE BOOL "" FORCE)
endforeach()
