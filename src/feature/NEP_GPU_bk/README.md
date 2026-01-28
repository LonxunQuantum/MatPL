This find neigh method is from https://github.com/bigd4/PyNEP/blob/master/nep_cpu.
We have retained the findneigh method, modified the Cartesian coordinate system to calculate fractional coordinates, and removed other irrelevant code in this method
This is for finding  nep neighbor

# make build
# cd build
# cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make

