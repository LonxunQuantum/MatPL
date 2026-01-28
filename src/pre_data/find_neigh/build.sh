c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/pynep.cpp src/nep.cpp -o findneigh.so

# make build
# cd build
# cmake -Dpybind11_DIR=$(python -m pybind11 --cmakedir) .. && make
