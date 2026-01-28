cmake_minimum_required(VERSION 3.15)
project(GenerateNew LANGUAGES CXX)

# Find the Python interpreter and development components
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Set pybind11_DIR to the directory containing pybind11Config.cmake
if(DEFINED ENV{PYBIND11_DIR})
    file(TO_CMAKE_PATH "$ENV{PYBIND11_DIR}" pybind11_DIR)
endif()

# Find the pybind11 package
find_package(pybind11 CONFIG REQUIRED)

# Create the pybind11 module
pybind11_add_module(findneigh src/pynep.cpp src/nep.cpp)

# Set the target properties to rename the output file
set_target_properties(findneigh PROPERTIES OUTPUT_NAME "findneigh")

