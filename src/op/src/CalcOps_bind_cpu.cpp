#include <torch/torch.h>
#include <torch/extension.h>

#include "../include/CalcOps.h"

TORCH_LIBRARY(CalcOps_cpu, m) {
    m.def("calculateForce", calculateForce_cpu);
    m.def("calculateVirial", calculateVirial_cpu);
    m.def("calculateCompress", calculateCompress_cpu);    
    m.def("calculate_maxneigh", calculate_maxneigh_cpu);
    m.def("calculate_neighbor", calculate_neighbor_cpu);
    m.def("calculate_descriptor", calculate_descriptor_cpu);
}