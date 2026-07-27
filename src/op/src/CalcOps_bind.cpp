#include <torch/torch.h>
#include <torch/extension.h>

#include "../include/CalcOps.h"

TORCH_LIBRARY(CalcOps_cuda, m) {
    m.def("calculateForce", calculateForce);
    m.def("calculateVirial", calculateVirial);
    m.def("calculateCompress", calculateCompress);
    m.def("calculateNepFeat", calculateNepFeat);
    m.def("calculateNepFeatWithGradContext", calculateNepFeatWithGradContext);
    m.def("calculateNepFeatInputGrad", calculateNepFeatInputGrad);
    m.def("calculateNepMbFeat", calculateNepMbFeat);
    m.def("calculateNepMbFeatWithGradContext", calculateNepMbFeatWithGradContext);
    m.def("calculateNepMbFeatInputGrad", calculateNepMbFeatInputGrad);
    m.def("calculateNepForce", calculateNepForce);
    m.def("calculateNepVirial", calculateNepVirial);
    m.def("calculate_maxneigh", calculate_maxneigh);
    m.def("calculate_neighbor", calculate_neighbor);
    m.def("calculate_descriptor", calculate_descriptor);
}
