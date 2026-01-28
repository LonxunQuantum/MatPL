#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nep3.cuh"

namespace py = pybind11;

PYBIND11_MODULE(nep_gpu, m) {
    py::class_<NEP3>(m, "NEP3")
        .def(py::init<>())  // 暴露默认构造函数
        .def("init_from_file", &NEP3::init_from_file, 
             py::arg("file_potential"), 
             py::arg("is_rank_0"), 
             py::arg("in_device_id"))
        .def("inference", [](NEP3& self, 
                                  py::array_t<int> itype_cpu, 
                                  py::array_t<double> box_cpu, 
                                  py::array_t<double> position_cpu) {
            // 获取 NumPy 数组的指针
            auto itype_ptr = itype_cpu.mutable_data();
            auto box_ptr = box_cpu.mutable_data();
            auto position_ptr = position_cpu.mutable_data();
            // 调用 NEP3 的 inference 方法
            int N = itype_cpu.size();
            // printf("=========input N is %d =====\n", N);
            self.inference(N, itype_ptr, box_ptr, position_ptr);
            size_t potential_size = self.nep_data.cpu_potential_per_atom.size();
            size_t force_size = self.nep_data.cpu_force_per_atom.size();
            size_t virial_size = self.nep_data.cpu_total_virial.size();
            // std::vector<size_t> force_shape = {force_size/3, 3};
            return py::make_tuple(
                py::array_t<double>(potential_size, self.nep_data.cpu_potential_per_atom.data()),
                py::array_t<double>(force_size, self.nep_data.cpu_force_per_atom.data()),
                py::array_t<double>(virial_size, self.nep_data.cpu_total_virial.data())
            );
        }, 
        py::arg("itype_cpu"), 
        py::arg("box_cpu"), 
        py::arg("position_cpu"));
}

