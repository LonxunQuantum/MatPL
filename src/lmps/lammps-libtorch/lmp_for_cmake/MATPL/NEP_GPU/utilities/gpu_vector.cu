/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "gpu_vector.cuh"

namespace {
template <typename T>
__global__ void gpu_fill(const size_t size, const T value, T* data) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        data[i] = value;
}
} // anonymous namespace

template <typename T>
GPU_Vector<T>::GPU_Vector() {
    size_ = 0;
    memory_ = 0;
    memory_type_ = Memory_Type::global;
    allocated_ = false;
}

template <typename T>
GPU_Vector<T>::GPU_Vector(const size_t size, const Memory_Type memory_type) {
    allocated_ = false;
    resize(size, memory_type);
}

template <typename T>
GPU_Vector<T>::GPU_Vector(const size_t size, const T value, const Memory_Type memory_type) {
    allocated_ = false;
    resize(size, value, memory_type);
}

template <typename T>
GPU_Vector<T>::~GPU_Vector() {
    if (allocated_) {
        CHECK_CUDA_NEP(cudaFree(data_));
        allocated_ = false;
    }
}

template <typename T>
void GPU_Vector<T>::resize(const size_t size, const Memory_Type memory_type) {
    size_ = size;
    memory_ = size_ * sizeof(T);
    memory_type_ = memory_type;
    if (allocated_) {
        CHECK_CUDA_NEP(cudaFree(data_));
        allocated_ = false;
    }
    if (memory_type_ == Memory_Type::global) {
        CHECK_CUDA_NEP(cudaMalloc((void**)&data_, memory_));
        allocated_ = true;
    } else {
        CHECK_CUDA_NEP(cudaMallocManaged((void**)&data_, memory_));
        allocated_ = true;
    }
}

template <typename T>
void GPU_Vector<T>::resize(const size_t size, const T value, const Memory_Type memory_type) {
    size_ = size;
    memory_ = size_ * sizeof(T);
    memory_type_ = memory_type;
    if (allocated_) {
        CHECK_CUDA_NEP(cudaFree(data_));
        allocated_ = false;
    }
    if (memory_type == Memory_Type::global) {
        CHECK_CUDA_NEP(cudaMalloc((void**)&data_, memory_));
        allocated_ = true;
    } else {
        CHECK_CUDA_NEP(cudaMallocManaged((void**)&data_, memory_));
        allocated_ = true;
    }
    // printf("size %d sizetype %d memory_ %d malloc %d M success\n", size, sizeof(T), memory_, memory_/1024/1024);
    fill(value);
}

template <typename T>
void GPU_Vector<T>::copy_from_host(const T* h_data) {
    CHECK_CUDA_NEP(cudaMemcpy(data_, h_data, memory_, cudaMemcpyHostToDevice));
}

template <typename T>
void GPU_Vector<T>::copy_from_host(const T* h_data, const size_t size) {
    const size_t memory = sizeof(T) * size;
    CHECK_CUDA_NEP(cudaMemcpy(data_, h_data, memory, cudaMemcpyHostToDevice));
}

template <typename T>
void GPU_Vector<T>::copy_from_device(const T* d_data) {
    CHECK_CUDA_NEP(cudaMemcpy(data_, d_data, memory_, cudaMemcpyDeviceToDevice));
}

template <typename T>
void GPU_Vector<T>::copy_from_device(const T* d_data, const size_t size) {
    const size_t memory = sizeof(T) * size;
    CHECK_CUDA_NEP(cudaMemcpy(data_, d_data, memory, cudaMemcpyDeviceToDevice));
}

template <typename T>
void GPU_Vector<T>::copy_to_host(T* h_data) {
    CHECK_CUDA_NEP(cudaMemcpy(h_data, data_, memory_, cudaMemcpyDeviceToHost));
}

template <typename T>
void GPU_Vector<T>::copy_to_host(T* h_data, const size_t size) {
    const size_t memory = sizeof(T) * size;
    CHECK_CUDA_NEP(cudaMemcpy(h_data, data_, memory, cudaMemcpyDeviceToHost));
}

template <typename T>
void GPU_Vector<T>::copy_to_device(T* d_data) {
    CHECK_CUDA_NEP(cudaMemcpy(d_data, data_, memory_, cudaMemcpyDeviceToDevice));
}

template <typename T>
void GPU_Vector<T>::copy_to_device(T* d_data, const size_t size) {
    const size_t memory = sizeof(T) * size;
    CHECK_CUDA_NEP(cudaMemcpy(d_data, data_, memory, cudaMemcpyDeviceToDevice));
}

template <typename T>
void GPU_Vector<T>::fill(const T value) {
    if (memory_type_ == Memory_Type::global) {
        const int block_size = 128;
        const int grid_size = (size_ + block_size - 1) / block_size;
        gpu_fill<<<grid_size, block_size>>>(size_, value, data_);
        CUDA_CHECK_KERNEL
    } else { // managed (or unified) memory
        for (int i = 0; i < size_; ++i)
            data_[i] = value;
    }
}

template <typename T>
T& GPU_Vector<T>::operator[](int index) {
    return data_[index];
}

template <typename T>
size_t GPU_Vector<T>::size() const {
    return size_;
}

template <typename T>
T const* GPU_Vector<T>::data() const {
    return data_;
}

template <typename T>
T* GPU_Vector<T>::data() {
    return data_;
}

// Explicit template instantiation
template class GPU_Vector<int>;
template class GPU_Vector<float>;
template class GPU_Vector<double>;
