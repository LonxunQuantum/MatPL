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

#pragma once

#include "error.cuh"

enum class Memory_Type {
    global = 0, // global memory, also called (linear) device memory
    managed     // managed memory, also called unified memory
};

template <typename T>
class GPU_Vector {
public:
    GPU_Vector();
    GPU_Vector(const size_t size, const Memory_Type memory_type = Memory_Type::global);
    GPU_Vector(const size_t size, const T value, const Memory_Type memory_type = Memory_Type::global);
    ~GPU_Vector();

    void resize(const size_t size, const Memory_Type memory_type = Memory_Type::global);
    void resize(const size_t size, const T value, const Memory_Type memory_type = Memory_Type::global);

    void copy_from_host(const T* h_data);
    void copy_from_host(const T* h_data, const size_t size);
    void copy_from_device(const T* d_data);
    void copy_from_device(const T* d_data, const size_t size);
    void copy_to_host(T* h_data);
    void copy_to_host(T* h_data, const size_t size);
    void copy_to_device(T* d_data);
    void copy_to_device(T* d_data, const size_t size);

    void fill(const T value);

    T& operator[](int index);

    size_t size() const;
    T const* data() const;
    T* data();

private:
    bool allocated_;
    size_t size_;
    size_t memory_;
    Memory_Type memory_type_;
    T* data_;
};
