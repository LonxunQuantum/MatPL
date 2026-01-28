#!/bin/bash

SRC_DIR=/workspace/MatPL-nep/src/op
BUILD_DIR=${SRC_DIR}/build

cd ${SRC_DIR}
rm ${BUILD_DIR} -rf
cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release && cmake --build ${BUILD_DIR} -j32
cd -

