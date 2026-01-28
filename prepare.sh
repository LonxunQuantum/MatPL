#!/bin/bash
pip install pwdata scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple

# 编译时可能找不到librt.so，可以临时做个软链接
ln -s /usr/lib/x86_64-linux-gnu/librt.so.1 /usr/lib/x86_64-linux-gnu/librt.so

# 编译时可能找不到libclang_rt.builtins-x86_64，临时修复
sed -i.bak '/get_filename_component(HIP_CLANG_INCLUDE_PATH/s/^/# /' /opt/dtk/lib/cmake/hip-lang/hip-lang-config.cmake
