cmake_minimum_required(VERSION 3.10)
project(assembly_v3 LANGUAGES CXX CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "--extended-lambda -D TIMINGS")

add_executable(assembly_v3 main.cu prelude.h assembly.cuh config.h blocks.cuh cqueue.cuh cqutils.cuh mempool.cuh timer.h singleton.h scratch.cu fingerprint.cu preprocess.cu overlap.cu kthread.h)

set_target_properties(assembly_v3 PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(assembly_v3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# for release builds, use the following
# cmake -DCMAKE_BUILD_TYPE=Release ..