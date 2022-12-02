cmake_minimum_required(VERSION 3.10)
project(empanada LANGUAGES CXX CUDA)

find_package(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_FLAGS "--extended-lambda -D TIMINGS")

add_executable(empanada main.cu prelude.h assembly.cuh config.h blocks.cuh cqueue.cuh cqutils.cuh mempool.cuh timer.h singleton.h scratch.cu fingerprint.cu preprocess.cu overlap.cu kthread.h)

set_target_properties(empanada PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(empanada PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

# for release builds, use the following
# cmake -DCMAKE_BUILD_TYPE=Release ..