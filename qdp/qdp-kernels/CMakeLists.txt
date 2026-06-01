cmake_minimum_required(VERSION 3.18)
project(qdp_kernels_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
include_directories(src)

add_executable(bench_kernels tests/bench_kernels.cu src/iqp.cu src/iqp_tc.cu src/AdaptiveOzaki.cu src/amplitude.cu src/angle.cu src/basis.cu src/phase.cu src/validation.cu)
set_target_properties(bench_kernels PROPERTIES CUDA_ARCHITECTURES "89;native")
target_link_libraries(bench_kernels CUDA::cudart)
