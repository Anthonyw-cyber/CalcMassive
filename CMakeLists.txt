#cmake_minimum_required(VERSION 3.29)
#project(CalcMassive CUDA)
#
#set(CMAKE_CUDA_STANDARD 20)
#
#add_executable(CalcMassive main.cu
#        main.cu)
#
#set_target_properties(CalcMassive PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)

cmake_minimum_required(VERSION 3.18)
project(FireflyCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

add_executable(firefly main.cpp firefly.cu
        src/main.cpp)

set_target_properties(firefly PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
)

target_include_directories(firefly PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(firefly ${CUDA_LIBRARIES})
