cmake_minimum_required(VERSION 3.6)

project(proto)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)

find_package(benchmark REQUIRED)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

# Finding eigen3 path flaky in some envs; use abs path
include_directories(/usr/local/include/eigen3)

add_executable(benchmarks benchmarks.cpp)
target_link_libraries(benchmarks benchmark::benchmark pthread)
