cmake_minimum_required(VERSION 3.6)

project(proto)

# Necessary for SWIG bindings; ignore cmake warnings
cmake_policy(SET CMP0078 OLD)
cmake_policy(SET CMP0086 OLD)
cmake_policy(SET CMP0148 OLD)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${PYTHON_INCLUDE_PATH})
# Finding eigen3 path flaky in some envs; use abs path
include_directories(/usr/local/include/eigen3)

find_package(GTest REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# gtest_all
add_executable(gtest_all gtest_all.cpp)
target_link_libraries(gtest_all LINK_PUBLIC LTSS DP score ${GTEST_LIBRARIES} pthread)

# include_directories(${GTEST_INCLUDE_DIRS})
# include_directories(${PYTHON_INCLUDE_PATH})

