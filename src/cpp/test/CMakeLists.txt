cmake_minimum_required(VERSION 3.6)

project(proto)

cmake_policy(SET CMP0078 OLD)
cmake_policy(SET CMP0086 OLD)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
include_directories(${PYTHON_INCLUDE_PATH})
# Finding eigen3 path flaky in some envs; use abs path
include_directories(/usr/local/include/eigen3)

include(GNUInstallDirs)

find_package(GTest REQUIRED)

# gtest_all
add_executable(gtest_all gtest_all.cpp)
target_link_libraries(gtest_all LTSS DP ${GTEST_LIBRARIES} pthread)

include_directories(${GTEST_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${PYTHON_INCLUDE_PATH})
