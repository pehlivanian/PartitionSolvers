cmake_minimum_required(VERSION 3.6)

###########################
# BEGIN COMMON DIRECTIVES #
###########################
project(proto)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})

# Toggle compiler support: C++11...C++17
option(USE_C++17 "Use C++17 compliant compiler, default is C++11" OFF)
option(USE_C++14 "Use C++14 compliant compiler, default is C++!1" OFF)
option(USE_C++11 "Use C++11 compliant compiler, default is C++11" ON)
option(GTEST "Compile google tests, default is OFF" OFF)
option(SWIG_BINDINGS "Compile SWIG bindings for python" ON)

if(CMAKE_CXX_COMPILER_LOADED)
  message(STATUS "The C++ compiler ID is: ${CMAKE_CXX_COMPILER_ID}")
  message(STATUS "The C++ compiler version is: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/python)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/examples)

if (USE_C++17)
  set(CMAKE_CXX_STANDARD 17)
  add_definitions(-DIS_CXX_17)
  message(STATUS "C++17-compliant CXX")
elseif (USE_C++14)
  set(CMAKE_CXX_STANDARD 14)
  add_definitions(-DIS_CXX_14)
  message(STATUS "C++14-compliant CXX")
else()
  set(CMAKE_CXX_STANDARD 11)
  add_definitions(-DIS_CXX_11)
  message(STATUS "C++11-compliant CXX")
endif()

# Debug symbols -- for example, config with cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Debug
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(src/cpp)
if (GTEST)
  add_subdirectory(src/cpp/test)
endif()

#########################################################
# To compile swig c++ python bindings from command line #
#########################################################
# SWIG bindings can be created outside of cmake, for those who just want the python
# interface and that's it. 
# 
# ---> Execute the following from <PROJECT_SOURCE_DIR>/src/cpp
#
# To generate swig bindings for [C++11] which do not link to eigen3 lib
# $ swig -c++ -python -I./include -outdir ../python proto.i
# $ g++ -std=c++11 -c -fPIC -O3 LTSS.cpp python_dpsolver.cpp DP.cpp python_ltsssolver.cpp proto_wrap.cxx -I/usr/include/python3.6 -I./include
# $ g++ -std=c++11 -O3 -shared python_dpsolver.o DP.o python_ltsssolver.o LTSS.o proto_wrap.o -o ../python/_proto.so -lstdc++

# To generate swig bindings [>=C++14] with eigen3
# $ swig -c++ -python -I./include -outdir ../python proto.i
# $ g++ -std=c++14 -c -fPIC -O3 -I/usr/local/include/eigen3 LTSS.cpp python_dpsolver.cpp DP.cpp python_ltsssolver.cpp proto_wrap.cxx -I/usr/include/python3.6 -I./include
# $ g++ -std=c++14 -O3 -shared python_dpsolver.o DP.o python_ltsssolver.o LTSS.o proto_wrap.o -o ../python/_proto.so -lstdc++
# $ rm *.o


# Test:
# $ python solver_ex.py

# Please replace the /usr/include/python3.6 directory above with the include directory on your host, as in
# In [1]: from sysconfig import get_paths                                                                                  
# In [2]: from pprint import pprint                                                                                        
# In [3]: pprint(get_paths())                                                                                                                
# {'data': '/usr',
# 
#  'include': '/usr/include/python3.6',			<<- "include_path"
# 
#  'platinclude': '/usr/include/python3.6', 
#  'platlib': '/usr/lib/python3.6/site-packages',
#  'platstdlib': '/usr/lib/python3.6',
#  'purelib': '/usr/lib/python3.6/site-packages',
#  'scripts': '/usr/bin',
#  'stdlib': '/usr/lib/python3.6'}
