# PartitionSolvers
Utilities to demonstrate use of the consecutive partitions property (CPP) in combinatorial optimization problems, notably those arising in the context of Spatial Scan Statistics. These were used to generate tables, figures for Charles A. Pehlivanian, Daniel B. Neill, **Efficient Optimization of Partition Scan Statistics via the Consecutive Partitions Property**, preprint, 2021 


- <img src="https://latex.codecogs.com/gif.latex?P=47 />

### Requirements:
- cmake
- swig
- google mock, test

## C++ api

### Compile to ./build as in 
```
$ cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release -DGTEST=ON
$ cmake --build build -- -j4
```

### Unittests:
```
$ ./build/bin/gtest_all
```

Examples of calling conventions are contained in DP_solver_test.

## Python api

### Ok, the swig bindings are now failing to compile from cmake directives, they must be generated from CL as follows:
```
$ swig -c++ -python proto.i
$ g++ -std=c++17 -c -fPIC -O3 LTSS.cpp python_dpsolver.cpp DP.cpp python_ltsssolver.cpp proto_wrap.cxx -I/usr/include/python3.6
$ g++ -std=c++17 -O3 -shared python_dpsolver.o DP.o python_ltsssolver.o LTSS.o proto_wrap.o -o _proto.so -lstdc++
```
#### Please replace the /usr/include/python3.6 directory above with the include directory on your host, as in
```
In [1]: from sysconfig import get_paths                                                                                  
In [2]: from pprint import pprint                                                                                        
In [3]: pprint(get_paths())                                                                                                                
{'data': '/usr',
 'include': '/usr/include/python3.6',			<<- include_path
 'platinclude': '/usr/include/python3.6', 
 'platlib': '/usr/lib/python3.6/site-packages',
 'platstdlib': '/usr/lib/python3.6',
 'purelib': '/usr/lib/python3.6/site-packages',
 'scripts': '/usr/bin',
 'stdlib': '/usr/lib/python3.6'}
```

#### Quick Test
$ python solver_ex.py

Examples of calling conventions are contained in pmlb_driver.py which performs successive fits over pmlb datasets.

