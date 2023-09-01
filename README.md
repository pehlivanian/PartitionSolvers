# PartitionSolvers
## Summary: 
We provide a set of solvers for the program
<img src="https://latex.codecogs.com/svg.image?\mathcal{V}&space;=&space;\{&space;1,&space;\dots,&space;n\}" title="\mathcal{V} = \{ 1, \dots, n\}" />
as described below. The routines and preset objective functions correspond to the Poisson, Gaussian population- and expectation-based Spatial Scan Statistic, see examples below. Features
- Choice of Gaussian, Poisson, or Rational score function
- Further specification of risk partitioning or multiple clustering objective
- OLS-based optimal paritition size determination
- C++, Python interface


## Description 
Utilities to demonstrate use of the consecutive partitions property (CPP) in combinatorial optimization problems, notably those arising in the context of Spatial Scan Statistics. The routines were used to generate tables, figures for 

Charles A. Pehlivanian, Daniel B. Neill, **Efficient Optimization of Partition Scan Statistics via the Consecutive Partitions Property**, preprint, 2021 

Let 
<img src="https://latex.codecogs.com/svg.image?\mathcal{V}&space;=&space;\{&space;1,&space;\dots,&space;n\}" title="\mathcal{V} = \{ 1, \dots, n\}" />
be the ground set, for some integer n and let 
<img src="https://latex.codecogs.com/svg.image?P&space;=&space;\{&space;S_1,&space;\dots,&space;S_t\}" title="P = \{ S_1, \dots, S_t\}" /> 
represent a partition of the ground set into t subsets. The C++ optimizer routines provide exact solutions in <img src="https://latex.codecogs.com/svg.image?\mathcal{O}\(n^2t\)" title="\mathcal{O}\(n^2t\)" /> time to the program
- <img src="https://latex.codecogs.com/svg.image?\max_{\substack{P&space;=&space;\left\lbrace&space;S_1,&space;\dots,&space;S_t\right\rbrace}}&space;{\sum_{j=1\ldots&space;t}f\left(&space;\sum_{i&space;\in&space;S_j}x_i,&space;\sum_{i&space;\in&space;S_j}y_i\right)}" title="P^{*}=\max_{\substack{P = \left\lbrace S_1, \dots, S_t\right\rbrace}} {\sum_{j=1\ldots t}f\left( \sum_{i \in S_j}x_i, \sum_{i \in S_j}y_i\right)}" />
for f satsifying certain regularity conditions. Note that the cardinality of the set of partitions of the ground set is a Stirling number of the second kind, which grows super-exponentially. The underlying optimizer engine is callable from Python via SWIG bindings provided, and asynchronous, distributed (using native C++ primitives) versions are also provided. Build instructions follow the theory.


In the spatial scan statistics partition setting, as opposed to the single subset case, the usual population-based and expectation-based approaches generalize to two objectives and two optimal problems: risk partitioning (optimal allocation of risk across subsets) and multiple clustering (optimal identification of highest scoring cluster configuration). Closed form objective functions can be computed for distributions belonging to a separable exponential family [](https://en.wikipedia.org/wiki/Exponential_family). For the risk partitioning problem, the objective is naturally expressed as an F-divergence, while in the multiple clustering case, it is a Bregman divergence. The resulting maximization problem above then admits an exact solution in <img src="https://latex.codecogs.com/svg.image?\mathcal{O}\(n^2t\)" title="\mathcal{O}\(n^2t\)" /> time. Note that a naive maximization over all partitions is infeasible. 

For example, for the NYC census track data multiple clustering problem below, there are 2089 tracts and 4 subsets per partition. The number of partitions of size 4 of a 2089 element set is a Stirling number of the second kind with more than 1256 digits:
 
```
In [1]: stir = Stirling_n_k(2089,4); stir
Out[1]: 21043144740980019129658372783222683906523792789112246030710636863335348411722359823811415381998892048609981119
4806302904295743586839744228213476513291173650425796293122933013115578706858760153919526875686176073853325449643320414
6485943073481676273653919898088413326076484396123683532414465887947673797299888087125592627468127962892632120837013146
9636059172353676266313766815736338760473655334058910911028543267684130334051858429546288247855676478633764359433913923
1619452837242688224628405816869812983358720423012434336108890489916370884851898274970453163490731427818364695134532080
6245838194317066974075381488333832294532940244886508862265360015566688997095259960001800654556788337330873910629352807
2125617036874802388199632399609689405909013161379001390118375786822877717585018603259290754646952155705256209769425479
1444966037038414255524677782230302321395494190002719126585317041532172643951244591923584906131585166857899297105226899
8804741451890294233532356703511684243059001758626138676848676504840094705767068595215748310978123161404332670058670974
2878132965472033883407672593053343239237195141728649457987778400474832159742398956191196062643245381075109904706119188
0075429075509288244465976655030524209008010059508898601427863013165223817872754402970

In [2]: math.log(stir,10)                                                                                                              
Out[2]: 1256.3231106424016

```

We display an exact solution over this space obtained in ~ 100 millis on a single Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz in the heat maps below.

### Prostate cancer incidence for NYC census tract data:
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/NYC_prostate_5_risk_part_Blues.jpg?raw=true)

### COVID confirmed cases, Minnesota, a/o 09-01-2020:
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Minnesota_09-01-2020_4_best_0_thresh.jpg?raw=true)

### COVID confirmed cases, Japan, a/o 09-01-2020:
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Japan_09-01-2020_3_best_0_thresh.jpg?raw=true)

Runtimes on Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz by varying n, T are shown as well as a least-squares 3-parameter power-rule fit with p(x) = a + b*x^c We expect c ~ 2.0 when T is held fixed, and c ~ 1.0 for fixed n.

### Runtimes for varying n
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Runtimes_by_n.jpg?raw=true)
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Runtimes_with_power_fit_by_n.jpg?raw=true)

### Runtimes for varying T
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Runtimes_by_T.jpg?raw=true)
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Runtimes_with_power_fit_by_T.jpg?raw=true)

NYC tract data, JHU CSSE Covid data plots can be generated by:
```
$ python render_NYC_tract_cancer.py 5 prostate -dist 1 -obj False
$ python render_region_COVID.py Minnesota 4 -d 09-01-2020 12-01-2020 -no-C
$ python render_region_COVID.py Japan 3 -d 09-01-2020 12-01-2020
```

CPU time in the document plots can be generated by:
```
$ ./src/scripts/runTimeGraphs.sh 5000 100 10 10
```

## C++ api

### Build requirements:
- cmake
- swig
- eigen >= 3.0
- google mock, test [optional]

### C++ cmake build options: {CMAKE_BUILD_TYPE, GTEST, SWIG_BINDINGS, USE_C++17}
### Compile to ./build with tests, SWIG bindings as in 
```
$ cmake -H. -Bbuild -DCMAKE_BUILD_TYPE=Release -DGTEST=ON -DSWIG_BINDINGS=ON
$ cmake --build build -- -j4
```

Demos, unittests are straightforward.

Google test suite for C++ engine run via
```
$ ./build/tests/gtest_all
```

Python unittests run via
```
$ python solver_tests.py
```


## Python api

Compile binarys with -DSWIG_BINDINGS=ON or from command line:

### Stand alone compilation of SWIG bindings
```

$ swig -c++ -python -I./include -outdir ../python proto.i
$ clang++ -std=c++17 -c -fPIC -mavx2 -O3 -I/usr/local/include/eigen3 LTSS.cpp \
  python_dpsolver.cpp DP.cpp python_ltsssolver.cpp proto_wrap.cxx -I/usr/include/python3.8 -I./include
$ clang++ -std=c++17 -O3 -shared python_dpsolver.o DP.o python_ltsssolver.o \ 
  LTSS.o proto_wrap.o -o ../python/_proto.so -lstdc++
```
#### Please replace the /usr/include/python3.8 path with the location of Python.h, as in
```
In [1]: from sysconfig import get_paths                                                                                  
In [2]: from pprint import pprint                                                                                        
In [3]: pprint(get_paths())                                                                                                                
{'data': '/usr',

 'include': '/usr/include/python3.8',

 'platinclude': '/usr/include/python3.8', 
 'platlib': '/usr/lib/python3.8/site-packages',
 'platstdlib': '/usr/lib/python3.8',
 'purelib': '/usr/lib/python3.8/site-packages',
 'scripts': '/usr/bin',
 'stdlib': '/usr/lib/python3.8'}
```

#### Test of SWIG bindings
```
$ python ./src/python/solver_tests.py
```

#### Python examples
```
$ python ./src/python/solver_ex.py
```

#### C++ excutables demonstrating solver API
```
./build/examples
|--DP_solver_demo   // Partition solver demo
|--LTSS_solver_demo // Single subset (LTSS) solver demo
|--solver_timer     // Timing utility; wrapped by runTimeGraphs.sh
```

