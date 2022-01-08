# PartitionSolvers
Utilities to demonstrate use of the consecutive partitions property (CPP) in combinatorial optimization problems, notably those arising in the context of Spatial Scan Statistics. These were used to generate tables, figures for Charles A. Pehlivanian, Daniel B. Neill, **Efficient Optimization of Partition Scan Statistics via the Consecutive Partitions Property**, preprint, 2021 

In particular, let 
<img src="https://latex.codecogs.com/svg.image?\mathcal{V}&space;=&space;\{&space;1,&space;\dots,&space;n\}" title="\mathcal{V} = \{ 1, \dots, n\}" />
be the ground set, for some integer n and let 
<img src="https://latex.codecogs.com/svg.image?P&space;=&space;\{&space;S_1,&space;\dots,&space;S_t\}" title="P = \{ S_1, \dots, S_t\}" /> 
represent a partition of the ground set. We provide exact solutions in <img src="https://latex.codecogs.com/svg.image?\mathcal{O}\(n^2t\)" title="\mathcal{O}\(n^2t\)" /> time to the program
- <img src="https://latex.codecogs.com/svg.image?\max_{\substack{P&space;=&space;\left\lbrace&space;S_1,&space;\dots,&space;S_t\right\rbrace}}&space;{\sum_{j=1\ldots&space;t}f\left(&space;\sum_{i&space;\in&space;S_j}x_i,&space;\sum_{i&space;\in&space;S_j}y_i\right)}" title="P^{*}=\max_{\substack{P = \left\lbrace S_1, \dots, S_t\right\rbrace}} {\sum_{j=1\ldots t}f\left( \sum_{i \in S_j}x_i, \sum_{i \in S_j}y_i\right)}" />
for f satsifying certain regularity conditions. Note that the cardinality of the set of partitions of the ground set is a Stirling number of the second kind, which grows super-exponentially. 


In the spatial scan statistics partition setting, as opposed to the single subset case, the usual population-based and expectation-based approaches generalize to two objectives and two optimal problems: risk partitioning (optimal allocation of risk across subsets) and multiple clustering (optimal identification of highest scoring cluster configuration). Closed form objective functions can be computed for distributions belonging to a separable exponential family [](https://en.wikipedia.org/wiki/Exponential_family). For the risk partitioning problem, the objective is naturally expressed as an F-divergence, while in the multiple clustering case, it is a Bregman divergence. The resulting maximization problem above then admits an exact solution in <img src="https://latex.codecogs.com/svg.image?\mathcal{O}\(n^2t\)" title="\mathcal{O}\(n^2t\)" /> time. Note that a naive maximization over all partitions is infeasible. For the NYC census track data below, there are 2089 tracts and 4 subsets per partition. The number of partitions of size 4 of a 2089 element set is a Stirling number of the second kind:
 
```
In [1]: stir = Stirling_n_k(2089,4); stir
Out[1]: 210431447409800191296583727832226839065237927891122460307106368633353484117223598238114153819988920486099811194806302904295743586839744228213476513291173650425796293122933013115578706858760153919526875686176073853325449643320414648594307348167627365391989808841332607648439612368353241446588794767379729988808712559262746812796289263212083701314696360591723536762663137668157363387604736553340589109110285432676841303340518584295462882478556764786337643594339139231619452837242688224628405816869812983358720423012434336108890489916370884851898274970453163490731427818364695134532080624583819431706697407538148833383229453294024488650886226536001556668899709525996000180065455678833733087391062935280721256170368748023881996323996096894059090131613790013901183757868228777175850186032592907546469521557052562097694254791444966037038414255524677782230302321395494190002719126585317041532172643951244591923584906131585166857899297105226899880474145189029423353235670351168424305900175862613867684867650484009470576706859521574831097812316140433267005867097428781329654720338834076725930533432392371951417286494579877784004748321597423989561911960626432453810751099047061191880075429075509288244465976655030524209008010059508898601427863013165223817872754402970

In [2]: math.log(stir,10)                                                                                                              
Out[2]: 1256.3231106424016

```

We display an exact solution below.

Prostate cancer incidence for NYC census tract data:
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/NYC_prostate_3_risk_part_Blues.jpg?raw=true)
COVID confirmed cases, Minnesota, a/o 09-01-2020
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Minnesota_09-01-2020_4_best_0_thresh.jpg?raw=true)
COVID confirmed cases, Japan, a/o 09-01-2020
![plot](https://github.com/pehlivanian/PartitionSolvers/blob/main/figures/Japan_09-01-2020_3_best_0_thresh.jpg?raw=true)


### Requirements:
- cmake
- swig
- google mock, test [optional]

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

