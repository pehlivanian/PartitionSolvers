import numpy as np
import solverSWIG_DP
import solverSWIG_LTSS
import proto

rng = np.random.RandomState(14)

num_partitions = 2
n = 15
a = proto.FArray()                  # wrapper for C++ float array type
b = proto.FArray()                  # wrapper for C++ float array type
objective_fn = 1                    # 1 ~ Poisson, 2 ~ Gaussian, 3 ~ RationalScore
risk_partitioning_objective = False # False => multiple clustering score function is used
optimized_score_calculation = False # Leave this False; only implemented for RationalScore case

a = rng.uniform(low=-10.0, high=10.0, size=n)
b = rng.uniform(low=1.0, high=10.0, size=n)


# all_results[0] ~ size n partition
# all_results[1] ~ cumulative score
all_results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                          a,
                                          b,
                                          objective_fn,
                                          risk_partitioning_objective,
                                          optimized_score_calculation)()
# single_result[0] ~ single best subset
# single_result[1] ~ score for best subset
single_result = solverSWIG_LTSS.OptimizerSWIG(a,
                                              b,
                                              objective_fn)()

print("MULT CLUST PARTITION")
print("====================")
print('{!r}'.format(all_results[0]))
print('SCORE: {}'.format(all_results[1]))
print("SINGLE BEST SUBSET")
print("==================")
print('{!r}'.format(single_result[0]))
print('SCORE: {}'.format(single_result[1]))
