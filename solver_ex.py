import numpy as np
import solverSWIG_DP
import solverSWIG_LTSS
import proto

rng = np.random.RandomState(136)

num_partitions = 2
n = 55
objective_fn = 1                    # 1 ~ Poisson, 2 ~ Gaussian
risk_partitioning_objective = False # False => multiple clustering score function is used
optimized_score_calculation = False # Leave this False; only implemented for RationalScore case
gamma = 1.0                         # coefficient for regularization term: score -= gamma * T^reg_power
reg_power = 1                       # power for regularization term: score -= gamma * T^reg_power

a_lower_limit = 0. if objective_fn == 1 else -10.; a_higher_limit = 10.
b_lower_limit = 0.; b_higher_limit = 10.
a = rng.uniform(low=a_lower_limit, high=a_higher_limit, size=n)
b = rng.uniform(low=b_lower_limit, high=b_higher_limit, size=n)


# all_results[0] ~ size n partition
# all_results[1] ~ cumulative score
all_results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                          a,
                                          b,
                                          objective_fn,
                                          risk_partitioning_objective,
                                          optimized_score_calculation)()
all_results_pen = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                              a,
                                              b,
                                              objective_fn,
                                              risk_partitioning_objective,
                                              optimized_score_calculation,
                                              gamma,
                                              reg_power)()
best_result_sweep = solverSWIG_DP.OptimizerSWIG(len(a)-10,
                                                a,
                                                b,
                                                objective_fn,
                                                risk_partitioning_objective,
                                                optimized_score_calculation,
                                                True)()


# single_result[0] ~ single best subset
# single_result[1] ~ score for best subset
single_result = solverSWIG_LTSS.OptimizerSWIG(a,
                                              b,
                                              objective_fn)()

print("OPTIMAL PARTITION")
print("=================")
print('{!r}'.format(all_results[0]))
print('SCORE: {}'.format(all_results[1]))
print("\nOPTIMAL PARTITION (WITH PENALTY: gamma: {}, reg_power: {})".format(gamma, reg_power))
print("=================")
print('{!r}'.format(all_results_pen[0]))
print('SCORE: {}'.format(all_results_pen[1]))
print("\nSINGLE BEST SUBSET")
print("==================")
print('{!r}'.format(single_result[0]))
print('SCORE: {}'.format(single_result[1]))
print("\nBEST RESULT SWEEP (Best partition of size [1, ..., {}]".format(len(a)-10))
print("=================")
print('{!r}'.format(best_result_sweep[0]))
print('SCORE: {}'.format(best_result_sweep[1]))
print('PARTITION SIZE: {}'.format(len(best_result_sweep[0])))

