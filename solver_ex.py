import numpy as np
import solverSWIG_DP
import solverSWIG_LTSS
import proto

SEED = 0xC0FFEE
rng = np.random.RandomState(SEED)

num_partitions = 2
n = 10
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
                                          optimized_score_calculation)
all_results_pen = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                              a,
                                              b,
                                              objective_fn,
                                              risk_partitioning_objective,
                                              optimized_score_calculation,
                                              gamma,
                                              reg_power)
all_results_sweep = solverSWIG_DP.OptimizerSWIG(len(a) - 5,
                                                a,
                                                b,
                                                objective_fn,
                                                risk_partitioning_objective,
                                                optimized_score_calculation,
                                                gamma=0.,
                                                reg_power=1.,
                                                sweep_all=True)
best_result_OLS_sweep = solverSWIG_DP.OptimizerSWIG(len(a) - 5,
                                                a,
                                                b,
                                                objective_fn,
                                                risk_partitioning_objective,
                                                optimized_score_calculation,
                                                gamma=0.,
                                                reg_power=1.,
                                                sweep_best=True)

# single_result[0] ~ single best subset
# single_result[1] ~ score for best subset
single_result = solverSWIG_LTSS.OptimizerSWIG(a,
                                              b,
                                              objective_fn)

# optimize
all_results_r = all_results()
all_results_pen_r = all_results_pen()
all_results_sweep_r = all_results_sweep()
best_result_OLS_sweep_r = best_result_OLS_sweep()
single_result_r = single_result()

print("OPTIMAL PARTITION (n = {}, t = {})".format(n, num_partitions))
print("=================")
print('{!r}'.format(all_results_r[0]))
print('SCORE: {}'.format(all_results_r[1]))
print("\nOPTIMAL PARTITION (n = {}, t = {} WITH PENALTY: gamma: {}, reg_power: {})".format(n, num_partitions, gamma, reg_power))
print("===================")
print('{!r}'.format(all_results_pen_r[0]))
print('SCORE: {}'.format(all_results_pen_r[1]))
print("\nSINGLE BEST SUBSET")
print("====================")
print('{!r}'.format(single_result_r[0]))
print('SCORE: {}'.format(single_result_r[1]))
