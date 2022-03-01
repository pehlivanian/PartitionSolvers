import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
from sklearn import linear_model
import proto

SEED = 0xC0FFEE3
rng = np.random.RandomState(SEED)

num_partitions = 2
n = 25
objective_fn = 1                    # 1 ~ Poisson, 2 ~ Gaussian
gamma = 1.0                         # coefficient for regularization term: score -= gamma * T^reg_power
reg_power = 1                       # power for regularization term: score -= gamma * T^reg_power


###############################################################
# Single subset, single partition cases                       #
###############################################################
# 3 Optimizers:                                               #
# single best partition, fixed partition size, no penalty     #
# single best partition, fixed partition size, with penalty   #
# single best subset LTSS                                     #
###############################################################
# generate uniform deviates
a_lower_limit = 0. if objective_fn == 1 else -10.; a_higher_limit = 10.
b_lower_limit = 0.; b_higher_limit = 10.
a = rng.uniform(low=a_lower_limit, high=a_higher_limit, size=n)
b = rng.uniform(low=b_lower_limit, high=b_higher_limit, size=n)


# all_results[0] ~ size n partition
# all_results[1] ~ cumulative score
all_results_mc = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                             a,
                                             b,
                                             objective_fn,
                                             False,
                                             True)
all_results_rp = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                             a,
                                             b,
                                             objective_fn,
                                             True,
                                             True)
all_results_mc_pen = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                                 a,
                                                 b,
                                                 objective_fn,
                                                 False,
                                                 True,
                                                 gamma,
                                                 reg_power)
# single_result[0] ~ single best subset
# single_result[1] ~ score for best subset
single_result = solverSWIG_LTSS.OptimizerSWIG(a,
                                              b,
                                              objective_fn)

# optimize
all_results_mc = all_results_mc()
all_results_rp = all_results_rp()
all_results_mc_pen_r = all_results_mc_pen()
single_result_r = single_result()

print("OPTIMAL PARTITION (RISK PART) FOR PRESPECIFIED t: (n = {}, t = {})".format(n, num_partitions))
print("=================")
print('{!r}'.format(all_results_rp[0]))
print('SCORE: {}'.format(all_results_rp[1]))
print("OPTIMAL PARTITION (MULT CLUST) FOR PRESPECIFIED t: (n = {}, t = {})".format(n, num_partitions))
print("=================")
print('{!r}'.format(all_results_mc[0]))
print('SCORE: {}'.format(all_results_mc[1]))
print("\nOPTIMAL PARTITION (MULT CLUST) FOR PRESPECIFIED t: (n = {}, t = {} WITH PENALTY: gamma: {}, reg_power: {})".format(n, num_partitions, gamma, reg_power))
print("===================")
print('{!r}'.format(all_results_mc_pen_r[0]))
print('SCORE: {}'.format(all_results_mc_pen_r[1]))
print("\nSINGLE BEST LTSS SUBSET")
print("====================")
print('{!r}'.format(single_result_r[0]))
print('SCORE: {}'.format(single_result_r[1]))


#######################################################################
# Cluster detection                                                   #
#######################################################################   
# 2 Optimizers:                                                       #
# t selection in C++                                                  #
# comprehensive partition, score output for t selection in Python     #
#######################################################################
SEED = 0xC0FFEA
rng = np.random.RandomState(SEED)

class Distribution:
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

n = 1000
max_num_partitions = 10
poisson_intensity = 100.
mu = 100.
sigma = 5.
epsilon = 0.45
objective_fn = Distribution.GAUSSIAN

NUM_TRIALS = 20

def fit(z):
    y = np.log(z)
    X = np.log(range(2,11)).reshape(-1,1)
    clf = linear_model.LinearRegression(fit_intercept=True)
    try:
        clf.fit(X,y)
    except ValueError:
        return np.nan
    residuals = y - clf.predict(X)
    return residuals.argmin() + 1

print('\n\n================================\n')
print('OPTIMAL NUMBER OF CLUSTER TRIALS\n')
print('================================\n')
for num_trial in range(NUM_TRIALS):
    num_true_clusters = rng.choice(range(2,max_num_partitions))

    split = int(n/num_true_clusters)
    resid = n - (split * num_true_clusters)
    resids = ([1] * int(resid)) + ([0] * (num_true_clusters - int(resid)))
    splits = [split + r for r in resids]
    levels = np.linspace(1-int(num_true_clusters/2)*epsilon,
                         1+int(num_true_clusters/2)*epsilon,
                         num_true_clusters)
    if 0 == num_true_clusters%2:
        levels = levels[levels!=0]
    q = np.concatenate([np.full(s,l) for s,l in zip(splits,levels)])
    

    if objective_fn == Distribution.GAUSSIAN:
        b = rng.normal(mu,sigma,size=n)
        a = rng.normal(q*b,sigma)
    elif objective_fn == Distribution.POISSON:
        b = rng.poisson(poisson_intensity,size=n).astype(float)
        a = rng.poisson(q*b).astype(float)
    elif objective_fn == Distribution.RATIONALSCORE:
        b = rng.normal(mu,sigma,size=n)
        a = rng.normal(q*b,sigma)

    all_results_sweep = solverSWIG_DP.OptimizerSWIG(max_num_partitions,
                                                    a,
                                                    b,
                                                    objective_fn,
                                                    True,
                                                    True,
                                                    gamma=0.,
                                                    reg_power=1.,
                                                    sweep_all=True)
    best_result_OLS_sweep = solverSWIG_DP.OptimizerSWIG(max_num_partitions,
                                                        a,
                                                        b,
                                                        objective_fn,
                                                        True,
                                                        True,
                                                        sweep_best=True)

    all_results_r = all_results_sweep()
    best_result_OLS_sweep_r = best_result_OLS_sweep()

    df = pd.DataFrame({'rp'+str(len(r[0])):[r[1]] for r in all_results_r})
    df = df.drop(columns=['rp0'])
    ddf = df.diff(axis=1)
    ddf = ddf.drop(columns=['rp1'])
    ddf['rp2'] = df['rp2']

    print("NATIVE C++ PYTHON OPTIMAL OLS T (RISK PART) VS THEORETICAL T: (n = {}, max t = {})".format(n, max_num_partitions))
    print("=============")
    print('Optimal t: {} Theoretical t: {}\n'.format(best_result_OLS_sweep_r[1], num_true_clusters))    
    print("OFFLINE PYTHON OPTIMAL OLS T (RISK PART) VS THEORETICAL T: (n = {}, max t = {})".format(n, max_num_partitions))
    print("=============")
    print('Optimal t: {} Theoretical t: {}\n'.format(fit(ddf.iloc[0,:].values), num_true_clusters))



#######################################################################
# Cluster detection                                                   #
#######################################################################   
# 2 clusters; $a \in \left{ -1, 1\right}$, $b \in \left{ 1\right}$    #
# t selection in C++; should identify 2 clusters                      #
#######################################################################
n = 2000
objective_fn = Distribution.RATIONALSCORE

# a grouped in 4 clusters
a = rng.choice([-20., -10.,10., 20.],size=n)
a += rng.normal(0., 1., size=n)
b = np.asarray([1.]*n)


print('PRIORITY UNIQUE VALUES: {!r}'.format(np.unique(a/b)))

for max_num_subsets in range(50,1,-1):
    best_result_OLS_sweep = solverSWIG_DP.OptimizerSWIG(max_num_subsets,
                                                        a,
                                                        b,
                                                        objective_fn,
                                                        True,
                                                        True,
                                                        sweep_best=True)
    
    best_result_OLS_sweep_r = best_result_OLS_sweep()
    print('OPTIMAL T: {} MAX_T: {}'.format(best_result_OLS_sweep_r[1], max_num_subsets))

