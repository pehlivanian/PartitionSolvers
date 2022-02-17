from enum import Enum
import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plot
import proto

SEED = 0xC0FFEE3
rng = np.random.RandomState(SEED)

class Distribution(Enum):
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

colorMap = ['m', 'r', 'g']

num_trials = 1
num_true_clusters = 7
max_num_clusters = 10
num_deviates = 1000
objective_fns = [Distribution.POISSON, Distribution.GAUSSIAN]
mu = 100.
sigma = 10.
intensity = 100.
k = 20
epsilon_min = .01
epsilon_max = .99
epsilon_delta = .001

epsilons = np.arange(epsilon_min, epsilon_max, epsilon_delta)
num_pred_clusters = list()

r = dict()
    
for objective_fn in objective_fns:
    xaxis, yaxis = list(), list()
    
    for epsilon in epsilons:
        split = int(num_deviates/num_true_clusters)
        resid = num_deviates - (split * num_true_clusters)
        resids = ([1] * int(resid)) + ([0] * (num_true_clusters - int(resid)))
        splits = [split + r for r in resids]
        levels = np.linspace(1-int(num_true_clusters/2)*epsilon,
                             1+int(num_true_clusters/2)*epsilon,
                             num_true_clusters)
        if 0 == num_true_clusters%2:
            levels = levels[levels!=0]

        if (levels[0] < 0):
            break
    
        q = np.concatenate([np.full(s,l) for s,l in zip(splits,levels)])
        
        # print('Multiplier levels: {!r}'.format(levels))
        
        cum_best_result_k = 0
    
        for num_trial in range(num_trials):
        
            if objective_fn == Distribution.GAUSSIAN:
                b = rng.normal(mu,sigma,size=num_deviates)
                a = rng.normal(q*b,sigma)
            elif objective_fn == Distribution.POISSON:
                b = rng.poisson(intensity,size=num_deviates).astype(float)
                a = rng.poisson(q*b).astype(float)
            elif objective_fn == Distribution.RATIONALSCORE:
                b = rng.normal(mu,sigma,size=num_deviates)
                a = rng.normal(q*b,sigma)
            
            kf = KFold(n_splits=k)
            for train_index, _ in kf.split(a):
                best_result_OLS_sweep = solverSWIG_DP.OptimizerSWIG(max_num_clusters,
                                                                    a[train_index],
                                                                    b[train_index],
                                                                    objective_fn.value,
                                                                    True,
                                                                    True,
                                                                sweep_best=True)
                best_result_k = best_result_OLS_sweep()
                cum_best_result_k += best_result_k

        best_result_OLS_sweep_r = float(cum_best_result_k)/float(k*num_trials)
        num_pred_clusters.append(best_result_OLS_sweep_r)

        xaxis.append(epsilon)
        yaxis.append(best_result_OLS_sweep_r)

        print('Optimal t: {} Theoretical t: {}\n'.format(best_result_OLS_sweep_r,  num_true_clusters))      
        print('CASE epsilon {} done'.format(epsilon))

    plot.plot(xaxis, yaxis)

    win_len = 25
    yaxis_smoothed = ([np.nan]*(win_len-1)) + list(moving_average(yaxis,win_len))
    r[objective_fn] = yaxis_smoothed

# Take care of smoothed series outside of loop
for k,v in r.items():
    plot.plot(xaxis, v, colorMap[k.value], label='predicted smoothed {}'.format(k.name.capitalize()), linewidth=3)

plot.plot(xaxis, [num_true_clusters]*len(yaxis), '-.', linewidth=2)
plot.title('Cluster detection by signal strength ({} clusters)'.format(num_true_clusters))
plot.legend()
plot.grid(True)
plot.pause(1e-2)
plot.savefig('Cluster_detection_{}_clusters.pdf'.format(num_true_clusters))
    

