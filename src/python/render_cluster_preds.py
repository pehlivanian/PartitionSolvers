import multiprocessing
from functools import partial
from itertools import islice, chain
from enum import Enum
import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
from sklearn import linear_model
from sklearn.model_selection import KFold
import matplotlib.pyplot as plot
import pickle
import proto

SEED = 0xC0FFEE5
rng = np.random.RandomState(SEED)

MU = 100.
SIGMA = 10.
POISSON_INTENSITY = 100.

NUM_WORKERS = multiprocessing.cpu_count() - 1

class Distribution(Enum):
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

colorMap = ['m', 'r', 'g']

class EpsilonTask(object):
    def __init__(self,
                 num_true_clusters,
                 max_num_clusters,
                 num_trials,
                 num_deviates,
                 objective_fn,
                 k,
                 epsilon_slice,
                 risk_partitioning_objective=True):
        self.num_true_clusters = num_true_clusters
        self.max_num_clusters = max_num_clusters
        self.num_trials = num_trials
        self.num_deviates = num_deviates
        self.objective_fn = objective_fn
        self.k = k
        self.epsilon_slice = epsilon_slice
        self.risk_partitioning_objective = risk_partitioning_objective
        self.task = self._task

        self.mu = MU
        self.sigma = SIGMA
        self.intensity = POISSON_INTENSITY

    def __call__(self):
        return self.task()

    def _form_splits_levels(self, epsilon):
        split = int(self.num_deviates/self.num_true_clusters)
        resid = self.num_deviates - (split * self.num_true_clusters)
        resids = ([1] * int(resid)) + ([0] * (self.num_true_clusters - int(resid)))
        splits = [split + r for r in resids]
        # levels = np.linspace(5. - 5.*epsilon,
        #                      5. + 5.*epsilon,
        #                      self.num_true_clusters)
        levels = np.linspace(1. - 1.*epsilon,
                             1. + 1.*epsilon,
                             self.num_true_clusters)
        # levels = np.linspace(max(0.,1-float(self.num_true_clusters/2)*epsilon),
        #                      min(2., 1+float(self.num_true_clusters/2)*epsilon),
        #                      self.num_true_clusters)
        return splits, levels

        
    def _task(self):        
        xaxis, yaxis = list(), list()
        std_error = dict()
        std_error = list()    

        for epsilon in self.epsilon_slice:
            
            splits, levels = self._form_splits_levels(epsilon)

            q = np.concatenate([np.full(s,l) for s,l in zip(splits,levels)])

            print('epsilon: {} levels: {}'.format(epsilon, levels))

            cum_best_result_k = 0
            all_estimates = list()
        
            for num_trial in range(self.num_trials):
        
                if objective_fn == Distribution.GAUSSIAN:
                    b = rng.normal(self.mu,self.sigma,size=self.num_deviates)
                    a = rng.normal(q*b,self.sigma)
                elif objective_fn == Distribution.POISSON:
                    b = rng.poisson(self.intensity,size=self.num_deviates).astype(float)
                    a = rng.poisson(q*b).astype(float)
                elif objective_fn == Distribution.RATIONALSCORE:
                    b = rng.normal(self.mu,self.sigma,size=self.num_deviates)
                    a = rng.normal(q*b,self.sigma)

                kf = KFold(n_splits=self.k)
                for train_index, _ in kf.split(a):
                    best_result_OLS_sweep = solverSWIG_DP.OptimizerSWIG(self.max_num_clusters,
                                                                        a[train_index],
                                                                        b[train_index],
                                                                        self.objective_fn.value,
                                                                        self.risk_partitioning_objective,
                                                                        True,
                                                                        sweep_best=True)
                    best_result_k = best_result_OLS_sweep()
                    cum_best_result_k += best_result_k[1]
                    all_estimates.append(best_result_k[1])

            best_result_OLS_sweep_r = float(cum_best_result_k)/float(self.k*self.num_trials)

            xaxis.append(epsilon)
            yaxis.append(best_result_OLS_sweep_r)
            std_error.append(2*np.sqrt(np.var(all_estimates)))
        
            print('Optimal t: {} Theoretical t: {}\n'.format(best_result_OLS_sweep_r,  self.num_true_clusters))      
            print('CASE epsilon {} done'.format(epsilon))

        return self.epsilon_slice, xaxis, yaxis, std_error

class EndTask(object):
    pass


class Worker(multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        proc_name = self.name
        while True:
            task = self.task_queue.get()
            if isinstance(task, EndTask):
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

class Baselines(object):
    def __init__(self,
                 num_true_clusters,
                 max_num_clusters,
                 num_trials,
                 num_deviates,
                 objective_fn,
                 k,
                 epsilon_min,
                 epsilon_max,
                 epsilon_delta,
                 risk_partitioning_objective=True
                 ):

        assert epsilon_min >= 0.
        assert epsilon_max <= 1.
        
        self.num_true_clusters = num_true_clusters
        self.max_num_clusters = max_num_clusters
        self.num_trials = num_trials
        self.num_deviates = num_deviates
        self.objective_fn = objective_fn
        self.k = k
        self.epsilons = np.arange(epsilon_min, epsilon_max, epsilon_delta)
        self.risk_partitioning_objective = risk_partitioning_objective

        self.full_range = range(num_trials)
        self.slices = Baselines._form_slices(self.epsilons)
        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()
        self.workers = [Worker(self.tasks, self.results) for _ in range(NUM_WORKERS)]

    @staticmethod
    def _form_slices(rang):
        rang = list(rang)
        num = len(rang)
        bin_ends = list(range(0,num,int(num/NUM_WORKERS)))
        bin_ends = bin_ends + [num] if num/NUM_WORKERS else bin_ends
        islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))
        slices = [list(islice(rang, *ind)) for ind in islice_on]
        return slices

    def create_cluster_preds(self):
        for worker in self.workers:
            worker.start()

        for i,slc in enumerate(self.slices):
            self.tasks.put(EpsilonTask(self.num_true_clusters,
                                       self.max_num_clusters,
                                       self.num_trials,
                                       self.num_deviates,
                                       self.objective_fn,
                                       self.k,
                                       slc,
                                       self.risk_partitioning_objective))

        for _ in self.workers:
            self.tasks.put(EndTask())

        self.tasks.join()

        left_on_queue = len(self.slices)
        xaxis = list(); yaxis = list(); std_error = list(); epsilon = list()
        while not self.results.empty():
            result = self.results.get(block=True)
            epsilon.append(result[0]); xaxis.append(result[1]); yaxis.append(result[2]); std_error.append(result[3])
            left_on_queue -= 1

        xaxis = np.array(list(chain.from_iterable(xaxis)))
        yaxis = np.array(list(chain.from_iterable(yaxis)))
        std_error = np.array(list(chain.from_iterable(std_error)))
        ind = np.argsort(xaxis)
        xaxis = xaxis[ind]; yaxis = yaxis[ind]; std_error=std_error[ind]
        
        assert left_on_queue == 0
        return xaxis, yaxis, std_error

class Plotter(object):
    colorMap = ['m', 'r', 'g']
    win_len = 5
        
    def moving_average(a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    
    @staticmethod
    def mainPlot(xaxis, yaxis, std_error, num_true_clusters, risk_partitioning_objective):
        yaxis_smoothed = ([np.nan]*(Plotter.win_len-1)) + list(Plotter.moving_average(yaxis,Plotter.win_len))
        std_error_smoothed = ([np.nan]*(Plotter.win_len-1)) + list(Plotter.moving_average(std_error,Plotter.win_len))

        plot.plot(xaxis, yaxis, Plotter.colorMap[1])

        # Remove for plots of raw std error bars
        # plot.plot(xaxis, [x-y for x,y in zip(yaxis,std_error)], Plotter.colorMap[1], linewidth=1)
        # plot.plot(xaxis, [x+y for x,y in zip(yaxis,std_error)], Plotter.colorMap[1], linewidth=1)
        plot.plot(xaxis, [x-y for x,y in zip(yaxis_smoothed,std_error_smoothed)], Plotter.colorMap[0], linewidth=1)
        plot.plot(xaxis, yaxis_smoothed, Plotter.colorMap[0], label='smoothed predicted', linewidth=3)        
        plot.plot(xaxis, [x+y for x,y in zip(yaxis_smoothed,std_error_smoothed)], Plotter.colorMap[0], linewidth=1)

        plot.plot(xaxis, [num_true_clusters]*len(yaxis), '-.', linewidth=2, label='theoretical')
        plot.title('Cluster detection by signal strength ({} clusters)'.format(num_true_clusters))
        plot.legend()
        plot.xlabel('signal strength (epsilon)')
        plot.ylabel('number of clusters')
        plot.grid(True)
        plot.pause(1e-2)
        suff = 'rp' if risk_partitioning_objective else 'mc'
        plot.savefig('Cluster_detection_{}_{}_clusters.pdf'.format(suff, num_true_clusters))
        plot.close()

if __name__ == '__main__':
    max_num_clusters = 15
    num_trials = 10
    num_deviates = 2500
    objective_fn = Distribution.POISSON
    k = 20
    epsilon_min = 0.
    epsilon_max = 1.00
    epsilon_delta = .01
    risk_partitioning_objective = True

    for num_true_clusters in (2,3,5,7):
        b = Baselines(num_true_clusters,
                      max_num_clusters,
                      num_trials,
                      num_deviates,
                      objective_fn,
                      k,
                      epsilon_min,
                      epsilon_max,
                      epsilon_delta,
                      risk_partitioning_objective
                  )
        xaxis, yaxis, std_error = b.create_cluster_preds()
        Plotter.mainPlot(xaxis, yaxis, std_error, num_true_clusters, risk_partitioning_objective)

        data = {'objective_fn': objective_fn,
                'xaxis': xaxis,
                'yaxis': yaxis,
                'std_error': std_error,
                }
        suff = 'rp' if risk_partitioning_objective else 'mc'
        with open('./cluster_pred_data/cd_{}_{}_{}_{}_{}_{}.pkl'.format(suff,
                                                                        num_trials,
                                                                        num_true_clusters,
                                                                        num_deviates,
                                                                        k,
                                                                        epsilon_delta), 'wb') as fh:
            d = pickle.dump(data,fh,protocol=pickle.HIGHEST_PROTOCOL)
    
    

