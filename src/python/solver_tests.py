import unittest
import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
from sklearn import linear_model
import proto

SEED = 0xC0FFEE3
rng = np.random.RandomState(SEED)

class Distribution:
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

class TestSolver(unittest.TestCase):

    def setUp(self):
        pass

    def test_solver_consistency_Gaussian(self):
        self.solver_consistency_helper(objective_fn=Distribution.GAUSSIAN)
    def test_solver_consistency_Poisson(self):
        self.solver_consistency_helper(objective_fn=Distribution.POISSON)        
    def test_solver_consistency_RationalScore(self):
        self.solver_consistency_helper(objective_fn=Distribution.RATIONALSCORE)        
        
    def solver_consistency_helper(self, objective_fn=0):

        num_partitions = 2
        n = 25
        gamma = 1.0
        reg_power = 1
        
        # generate uniform deviates
        a_lower_limit = 0. if objective_fn == 1 else -10.; a_higher_limit = 10.
        b_lower_limit = 0.; b_higher_limit = 10.
        a = rng.uniform(low=a_lower_limit, high=a_higher_limit, size=n)
        b = rng.uniform(low=b_lower_limit, high=b_higher_limit, size=n)

        # all_results[0] ~ size n partition
        # all_results[1] ~ cumulative score
        # rp ~ risk partition, mc ~ multiple cluster (problem setting)
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


        # Assert both unpenalized, penalized mc return same subsets
        for i,_ in enumerate(all_results_mc[0]):
            self.assertTupleEqual(all_results_mc[0][i], all_results_mc_pen_r[0][i])

        # Assert multiple cluster unpenalized score greater than
        # penalized score
        self.assertGreater(all_results_mc[1], all_results_mc_pen_r[1])

        # Assert single solver returns corresponding subset of mc solver
        self.assertTupleEqual(single_result_r[0], all_results_mc[0][-1])

    def test_cluster_detection_Gaussian(self):
        self.cluster_detection_helper(objective_fn=Distribution.GAUSSIAN)
    def test_cluster_detection_RationalScore(self):
        self.cluster_detection_helper(objective_fn=Distribution.RATIONALSCORE)
        
    def cluster_detection_helper(self, objective_fn=0):
        n = 1000
        max_num_partitions = 10
        poisson_intensity = 100.
        mu = 100.
        sigma = 5.
        epsilon = 0.45

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

        def tabular_resids(all_results_r):
            df = pd.DataFrame({'rp'+str(len(r[0])):[r[1]] for r in all_results_r})
            df = df.drop(columns=['rp0'])
            ddf = df.diff(axis=1)
            ddf = ddf.drop(columns=['rp1'])
            ddf['rp2'] = df['rp2']
            return ddf

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

            # Assert C++ bindings optimal OLS T matches theoretical
            self.assertEqual(best_result_OLS_sweep_r[1], num_true_clusters)

            # Assert C++ bindings optimal OLS T matches offline fit
            ddf = tabular_resids(all_results_r)            
            offline_fit = fit(ddf.iloc[0,:].values)
            self.assertEqual(offline_fit, num_true_clusters)

    def test_optimal_T_Gaussian(self):
        self.optimal_T_helper(objective_fn=Distribution.GAUSSIAN)
    def test_optimal_T_RationalScore(self):
        self.optimal_T_helper(objective_fn=Distribution.RATIONALSCORE)
    
    def optimal_T_helper(self, objective_fn=0):
        n = 2000
        
        # a grouped in 4 clusters
        a = rng.choice([-20., -10.,10., 20.],size=n)
        a += rng.normal(0., 1., size=n)
        b = np.asarray([1.]*n)
        
        for max_num_subsets in range(50,5,-1):
            best_result_OLS_sweep = solverSWIG_DP.OptimizerSWIG(max_num_subsets,
                                                                a,
                                                                b,
                                                                objective_fn,
                                                                True,
                                                                True,
                                                                sweep_best=True)
            
            best_result_OLS_sweep_r = best_result_OLS_sweep()

            # Assert optimal T is 4
            self.assertEqual(best_result_OLS_sweep_r[1], 4)
        
        
if __name__ == '__main__':
    unittest.main()
