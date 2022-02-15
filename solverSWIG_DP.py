import multiprocessing
import proto

class Distribution:
    GAUSSIAN = 0
    POISSON = 1
    RATIONALSCORE = 2

class OptimizerSWIG(object):
    ''' Task-based C++ optimizer.
    '''
    def __init__(self,
                 num_partitions,
                 g,
                 h,
                 objective_fn=Distribution.GAUSSIAN,
                 risk_partitioning_objective=False, # So multiple clustering
                 use_rational_optimization=False,
                 gamma=0.,
                 reg_power=1.,
                 parallel_sweep=False, # find best score among all s <= num_partitions, C++ threadpool
                 sweep_all=False,      # return all partitions, scores for s <= num_partitions
                 sweep_best=False):    # find best partition, s <= num_partitions, according to OLS criterion
        self.N = len(g)
        self.num_partitions = num_partitions
        self.objective_fn = objective_fn
        self.risk_partitioning_objective = risk_partitioning_objective
        self.use_rational_optimization = use_rational_optimization
        self.gamma = gamma
        self.reg_power = reg_power
        self.g_c = proto.FArray()
        self.h_c = proto.FArray()        
        self.g_c = g
        self.h_c = h

        self.parallel_sweep = parallel_sweep
        self.sweep_all = sweep_all
        self.sweep_best = sweep_best

        assert not (self.parallel_sweep and self.sweep_all)
        assert not (self.sweep_all and self.sweep_best)

    def compute_score(self):
        return proto.compute_score(self.g_c,
                                   self.h_c,
                                   self.objective_fn,
                                   self.risk_partitioning_objective,
                                   self.use_rational_optimization)

    def compute_partial_score(self, i, j):
        return proto.compute_score(self.g_c[i:j],
                                   self.h_c[i:j],
                                   self.objective_fn,
                                   self.risk_partitioning_objective,
                                   self.use_rational_optimization)
        
    def __call__(self):
        if self.parallel_sweep:
            return proto.sweep_parallel__DP(self.N,
                                            self.num_partitions,
                                            self.g_c,
                                            self.h_c,
                                            self.objective_fn,
                                            self.risk_partitioning_objective,
                                            self.use_rational_optimization,
                                            self.gamma,
                                            int(self.reg_power))
        elif self.sweep_best:
            return proto.compute_optimal_num_clusters_OLS(self.N,
                                                          self.num_partitions,
                                                          self.g_c,
                                                          self.h_c,
                                                          self.objective_fn,
                                                          self.risk_partitioning_objective,
                                                          self.use_rational_optimization,
                                                          self.gamma,
                                                          int(self.reg_power))
        elif self.sweep_all:
            return proto.optimize_all__DP(self.N,
                                          self.num_partitions,
                                          self.g_c,
                                          self.h_c,
                                          self.objective_fn,
                                          self.risk_partitioning_objective,
                                          self.use_rational_optimization,
                                          self.gamma,
                                          int(self.reg_power))
        else:
            return proto.optimize_one__DP(self.N,
                                          self.num_partitions,
                                          self.g_c,
                                          self.h_c,
                                          self.objective_fn,
                                          self.risk_partitioning_objective,
                                          self.use_rational_optimization,
                                          self.gamma,
                                          int(self.reg_power))

class EndTask(object):
    pass

class OptimizerTask(object):
    def __init__(self,
                 N,
                 num_partitions,
                 g,
                 h,
                 objective_fn,
                 risk_partitioning_objective,
                 use_rational_optimization,
                 gamma,
                 reg_power):
        g_c = proto.FArray()
        h_c = proto.FArray()        
        g_c = g
        h_c = h
        self.task = partial(self._task,
                            N,
                            num_partitions,
                            g_c,
                            h_c,
                            objective_fn,
                            risk_partitioning_objective,
                            use_rational_optimization,
                            gamma,
                            reg_power)

    def __call__(self):
        return self.task()
        
    @staticmethod
    def _task(N,
              num_partitions,
              g,
              h,
              objective_fn,
              risk_partitioning_objective,
              use_rational_optimization,
              gamma,
              reg_power):
        s, w = proto.optimize_one__DP(N,
                                      num_partitions,
                                      g,
                                      h,
                                      objective_fn,
                                      risk_partitioning_objective,
                                      use_rational_optimization,
                                      gamma,
                                      int(reg_power))
        return s, w

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

    

