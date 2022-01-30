import multiprocessing
from functools import partial
from itertools import islice
import numpy as np
import pandas as pd
import solverSWIG_DP
import proto

SEED = 1868
POISSON_INTENSITY = 100
DEVIATE_SIZE = 5000
NUM_NULL_EXPERIMENTS = 1000
NUM_THRESHOLD_CALCULATIONS = 1000
NUM_EXPERIMENTS_PER_EPSILON = 1000
NUM_EPSILON_VALUES = 11
NUM_WORKERS = multiprocessing.cpu_count() - 1

assert not DEVIATE_SIZE%2
assert NUM_EPSILON_VALUES == 11

rng = np.random.RandomState(SEED)

def ranking_quality(m):
  rq = 0
  num_rows = m.shape[0]
  num_cols = m.shape[1]
  for i1 in range(num_rows):
    for j1 in range(num_cols):
      for i2 in range(num_rows):
        for j2 in range(num_cols):
          if ((i1 < i2) & (j1 < j2)) | ((i1 > i2) & (j1 > j2)) | ((i1 == i2) & (j1 == j2)):
            rq += m[i1,j1]*m[i2,j2]
  return rq/(np.sum(m)*np.sum(m))

def bootstrap_95th_percentile(null_scores):
  thresholds = np.zeros(NUM_THRESHOLD_CALCULATIONS)
  for i in range(NUM_THRESHOLD_CALCULATIONS):
    thresholds[i] = np.quantile(rng.choice(null_scores, size=DEVIATE_SIZE, replace=True),0.95)
  thresholds = np.sort(thresholds)
  return thresholds

def slice_range(rang):
    rang = list(rang)
    num = len(rang)
    bin_ends = list(range(0,num,int(num/NUM_WORKERS)))
    bin_ends = bin_ends + [num] if num/NUM_WORKERS else bin_ends
    islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))
    slices = [list(islice(rang, *ind)) for ind in islice_on]
    return slices

class NullTask(object):
    def __init__(self, num_null_experiments, num_deviates, poisson_intensity, range):
        self.num_null_experiments = num_null_experiments
        self.poisson_intensity = poisson_intensity
        self.deviate_size = num_deviates
        self.range = range
        self.task = partial(self._task)

    def __call__(self):
        return self.task()

    def _task(self):
        count = 0
        null_scores = np.zeros([len(self.range),5])
        for i in self.range:
            b = rng.poisson(self.poisson_intensity, size=self.deviate_size).astype(float)
            a = rng.poisson(b).astype(float)
            result_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()[1]
            result_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()[1]
            result_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()[1]
            result_mcd2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,False,True)()[1]
            result_mcd3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,False,True)()[1]
            result_mcd10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,False,True)()[1]
            null_scores[count,:] = [result_rp2,result_rp3,result_rp10,result_mcd2,result_mcd3]
            count += 1
        null_scores_df = pd.DataFrame(null_scores,columns=['rp2','rp3','rp10','mcd2','mcd3'])
        return null_scores_df

class QATask(object):
  def __init__(self, num_experiments_per_epsilon, deviate_size, poisson_intensity, num_epsilon_values):
    self.num_experiments_per_epsilon = num_experiments_per_epsilon
    self.deviate_size = deviate_size
    self.poisson_intensity = poisson_intensity
    self.num_epsilon_values = num_epsilon_values
    self.task = partial(self._task)

  def __call__(self):
    return self.task()

  def _task(self):
    exp0_scores = np.zeros([self.num_experiments_per_epsilon*self.num_epsilon_values,4])
    exp0_ranking_quality = np.zeros([self.num_epsilon_values,4])
    for j in range(self.num_epsilon_values):      
      theepsilon = 0.05*j
      ranking_quality_rp2 = 0
      ranking_quality_rp3 = 0
      ranking_quality_rp10 = 0
      for i in range(self.num_experiments_per_epsilon):
        b = rng.poisson(self.poisson_intensity,size=self.deviate_size).astype(float)
        q = np.concatenate([np.full(int(self.deviate_size/2),1.0-theepsilon),
                            np.full(int(self.deviate_size/2),1.0+theepsilon)])
        a = rng.poisson(q*b).astype(float)
        all_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()
        all_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()
        all_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()
        score_rp2 = all_rp2[1]
        score_rp3 = all_rp3[1]
        score_rp10 = all_rp10[1]
        confusion_rp2 = np.zeros([2,2])
        confusion_rp3 = np.zeros([2,3])
        confusion_rp10 = np.zeros([2,10])

        for k in range(2):
            thepartition = np.array(all_rp2[0][k])
            for m in range(2):
                confusion_rp2[m,k] = len(thepartition[(thepartition >= (self.deviate_size/2)*m) & (thepartition < (self.deviate_size/2)*(m+1))])
        ranking_quality_rp2 += ranking_quality(confusion_rp2)
            
        for k in range(3):
            thepartition = np.array(all_rp3[0][k])
            for m in range(2):
                confusion_rp3[m,k] = len(thepartition[(thepartition >= (self.deviate_size/2)*m) & (thepartition < (self.deviate_size/2)*(m+1))])
        ranking_quality_rp3 += ranking_quality(confusion_rp3)

        for k in range(10):
            thepartition = np.array(all_rp10[0][k])
            for m in range(2):
                confusion_rp10[m,k] = len(thepartition[(thepartition >= (self.deviate_size/2)*m) & (thepartition < (self.deviate_size/2)*(m+1))])
        ranking_quality_rp10 += ranking_quality(confusion_rp10)

        theindex = i + j*self.num_experiments_per_epsilon
        exp0_scores[theindex,0] = theepsilon
        exp0_scores[theindex,1] = score_rp2        
        exp0_scores[theindex,2] = score_rp3
        exp0_scores[theindex,3] = score_rp10

    exp0_ranking_quality[j,0] = theepsilon
    exp0_ranking_quality[j,1] = ranking_quality_rp2 / self.num_experiments_per_epsilon
    exp0_ranking_quality[j,2] = ranking_quality_rp3 / self.num_experiments_per_epsilon
    exp0_ranking_quality[j,3] = ranking_quality_rp10 / self.num_experiments_per_epsilon

    print('Finished {} out of {}'.format(j, self.num_epsilon_values))
    
    exp0_scores_df = pd.DataFrame(exp0_scores,columns=['epsilon','rp2','rp3','rp10'])
    exp0_scores_df.to_csv("exp0_scores.csv",index_label='i')
    exp0_ranking_quality_df = pd.DataFrame(exp0_ranking_quality,columns=['epsilon','rp2','rp3','rp10'])

    return exp0_ranking_quality_df
  
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
  def __init__(self):
    self.full_range = range(NUM_NULL_EXPERIMENTS)
    self.slices = slice_range(self.full_range)
    self.tasks = multiprocessing.JoinableQueue()
    self.results = multiprocessing.Queue()
    self.workers = [Worker(self.tasks, self.results) for i in range(NUM_WORKERS)]

  def create_null_scores(self):
    for worker in self.workers:
      worker.start()
    for i,slc in enumerate(self.slices):
      self.tasks.put(NullTask(NUM_NULL_EXPERIMENTS, DEVIATE_SIZE, POISSON_INTENSITY, slc))
    for _ in self.workers:
      self.tasks.put(EndTask())

    self.tasks.join()

    left_on_queue = len(self.slices)
    null_scores_df = pd.DataFrame(columns=['rp2','rp3','rp10','mcd2','mcd3'])
    while not self.results.empty():
      result = self.results.get(block=True)
      null_scores_df = pd.concat([null_scores_df, result])
      left_on_queue -= 1
      
    assert left_on_queue == 0
    null_scores_df.to_csv("null_scores.csv",index_label='i')

  def create_thresholds(self):
    null_scores_df = pd.read_csv("null_scores.csv")
    null_scores_rp2 = null_scores_df.loc[:,'rp2']
    null_scores_rp3 = null_scores_df.loc[:,'rp3']
    null_scores_rp10 = null_scores_df.loc[:,'rp10']
    null_scores_mcd2 = null_scores_df.loc[:,'mcd2']
    null_scores_mcd3 = null_scores_df.loc[:,'mcd3']
    thresholds_rp2 = bootstrap_95th_percentile(null_scores_rp2)
    thresholds_rp3 = bootstrap_95th_percentile(null_scores_rp3)
    thresholds_rp10 = bootstrap_95th_percentile(null_scores_rp10)
    thresholds_mcd2 = bootstrap_95th_percentile(null_scores_mcd2)
    thresholds_mcd3 = bootstrap_95th_percentile(null_scores_mcd3)
    thresholds_df = pd.DataFrame({'rp2':thresholds_rp2,
                                  'rp3':thresholds_rp3,
                                  'rp10':thresholds_rp10,
                                  'mcd2':thresholds_mcd2,
                                  'mcd3':thresholds_mcd3})
    thresholds_df.to_csv("null_thresholds.csv")
    
class QA(object):
  def __init__(self):
    self.full_range = range(NUM_EPSILON_VALUES)
    self.slices = slice_range(self.full_range)
    self.tasks = multiprocessing.JoinableQueue()
    self.results = multiprocessing.Queue()
    self.workers = [Worker(self.tasks, self.results) for i in range(NUM_WORKERS)]

  def compute_ranking_quality(self):
    for worker in self.workers:
      worker.start()
    for i,slc in enumerate(self.slices):
      self.tasks.put(QATask(NUM_EXPERIMENTS_PER_EPSILON, DEVIATE_SIZE, POISSON_INTENSITY, NUM_EPSILON_VALUES))
    for _ in self.workers:
      self.tasks.put(EndTask())

    self.tasks.join()

    left_on_queue = len(self.slices)
    ranking_quality_df = pd.DataFrame(columns=['epsilon','rp2','rp3','rp10'])
    while not self.results.empty():
      result = self.results.get(block=True)
      ranking_quality_df = pd.concat([ranking_quality_df, result])
      left_on_queue -= 1

    assert left_on_queue == 0
  
    ranking_quality_df.to_csv("exp0_ranking_quality.csv",index=False)    
                    
b = Baselines()
b.create_null_scores()
b.create_thresholds()

################################################################
################################################################
# experiment 0: two partitions
#
num_epsilon_values = 11 # XXX ???
exp0_scores = np.zeros([NUM_EXPERIMENTS_PER_EPSILON*num_epsilon_values,4])
exp0_ranking_quality = np.zeros([num_epsilon_values,4])
for j in range(num_epsilon_values):
    theepsilon = 0.05*j
    ranking_quality_rp2 = 0
    ranking_quality_rp3 = 0
    ranking_quality_rp10 = 0
    for i in range(NUM_EXPERIMENTS_PER_EPSILON):
        b = rng.poisson(POISSON_INTENSITY,size=DEVIATE_SIZE).astype(float)
        q = np.concatenate([np.full(int(DEVIATE_SIZE/2),1.0-theepsilon),
                            np.full(int(DEVIATE_SIZE/2),1.0+theepsilon)])
        a = rng.poisson(q*b).astype(float)
        all_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()
        all_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()
        all_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()
        score_rp2 = all_rp2[1]
        score_rp3 = all_rp3[1]
        score_rp10 = all_rp10[1]
        confusion_rp2 = np.zeros([2,2])
        confusion_rp3 = np.zeros([2,3])
        confusion_rp10 = np.zeros([2,10])

        for k in range(2):
            thepartition = np.array(all_rp2[0][k])
            for m in range(2):
                confusion_rp2[m,k] = len(thepartition[(thepartition >= (DEVIATE_SIZE/2)*m) & (thepartition < (DEVIATE_SIZE/2)*(m+1))])
        ranking_quality_rp2 += ranking_quality(confusion_rp2)
            
        for k in range(3):
            thepartition = np.array(all_rp3[0][k])
            for m in range(2):
                confusion_rp3[m,k] = len(thepartition[(thepartition >= (DEVIATE_SIZE/2)*m) & (thepartition < (DEVIATE_SIZE/2)*(m+1))])
        ranking_quality_rp3 += ranking_quality(confusion_rp3)

        for k in range(10):
            thepartition = np.array(all_rp10[0][k])
            for m in range(2):
                confusion_rp10[m,k] = len(thepartition[(thepartition >= (DEVIATE_SIZE/2)*m) & (thepartition < (DEVIATE_SIZE/2)*(m+1))])
        ranking_quality_rp10 += ranking_quality(confusion_rp10)

        theindex = i + j*NUM_EXPERIMENTS_PER_EPSILON
        exp0_scores[theindex,0] = theepsilon
        exp0_scores[theindex,1] = score_rp2        
        exp0_scores[theindex,2] = score_rp3
        exp0_scores[theindex,3] = score_rp10

    exp0_ranking_quality[j,0] = theepsilon
    exp0_ranking_quality[j,1] = ranking_quality_rp2 / NUM_EXPERIMENTS_PER_EPSILON
    exp0_ranking_quality[j,2] = ranking_quality_rp3 / NUM_EXPERIMENTS_PER_EPSILON
    exp0_ranking_quality[j,3] = ranking_quality_rp10 / NUM_EXPERIMENTS_PER_EPSILON

    print('Finished {} out of {}'.format(j, num_epsilon_values))
    
exp0_scores_df = pd.DataFrame(exp0_scores,columns=['epsilon','rp2','rp3','rp10'])
exp0_scores_df.to_csv("exp0_scores.csv",index_label='i')
exp0_ranking_quality_df = pd.DataFrame(exp0_ranking_quality,columns=['epsilon','rp2','rp3','rp10'])
exp0_ranking_quality_df.to_csv("exp0_ranking_quality.csv",index=False)
########################################################################

########################################################################
# Compute detection power for experiment 0
#
null_thresholds_df = pd.read_csv("null_thresholds.csv")
exp0_scores_df = pd.read_csv("exp0_scores.csv")
num_epsilon_values = 11
methods = ['rp2','rp3','rp10']
detection_power = np.zeros([num_epsilon_values,1+len(methods)])
for j in range(num_epsilon_values):
    theepsilon = 0.05*j
    detection_power[j,0] = theepsilon
    i = 1
    for themethod in methods:
        thethresholds = null_thresholds_df.loc[:,themethod]
        thescores = exp0_scores_df[np.abs(exp0_scores_df['epsilon'] - theepsilon) < 1E-6].loc[:,themethod]
        thepower = 0
        for k in thescores:
            if (k <= np.min(thethresholds)):
              thepower += 0.
            elif (k >= np.max(thethresholds)):
              thepower += 1.
            else:
                for m in thethresholds:
                    if (k > m):
                        thepower += 1./len(thethresholds)
        detection_power[j,i] = thepower/len(thescores)
        i += 1

detection_power_df = pd.DataFrame(detection_power,columns=['epsilon','rp2','rp3','rp10'])
detection_power_df.to_csv("exp0_detection_power.csv",index=False)
#
########################################################################
