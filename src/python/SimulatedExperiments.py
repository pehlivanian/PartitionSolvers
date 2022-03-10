import multiprocessing
from functools import partial
from itertools import islice
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import solverSWIG_DP
import proto


SEED = 1869
rng = np.random.RandomState(SEED)

def compute_prob_sup(v1,v2):
   prob_sup = 0.
   thelength = len(v1)
   for j in range(thelength):
     for i in range(j+1,len(v2)):
       prob_sup += v1[i]*v2[j]
   prob_sup /= np.sum(v1)*np.sum(v2)
   return prob_sup

def overlap_coeff(m):
  true_pos = np.sum(m[1:,1:])
  false_neg = np.sum(m[1:,0])
  false_pos = np.sum(m[0,1:])
  precision = true_pos / (true_pos+false_pos)
  recall = true_pos / (true_pos+false_neg)
  overlap = true_pos / (true_pos+false_pos+false_neg)
  prob_sup_primary = compute_prob_sup(m[2,:],m[0,:])
  prob_sup_secondary = compute_prob_sup(m[1,:],m[0,:])
  prob_sup_distinguish = compute_prob_sup(m[2,:],m[1,:])
  return precision,recall,overlap,prob_sup_primary,prob_sup_secondary,prob_sup_distinguish

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
        self.task = self._task

    def __call__(self):
        return self.task()

    def _task(self):
        count = 0
        cluster_list = CLUSTER_LIST
        null_scores = np.zeros([len(self.range),2*len(cluster_list)])
        for i in self.range:
            b = rng.poisson(self.poisson_intensity, size=self.deviate_size).astype(float)
            a = rng.poisson(b).astype(float)
            results_rp = [0.]*len(cluster_list)
            results_mcd = [0.]*len(cluster_list)
            for ind,n in enumerate(cluster_list):
              results_rp[ind] = solverSWIG_DP.OptimizerSWIG(n,a,b,1,True,True)()[1]
              results_mcd[ind] = solverSWIG_DP.OptimizerSWIG(n,a,b,1,False,True)()[1]              
            null_scores[count,:] = results_rp + results_mcd
            count += 1
        null_scores_df = pd.DataFrame(null_scores, columns=['rp'+str(x) for x in cluster_list]+['mcd'+str(x) for x in cluster_list])
        return null_scores_df

class QATask(object):
  def __init__(self, num_true_clusters, num_experiments_per_epsilon, deviate_size,
               poisson_intensity, num_epsilon_values, figures_for_submission=False):
    self.num_true_clusters = num_true_clusters
    self.num_experiments_per_epsilon = num_experiments_per_epsilon
    self.deviate_size = deviate_size
    self.poisson_intensity = poisson_intensity
    self.num_epsilon_values = num_epsilon_values
    self.task = partial(self._task)
    self.rng = np.random.RandomState()
    self.figures_for_submission = figures_for_submission

  def __call__(self):
    return self.task()

  def _task(self):
    cluster_list = CLUSTER_LIST
    exp0_scores = np.zeros([self.num_experiments_per_epsilon*self.num_epsilon_values,len(cluster_list)+1])
    exp0_ranking_quality = np.zeros([self.num_epsilon_values,len(cluster_list)+1])
    for j in range(self.num_epsilon_values):      
      theepsilon = 0.05*j
      ranking_quality_ind = [0.]*len(cluster_list)
      for i in range(self.num_experiments_per_epsilon):
        b = self.rng.poisson(self.poisson_intensity,size=self.deviate_size).astype(float)
        # Branch on number of true clusters
        # RISK PARTITIONING CASE        
        split = int(self.deviate_size/self.num_true_clusters)
        resid = self.deviate_size - (split * self.num_true_clusters)
        resids = ([1] * int(resid)) + ([0] * (self.num_true_clusters - int(resid)))
        splits = [split + r for r in resids]
        if (self.figures_for_submission):
           if (self.num_true_clusters == 2):
              q = np.concatenate([np.full(splits[0],1.0-theepsilon),
                                  np.full(splits[1],1.0+theepsilon)])
           elif (self.num_true_clusters == 3):
              q = np.concatenate([np.full(splits[0],1.0-theepsilon),np.full(splits[1],1.0),np.full(splits[2],1.0+theepsilon)])
           elif (self.num_true_clusters == 10):
              q = np.concatenate([np.full(splits[0],1.0-theepsilon),
                                  np.full(splits[1],1.0-4.*theepsilon/5.),
                                  np.full(splits[2],1.0-3.*theepsilon/5.),
                                  np.full(splits[3],1.0-2.*theepsilon/5.),
                                  np.full(splits[4],1.0-1.*theepsilon/5.),
                                  np.full(splits[5],1.0+1.*theepsilon/5.),
                                  np.full(splits[6],1.0+2.*theepsilon/5.),
                                  np.full(splits[7],1.0+3.*theepsilon/5.),
                                  np.full(splits[8],1.0+4.*theepsilon/5.),
                                  np.full(splits[9],1.0+theepsilon)])
        else:
           levels = np.linspace(theepsilon/self.num_true_clusters,
                                2-theepsilon/self.num_true_clusters,
                                self.num_true_clusters)
           q = np.concatenate([np.full(s,l) for s,l in zip(splits,levels)])
           
        a = rng.poisson(q*b).astype(float)
           
        all_rp = [0.]*len(cluster_list)
        score_rp = [0.]*len(cluster_list)
        confusion = [0.]*len(cluster_list)        
        for ind,n in enumerate(cluster_list):
          all_rp[ind] = solverSWIG_DP.OptimizerSWIG(n,a,b,1,True,True)()
          score_rp[ind] = all_rp[ind][1]
          confusion[ind] = np.zeros([self.num_true_clusters,n])
        for ind,n in enumerate(cluster_list):
          for k in range(n):
            thepartition = np.array(all_rp[ind][0][k])            
            for m in range(self.num_true_clusters):
              confusion[ind][m,k] = len(thepartition[(thepartition >= splits[m]*m) & (thepartition < splits[m]*(m+1))])
          ranking_quality_ind[ind] += ranking_quality(confusion[ind])

        theindex = i + j*self.num_experiments_per_epsilon
        exp0_scores[theindex,0] = theepsilon        
        for ind,n in enumerate(cluster_list):          
          exp0_scores[theindex,1+ind] = score_rp[ind]

      exp0_ranking_quality[j,0] = theepsilon
      for ind,n in enumerate(cluster_list):
        exp0_ranking_quality[j,1+ind] = ranking_quality_ind[ind] / self.num_experiments_per_epsilon

    exp0_scores_df = pd.DataFrame(exp0_scores,columns=['epsilon']+['rp'+str(x) for x in cluster_list])
    exp0_scores_df.to_csv("exp0_scores.csv",index_label='i')
    exp0_ranking_quality_df = pd.DataFrame(exp0_ranking_quality,columns=['epsilon']+['rp'+str(x) for x in cluster_list])

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
    self.workers = [Worker(self.tasks, self.results) for _ in range(NUM_WORKERS)]

  def create_null_scores(self):
    cluster_list = CLUSTER_LIST
    for worker in self.workers:
      worker.start()
    for i,slc in enumerate(self.slices):
      self.tasks.put(NullTask(NUM_NULL_EXPERIMENTS, DEVIATE_SIZE, POISSON_INTENSITY, slc))
    for _ in self.workers:
      self.tasks.put(EndTask())

    self.tasks.join()

    left_on_queue = len(self.slices)
    null_scores_df = pd.DataFrame(columns=['rp'+str(x) for x in cluster_list]+['mcd'+str(x) for x in cluster_list])
    while not self.results.empty():
      result = self.results.get(block=True)
      null_scores_df = pd.concat([null_scores_df, result])
      left_on_queue -= 1
      
    assert left_on_queue == 0
    null_scores_df.to_csv("null_scores.csv",index_label='i')

  def create_thresholds(self):
    cluster_list = CLUSTER_LIST
    null_scores_df = pd.read_csv("null_scores.csv")
    null_scores_ind = np.zeros([null_scores_df.shape[0], len(cluster_list)])
    thresholds = dict()
    for ind,n in enumerate(cluster_list):
      rpCol = 'rp'+str(n)
      mcdCol = 'mcd'+str(n)
      thresholds[rpCol] = bootstrap_95th_percentile(null_scores_df.loc[:,rpCol])
      thresholds[mcdCol] = bootstrap_95th_percentile(null_scores_df.loc[:,mcdCol])
    thresholds_df = pd.DataFrame(thresholds)
    thresholds_df = thresholds_df[['rp'+str(x) for x in cluster_list]+['mcd'+str(x) for x in cluster_list]]
    thresholds_df.to_csv("null_thresholds.csv")

class Plotter(object):
   colorMap = ['m', 'r', 'g']
   win_len = 5

   linestyles = [
      'solid',
      'dotted',
      'dashed',
      'dashdot'
      ]

   def moving_average(a, n) :
      ret = np.cumsum(a, dtype=float)
      ret[n:] = ret[n:] - ret[:-n]
      return ret[n - 1:] / n

   @staticmethod
   def plotRankingQuality(clusterList, num_true_clusters):
      df = pd.read_csv('exp0_ranking_quality.csv')
      grouped = df.groupby(by=['epsilon'], as_index=False).mean()
      columns = ['rp'+str(x) for x in clusterList]
      xaxis = grouped['epsilon']
      for ind,cluster in enumerate(clusterList):
         column = 'rp'+str(cluster)
         yaxis = grouped.loc[:,column].values
         plot.plot(xaxis,yaxis,label='t = {}'.format(cluster), linestyle=Plotter.linestyles[ind])
      plot.xlabel('signal strength (epsilon)')
      plot.ylabel('ranking quality')
      plot.legend()
      plot.grid(True)
      plot.title('Ranking quality - {} risk partitions'.format(num_true_clusters))
      plot.pause(1)
      plot.savefig('Figure_4_3_partitions.pdf')
      plot.close()
      
      
class QA(object):
  def __init__(self):
    self.full_range = range(NUM_EPSILON_VALUES)
    self.slices = slice_range(self.full_range)
    self.tasks = multiprocessing.JoinableQueue()
    self.results = multiprocessing.Queue()
    self.workers = [Worker(self.tasks, self.results) for i in range(NUM_WORKERS)]

  def compute_ranking_quality(self, num_true_clusters):
    cluster_list = CLUSTER_LIST
    for worker in self.workers:
      worker.start()
    for i,slc in enumerate(self.slices):
      self.tasks.put(QATask(num_true_clusters, NUM_EXPERIMENTS_PER_EPSILON, DEVIATE_SIZE, POISSON_INTENSITY, NUM_EPSILON_VALUES))
    for _ in self.workers:
      self.tasks.put(EndTask())

    self.tasks.join()

    left_on_queue = len(self.slices)
    ranking_quality_df = pd.DataFrame(columns=['epsilon']+['rp'+str(x) for x in cluster_list])
    while not self.results.empty():
      result = self.results.get(block=True)
      ranking_quality_df = pd.concat([ranking_quality_df, result])
      left_on_queue -= 1

    assert left_on_queue == 0
  
    ranking_quality_df.to_csv("exp0_ranking_quality.csv",index=False)    

  def compute_detection_power(self):
    cluster_list = CLUSTER_LIST
    null_thresholds_df = pd.read_csv("null_thresholds.csv")
    exp0_scores_df = pd.read_csv("exp0_scores.csv")
    num_epsilon_values = NUM_EPSILON_VALUES
    
    methods = ['rp'+str(x) for x in cluster_list]
    detection_power = np.zeros([num_epsilon_values,1+len(methods)])
    for j in range(num_epsilon_values):
      theepsilon = 0.05*j
      detection_power[j,0] = theepsilon
      i = 1
      for method in methods:
        thresholds = null_thresholds_df.loc[:,method]
        scores = exp0_scores_df[np.abs(exp0_scores_df['epsilon'] - theepsilon) < 1E-6].loc[:,method]
        power = 0
        for k in scores:
            if (k <= np.min(thresholds)):
              power += 0.
            elif (k >= np.max(thresholds)):
              power += 1.
            else:
                for m in thresholds:
                    if (k > m):
                        power += 1./len(thresholds)
        detection_power[j,i] = power/len(scores)
        i += 1

    detection_power_df = pd.DataFrame(detection_power,columns=['epsilon']+['rp'+str(x) for x in cluster_list])
    detection_power_df.to_csv("exp0_detection_power.csv",index=False)

if __name__ == '__main__':
  import sys
  num_true_clusters = int(sys.argv[1])

  POISSON_INTENSITY = 100             # 100  in paper
  DEVIATE_SIZE = 1000                 # 5000 in paper
  NUM_NULL_EXPERIMENTS = 1000         # 1000 in paper
  NUM_THRESHOLD_CALCULATIONS = 1000   # 1000 in paper
  NUM_EXPERIMENTS_PER_EPSILON = 1000  # 1000 in paper
  NUM_EPSILON_VALUES = 11             # 11 in paper
  CLUSTER_LIST = (2,3,10)    # (2,3,5,10) in paper
  NUM_WORKERS = multiprocessing.cpu_count() - 1

  assert not DEVIATE_SIZE%2

  # Create baselines: null scores and 95% thresholds, only need to run once to persist
  # b = Baselines()
  # b.create_null_scores()
  # b.create_thresholds()

  # Generate statistics
  Q = QA()
  Q.compute_ranking_quality(num_true_clusters)
  Q.compute_detection_power()
  P = Plotter()
  P.plotRankingQuality(CLUSTER_LIST, num_true_clusters)

