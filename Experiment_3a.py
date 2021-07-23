import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
import proto

NUM_EXPERIMENTS = 10000 # 100
NUM_EXPERIMENTS_PER_EPSILON = 500 # 100
NUM_THRESHOLD_EXPERIMENTS = 1000 # 100
N = 2500 # 500

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
  thresholds = np.zeros(NUM_THRESHOLD_EXPERIMENTS)
  for i in range(NUM_THRESHOLD_EXPERIMENTS):
    thresholds[i] = np.quantile(rng.choice(null_scores, size=N, replace=True),0.95)
  thresholds = np.sort(thresholds)
  return thresholds

rng = np.random.default_rng(159)

a = proto.FArray()                  # wrapper for C++ float array type
b = proto.FArray()                  # wrapper for C++ float array type

################################################################
# null experiments
rng = np.random.default_rng(12345)
num_null_experiments = NUM_EXPERIMENTS
null_scores = np.zeros([num_null_experiments,5])
for i in range(num_null_experiments):
    b = rng.poisson(100,size=N).astype(float)
    a = rng.poisson(b).astype(float)
    result_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()[1]
    result_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()[1]
    result_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()[1]
    result_mcd2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,False,True)()[1]
    result_mcd3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,False,True)()[1]
    null_scores[i,:] = [result_rp2,result_rp3,result_rp10,result_mcd2,result_mcd3]
null_scores_df = pd.DataFrame(null_scores,columns=['rp2','rp3','rp10','mcd2','mcd3'])
null_scores_df.to_csv("null_scores.csv",index_label='i')
################################################################

################################################################
# bootstrapped 95th percentile distribution for detection power
rng = np.random.default_rng(95)
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
thresholds_df = pd.DataFrame({'rp2':thresholds_rp2,'rp3':thresholds_rp3,'rp10':thresholds_rp10,'mcd2':thresholds_mcd2,'mcd3':thresholds_mcd3})
thresholds_df.to_csv("null_thresholds.csv")
################################################################

################################################################
# experiment 3: no partitions; all q are uniform on [1-eps, 1+eps]
#
num_experiments_per_epsilon = NUM_EXPERIMENTS_PER_EPSILON
num_epsilon_values = 11
exp3_scores = np.zeros([num_experiments_per_epsilon*num_epsilon_values,4])
for j in range(num_epsilon_values):
    theepsilon = 0.01*j
    for i in range(num_experiments_per_epsilon):
        b = rng.poisson(100,size=N).astype(float)
        q = rng.uniform(1.0-theepsilon,1.0+theepsilon,size=N)
        a = rng.poisson(q*b).astype(float)
        all_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()
        all_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()
        all_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()
        score_rp2 = all_rp2[1]
        score_rp3 = all_rp3[1]
        score_rp10 = all_rp10[1]

        theindex = i + j*num_experiments_per_epsilon
        exp3_scores[theindex,0] = theepsilon
        exp3_scores[theindex,1] = score_rp2        
        exp3_scores[theindex,2] = score_rp3
        exp3_scores[theindex,3] = score_rp10

exp3_scores_df = pd.DataFrame(exp3_scores,columns=['epsilon','rp2','rp3','rp10'])
exp3_scores_df.to_csv("exp3a_scores.csv",index_label='i')
########################################################################

########################################################################
# Compute detection power for experiment 3
#
null_thresholds_df = pd.read_csv("null_thresholds.csv")
exp3_scores_df = pd.read_csv("exp3a_scores.csv")
num_epsilon_values = 11
methods = ['rp2','rp3','rp10']
detection_power = np.zeros([num_epsilon_values,1+len(methods)])
for j in range(num_epsilon_values):
    theepsilon = 0.01*j
    detection_power[j,0] = theepsilon
    i = 1
    for themethod in methods:
        thethresholds = null_thresholds_df.loc[:,themethod]
        thescores = exp3_scores_df[np.abs(exp3_scores_df['epsilon'] - theepsilon) < 1E-6].loc[:,themethod]
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
detection_power_df.to_csv("exp3a_detection_power.csv",index=False)
#
########################################################################

# Plot detection power
import matplotlib.pyplot as plot
exp3a_detection_power_df = pd.read_csv("exp3a_detection_power.csv")
exp3a_detection_power_df = exp3a_detection_power_df.set_index('epsilon')
exp3a_detection_power_df = exp3a_detection_power_df.rename(columns={'rp2': 't = 2', 'rp3': 't = 3', 'rp10': 't = 10'})
exp3a_detection_power_df.plot(kind='line', grid=True, style='.-', title='Detection power - q ~ Uniform[1-eps,1+eps]', ylabel='detection power', xlabel='signal strength (epsilon)')
plot.pause(1e-3)
plot.savefig('detection_power_exp3a.pdf')
