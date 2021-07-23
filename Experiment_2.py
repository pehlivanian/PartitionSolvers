import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
import proto
import seaborn as sns
import matplotlib.pyplot as plot

NUM_EXPERIMENTS = 10000 # 100
NUM_EXPERIMENTS_PER_EPSILON = 500 # 100
NUM_THRESHOLD_EXPERIMENTS = 1000 # 100
N = 2500 # 500

##############
## Graphics ##
##############

def plot_confusion(confusion_matrix, source_class_names, target_class_names, figsize=(10,7), fontsize=14, title='Confusion Matrix'):
    df_cm = pd.DataFrame(
        confusion_matrix, index=source_class_names, columns=target_class_names, 
    )
    fig = plot.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", linewidths=0.5)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plot.ylabel('True cluster')
    plot.xlabel('Predicted cluster')
    plot.title(title)
    return fig

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

rng = np.random.default_rng(59)

a = proto.FArray()                  # wrapper for C++ float array type
b = proto.FArray()                  # wrapper for C++ float array type

################################################################
# null experiments
# rng = np.random.default_rng(12345)
# num_null_experiments = NUM_EXPERIMENTS
# null_scores = np.zeros([num_null_experiments,5])
# for i in range(num_null_experiments):
#     b = rng.poisson(100,size=N).astype(float)
#     a = rng.poisson(b).astype(float)
#     result_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()[1]
#     result_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()[1]
#     result_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()[1]
#     result_mcd2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,False,True)()[1]
#     result_mcd3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,False,True)()[1]
#     null_scores[i,:] = [result_rp2,result_rp3,result_rp10,result_mcd2,result_mcd3]
# null_scores_df = pd.DataFrame(null_scores,columns=['rp2','rp3','rp10','mcd2','mcd3'])
# null_scores_df.to_csv("null_scores.csv",index_label='i')
################################################################

################################################################
# bootstrapped 95th percentile distribution for detection power
# rng = np.random.default_rng(95)
# null_scores_df = pd.read_csv("null_scores.csv")
# null_scores_rp2 = null_scores_df.loc[:,'rp2']
# null_scores_rp3 = null_scores_df.loc[:,'rp3']
# null_scores_rp10 = null_scores_df.loc[:,'rp10']
# null_scores_mcd2 = null_scores_df.loc[:,'mcd2']
# null_scores_mcd3 = null_scores_df.loc[:,'mcd3']
# thresholds_rp2 = bootstrap_95th_percentile(null_scores_rp2)
# thresholds_rp3 = bootstrap_95th_percentile(null_scores_rp3)
# thresholds_rp10 = bootstrap_95th_percentile(null_scores_rp10)
# thresholds_mcd2 = bootstrap_95th_percentile(null_scores_mcd2)
# thresholds_mcd3 = bootstrap_95th_percentile(null_scores_mcd3)
# thresholds_df = pd.DataFrame({'rp2':thresholds_rp2,'rp3':thresholds_rp3,'rp10':thresholds_rp10,'mcd2':thresholds_mcd2,'mcd3':thresholds_mcd3})
# thresholds_df.to_csv("null_thresholds.csv")
################################################################

################################################################
# experiment 2: ten partitions
#
num_experiments_per_epsilon = NUM_EXPERIMENTS_PER_EPSILON
num_epsilon_values = 11
exp2_scores = np.zeros([num_experiments_per_epsilon*num_epsilon_values,4])
exp2_ranking_quality = np.zeros([num_epsilon_values,4])
for j in range(num_epsilon_values):
    n1 = int(N/10)
    n2,n3,n4,n5,n6,n7,n8,n9 = 2*n1,3*n1,4*n1,5*n1,6*n1,7*n1,8*n1,9*n1
    n10 = N-n9
    theepsilon = 0.05*j
    ranking_quality_rp2 = 0
    ranking_quality_rp3 = 0
    ranking_quality_rp10 = 0
    for i in range(num_experiments_per_epsilon):
        b = rng.poisson(100,size=N).astype(float)
        q = np.concatenate([np.full(n1,1.0-theepsilon),
                            np.full(n1,1.0-4.*theepsilon/5.),
                            np.full(n1,1.0-3.*theepsilon/5.),
                            np.full(n1,1.0-2.*theepsilon/5.),
                            np.full(n1,1.0-1.*theepsilon/5.),
                            np.full(n1,1.0+1.*theepsilon/5.),
                            np.full(n1,1.0+2.*theepsilon/5.),
                            np.full(n1,1.0+3.*theepsilon/5.),
                            np.full(n1,1.0+4.*theepsilon/5.),
                            np.full(n1,1.0+theepsilon)])
        a = rng.poisson(q*b).astype(float)
        all_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()
        all_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()
        all_rp10 = solverSWIG_DP.OptimizerSWIG(10,a,b,1,True,True)()
        score_rp2 = all_rp2[1]
        score_rp3 = all_rp3[1]
        score_rp10 = all_rp10[1]
        confusion_rp2 = np.zeros([10,2])
        confusion_rp3 = np.zeros([10,3])
        confusion_rp10 = np.zeros([10,10])

        for k in range(2):
            thepartition = np.array(all_rp2[0][k])
            for m in range(10):
                confusion_rp2[m,k] = len(thepartition[(thepartition >= n1*m) & (thepartition < n1*(m+1))])
        ranking_quality_rp2 += ranking_quality(confusion_rp2)
            
        for k in range(3):
            thepartition = np.array(all_rp3[0][k])
            for m in range(10):
                confusion_rp3[m,k] = len(thepartition[(thepartition >= n1*m) & (thepartition < n1*(m+1))])
        ranking_quality_rp3 += ranking_quality(confusion_rp3)

        for k in range(10):
            thepartition = np.array(all_rp10[0][k])
            for m in range(10):
                confusion_rp10[m,k] = len(thepartition[(thepartition >= n1*m) & (thepartition < n1*(m+1))])
        ranking_quality_rp10 += ranking_quality(confusion_rp10)

        theindex = i + j*num_experiments_per_epsilon
        exp2_scores[theindex,0] = theepsilon
        exp2_scores[theindex,1] = score_rp2        
        exp2_scores[theindex,2] = score_rp3
        exp2_scores[theindex,3] = score_rp10

    exp2_ranking_quality[j,0] = theepsilon
    exp2_ranking_quality[j,1] = ranking_quality_rp2 / num_experiments_per_epsilon
    exp2_ranking_quality[j,2] = ranking_quality_rp3 / num_experiments_per_epsilon
    exp2_ranking_quality[j,3] = ranking_quality_rp10 / num_experiments_per_epsilon
exp2_scores_df = pd.DataFrame(exp2_scores,columns=['epsilon','rp2','rp3','rp10'])
exp2_scores_df.to_csv("exp2_scores.csv",index_label='i')
exp2_ranking_quality_df = pd.DataFrame(exp2_ranking_quality,columns=['epsilon','rp2','rp3','rp10'])
exp2_ranking_quality_df.to_csv("exp2_ranking_quality.csv",index=False)
########################################################################

# Plot confusion
rp10_label = ['1','2','3','4','5','6','7','8','9','10']
rp10_title = 'Confusion matrix - 10 true risk partitions, t = 10'
rp3_label  = ['1','2','3']
rp3_title  = 'Confusion matrix - 10 true risk partitions, t = 3'
rp2_label  = ['1','2']
rp2_title  = 'confusion matrix - 10 true risk partitions, t = 2'

fig = plot_confusion(confusion_rp10.astype('int'), ['1','2','3','4','5','6','7','8','9','10'],rp10_label, title=rp10_title)
plot.pause(1e-3)
plot.close()
fig.savefig('Confusion_matrix_rp10_exp2.pdf')
fig = plot_confusion(confusion_rp3.astype('int'), ['1','2','3','4','5','6','7','8','9','10'],rp3_label, title=rp3_title)
fig.show
plot.close()
fig.savefig('Confusion_matrix_rp3_exp2.pdf')
fig = plot_confusion(confusion_rp2.astype('int'), ['1','2','3','4','5','6','7','8','9','10'],rp2_label, title=rp2_title)
plot.pause(1e-3)
plot.close()
fig.savefig('Confusion_matrix_rp2_exp2.pdf')

# Plot ranking quality
import matplotlib.pyplot as plot
exp2_ranking_quality_df = pd.read_csv("exp2_ranking_quality.csv")
exp2_ranking_quality_df = exp2_ranking_quality_df.set_index('epsilon')
exp2_ranking_quality_df = exp2_ranking_quality_df.rename(columns={'rp2': 't = 2', 'rp3': 't = 3', 'rp10': 't = 10'})
exp2_ranking_quality_df.plot(kind='line', grid=True, style='.-', title='Ranking quality - 10 risk partitions', ylabel='ranking quality', xlabel='signal strength (epsilon)')
plot.pause(1e-3)
# plot.savefig('ranking_quality_exp2.pdf')
plot.close()

########################################################################
# Compute detection power for experiment 2
#
null_thresholds_df = pd.read_csv("null_thresholds.csv")
exp2_scores_df = pd.read_csv("exp2_scores.csv")
num_epsilon_values = 11
methods = ['rp2','rp3','rp10']
detection_power = np.zeros([num_epsilon_values,1+len(methods)])
for j in range(num_epsilon_values):
    theepsilon = 0.05*j
    detection_power[j,0] = theepsilon
    i = 1
    for themethod in methods:
        thethresholds = null_thresholds_df.loc[:,themethod]
        thescores = exp2_scores_df[np.abs(exp2_scores_df['epsilon'] - theepsilon) < 1E-6].loc[:,themethod]
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
detection_power_df.to_csv("exp2_detection_power.csv",index=False)
#
########################################################################

# Plot detection power
import matplotlib.pyplot as plot
exp2_detection_power_df = pd.read_csv("exp2_detection_power.csv")
exp2_detection_power_df = exp2_detection_power_df.set_index('epsilon')
exp2_detection_power_df = exp2_detection_power_df.rename(columns={'rp2': 't = 2', 'rp3': 't = 3', 'rp10': 't = 10'})
exp2_detection_power_df.plot(kind='line', grid=True, style='.-', title='Detection power - 10 risk partitions', ylabel='detection power', xlabel='signal strength (epsilon)')
plot.pause(1e-3)
# plot.savefig('detection_power_exp2.pdf')
plot.close()
