import numpy as np
import pandas as pd
import solverSWIG_DP
import solverSWIG_LTSS
import proto

NUM_EXPERIMENTS = 10000 # 100
NUM_EXPERIMENTS_PER_EPSILON = 500 # 10
NUM_THRESHOLD_EXPERIMENTS = 1000 # 100
N = 500 # 500

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
  thresholds = np.zeros(NUM_THRESHOLD_EXPERIMENTS)
  for i in range(NUM_THRESHOLD_EXPERIMENTS):
    thresholds[i] = np.quantile(rng.choice(null_scores, size=N, replace=True),0.95)
  thresholds = np.sort(thresholds)
  return thresholds

rng = np.random.default_rng(9500)

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
# experiment 4: multiple cluster detection, two clusters
#
num_experiments_per_q = NUM_EXPERIMENTS_PER_EPSILON
exp4_scores = np.zeros([num_experiments_per_q*60,6])
exp4_precision = np.zeros([60,6])
exp4_recall = np.zeros([60,6])
exp4_overlap = np.zeros([60,6])
exp4_primary = np.zeros([60,6])
exp4_secondary = np.zeros([60,6])
exp4_distinguish = np.zeros([60,6])
j = 0
theindex = 0
for q1 in np.linspace(1.01,1.2,20):
  for q2 in [1+0.25*(q1-1),1+0.5*(q1-1),1+0.75*(q1-1)]:
    precision_rp2 = recall_rp2 = overlap_rp2 = primary_rp2 = secondary_rp2 = distinguish_rp2 = 0
    precision_rp3 = recall_rp3 = overlap_rp3 = primary_rp3 = secondary_rp3 = distinguish_rp3 = 0
    precision_mcd2 = recall_mcd2 = overlap_mcd2 = primary_mcd2 = secondary_mcd2 = distinguish_mcd2 = 0
    precision_mcd3 = recall_mcd3 = overlap_mcd3 = primary_mcd3 = secondary_mcd3 = distinguish_mcd3 = 0
    for i in range(num_experiments_per_q):
        n1 = int(N/10)
        n2 = N-n1-n1       
        b = rng.poisson(100,size=N).astype(float)
        q = np.concatenate([np.full(n2,1.0),
                            np.full(n1,q2),
                            np.full(n1,q1)])
        a = rng.poisson(q*b).astype(float)
        all_rp2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,True,True)()
        all_rp3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,True,True)()
        all_mcd2 = solverSWIG_DP.OptimizerSWIG(2,a,b,1,False,True)()
        all_mcd3 = solverSWIG_DP.OptimizerSWIG(3,a,b,1,False,True)()

        score_rp2 = all_rp2[1]
        score_rp3 = all_rp3[1]
        score_mcd2 = all_mcd2[1]
        score_mcd3 = all_mcd3[1]
        confusion_rp2 = np.zeros([3,2])
        confusion_rp3 = np.zeros([3,3])
        confusion_mcd2 = np.zeros([3,2])
        confusion_mcd3 = np.zeros([3,3])

        for k in range(2):
            thepartition_rp = np.array(all_rp2[0][k])
            thepartition_mcd = np.array(all_mcd2[0][k])
            confusion_rp2[0,k] = len(thepartition_rp[(thepartition_rp < 400)])
            confusion_rp2[1,k] = len(thepartition_rp[(thepartition_rp >= 400) & (thepartition_rp < 450)])
            confusion_rp2[2,k] = len(thepartition_rp[(thepartition_rp >= 450)])
            confusion_mcd2[0,k] = len(thepartition_mcd[(thepartition_mcd < 400)])
            confusion_mcd2[1,k] = len(thepartition_mcd[(thepartition_mcd >= 400) & (thepartition_mcd < 450)])
            confusion_mcd2[2,k] = len(thepartition_mcd[(thepartition_mcd >= 450)])
        theprecision,therecall,theoverlap,theprimary,thesecondary,thedistinguish = overlap_coeff(confusion_rp2)
        precision_rp2 += theprecision
        recall_rp2 += therecall
        overlap_rp2 += theoverlap
        primary_rp2 += theprimary
        secondary_rp2 += thesecondary
        distinguish_rp2 += thedistinguish
        theprecision,therecall,theoverlap,theprimary,thesecondary,thedistinguish = overlap_coeff(confusion_mcd2)
        precision_mcd2 += theprecision
        recall_mcd2 += therecall
        overlap_mcd2 += theoverlap
        primary_mcd2 += theprimary
        secondary_mcd2 += thesecondary
        distinguish_mcd2 += thedistinguish
            
        for k in range(3):
            thepartition_rp = np.array(all_rp3[0][k])
            thepartition_mcd = np.array(all_mcd3[0][k])
            confusion_rp3[0,k] = len(thepartition_rp[(thepartition_rp < n2)])
            confusion_rp3[1,k] = len(thepartition_rp[(thepartition_rp >= n2) & (thepartition_rp < n1+n2)])
            confusion_rp3[2,k] = len(thepartition_rp[(thepartition_rp >= n1+n2)])
            confusion_mcd3[0,k] = len(thepartition_mcd[(thepartition_mcd < n2)])
            confusion_mcd3[1,k] = len(thepartition_mcd[(thepartition_mcd >= n2) & (thepartition_mcd < n1+n2)])
            confusion_mcd3[2,k] = len(thepartition_mcd[(thepartition_mcd >= n1+n2)])
        theprecision,therecall,theoverlap,theprimary,thesecondary,thedistinguish = overlap_coeff(confusion_rp3)
        precision_rp3 += theprecision
        recall_rp3 += therecall
        overlap_rp3 += theoverlap
        primary_rp3 += theprimary
        secondary_rp3 += thesecondary
        distinguish_rp3 += thedistinguish
        theprecision,therecall,theoverlap,theprimary,thesecondary,thedistinguish = overlap_coeff(confusion_mcd3)
        precision_mcd3 += theprecision
        recall_mcd3 += therecall
        overlap_mcd3 += theoverlap
        primary_mcd3 += theprimary
        secondary_mcd3 += thesecondary
        distinguish_mcd3 += thedistinguish

        exp4_scores[theindex,:] = [q1,q2,score_rp2,score_rp3,score_mcd2,score_mcd3]
        theindex += 1

    exp4_precision[j,:] = [q1,q2,precision_rp2/num_experiments_per_q,precision_rp3/num_experiments_per_q,precision_mcd2/num_experiments_per_q,precision_mcd3/num_experiments_per_q]
    exp4_recall[j,:] = [q1,q2,recall_rp2/num_experiments_per_q,recall_rp3/num_experiments_per_q,recall_mcd2/num_experiments_per_q,recall_mcd3/num_experiments_per_q]
    exp4_overlap[j,:] = [q1,q2,overlap_rp2/num_experiments_per_q,overlap_rp3/num_experiments_per_q,overlap_mcd2/num_experiments_per_q,overlap_mcd3/num_experiments_per_q]
    exp4_primary[j,:] = [q1,q2,primary_rp2/num_experiments_per_q,primary_rp3/num_experiments_per_q,primary_mcd2/num_experiments_per_q,primary_mcd3/num_experiments_per_q]
    exp4_secondary[j,:] = [q1,q2,secondary_rp2/num_experiments_per_q,secondary_rp3/num_experiments_per_q,secondary_mcd2/num_experiments_per_q,secondary_mcd3/num_experiments_per_q]
    exp4_distinguish[j,:] = [q1,q2,distinguish_rp2/num_experiments_per_q,distinguish_rp3/num_experiments_per_q,distinguish_mcd2/num_experiments_per_q,distinguish_mcd3/num_experiments_per_q]
    j += 1

exp4_scores_df = pd.DataFrame(exp4_scores,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_scores_df.to_csv("exp4a_scores.csv",index_label='i')
exp4_precision_df = pd.DataFrame(exp4_precision,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_precision_df.to_csv("exp4a_precision.csv",index=False)
exp4_recall_df = pd.DataFrame(exp4_recall,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_recall_df.to_csv("exp4a_recall.csv",index=False)
exp4_overlap_df = pd.DataFrame(exp4_overlap,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_overlap_df.to_csv("exp4a_overlap.csv",index=False)
exp4_primary_df = pd.DataFrame(exp4_primary,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_primary_df.to_csv("exp4a_primary.csv",index=False)
exp4_secondary_df = pd.DataFrame(exp4_secondary,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_secondary_df.to_csv("exp4a_secondary.csv",index=False)
exp4_distinguish_df = pd.DataFrame(exp4_distinguish,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
exp4_distinguish_df.to_csv("exp4a_distinguish.csv",index=False)

########################################################################

########################################################################
# Compute detection power for experiment 4
#
null_thresholds_df = pd.read_csv("null_thresholds.csv")
exp4_scores_df = pd.read_csv("exp4a_scores.csv")
methods = ['rp2','rp3','mcd2','mcd3']
detection_power = np.zeros([60,2+len(methods)])
therow = 0
for q1 in np.linspace(1.01,1.2,20):
  for q2 in [1+0.25*(q1-1),1+0.5*(q1-1),1+0.75*(q1-1)]:
    detection_power[therow,0] = q1
    detection_power[therow,1] = q2
    thecol = 2
    for themethod in methods:
        thethresholds = null_thresholds_df.loc[:,themethod]
        thescores = exp4_scores_df[(np.abs(exp4_scores_df['q1'] - q1) < 1E-6) & (np.abs(exp4_scores_df['q2'] - q2) < 1E-6)].loc[:,themethod]
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
        detection_power[therow,thecol] = thepower/len(thescores)
        thecol += 1
    therow += 1

detection_power_df = pd.DataFrame(detection_power,columns=['q1','q2','rp2','rp3','mcd2','mcd3'])
detection_power_df.to_csv("exp4a_detection_power.csv",index=False)
#
########################################################################
