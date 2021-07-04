import numpy as np
import solverSWIG_DP
import solverSWIG_LTSS
import proto

#############################
# Poisson only

def thescore(C,B,risk_partitioning_objective):
    if risk_partitioning_objective:
        return C*np.log(C/B)
    if (C <= B):
        return 0
    return C*np.log(C/B)+B-C

def risk_partitioning_allpython(C,B,Tmax,risk_partitioning_objective):
    multclustscore = 0
    ratios = C/B
    forsorting = ratios.argsort()
    sorted_C = C[forsorting]
    sorted_B = B[forsorting]
    sorted_ratios = ratios[forsorting]
    C = sorted_C
    B = sorted_B
    ratios = sorted_ratios
    if risk_partitioning_objective:
        offset = np.sum(C)*np.log(np.sum(C)/np.sum(B)) 
    else:
        offset = 0

    maxes = {}
    argmaxes = {}

    # maxes[(n,T)] is the maximum score for dividing elements n..(N-1) into T partitions
    # argmaxes[(n,T)] is the start of the second partition, i.e., first partition goes from n..(argmaxes[(n,T)]-1)

    # base case.  maxes[(n,1)] = (\sum_{i=n..N-1} C_i)^2/(\sum_{i-n..N-1} B_i)
    cumulative_C = 0.
    cumulative_B = 0.
    for n in np.arange(N-1,-1,-1):
        cumulative_C += C[n]
        cumulative_B += B[n]
        maxes[(n,1)] = thescore(cumulative_C,cumulative_B,risk_partitioning_objective)
        if (maxes[(n,1)] > multclustscore):
            multclustscore = maxes[(n,1)]
        argmaxes[(n,1)] = N
    
    # increment case.  maxes[(n,T)] = max_{k=n+1..N-1} (\sum_{i=n..k-1} C_i)^2/(\sum_{i-n..k-1} B_k) + maxes[(k,T-1)] 
    for T in np.arange(2,Tmax+1,1):
        for n in range(N):
            bestscore = -1E10
            bestk = -1
            cumulative_C = 0.
            cumulative_B = 0.
            for k in np.arange(n+1,N):
                cumulative_C += C[k-1]
                cumulative_B += B[k-1]
                tempscore = thescore(cumulative_C,cumulative_B,risk_partitioning_objective)
                tempscore = tempscore + maxes[(k,T-1)]
                if (tempscore > bestscore):
                    bestscore = tempscore
                    bestk = k
            maxes[(n,T)] = bestscore
            if (T < Tmax) & (bestscore > multclustscore):
                multclustscore = bestscore
            argmaxes[(n,T)] = bestk

    # best partitions
    #for T in np.arange(1,Tmax+1,1):
    #    print("Best partition of size",T,"with score",maxes[(0,T)] - offset,":")
    #    current = 0
    #    for t in np.arange(T,0,-1):
    #        thenext = argmaxes[(current,t)]
    #        print("(",current,",",thenext-1,")")
    #        current = thenext
        
    #for T in np.arange(1,Tmax+1,1):
    #    print(maxes[(0,T)] - offset)
    if risk_partitioning_objective:
        return maxes[(0,Tmax)] - offset
    return multclustscore

################################

rng = np.random.RandomState(23)

num_partitions = 7
N = 178
a = proto.FArray()                  # wrapper for C++ float array type
b = proto.FArray()                  # wrapper for C++ float array type
objective_fn = 1                    # 1 ~ Poisson, 2 ~ Gaussian
risk_partitioning_objective = False # False => multiple clustering score function is used
optimized_score_calculation = False # Leave this False; only implemented for RationalScore case

a_lower_limit = 0. if objective_fn == 1 else -10.; a_higher_limit = 10.
b_lower_limit = 0.; b_higher_limit = 10.
a = rng.uniform(low=a_lower_limit, high=a_higher_limit, size=N)
b = rng.uniform(low=b_lower_limit, high=b_higher_limit, size=N)

#ratios = a/b
#forsorting = ratios.argsort()
#sorted_a = a[forsorting]
#sorted_b = b[forsorting]
#a = sorted_a
#b = sorted_b

#print(a)
#print(b)

# all_results[0] ~ size N partition
# all_results[1] ~ cumulative score
all_results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                          a,
                                          b,
                                          objective_fn,
                                          risk_partitioning_objective,
                                          optimized_score_calculation)()

print("OPTIMAL PARTITION")
print('{!r}'.format(all_results[0]))
print('SCORE: {}'.format(all_results[1]))

allpython_score = risk_partitioning_allpython(a,b,num_partitions,risk_partitioning_objective)
print("OPTIMAL PARTITION ALL PYTHON")
print('SCORE: {}'.format(allpython_score))
