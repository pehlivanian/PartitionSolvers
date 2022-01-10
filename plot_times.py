import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import curve_fit
import scipy.odr.odrpack as odr


parser = argparse.ArgumentParser(description="Generate run times plots")
parser.add_argument('n', metavar='n', type=int,
                    help='size of ground set')
parser.add_argument('T', metavar='T', type=int,
                    help='number of subsets')
parser.add_argument('nStride', metavar='MStride', type=int,
                     help='stride for range of n values')
parser.add_argument('TStride', metavar='TStride', type=int,
                    help='stride for range of T values')
parser.add_argument('CPU', metavar='CPU', type=str,
                    help='appropriate CPU info from lscpu, e.g.')
parser.add_argument('p', metavar='path', type=str,
                     help='path to raw input file')

args = parser.parse_args(sys.argv[1:])


n = args.n
T = args.T
NStride = args.nStride
TStride = args.TStride
CPUInfo = args.CPU
path = args.p

BY_N = True
BY_T = False
FIT_POWER_CURVE = True

def power_law(x,a,b):
    return a*x**b

def power_law_ODR(C,x):
    return C[0]*x**C[1]

def power_law_const(x,a,b,c):
    return a + b*x**c

def power_law_const_ODR(C,x):
    return C[0] + C[1] * x**C[2]

start_n = T + NStride
Ts = range(TStride, T+1, TStride)
Ns = range(T, n+1, NStride)

# df gives run times in microseconds
# columns ~ T values ~ number of subsets
# rows ~ n values ~ size of ground set
df = pd.read_csv(path, sep=',', header=None, index_col=None)
df = df.drop(columns=[df.columns[-1]])

df = df.loc[Ns,Ts]

if (BY_N):
    xaxis = df.index.to_list()
    xlabel = 'n ~ Size of ground set'        
    for ind in range(0,df.shape[1]):
        yaxis = [x/1000/1000 for x in df.iloc[:,ind].values]
        plot.plot(xaxis, yaxis, label='t = {}'.format(Ts[ind]))
elif (BY_T):
    df = df.T
    xaxis = df.index.to_list()
    xlabel = 'T ~ Number of subsets'        
    for ind in range(0, df.shape[1]):
        yaxis = [x/1000/1000 for x in df.iloc[:,ind].values]
        plot.plot(xaxis, yaxis, label='n = {}'.format(Ns[ind]))

plot.xlabel(xlabel)
plot.ylabel('CPU time (seconds)')
plot.title('Run Times [{}]'.format(CPUInfo))
plot.grid('on')
plot.legend()
plot.pause(1e-3)
plot.savefig('Runtimes.pdf')
plot.close()
    
if (FIT_POWER_CURVE):
    xaxis = df.index.to_list()
    # average across all cases...???
    yaxis = [y/1000/1000 for y in df.mean(axis=1).values]
    plot.plot(xaxis, yaxis, label='emprical avg runtime')
    try:
        popt,pcov = curve_fit(power_law_const, xaxis, yaxis)
    except Exception as e:
        popt = [0.0, 0.0, 1.0]
    func = odr.Model(power_law_const_ODR)
    odrdata = odr.Data(xaxis,yaxis)
    odrmodel = odr.ODR(odrdata,func,beta0=popt,maxit=500,ifixx=[0])
    try:
        o = odrmodel.run()
    except Exception as e:
        print('Exception in ODR fit')
        import pdb; pdb.set_trace()
    yfitaxis = [power_law_const_ODR(o.beta.tolist(), x) for x in xaxis]
    plot.plot(xaxis, yfitaxis, label='fit: power: {:2.4f}'.format(o.beta[2]))

    plot.xlabel(xlabel)
    plot.ylabel('CPU time (seconds)')
    plot.title('Power law fit: ({:2.4e}) + ({:2.4e})x^({:2.4f})'.format(*o.beta.tolist()))
    plot.grid('on')
    plot.legend()
    plot.pause(1e-3)
    plot.savefig('Runtimes_with_power_fit.pdf')
    plot.close()
