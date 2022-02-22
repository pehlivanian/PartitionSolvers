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
parser.add_argument('--by-N', dest='byN', action='store_true', default=True)
parser.add_argument('--by-T', dest='byT', action='store_true', default=False)

args = parser.parse_args(sys.argv[1:])

n = args.n
T = args.T
NStride = args.nStride
TStride = args.TStride
CPUInfo = args.CPU
path = args.p
byN = args.byN
byT = args.byT

BY_N = True
BY_T = True
FIT_POWER_CURVE = True
POWER_LAW_WITH_CONSTANT = True

def plot_single(beta, xlabel, destPath):
    if (POWER_LAW_WITH_CONSTANT):
        plot.plot(xaxis, yfitaxis, label='fit: power: {:2.4f}'.format(o.beta[2]))
    else:
        plot.plot(xaxis, yfitaxis, label='fit: power: {:2.4f}'.format(o.beta[1]))
    plot.xlabel(xlabel)
    plot.ylabel('CPU time (seconds)')
    if (POWER_LAW_WITH_CONSTANT):
        plot.title('Power law fit: ({:2.4e}) + ({:2.4e})x^({:2.4f})'.format(*o.beta.tolist()))
    else:
        plot.title('Power law fit: (({:2.4e})x^({:2.4f})'.format(*o.beta.tolist()))
    plot.grid('on')
    plot.legend()
    plot.pause(1e-3)
    plot.savefig(destPath)
    plot.close()

def plot_multiple(xlabel, CPUInfo, destPath):
    plot.xlabel(xlabel)
    plot.ylabel('CPU time (seconds)')
    plot.title('Run Times [{}]'.format(CPUInfo))
    plot.grid('on')
    plot.legend()
    plot.pause(1e-3)
    plot.savefig(destPath)
    plot.close()
    
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
df_bkup = df.copy()

if (BY_N):
    xaxis = df.index.to_list()
    xlabel = 'n ~ Size of ground set'
    maxLabels = 12; labelCount = 1        
    for ind in range(0,df.shape[1]):
        yaxis = [x/1000/1000 for x in df.iloc[:,ind].values]
        if labelCount <= maxLabels:
            plot.plot(xaxis, yaxis, label='t = {}'.format(Ts[ind]))
        else:
            plot.plot(xaxis, yaxis)
        labelCount+=1
plot_multiple(xlabel, CPUInfo, 'Runtimes_by_n.pdf')

if (BY_T):
    df = df_bkup.T
    xaxis = df.index.to_list()
    xlabel = 'T ~ Number of subsets'
    maxLabels = 12; labelCount = 1    
    for ind in range(0, df.shape[1]):
        yaxis = [x/1000/1000 for x in df.iloc[:,ind].values]
        if labelCount <= maxLabels:
            plot.plot(xaxis, yaxis, label='n = {}'.format(Ns[ind]))
        else:
            plot.plot(xaxis, yaxis)
        labelCount+=1
plot_multiple(xlabel, CPUInfo, 'Runtimes_by_T.pdf')

if (FIT_POWER_CURVE):
    df = df_bkup
    for var in ('n', 'T'):
        if var == 'T':
            df = df.T
            xlabel = 'T ~ Number of subsets'
            destPath = 'Runtimes_with_power_fit_by_T.pdf'            
        else:
            xlabel = 'N ~ Size of ground set'            
            destPath = 'Runtimes_with_power_fit_by_n.pdf'
        xaxis = df.index.to_list()
        
        # average across all cases...???
        yaxis = [y/1000/1000 for y in df.mean(axis=1).values]
        plot.plot(xaxis, yaxis, label='emprical avg runtime')
        if (POWER_LAW_WITH_CONSTANT):
            try:
                popt,pcov = curve_fit(power_law_const, xaxis, yaxis)
            except Exception as e:
                popt = [0.0, 0.0, 1.0]
            func = odr.Model(power_law_const_ODR)
        else:
            try:
                popt,pcov = curve_fit(power_law, xaxis, yaxis)
            except Exception as e:
                popt = [0.0, 0.0, 1.0]
            func = odr.Model(power_law_ODR)
        odrdata = odr.Data(xaxis,yaxis)
        odrmodel = odr.ODR(odrdata,func,beta0=popt,maxit=500,ifixx=[0])
        try:
            o = odrmodel.run()
        except Exception as e:
            print('Exception in ODR fit')
        if (POWER_LAW_WITH_CONSTANT):
            yfitaxis = [power_law_const_ODR(o.beta.tolist(), x) for x in xaxis]
        else:
            yfitaxis = [power_law_ODR(o.beta.tolist(), x) for x in xaxis]
        plot_single(o.beta, xlabel, destPath)

