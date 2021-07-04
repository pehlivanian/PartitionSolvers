import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from scipy.optimize import curve_fit
import scipy.odr.odrpack as odr


BY_N = True
BY_T = False

def power_law(x,a,b):
    return a*x**b

def power_law_ODR(C,x):
    return C[0]*x**C[1]

def power_law_const(x,a,b,c):
    return a + b*x**c

def power_law_const_ODR(C,x):
    return C[0] + C[1] * x**C[2]

if (BY_N):
    Ts = (100,200,300,400,500)
    for T in Ts:

        # path = './times_100_30.dat'
        # path = './times_1100_100_100.dat'
        # path = './times_1100_1000_100_10.dat'
        # path = './times_2100_400_10_10.dat'
        # path = './t_2500_{}_10_1_new_new.dat'.format(T)
        path = './t_5000_{}_100_1.dat'.format(T)
        n = 5000;stride=T;partsStride=1
        
        df = pd.read_csv(path, sep=',', header=None, index_col=None)
        df = df.drop(columns=[df.columns[-1]])
        
        n_values = [i for i in range(T+5,n+5,stride)]
        df = df.loc[n_values]
        df = df.replace(0, np.nan)
        
        # xaxis = [i for i in range(T+5,n-5,stride)] 
        # yaxis = df[00][xaxis]
        
        # xaxis = [i for i in df.columns if not (i-2)%partsStride]
        # factor = None
        # for rowNum in range(stride+5,n,stride):
        #     label ='n : '+str((rowNum-5)*100)
        #     yaxis=df.loc[rowNum].values[xaxis]
        #     if factor is None:
        #         factor = yaxis[0]
        #     yaxis = yaxis * (1./factor)
        #     plot.plot(xaxis, yaxis, label=label)
        
        xaxis = df.index.to_list()
        yaxis = list(df[T].values)
        yaxis = [x/1000/1000 for x in yaxis]
        
        # first_el = yaxis[0]
        # yaxis = [x/first_el for x in yaxis]
        plot.plot(xaxis, yaxis, label='t = {}'.format(str(T)))

        if (True):
            popt,pcov = curve_fit(power_law_const, xaxis, yaxis)
            func = odr.Model(power_law_const_ODR)
            # popt,pcov = curve_fit(power_law, xaxis, yaxis)
            # func = odr.Model(power_law_ODR)            
            odrdata = odr.Data(xaxis,yaxis)
            odrmodel = odr.ODR(odrdata,func,beta0=popt,maxit=500,ifixx=[0])
            o = odrmodel.run()
        print('T: {} o.beta: {}'.format(T, o.beta))

    plot.xlabel('n')
    plot.ylabel('CPU time (seconds)')
    plot.title('Run time')
    plot.grid('on')
    plot.legend()
    plot.pause(1e-3)
    plot.savefig('Runtime_by_n.png')
    plot.close()
else:
    ns = (1000,2000,3000,4000,5000,6000,7000,8000,9000,10000)
    for n in ns:

        # path = './times_100_30.dat'
        # path = './times_1100_100_100.dat'
        # path = './times_1100_1000_100_10.dat'
        # path = './times_2100_400_10_10.dat'
        path = './t2_{}_1000_100_10.dat'.format(n)
        T=1000;stride=10;partsStride=1
        
        df = pd.read_csv(path, sep=',', header=None, index_col=None)
        df = df[n:]

        n_values = [i for i in range(10,T,stride)]
        df = df[n_values]

        xaxis = df.columns.to_list()
        yaxis = list(df.values[0])
        plot.plot(xaxis, yaxis, label='n = {}'.format(str(n)))
