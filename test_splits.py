import os
import sys
import numpy as np
import pickle
import multiprocessing
from scipy.special import comb
from functools import partial
from itertools import chain, islice, combinations
import matplotlib.pyplot as plot
from scipy.spatial import ConvexHull, Delaunay

SEED = 1856
rng = np.random.RandomState(SEED)

def subsets(ns):
    return list(chain(*[[[list(x)] for x in combinations(range(ns), i)] for i in range(1,ns+1)]))

def knuth_partition(ns, m):
    if m == 1:
        return [[ns]]
    
    def visit(n, a):
        ps = [[] for i in range(m)]
        for j in range(n):
            ps[a[j + 1]].append(ns[j])
        return ps

    def f(mu, nu, sigma, n, a):
        if mu == 2:
            yield visit(n, a)
        else:
            for v in f(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v
        if nu == mu + 1:
            a[mu] = mu - 1
            yield visit(n, a)
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                yield visit(n, a)
        elif nu > mu + 1:
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = mu - 1
            else:
                a[mu] = mu - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] > 0:
                a[nu] = a[nu] - 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v

    def b(mu, nu, sigma, n, a):
        if nu == mu + 1:
            while a[nu] < mu - 1:
                yield visit(n, a)
                a[nu] = a[nu] + 1
            yield visit(n, a)
            a[mu] = 0
        elif nu > mu + 1:
            if (a[nu] + sigma) % 2 == 1:
                for v in f(mu, nu - 1, 0, n, a):
                    yield v
            else:
                for v in b(mu, nu - 1, 0, n, a):
                    yield v
            while a[nu] < mu - 1:
                a[nu] = a[nu] + 1
                if (a[nu] + sigma) % 2 == 1:
                    for v in f(mu, nu - 1, 0, n, a):
                        yield v
                else:
                    for v in b(mu, nu - 1, 0, n, a):
                        yield v
            if (mu + sigma) % 2 == 1:
                a[nu - 1] = 0
            else:
                a[mu] = 0
        if mu == 2:
            yield visit(n, a)
        else:
            for v in b(mu - 1, nu - 1, (mu + sigma) % 2, n, a):
                yield v

    n = len(ns)
    a = [0] * (n + 1)
    for j in range(1, m + 1):
        a[n - m + j] = j - 1
    return f(m, n, 0, n, a)

def Bell_n_k(n, k):
    ''' Number of partitions of  1,...,n} into
        k subsets, a restricted Bell number
    '''
    if (n == 0 or k == 0 or k > n): 
        return 0
    if (k == 1 or k == n): 
        return 1
      
    return (k * Bell_n_k(n - 1, k) + 
                Bell_n_k(n - 1, k - 1))

def _Mon_n_k(n, k):
    return comb(n-1, k-1, exact=True)

def slice_partitions(partitions):
    # Have to consume it; can't split work on generator
    partitions = list(partitions)
    num_partitions = len(partitions)
    
    bin_ends = list(range(0,num_partitions,int(num_partitions/NUM_WORKERS)))
    bin_ends = bin_ends + [num_partitions] if num_partitions/NUM_WORKERS else bin_ends
    islice_on = list(zip(bin_ends[:-1], bin_ends[1:]))

    rng.shuffle(partitions)
    slices = [list(islice(partitions, *ind)) for ind in islice_on]
    return slices

def reduce(return_values, fn):
    return fn(return_values, key=lambda x: x[0])

def simple_power_score_fn(a,b,gamma,p):
    return np.sum(a[p])**gamma

def power_score_fn(a,b,gamma,p):
    return np.sum(a[p])**gamma/np.sum(b[p])

def double_power_score_fn(a,b,gamma,p):
    return (np.sum(a[p])**gamma)/(np.sum(b[p])**1.00)

def neg_x_times_y(a,b,gamma,p):
    return -1.*np.sum(a[p])*np.sum(b[p])

def sum_of_powers_of_x_fn(a,b,gamma,p):
    return np.sum(a[p])**gamma

def sum_of_powers_fn(a,b,gamma,p):
    return np.sum(a[p])**gamma + np.sum(b[p])**gamma

def neg_sum_of_powers_fn(a,b,gamma,p):
    return -np.sum(a[p])**gamma - np.sum(b[p])**gamma

def power_of_sums_fn(a,b,gamma,p):
    return np.power(np.sum(a[p])+np.sum(b[p]), 2.)

def sqrt_of_sum_of_powers_fn(a,b,gamma,p):
    return np.sqrt(np.sum(a[p])**gamma + np.sum(b[p])**gamma)

def linear_fn(a,b,gamma,p):
    return np.sum(a[p])

def sum_power(a,b,gamma,p):
    return (np.sum(a[p]) + np.sum(b[p]))**gamma

def mixed_exp_fn(a,b,gamma,p):
    return np.exp(-np.sum(b[p]))*np.power(2.0,np.sum(a[p]))
    # return np.exp(-q*np.sum(b[p]))*np.power(q,np.sum(a[p]))/np.exp(-np.sum(b[p]))
    
def Gaussian_llr(a,b,gamma,p):
    asum = np.sum(a[p])
    bsum = np.sum(b[p])
    if asum > bsum:
        return np.power((asum-bsum),2.0)/(2*bsum)
    else:
        return 0
    
def Poisson_llr(a,b,gamma,p):
    asum = np.sum(a[p])
    bsum = np.sum(b[p])
    if asum > bsum:
        return asum*np.log(asum/bsum) + bsum - asum
    else:
        return 0
    
def log_score_fn(a,b,gamma,p):
    return -1.*np.log(1. + np.sum(a[p]))

def log_prod_fn(a,b,gamma,p):
    return np.log((1.+np.sum(a[p]))*(1.+np.sum(b[p])))

class Task(object):
    def __init__(self, a, b, partition, power=2, cond=max, score_fn=power_score_fn, fnArgs=()):
        self.partition = partition
        self.cond = cond
        self.score_fn = score_fn
        self.task = partial(self._task, a, b, power)

    def __call__(self):
        return self.task(self.partition)

    def _task(self, a, b, power, partitions, report_each=1000):

        if self.cond == min:
            max_sum = float('inf')
        else:
            max_sum = float('-inf')            
        
        arg_max = -1
        
        for ind,part in enumerate(partitions):
            val = 0
            part_val = [0] * len(part)
            # print('PARTITION: {!r}'.format(part))
            for part_ind, p in enumerate(part):
                fnArgs = (a,b,power,p)
                part_sum = self.score_fn(*fnArgs)
                part_val[part_ind] = part_sum
                val += part_sum
                # print('    SUBSET: {!r} SUBSET SCORE: {:4.4f}'.format(p, part_sum))
            if self.cond(val, max_sum) == val:
                max_sum = val
                arg_max = part
            # print('    PARTITION SCORE: {:4.4f}'.format(val))
        # print('MAX PARTITION SCORE: {:4.4f}, MAX_PARTITION: {}'.format(max_sum, list(arg_max)))
        return (max_sum, arg_max)

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
            # print('{} : Fetched task of type {}'.format(proc_name, type(task)))
            if isinstance(task, EndTask):
                # print('Exiting: {}'.format(proc_name))
                self.task_queue.task_done()
                break
            result = task()
            self.task_queue.task_done()
            self.result_queue.put(result)

def plot_convex_hull(a0,
                     b0,
                     fig1=None,
                     ax1=None,
                     score_fn=power_score_fn,
                     plot_extended=False,
                     plot_symmetric=False,
                     show_plot=True,
                     show_contours=True,
                     label_interior_points=True,
                     label_no_points=False,
                     include_colorbar=True):
    NUM_AXIS_POINTS = 201

    def in_hull(dhull, x, y):
        return dhull.find_simplex((x,y)) >= 0

    def F_symmetric(x,y,gamma):
        import warnings
        warnings.filterwarnings('ignore')
        ret = x**gamma/y + (Cx-x)**gamma/(Cy-y)
        # ret = (np.log((1. + x)*(1. + y)) + np.log((1. + (Cx-x))*(1 + (Cy-y))))
        # ret = (x**gamma)/(y**1.0) + ((Cx-x)**gamma)/((Cy-y)**1.0)
        # ret = np.exp(-2.0*y)*np.power(2.0,x)/np.exp(-y) + np.exp(-2.0*(Cy-y))*np.power(2.0,(Cx-x))/np.exp(-(Cy-y))
        # ret = (x**gamma) + ((Cx-x)**gamma)
        # ret = F_orig(x,y,gamma) + F_orig(Cx-x,Cy-y,gamma)
        
        warnings.resetwarnings()
        return ret

    def F_orig(x,y,gamma):
        import warnings
        warnings.filterwarnings('ignore')
        ret = x**gamma/y
        # ret = -1.*np.log(1. + x)
        # ret = (x**gamma)/(y**1.0)
        # ret = np.exp(-2.0*y)*np.power(2.0,x)/np.exp(-y)        
        # ret = x**gamma
        # XXX
        # if x > y:
        # ret = x*np.log(x/y) + y - x
        # else:
        #     ret = 0
        # ret = np.log((1.+x)*(1.+y))
        warnings.resetwarnings()
        return ret

    ind = np.argsort(a0**PRIORITY_POWER/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))

    pi = subsets(len(a))
    if not plot_extended:
        mp = [p[0] for p in pi if len(p[0]) == len(a0)]
        pi.remove(mp)

    if plot_symmetric:
        F = F_symmetric
        title = 'F Symmetric, '
    else:
        F = F_orig
        title = 'F Non-Symmetric, '

    if plot_extended:
        title += 'Full Hull'
    else:
        title += 'Constrained Hull'

    title += '  Case: (n, t) = : ( ' + str(len(a0)) + ', ' + str(PARTITION_SIZE) + ' )'
        
    X = list()
    Y = list()
    txt = list()

    for subset in pi:
        s = subset[0]
        X.append(np.sum(a[s]))
        Y.append(np.sum(b[s]))
        txt.append(str([x+1 for x in s]))

    if plot_extended:
        X = [0.] + X
        Y = [0.] + Y
        txt = ['-0-'] + txt

    points = np.stack([X,Y]).transpose()

    Xm, XM = np.min(X), np.max(X)
    Ym, YM = np.min(Y), np.max(Y)
    Cx, Cy = np.sum(a), np.sum(b)        

    hull = ConvexHull(points)
    vertices = [points[v] for v in hull.vertices]
    dhull = Delaunay(vertices)

    if show_plot:
        cmap = plot.cm.RdYlBu
        if fig1 is None: # Assume ax1 is None also
            fig1 = plot.figure(figsize=(10,8))
            ax1 = fig1.add_subplot(1,1,1)            
            # fig1, ax1 = plot.subplots(1,1)

        xaxis = np.linspace(Xm, XM, NUM_AXIS_POINTS)
        yaxis = np.linspace(Ym, YM, NUM_AXIS_POINTS)
        xaxis, yaxis = xaxis[:-1], yaxis[:-1]
        Xgrid,Ygrid = np.meshgrid(xaxis, yaxis)
        Zgrid = F(Xgrid, Ygrid, POWER)

        for xi,xv in enumerate(xaxis):
            for yi,yv in enumerate(yaxis):
                if in_hull(dhull, xv, yv):
                    continue
                else:
                    Zgrid[yi,xi] = -99.

        Zgrid[Zgrid < 0] = .5*(np.min(Zgrid[Zgrid>0]) + np.max(Zgrid[Zgrid>0]))

        if show_contours:
            cp = ax1.contourf(Xgrid, Ygrid, Zgrid, cmap=cmap)            
            cp.changed()
            if include_colorbar:
                fig1.colorbar(cp)
            

        ax1.scatter(X, Y)
        for i,t in enumerate(txt):
            if i in hull.vertices:
                t = t.replace('[','<').replace(']','>')
            elif label_interior_points:
                t = t.replace('[','').replace(']','')
            else:
                t = ''
            t = t.replace(', ', ',')
            if label_no_points:
                t = ''
            ax1.annotate(t, (X[i], Y[i]))

        for simplex in hull.simplices:
            ax1.plot(points[simplex,0], points[simplex,1], 'k-')

        plot.title(title)    

    vertices_txt = [txt[v] for v in hull.vertices]
    header = 'FULL_' if plot_extended else 'CONST_'
    header += 'SYM' if plot_symmetric else 'NONSYM'
    print('{:12} : {!r}'.format(header,vertices_txt))

    return fig1, ax1, vertices_txt
    

def plot_constrained_unconstrained_overlay(a0, b0, plot_constrained=True, score_fn=power_score_fn, show_plot=True, save_plot=False):
    for labeled_case in (True, False):
        for symmetric_case in (False, True):
            fig1, ax1,vert_const_asym = plot_convex_hull(a0,
                                                         b0,
                                                         score_fn=power_score_fn,
                                                         plot_extended=False,
                                                         plot_symmetric=symmetric_case,
                                                         show_plot=show_plot,
                                                         label_interior_points=False,
                                                         label_no_points=not labeled_case,
                                                         include_colorbar=True)
            fig2, ax2,vert_ext_asym = plot_convex_hull(a0,
                                                       b0,
                                                       fig1,
                                                       ax1,
                                                       score_fn=power_score_fn,
                                                       plot_extended=True,
                                                       plot_symmetric=symmetric_case,
                                                       show_plot=show_plot,
                                                       label_interior_points=False,
                                                       label_no_points=not labeled_case,                                                 
                                                       include_colorbar=False)
            
            title = 'Constrained and unconstrained hull partition polytopes, '
            
            if symmetric_case:
                title += 'symm. score, '
                if labeled_case:
                    path = 'Hull_overlay_Sym_labeled_{}_{}.pdf'.format(len(a0), PARTITION_SIZE)
                else:
                    path = 'Hull_overlay_Sym_unlabeled_{}_{}.pdf'.format(len(a0), PARTITION_SIZE)
            else:
                if labeled_case:
                    path = 'Hull_overlay_Nonsym_labeled_{}_{}.pdf'.format(len(a0), PARTITION_SIZE)
                else:
                    path = 'Hull_overlay_Nonsym_unlabeled_{}_{}.pdf'.format(len(a0), PARTITION_SIZE)

            title+= 'n = {}'.format(len(a0))
            plot.title(title)
            
            plot.xlabel('X')
            plot.ylabel('Y')
            
            plot.savefig(path)
            plot.close()
    import pdb; pdb.set_trace()

def plot_polytope(a0, b0, plot_constrained=True, score_fn=power_score_fn, show_plot=True, save_plot=False):

    fig1, ax1,vert_const_asym = plot_convex_hull(a0,
                                                 b0,
                                                 score_fn=power_score_fn,
                                                 plot_extended=False,
                                                 plot_symmetric=False,
                                                 show_plot=show_plot)
    if save_plot:
        plot.savefig('plot1.pdf')
    else:
        plot.pause(1e-3)
        
    fig2, ax2,vert_const_sym = plot_convex_hull(a0,
                                                b0,
                                                score_fn=power_score_fn,
                                                plot_extended=False,
                                                plot_symmetric=True,
                                                show_plot=show_plot)

    if save_plot:
        plot.savefig('plot2.pdf')
    else:
        plot.pause(1e-3)
        
    fig3, ax3,vert_ext_asym = plot_convex_hull(a0,
                                               b0,
                                               score_fn=power_score_fn,
                                               plot_extended=True,
                                               plot_symmetric=False,
                                               show_plot=show_plot)
    if save_plot:
        plot.savefig('plot3.pdf')
    else:
        plot.pause(1e-3)
        
    fig4, ax4,vert_ext_sym = plot_convex_hull(a0,
                                              b0,
                                              score_fn=power_score_fn,
                                              plot_extended=True,
                                              plot_symmetric=True,
                                              show_plot=show_plot)
    if save_plot:
        plot.savefig('plot4.pdf')
    else:
        plot.pause(1e-3)    

    if show_plot:
        plot.close()
        plot.close()
        plot.close()
        plot.close()

    return vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym

def optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER, CONSEC_ONLY=False, cond=max):
    ind = np.argsort(a0**PRIORITY_POWER/b0)
    (a,b) = (seq[ind] for seq in (a0,b0))

    # XXX
    # if num_mon_partitions > 100:
    # Doesn't work in certain cases?
    if False:
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        workers = [Worker(tasks, results) for i in range(NUM_WORKERS)]
        num_slices = len(slices)
        
        for worker in workers:
            worker.start()

        for i,slice in enumerate(slices):
            tasks.put(Task(a, b, slice, power=POWER, cond=cond, score_fn=SCORE_FN))

        for i in range(NUM_WORKERS):
            tasks.put(EndTask())

        tasks.join()
        
        allResults = list()
        slices_left = num_slices
        while not results.empty():
            result = results.get(block=True)
            allResults.append(result)
            slices_left -= 1
    else:
        partitions = list(knuth_partition(range(0, len(a)), PARTITION_SIZE))
        if CONSEC_ONLY:
            partitions = [p for p in partitions if all(np.diff([x for x in chain.from_iterable(p)]) == 1)]
        allResults = [Task(a, b, partitions, power=POWER, cond=cond, score_fn=SCORE_FN)()]
    
            
    r_max = reduce(allResults, cond)

    return r_max

# Maximal ordered partition demonstration
if __name__ == '__main__':
    NUM_POINTS =        int(sys.argv[1]) or 3          # N
    PARTITION_SIZE =    int(sys.argv[2]) or 2          # T
    POWER =             float(sys.argv[3]) or 2.2      # gamma
    PRIORITY_POWER =    float(sys.argv[4]) or 1.0      # tau
    UNCONSTRAINED =     float(sys.argv[5]) or False    # full upper-half plane

    NUM_WORKERS = min(NUM_POINTS, multiprocessing.cpu_count() - 1)

    SCORE_FN = power_score_fn
    # SCORE_FN = log_score_fn
    # SCORE_FN = double_power_score_fn
    # SCORE_FN = sum_of_powers_of_x_fn
    # SCORE_FN = sum_of_powers_fn
    # SCORE_FN = neg_x_times_y
    # SCORE_FN = neg_sum_of_powers_fn
    # SCORE_FN = power_of_sums_fn
    # SCORE_FN = sqrt_of_sum_of_powers_fn
    # SCORE_FN = simple_power_score_fn
    # SCORE_FN = sum_power
    # SCORE_FN = Poisson_llr
    # SCORE_FN = Gaussian_llr
    # SCORE_FN = mixed_exp_fn
    # SCORE_FN = linear_fn
    # SCORE_FN = log_prod_fn
    
    num_partitions = Bell_n_k(NUM_POINTS, PARTITION_SIZE)
    num_mon_partitions = _Mon_n_k(NUM_POINTS, PARTITION_SIZE)
    partitions = knuth_partition(range(0, NUM_POINTS), PARTITION_SIZE)
    
    slices = slice_partitions(partitions)

    trial = 0
    bad_cases = 0
    while True:
        # a0 = rng.choice(range(1,11), NUM_POINTS, True)
        # b0 = rng.choice(range(1,11), NUM_POINTS, True)

        a0 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))
        b0 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))

        if UNCONSTRAINED:
            a0 = rng.uniform(low=-10., high=10.0, size=int(NUM_POINTS))
            b0 = rng.uniform(low=0., high=10.0, size=int(NUM_POINTS))                

        a0 = np.round(a0, 8)
        b0 = np.round(b0, 8)

        sortind = np.argsort(a0**PRIORITY_POWER/b0)
        a0 = a0[sortind]
        b0 = b0[sortind]

        r_max_raw = optimize(a0, b0, PARTITION_SIZE, POWER, NUM_WORKERS, PRIORITY_POWER)

        # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)
        # plot_constrained_unconstrained_overlay(a0, b0, score_fn=SCORE_FN, show_plot=True)        

        # print('=====')

        if False:
            print('TRIAL: {} : max_raw: {:4.6f} pttn: {!r}'.format(trial, *r_max_raw))

        test_parts = [range(0,i) for i in range(1,len(a0))] + [list(range(0,len(a0)))] + [range(i,len(a0)) for i in range(1,len(a0))]
        test_cparts = [list(set(range(0,len(a0))).difference(set(test_part))) for test_part in test_parts]
        comp_parts = [(list(x[0]),list(x[1])) for x in zip(test_parts[:len(a0)-1][::-1],test_cparts[:len(a0)-1])]        
        all_scores = [(list(test_part),SCORE_FN(a0,b0,POWER,test_part)) for test_part in test_parts]
        all_sym_scores = [(list(test_part),SCORE_FN(a0, b0,POWER,test_part)+SCORE_FN(a0,b0,POWER,test_cpart)) \
                          for test_part,test_cpart in zip(test_parts,test_cparts)]
        all_skew_scores = [(list(test_part),SCORE_FN(a0, b0,POWER,test_part)-SCORE_FN(a0,b0,POWER,test_cpart)) \
                           for test_part,test_cpart in zip(test_parts,test_cparts)]
        all_comp_scores = [(list(x[0]),list(x[1]), SCORE_FN(a0,b0,POWER,x[0])-SCORE_FN(a0,b0,POWER,x[1])) for x in comp_parts]
        # all_asc_scores = [(list(x[1]),SCORE_FN(a0,b0,POWER,x[0])) for x in comp_parts]
        # all_dec_scores = [(list(x[0]),SCORE_FN(a0,b0,POWER,x[1])) for x in comp_parts]

        if (False):
            print('ALL_SCORES')
            print('==========')
            [print('{0:>36} {1:>12}'.format(str(o[0]),str(o[1]))) for o in all_scores]
            print('ALL_SYM_SCORES')
            print('==============')
            [print('{0:>36} {1:>12}'.format(str(o[0]),str(o[1]))) for o in all_sym_scores]
            print('ALL_SKEW_SCORES')
            print('==============')
            [print('{0:>36} {1:>12}'.format(str(o[0]),str(o[1]))) for o in all_skew_scores]
            print('ALL_COMP_SCORES')
            print('==============')
            [print('{0:>36} {1:>36} {2:>12}'.format(str(o[0]),str(o[1]),str(o[2]))) for o in all_comp_scores]

        # if np.max([x[1] for x in all_asc_scores]) > np.min([x[1] for x in all_dec_scores]):
        #     import pdb
        #     pdb.set_trace()
            
        # if (np.any(np.array([x[2] for x in all_comp_scores])>0)):
        #     import pdb
        #     pdb.set_trace()
        
        if (False):
            ss_asc = [list(range(0,i)) for i in range(1,len(a0))]
            ss_des = [list(range(i,len(a0))) for i in range(len(a0)-1,-1,-1)]
            ss = ss_asc + ss_des
            
            # res = Task(a0,b0,ss,power=POWER,cond=max,score_fn=SCORE_FN)()
            consec_scores = [(p,SCORE_FN(a0,b0,POWER,p)) for p in ss_asc]
            # [print((p,SCORE_FN(a0,b0,POWER,p))) for p in ss]
            # print('OPTIMIZATION OVER ALL PARTITIONS')
            # print('================================')        
            optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER) for i in range(1, 1+len(a0))]
            # print('OPTIMIZATION OVER CONSECUTIVE PARTITIONS')
            # print('========================================')
            con_optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER, CONSEC_ONLY=True) for i in range(1, 1+len(a0))]
            # print('SUMMARY')
            # print('=======')
            # print('a0 = {}'.format(a0))
            # print('b0 = {}'.format(b0))        
            
            # import pdb
            # pdb.set_trace()
            # if 1+np.argmax([SCORE_FN(a0,b0,POWER,p) for p in ss]) == len(ss):
            # if 1+np.argmax([SCORE_FN(a0,b0,POWER,p) for p in ss]) == 0:            
            #     import pdb
            #     pdb.set_trace()
            
            if any(np.diff([x[1] for x in consec_scores])<0):
                print('MONOTONICITIY VIOLATED')
                # import pdb
                # pdb.set_trace()
                
            if not all(np.diff([o[1][1][0] for o in con_optim_all[1:]]) <= 0):
                print('INTERLEAVING VIOLATED')            
                # import pdb
                # pdb.set_trace()

        # Check quasiconvexity
        optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER) for i in range(1, 1+len(a0))]
        con_optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER, CONSEC_ONLY=True) for i in range(1, 1+len(a0))]        
        con_scores = [o[0] for o in con_optim_all]
        conv = [x[0]-min([x[1],x[2]]) for x in zip(con_scores[1:-1], con_scores[:-2], con_scores[2:])]
        # conv2 = [2*x[0]-(x[1]+x[2]) for x in zip(con_scores[1:-1], con_scores[:-2], con_scores[2:])]

        # if not all([x>0 for x in conv]): # quasiconvexity
        # if not all(np.diff(con_scores)<=0.): # decreasing
        if True:
            import matplotlib.pyplot as plot
            # yaxis = [o[0] for o in optim_all]
            yaxis = [o[0] for o in con_optim_all]
            xaxis = list(range(1,1+len(yaxis)))
            # plot.plot(xaxis, yaxis)
            # plot.pause(1e-3)
            # import pdb
            # pdb.set_trace()
            # plot.close()

        if all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1):
            print('OPTIMAL PARTITION: {}'.format(r_max_raw[1]))
            print('=================')            

        # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)
        # import pdb; pdb.set_trace()

        plot_constrained_unconstrained_overlay(a0, b0, score_fn=SCORE_FN, show_plot=True)        
        import pdb; pdb.set_trace()
        
        try:
            # assert False
            assert all(np.diff(list(chain.from_iterable(r_max_raw[1]))) == 1)
            # assert np.max([o[0] for o in con_optim_all]) >= np.max([o[0] for o in optim_all])
        except AssertionError as e:

            print('OPTIMAL PARTITION ***: {}'.format(r_max_raw[1]))
            print('=================')

            vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=False)
            if len(set(vert_const_sym).difference(set(vert_ext_sym))) == 0:
                print( set(vert_const_sym).difference(set(vert_ext_sym)) )
                import pdb
                pdb.set_trace()
            
            # optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER) for i in range(1, 1+len(a0))]

            # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)
            # ss = [list(range(0,i)) for i in range(1,len(a0))] + [list(range(i,len(a0))) for i in range(len(a0)-1,-1,-1)]
            # _ = [print((p,SCORE_FN(a0,b0,POWER,p))) for p in ss]                        
            # res = Task(a0,b0,ss,power=POWER,cond=max,score_fn=SCORE_FN)()

            
            if True:
                print('OPTIMIZATION OVER ALL PARTITIONS')
                print('================================')        
                optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER) for i in range(1, 1+len(a0))]
                print('OPTIMIZATION OVER CONSECUTIVE PARTITIONS')
                print('========================================')
                con_optim_all = [optimize(a0, b0, i, POWER, NUM_WORKERS, PRIORITY_POWER, CONSEC_ONLY=True) for i in range(1, 1+len(a0))]
                print('SUMMARY')
                print('=======')
                print('a0 = {}'.format(a0))
                print('b0 = {}'.format(b0))

            if False:
                def convex_comb(S):
                    CONSEC_SETS_ONLY = False
                    CONSEC_SPLITTING_SETS_ONLY = True
                    
                    b = np.array([np.sum(a0[S]), np.sum(b0[S]), 1.]).reshape((3,1))
                    sets = subsets(len(a0))
                    if CONSEC_SETS_ONLY:
                        sets = [s for s in sets if np.all(np.diff(s)==1)]
                    if CONSEC_SPLITTING_SETS_ONLY:
                        sets = [s for s in sets if np.all(np.diff(s)==1)]
                        sets = [s for s in sets if s[0][0]==0 or s[0][-1]==-1+len(a0)]
                
                

                import pdb
                pdb.set_trace()
                    
                count = 0
                while (count < 100):
                    m,n,o = rng.choice(len(sets),3, replace=False)
                    s1,s2,s3 = sets[m],sets[n],sets[o]
                    p1 = np.array([np.sum(a0[s1]), np.sum(b0[s1])])
                    p2 = np.array([np.sum(a0[s2]), np.sum(b0[s2])])
                    p3 = np.array([np.sum(a0[s3]), np.sum(b0[s3])])
                    A = np.concatenate([np.stack([p1,p2,p3],axis=1), np.array([1.,1.,1.]).reshape((1,3))])
                    x = np.linalg.solve(A, b)
                    if np.all(x>0) and not np.any(np.isclose(x,1)):
                        c = np.array([SCORE_FN(a0,b0,POWER,p) for p in (s1,s2,s3)])
                        cc = np.dot(x.T, c)
                        if cc < SCORE_FN(a0,b0,POWER,S):
                            print('FAIL')
                            import pdb
                            pdb.set_trace()
                        count += 1
                        print('count: {}'.format(count))
                
            def convex_closure(p, st=None, numSummands=2):
                CONSEC_SETS_ONLY = False
                CONSEC_NONSPLITTING_SETS_ONLY = True
                CONSEC_NONSPLITTING_ASC_ONLY = False
                NUM_POINTS = 3

                sets = subsets(len(a0))
                if CONSEC_SETS_ONLY:
                    sets = [s for s in sets if np.all(np.diff(s)==1)]
                if CONSEC_NONSPLITTING_SETS_ONLY:
                    sets = [s for s in sets if np.all(np.diff(s)==1)]
                    sets = [s for s in sets if s[0][0]==0 or s[0][-1]==-1+len(a0)]
                if CONSEC_NONSPLITTING_ASC_ONLY:
                    sets = [s for s in sets if np.all(np.diff(s)==1)]                    
                    sets = [s for s in sets if s[0][0]==0]
                fn = float(np.inf)
                s_opt = []
                x_opt = []
                for numPoints in (numSummands,):
                    combs = [c for c in combinations(sets, numPoints)]
                    fn = float(np.inf)
                    if numPoints == 3:
                        try:
                            p = np.concatenate([np.array(p).reshape(2,1), np.array([1.]).reshape(1,1)])
                        except Exception as e:
                            import pdb; pdb.set_trace()
                    for comb in combs:
                        ps = [np.array([np.sum(a0[s[0]]),np.sum(b0[s[0]])]) for s in comb]
                        A = np.stack(ps,axis=1).reshape((2,numPoints))
                        if numPoints == 3:
                            A = np.concatenate([A, np.array([1.,1.,1.]).reshape((1,3))])
                        x = np.linalg.solve(A, p)
                        # print('CANDIDATE: S: {} comb: {} weights: {}'.
                        #       format(st, str(comb), x.reshape(1,-1)))
                        if numSummands == 3 and [st] in comb:
                            x = np.array([0.,0.,0.]).reshape(3,1)
                            x[comb.index([st])] = 1.
                        if (numPoints == 2 and np.all(x>=0.)) or (numPoints == 3 and np.all((x>=0.)|np.isclose(x,0.))):
                            c  = np.array([SCORE_FN(a0,b0,POWER,p) for p in comb])
                            score = np.dot(x.T, c)
                            # print('S: {} comb: {} SCORE_FN: {} score-SCORE_FN: {} weights: {}'.
                            #       format(st, str(comb), SCORE_FN(a0,b0,POWER,st), score-SCORE_FN(a0,b0,POWER,st), x.reshape(1,-1)))
                            # Only fails for non-partition points
                            # XXX
                            # if score < SCORE_FN(a0,b0,POWER,st) and not np.isclose(score, SCORE_FN(a0,b0,POWER,st)):
                            #     print('FAIL_o1')
                            #     import pdb
                            #     pdb.set_trace()
                            if score < fn:
                                fn = score
                                s_opt=comb
                                x_opt = x
                if not s_opt and not st:
                    print('FAILED TO FIND COMBINATION for set: {}'.format(st))
                    import pdb; pdb.set_trace()

                if (st):
                    from scipy.optimize import linprog
                    c_ = [-np.sum(a0[st]),-np.sum(b0[st])]
                    A_ = [[np.sum(a0[0:i]),np.sum(b0[0:i])] for i in range(1,1+len(a0))] + [[np.sum(a0[i:]),np.sum(b0[i:])] for i in range(1,len(a0))]
                    b_ = [SCORE_FN(a0,b0,POWER,range(0,i)) for i in range(1,1+len(a0))] + [SCORE_FN(a0,b0,POWER,range(i,len(a0))) for i in range(1,len(a0))]
                    bounds_ = [(None,None)] * len(c_)
                    opt_poly = linprog(c_, A_ub=A_, b_ub=b_, bounds=bounds_, method='revised simplex')
                    fn_opt_poly = -opt_poly.fun
                    c_base_ = [-np.sum(a0[st]),-np.sum(b0[st])]
                    A_base_ = [[np.sum(a0[0:i]),np.sum(b0[0:i])] for i in range(1,len(a0))] + [[np.sum(a0[i:]),np.sum(b0[i:])] for i in range(1,len(a0))]
                    b_base_ = [SCORE_FN(a0,b0,POWER,range(0,i)) for i in range(1,len(a0))] + [SCORE_FN(a0,b0,POWER,range(i,len(a0))) for i in range(1,len(a0))]
                    A_eq_base_ = [[np.sum(a0), np.sum(b0)]]
                    b_eq_base_ = [[SCORE_FN(a0,b0,POWER,range(0,len(a0)))]]
                    bounds_base_ = [(None,None)] * len(c_)
                    opt_base_poly = linprog(c_base_, A_ub=A_base_, b_ub=b_base_, A_eq=A_eq_base_, b_eq=b_eq_base_, bounds=bounds_base_, method='revised simplex')
                    fn_opt_base_poly = -opt_base_poly.fun

                if (False):
                    from fillplots import plot_regions, boundary
                    from functools import partial
                    import matplotlib.pyplot as plt
                    con_parts = [list(range(0,i)) for i in range(1,1+len(a0))] + [list(range(i,len(a0))) for i in range(1,len(a0))]
                    def f_i_test(i,s):
                        p_i = np.array([np.sum(a0[con_parts[i]]), np.sum(b0[con_parts[i]])])
                        F_i = SCORE_FN(a0,b0,POWER,con_parts[i])
                        return np.dot(s, p_i) - F_i
                    def f_S_test(ss,s):
                        p_S = np.array([np.sum(a0[ss]), np.sum(b0[ss])])
                        F_S = SCORE_FN(a0,b0,POWER,ss)
                        return np.dot(s, p_S) - F_S
                    def fsub(i,x):
                        return -(A_[i][0]/A_[i][1])*x + SCORE_FN(a0,b0,POWER,con_parts[i])/A_[i][1]
                    def fS(x):
                        return -(np.sum(a0[st])/np.sum(b0[st]))*x + SCORE_FN(a0,b0,POWER,st)/np.sum(b0[st])
                    regions = list()
                    for i,_ in enumerate(con_parts):
                        regions.append( (partial(fsub, i), True))
                    # Add region corresponding to S
                    # regions.append( (fS, True))
                    xmax = int(opt_poly.x[0])+2 # 2
                    # ymax = -int(opt_poly.x[1])+2 # 2
                    ymax = SCORE_FN(a0,b0,POWER,[-1])+1
                    plotter = plot_regions( [regions], xlim=(-xmax,xmax), ylim=(-ymax,ymax))
                    for i,ineq in enumerate(plotter.regions[0].inequalities):
                        ineq.boundary.config.line_args['label'] = str(con_parts[i])
                    # plotter.regions[0].inequalities[-1].boundary.config.line_args['label'] = 'S'
                    # title = 'S: ' + str(S[0]) + ' ~ ' + str(tuple([np.round(x[0],2) for x in p])) + \
                    #         ' Binding: ' + str([con_parts[i] for i,v in enumerate(opt_poly.slack) if np.isclose(v,0.)]) \
                    #         + '  opt s: ' + str([np.round(x,2) for x in opt_poly.x])
                    title = 'Base Polytope, n = ' + str(len(a0))
                    plotter.plot()
                    plotter.ax.legend(loc='best')
                    plt.grid()
                    plt.title(title)
                    plt.xlabel('s1')
                    plt.ylabel('s2')
                    plt.pause(1e-3)
                    vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)                    

                    plot.savefig('base_polytope.pdf')

                    import pdb
                    pdb.set_trace()
                    plt.close()


                # if (fn < SCORE_FN(a0,b0,POWER,st)) and not np.isclose(fn,SCORE_FN(a0,b0,POWER,st)):
                #     print('INNER FAIL')
                #     import pdb
                #     pdb.set_trace()
                if st:
                    # print('OPTIMIZATION: {}'.format(opt_poly))
                    # XXX
                    return fn_opt_poly,s_opt,x_opt
                else:
                    return fn,s_opt,x_opt                 

                
            # import matplotlib.pyplot as plot
            # yaxis = [o[0] for o in optim_all]
            # # yaxis = [o[0] for o in con_optim_all]
            # xaxis = list(range(1,1+len(yaxis)))
            # plot.plot(xaxis, yaxis)
            # plot.pause(1e-3)
            # plot.close()

            # convex_comb([0,1,3, 5, 6])
            # Convex inequality
            # XXX
            TRIALS = 10
            sets = subsets(len(a0))
            sets.remove([[0]]);
            # sets.remove([[-1+len(a0)]]);
            sets.remove([[x for x in range(0,len(a0))]])
            con_parts = [list(range(0,i)) for i in range(1,1+len(a0))] + [list(range(i,len(a0))) for i in range(1,len(a0))]            
            for trial in range(TRIALS):
                # XXX
                CONSEC_SPLITTING_SETS_ONLY = False
                sets = subsets(len(a0))
                sets.remove([[0]]);                
                if CONSEC_SPLITTING_SETS_ONLY:
                    sets = [s for s in sets if np.all(np.diff(s)==1)]
                    sets = [s for s in sets if not s[0][0] == 0 and not s[0][-1] == -1+len(a0)]
                    print('CONSECUTIVE SUBSETS ONLY')
                S = sets[rng.choice(len(sets))]
                
                p = np.array([np.sum(a0[S[0]]), np.sum(b0[S[0]])]).reshape((2,1))
                score,s_opt,x_opt = convex_closure(p,st=S)
                # print('S: {}'.format(S))
                # print('s_opt: {}'.format(s_opt))
                # print('x_opt: {}'.format(x_opt.tolist()))
                # print('conv_cl: {} F: {}'.format(score, SCORE_FN(a0,b0,POWER,S)))


                # Extension dominates score function on partition points
                if (score < SCORE_FN(a0,b0,POWER,S[0])) and not np.isclose(score,SCORE_FN(a0,b0,POWER,S)):
                    print('FAIL_c1')
                    print('S: {} extension score: {} score: {}'.format(S[0], score, SCORE_FN(a0,b0,POWER,S[0])))
                    print('s_opt: {}'.format(s_opt))
                    print('x_opt: {}'.format(x_opt.tolist()))
                    # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)                                       
                    # import pdb
                    # pdb.set_trace()

                # Extension is subadditive on partition points
                S1=[];S2=[]
                while not S1 or not S2:
                    ind1,ind2 = rng.choice(len(sets), size=2)
                    S1 = sets[ind1][0]; S2 = sets[ind2][0];
                    S2 = list(set(S2).difference(set(S1)))
                S_union = list(set(S1).union(set(S2)))
                p1 = np.array([np.sum(a0[S1]), np.sum(b0[S1])]).reshape((2,1))
                p2 = np.array([np.sum(a0[S2]), np.sum(b0[S2])]).reshape((2,1))
                p_union = np.array([np.sum(a0[S_union]), np.sum(b0[S_union])]).reshape((2,1))
                lhs1_base = SCORE_FN(a0,b0,POWER,S1)
                lhs2_base = SCORE_FN(a0,b0,POWER,S2)
                rhs1_base = SCORE_FN(a0,b0,POWER,S_union)
                rhs2_base = 0.
                lhs1,_,_ = convex_closure(p1,st=S1)
                lhs2,_,_ = convex_closure(p2,st=S2)
                rhs1,_,_ = convex_closure(p_union,st=S_union)
                rhs2 = 0.
                if (lhs1+lhs2)<(rhs1+rhs2) and not np.isclose(lhs1+lhs2,rhs1+rhs2):
                    print('FAIL_c2')
                    # import pdb
                    # pdb.set_trace()

                if SCORE_FN(a0,b0,POWER,S) > 0:

                    # Intersection of optimal sets is nonempty
                    # if (S not in s_opt):                    
                    #     if not len(set(s_opt[0][0]).intersection(set(s_opt[1][0]))):
                    #         print('FAIL_c2')
                    #         # import pdb
                    #         # pdb.set_trace()

                    # Optimal sets sit adjacent in standard ordering
                    # if np.diff(np.sort([con_parts.index(s_[0]) for s_ in s_opt]))[0] != 1:
                    #     print('FAIL_c3')
                    #     # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)                    
                    #     # import pdb
                    #     # pdb.set_trace()

                    # Sum of weights is \leq 1
                    if np.sum(x_opt) > 1.:
                        print('FAIL_c4')
                        # import pdb
                        # pdb.set_trace()
                    coverage = set(chain.from_iterable([subs[0] for subs in s_opt]))

                    # Original set in set of optimal sets
                    # if not set(S[0]).issubset(coverage):
                    #     print('FAIL_c5')
                    #     # vert_const_asym, vert_const_sym, vert_ext_asym, vert_ext_sym = plot_polytope(a0, b0, score_fn=SCORE_FN, show_plot=True)                    
                    #     # import pdb
                    #     # pdb.set_trace()

                    # Original set is subset of intersection of optimal sets
                    # if not set(S[0]).issubset(set(s_opt[0][0]).intersection(set(s_opt[1][0]))):
                    #     print('FAIL_c6')
                    #     # import pdb
                    #     # pdb.set_trace()

                    # Score of all optimal sets smaller than score of original set
                    if (SCORE_FN(a0,b0,POWER,s_opt[0][0]) < SCORE_FN(a0,b0,POWER,S[0])) and (SCORE_FN(a0,b0,POWER,s_opt[1][0]) < SCORE_FN(a0,b0,POWER,S[0])):
                        print('FAIL_c8')
                        # import pdb
                        # pdb.set_trace()

            # Convexity of extension
            # XXX
            TRIALS = 10
            sets = subsets(len(a0))
            sets.remove([[0]])
            # sets.remove([[-1+len(a0)]])            
            for trial in range(TRIALS):
                m,n = rng.choice(len(sets), 2, replace=False)
                s1 = sets[m][0]
                s2 = sets[n][0]
                # print('DEBUG: s1: {}'.format(s1))
                # print('DEBUG: s2: {}'.format(s2))                        
                p1 = [np.sum(a0[s1]),np.sum(b0[s1])]
                p2 = [np.sum(a0[s2]),np.sum(b0[s2])]
                lam1,lam2 = rng.uniform(low=0.,high=1.,size=2)
                p1 = [lam1*x for x in p1]
                p2 = [lam2*x for x in p2]
                lam = rng.uniform(low=0.,high=1.)
                f1,_,_ = convex_closure(p1,s1)
                f2,_,_ = convex_closure(p2,s2)
                rhs = lam*f1+(1-lam)*f2
                lhs,_,_ = convex_closure([lam*x[0]+(1-lam)*x[1] for x in zip(p1,p2)])
                if lhs>rhs and not np.isclose(lhs,rhs):
                    print('FAIL_c2')
                    # import pdb
                    # pdb.set_trace()

            # Subadditivity of extension
            # XXX
            TRIALS = 10
            sets = subsets(len(a0))
            sets.remove([[0]])
            # sets.remove([[-1+len(a0)]])            
            for trial in range(TRIALS):
                s = sets[rng.choice(len(sets))][0]
                p = [np.sum(a0[s]),np.sum(b0[s])]
                p_half = [.5*x for x in p]
                lhs = convex_closure(p)[0]
                rhs = convex_closure(p_half)[0]
                if (lhs > 2*rhs) and not np.isclose(lhs,2.*rhs):
                    print('FAIL_s1')
                    import pdb
                    pdb.set_trace()

            # Homogeneity of extension
            TRIALS = 10
            sets = subsets(len(a0))
            sets.remove([[0]])
            # sets.remove([[-1+len(a0)]])            
            for trial in range(TRIALS):
                s = sets[rng.choice(len(sets))][0]
                p = [np.sum(a0[s]),np.sum(b0[s])]
                p_half = [.5*x for x in p]
                lhs = convex_closure(p)[0]
                rhs = convex_closure(p_half)[0]
                if not np.isclose(lhs,2.*rhs):
                    print('FAIL_s2')
                    # import pdb
                    # pdb.set_trace()
                    

            # ascending
            if True and not np.all( np.diff([-np.sum(a0[0:i])/np.sum(b0[0:i]) for i in range(1,1+len(a0))])<=0):
                print('FAIL_a1')
                # import pdb
                # pdb.set_trace()

            # descending
            if True and not np.all( np.diff([-np.sum(a0[i:])/np.sum(b0[i:]) for i in range(0,len(a0))])<=0):
                print('FAIL_d1')
                # import pdb
                # pdb.set_trace()

            if True and not np.all( np.diff([np.sum(a0[range(0,i)])/np.sum(b0[range(0,i)])/np.sum(a0[range(i,len(a0))])/np.sum(b0[range(i,len(a0))]) for i in range(1,len(a0))])>0):
                print('FAIL_ad1')
                # import pdb
                # pdb.set_trace()

            # Although this fails in UNCONSTRAINED case, the diffs once positive stay positive
            # boundary expanding
            if not np.all(np.diff([SCORE_FN(a0,b0,POWER,con_parts[i])/np.sum(b0[con_parts[i]]) for i,_ in enumerate(con_parts)]) >= 0):
                
            # ascending boundary expanding
            # if not np.all(np.diff([SCORE_FN(a0,b0,POWER,con_parts[i])/np.sum(b0[con_parts[i]]) for i,_ in enumerate(con_parts[:len(a0)])]) > 0):
            # descending boundary expanding
            # if not np.all(np.diff([SCORE_FN(a0,b0,POWER,con_parts[i])/np.sum(b0[con_parts[i]]) for i,_ in enumerate(con_parts) if i >= len(a0)]) >= 0):
            # if not np.all(np.diff(np.diff([SCORE_FN(a0,b0,POWER,con_parts[i])/np.sum(b0[con_parts[i]]) for i,_ in enumerate(con_parts)])) > 0):
                print('FAIL_A1')
                _ = [print(con_parts[i], SCORE_FN(a0,b0,POWER,con_parts[i])/np.sum(b0[con_parts[i]])) for i,_ in enumerate(con_parts)]            
                _ = [print(con_parts[i], v) for i,v in enumerate(np.diff([SCORE_FN(a0,b0,POWER,con_parts[i])/np.sum(b0[con_parts[i]]) for i,_ in enumerate(con_parts)]))]

            if not np.all(np.diff([np.sum(a0[con_part])/np.sum(b0[con_part]) for con_part in con_parts]) > 0):
                print('FAIL_A2')
                import pdb
                pdb.set_trace()

            j,k = np.sort(rng.choice(len(a0), 2, replace=False))
            j+=1;k+=1
            part_lhs = [x for x in range(j,k)]
            part_rhs1 = list(range(k-1,len(a0)))
            part_rhs2 = list(range(0,j+1))

            lhs = SCORE_FN(a0,b0,POWER,part_lhs)/np.sum(b0[part_lhs])
            rhs1 = SCORE_FN(a0,b0,POWER,part_rhs1)/np.sum(b0[part_rhs1])
            rhs2 = SCORE_FN(a0,b0,POWER,part_rhs2)/np.sum(b0[part_rhs2])
            if (lhs < rhs2) or (lhs > rhs1):
                print('FAIL_A3')
                # import pdb
                # pdb.set_trace()

            part_rhs1 = list(range(0,k))
            part_rhs2 = list(range(j,len(a0)))
            lhs = SCORE_FN(a0,b0,POWER,part_lhs)/np.sum(b0[part_lhs])
            rhs1 = SCORE_FN(a0,b0,POWER,part_rhs1)/np.sum(b0[part_rhs1])
            rhs2 = SCORE_FN(a0,b0,POWER,part_rhs2)/np.sum(b0[part_rhs2])
                            
            if (lhs < rhs1) or (lhs > rhs2):
                print('FAIL_A4')
                # import pdb
                # pdb.set_trace()

            if False:
                if not os.path.exists('./violations'):
                    os.mkdir('./violations')
                with open('_'.join(['./violations/a', str(SEED),
                                    str(trial),
                                    str(PARTITION_SIZE)]), 'wb') as f:
                    pickle.dump(a0, f)
                with open('_'.join(['./violations/b', str(SEED),
                                    str(trial),
                                    str(PARTITION_SIZE)]), 'wb') as f:
                    pickle.dump(b0, f)
                with open('_'.join(['./violations/rmax', str(SEED),
                                    str(trial),
                                    str(PARTITION_SIZE)]), 'wb') as f:
                    pickle.dump(r_max_raw, f)
                bad_cases += 1
                if bad_cases == 10:
                    import sys
                    sys.exit()
    
        trial += 1
