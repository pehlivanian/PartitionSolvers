import argparse
import sys
import numpy as np
import pandas as pd
from dbfread import DBF

from mpl_toolkits.basemap import Basemap
import shapefile
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import solverSWIG_DP
import proto

MAXIMAL_T = 100

dict_to_add = {'36005002000':'36005002400', 
               '36005003500':'36005003700',
               '36005009000':'36005011000',
               '36005015300':'36005016300',
               '36005024502':'36005024900',
               '36005030000':'36005027600',
               '36005030900':'36005031900',
               '36005039300':'36005033400',
               '36005043100':'36005043500',
               '36005045600':'36005050400',
               '36047000200':'36047001800',
               '36047001500':'36047001100',
               '36047003300':'36047003700',
               '36047015200':'36047016400',
               '36047021100':'36047054300',
               '36047028502':'36047028501',
               '36047040500':'36047040700',
               '36047046202':'36047045000',
               '36047070201':'36047070202',
               '36047085000':'36047085200',
               '36047093000':'36047096000',
               '36047114202':'36047118000',
               '36047150200':'36047017500',
               '36061000900':'36061031900',
               '36061008601':'36061008602',
               '36061010100':'36061010900',
               '36061010400':'36061010200',
               '36061016200':'36061024000',
               '36061016800':'36061014300',              
               '36061020101':'36061019701',
               '36061020500':'36061020300',
               '36061021303':'36061021703',
               '36061027700':'36061031100',
               '36061029500':'36061029700',
               '36081000100':'36081019900',
               '36081003900':'36081003700',
               '36081004401':'36081005000',
               '36081018102':'36081017100',
               '36081020500':'36081022900',
               '36081031700':'36081033100',
               '36081043200':'36081042600',
               '36081052100':'36081060701',
               '36081053100':'36081021900',
               '36081055500':'36081055900',
               '36081056700':'36081056100',
               '36081061301':'36081061302',
               '36081063900':'36081064102',
               '36081077905':'36081038302',
               '36081087100':'36081038301',
               '36081088400':'36081071600',
               '36081092200':'36081091800',
               '36081099703':'36081099900',
               '36081129103':'36081128300',
               '36081137700':'36081138502',               
               '36085007400':'36085001800',
               '36085029102':'36085022800'}

def get_cancer_data(cancer_type):
    ''' Get raw data, filter by cancer type
    '''
       
    data = pd.read_csv("../../NYS_data/NYC Cancer Rates 2013-2017.csv")
    
    g, h, geoid = list(), list(), list()
    for i,r in data.iterrows():
        if cancer_type == 'breast':
            g.append(r.Breast_observed)
            h.append(r.Breast_expected)
        elif cancer_type == 'prostate':
            g.append(r.Prostate_observed)
            h.append(r.Prostate_expected)
        elif cancer_type == 'lung':
            g.append(r.Lung_observed)
            h.append(r.Lung_expected)
        geoid.append(int(r.geoid))

    g_ = np.asarray(g)
    h_ = np.asarray(h)
    geoid_ = np.asarray(geoid)
    return g_,h_,geoid_

def find_partitions(num_partitions, g, h, distribution, risk_partitioning_objective):
    ''' Optimizer
    '''
    all_results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                              g,
                                              h,
                                              distribution,
                                              risk_partitioning_objective,
                                              True)()
    return all_results

def find_optimal_t(max_num_partitions, g, h, distribution, risk_partitioning_objective):
    optimal_t = solverSWIG_DP.OptimizerSWIG(max_num_partitions,
                                            g,
                                            h,
                                            distribution,
                                            risk_partitioning_objective,
                                            True,
                                            sweep_best=True)()
    return optimal_t
    

def label_clusters(results, geoid):
    ''' associate tract with cluster number
    '''
    def cluster_no(i, clusters):
        ind = 0
        while i not in clusters[ind]:
            ind+=1
        return ind

    res1 = pd.DataFrame({'geoid': geoid, 'cluster': 0})
    for i,r in res1.iterrows():
        res1.iloc[i].cluster = cluster_no(i, results[0])

    res2 = pd.DataFrame(columns=['geoid', 'cluster'])
    for from_geo,to_geo in dict_to_add.items():
        theclust = res1[res1['geoid']==int(from_geo)].iloc[0,1]
        res2=res2.append({'geoid':to_geo,'cluster':theclust},ignore_index=True)
    res = pd.concat([res1,res2],ignore_index=True).astype({'cluster':int})
    return res

def visualization(raw_result, result, num_partitions, cancer_type='breast', risk_partitioning_objective=True, colormap='Blues'):
    ''' render visuals
    '''
    nyc_geoid=list(result.geoid.unique())
    sf = shapefile.Reader("../../NYS_data/nyct2010_21b/nyct2010.shp")
    recs    = sf.records()
    test=pd.DataFrame(recs)
    test.loc[:,'county']=0
    test.loc[test.iloc[:,2]=="Staten Island","county"]="085"
    test.loc[test.iloc[:,2]=="Manhattan","county"]="061"
    test.loc[test.iloc[:,2]=="Brooklyn","county"]="047"
    test.loc[test.iloc[:,2]=="Bronx","county"]="005"
    test.loc[test.iloc[:,2]=="Queens","county"]="081"
    test.loc[:,"geoid"]=0
    test.loc[:,"geoid"]="36"+test.county+test.iloc[:,3].apply(str)

    qs = [np.sum(g[list(i)])/np.sum(h[list(i)]) for i in raw_result[0]]
    proportions = [len(i)/len(g) for i in raw_result[0]]

    sortind = np.argsort(qs)
    sorted_results = [raw_result[0][i] for i in sortind]
    sorted_qs = [qs[i] for i in sortind]
    sorted_proportions = [proportions[i] for i in sortind]


    shapes  = sf.shapes()
    Nshp    = len(shapes)
    cns     = []
    for nshp in range(Nshp):
        cns.append(recs[nshp][1])
    cns = np.array(cns)

    cmap = getattr(plt.cm, colormap)(np.linspace(0,1,max(result.iloc[:,-1])+1))
    cmap[0] = [1., 1., 1., 1.]

    fig=plt.figure(figsize = (10,10)) 
    fig.add_subplot(111)
    ax = fig.gca()
    for nshp in list(range(Nshp))[1:]:
        if int(test.iloc[nshp,-1]) in nyc_geoid:
            k=result[result.geoid==int(test.iloc[nshp,-1])].iloc[0,-1]
            c=cmap[k][0:3]  
            ptchs   = []
            pts     = np.array(shapes[nshp].points)
            prt     = shapes[nshp].parts
            par     = list(prt) + [pts.shape[0]]
            for pij in range(len(prt)):
                ptchs.append(Polygon(pts[par[pij]:par[pij+1]]))
            pc = PatchCollection(ptchs, facecolor=c,edgecolor='k',linewidths=.5)
            ax.add_collection(pc)
            ax.add_collection(PatchCollection(ptchs,facecolor=c,edgecolor='k', linewidths=.5))            
        ax.axis('scaled')

    import matplotlib.patches as mpatches
    clum_num=len(result.iloc[:,-1].unique())

    handles=[]
    for t in list(range(clum_num))[1:]:
        props = str(round(100.*sorted_proportions[t],2))
        props = props + '0'*(2-len(props.split('.;')[-1]))
        locals()["patch_{}".format(t)] = mpatches.Patch(color=cmap[t][0:3] ,
                                                        label='Cluster '+str(t+1-1)+': (q = '+str(round(sorted_qs[t], 2))
                                                        +', '
                                                        +props
                                                        +'% of census tracts)')
        handles.append(locals()["patch_{}".format(t)])
    plt.xticks([], [])
    plt.yticks([],[])
    algo_type = 'risk_part' if risk_partitioning_objective else 'MULT'    
    plt.title('NYC {} cancer incidence 2013-2017 {} t = {} partitions'.format(cancer_type, algo_type, num_partitions))
    plt.legend(handles=handles,loc='upper left',prop={'size':8})
    plt.pause(1e-3)
    plt.savefig('NYC_{}_{}_{}_{}.pdf'.format(cancer_type, num_partitions, algo_type, colormap))
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate NYC census tract figures')
    parser.add_argument('T', metavar='T', type=int,
                        help='number of subsets')
    parser.add_argument('type', metavar='Cancer type', type=str,
                        help='''type of cancer, one of ("breast", "prostate", "lung")''')
    parser.add_argument('-dist', metavar='Distribution', type=int, default=1,
                        help='distribution specification, 1 ~ Poisson, 2 ~ Gaussian DEFAULT: 1')
    parser.add_argument('-obj', metavar='Objective', type=bool, default=True,
                        help='true ~ risk partitioning objective, false ~ multiple cluster detection DEFAULT: True')
    parser.add_argument('-color', metavar='Colormap', type=str, default='Blues',
                        help='''colormap specification for visualization, choices are not limited to
                        ('Blues',
                        'Spectral',
                        'Purples',
                        'Reds',
                        'Greens',
                        'Greys',
                        'plasma',
                        'Orange'). See matplot.pyplot.cm attributes for more information DEFAULT: Blues''')

    args = parser.parse_args(sys.argv[1:])

    # get raw cancer data by type
    g, h, geoid = get_cancer_data(args.type)

    # find optimal t
    optimal_t = find_optimal_t(MAXIMAL_T, g, h, args.dist, args.obj)
    print('OPTIMAL M: {} t: {}'.format(MAXIMAL_T, optimal_t))

    # optimize
    parts = find_partitions(optimal_t, g, h, args.dist, args.obj)

    # label things
    result = label_clusters(parts, geoid)

    # render graph
    visualization(parts, result, optimal_t, cancer_type=args.type, risk_partitioning_objective=args.obj, colormap=args.color)
