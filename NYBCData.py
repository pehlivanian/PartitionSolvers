import numpy as np
import pandas as pd
from dbfread import DBF
import solverSWIG_DP
import solverSWIG_LTSS
import proto
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def set_display(max_columns=100, max_rows=500, max_colwidth=64, prec=6):
    import pandas as pd
    pd.options.display.expand_frame_repr = False
    pd.options.display.max_columns = max_columns
    pd.options.display.max_rows = max_rows
    pd.options.display.max_colwidth=max_colwidth
    if prec>=0:
        pd.options.display.float_format = ('{:20,.%df}'%prec).format

set_display()

def Poisson_llr(a,b,p):
    p = list(p)
    asum = np.sum(a[p])
    bsum = np.sum(b[p])
    if asum > bsum:
        return asum*np.log(asum/bsum) + bsum - asum
    else:
        return 0

SCORE_FN = Poisson_llr

file1 = './NYBCData/BlockGroup_Crosswalk_region.dbf'
file2 = './NYBCData/NYSCAncer_region.dbf'
file3 = './NYBCData/NYS_SES_DAta_region.dbf'

records1 = DBF(file1)
records2 = DBF(file2)
records3 = DBF(file3)

data_by_geo = dict()
for rec1 in records1:
    if data_by_geo.get(rec1['GEOID10'],None):
        print('LOCATION ALREADY SEEN')
    else:
        id = rec1.pop('GEOID10')
        data_by_geo[id] = rec1

breast_by_geo = dict()
for rec2 in records2:
    id = rec2['DOHREGION']
    # XXX
    # breast_by_geo[id] = rec2['OBREAST']
    # breast_by_geo[id] = rec2['OBRAIN']
    # breast_by_geo[id] = rec2['OTOTAL']
    breast_by_geo[id] = rec2['OLUNG']
    
f_pop_by_geo = dict()
for rec3 in records3:
    id = rec3['DOHREGION']
    f_pop_by_geo[id] = rec3['F_TOT']

g,h,loc = list(),list(),list()
for k in f_pop_by_geo:
    if k in data_by_geo:
        g.append(breast_by_geo[k])
        h.append(f_pop_by_geo[k])
        loc.append(data_by_geo[k])
g = np.array(g).astype('float64')
h = np.array(h).astype('float64')

g*=100.

g_c = proto.FArray()
h_c = proto.FArray()

g_c = g
h_c = h

num_partitions = 6
distribution = 1
risk_partitioning_objective = True
optimized = True

all_results = solverSWIG_DP.OptimizerSWIG(num_partitions,
                                          g_c,
                                          h_c,
                                          distribution,
                                          risk_partitioning_objective,
                                          optimized)()
sortind = np.argsort([np.sum(g[list(all_results[0][i])])/np.sum(h[list(all_results[0][i])]) for i,_ in enumerate(all_results[0])])
sorted_results = [all_results[0][i] for i in sortind]
all_results = (sorted_results, all_results[1])

single_result = solverSWIG_LTSS.OptimizerSWIG(g_c, h_c, distribution)()

def plot_spatial_data(g, h, loc, results, part_num_thresh=0, title=None):
    # Draw the map background
    region = 'NY'
    coords = {'Japan': dict(lat_0=36.2048, lon_0=138.2529, width=2E6, height=2.3E6),
              'Russia': dict(lat_0=61.5240, lon_0=105.3188, width=10E6, height=6.3E6),
              'US': dict(lat_0=37.0902, lon_0=-95.7129, width=8E6, height=5.0E6),
              'NY': dict(lat_0=44.0, lon_0=-75.0, width=1E6, height=1E6)
              }[region]

    fig = plt.figure(figsize=(8,8))
    m = Basemap(projection='lcc', resolution='h',
                lat_0=coords['lat_0'], lon_0=coords['lon_0'],
                width=coords['width'], height=coords['height'])
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(linewidth=0.5, color='gray')
    m.drawstates(color='gray')

    scores = [SCORE_FN(g, h, results[0][i]) for i,_ in enumerate(results[0])]
    qs = [np.sum(g[list(results[0][i])])/np.sum(h[list(results[0][i])])
          for i,_ in enumerate(results[0])]

    # Desired range 1-50 ish
    mn = min(scores)
    dil = max(scores) + (-mn)
    rng = 49
    if len(scores) > 1 and not all([x==0 for x in scores]):
        sizes = [0.1*(10+((rng+100)/dil)*(s+(-mn))) for s in scores]
    else:
        sizes = [10.] * len(scores)

    colors = list(plt.rcParams['axes.prop_cycle'])
    num_colors = len(colors)
    
    top_parts = list(np.argsort(qs))[part_num_thresh:]

    for ind in top_parts:
        part = all_results[0][ind]
        sze = sizes[ind]
        color = colors[ind]['color']
        for p in part:
            m.scatter(loc[p]['INTPTLONG'], loc[p]['INTPTLAT'], latlon=True,
                      c=color, s=sze, cmap='Reds', alpha=0.95)
            
    for top_parts_ind, ind in enumerate(top_parts):
        sze = sizes[ind]
        plt.scatter([], [], c=colors[ind%num_colors]['color'], alpha=0.95, s=sze,
                    label='Region: {:2d} q: {:>4.2f}'.format(part_num_thresh+top_parts_ind,
                                                                 round(qs[ind],2)))
    plt.legend(scatterpoints=1, frameon=False,
               labelspacing=1, loc='upper left')
    title = title or 'NYS Brain Cancer Clusters'
    plt.title(title)
    plt.pause(1e-3)
    plt.savefig('NYS_BC_{}.pdf'.format(num_partitions))
    plt.close()

if __name__ == '__main__':
    # results = [[single_result[0]]]
    results = all_results
    title = 'NYS lung cancer clusters - {} partitions'.format(num_partitions-1)
    part_num_thresh = 1
    # title = 'NYS breast cancer clusters - KULL'
    plot_spatial_data(g_c, h_c, loc, results, part_num_thresh=part_num_thresh, title=title)
