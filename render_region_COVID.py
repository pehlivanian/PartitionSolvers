import argparse
import sys
import numpy as np
import pandas as pd
import os
import datetime
import solverSWIG_DP
import solverSWIG_LTSS
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap


def Poisson_llr(a,b,p):
    p = list(p)
    asum = np.sum(a[p])
    bsum = np.sum(b[p])
    if asum > bsum:
        return asum*np.log(asum/bsum) + bsum - asum
    else:
        return 0

COUNTRIES = ['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola',
             'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia',
             'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
             'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
             'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil',
             'Brunei', 'Bulgaria', 'Burkina Faso', 'Burma', 'Burundi',
             'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada',
             'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia',
             'Comoros', 'Congo (Brazzaville)', 'Congo (Kinshasa)', 'Costa Rica',
             "Cote d'Ivoire", 'Croatia', 'Cuba', 'Cyprus', 'Czechia', 'Denmark',
             'Diamond Princess', 'Djibouti', 'Dominica', 'Dominican Republic',
             'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea',
             'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France',
             'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece',
             'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana',
             'Haiti', 'Holy See', 'Honduras', 'Hungary', 'Iceland', 'India',
             'Indonesia', 'Iran', 'Iraq', 'Ireland', 'Israel', 'Italy',
             'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya',
             'Korea, South', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia',
             'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein',
             'Lithuania', 'Luxembourg', 'MS Zaandam', 'Madagascar', 'Malawi',
             'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands',
             'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco',
             'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia',
             'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger',
             'Nigeria', 'North Macedonia', 'Norway', 'Oman', 'Pakistan',
             'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
             'Poland', 'Portugal', 'Qatar', 'Romania', 'Russia', 'Rwanda',
             'Saint Kitts and Nevis', 'Saint Lucia',
             'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
             'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
             'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
             'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
             'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland',
             'Syria', 'Taiwan*', 'Tajikistan', 'Tanzania', 'Thailand',
             'Timor-Leste', 'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
             'US', 'Uganda', 'Ukraine', 'United Arab Emirates',
             'United Kingdom', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela',
             'Vietnam', 'West Bank and Gaza', 'Yemen', 'Zambia', 'Zimbabwe']

SCORE_FN = Poisson_llr
PATH = os.path.join('/home/charles/git/COVID-19', 'csse_covid_19_data',
'csse_covid_19_daily_reports')

def simulate_country_data(county, dt, use_province_state=False, provinceState=None):
    # Read all data for COUNTRY
    df_all = pd.DataFrame()
    for _date in allDates:
        path = os.path.join(PATH,'.'.join([_date, 'csv']))
        df = pd.read_csv(path, sep=',',header='infer',index_col=None)
        try:
            df_country = df[df['Country_Region'] == country]
        except KeyError:
            continue
        df_country['date']= _date
        df_all = pd.concat([df_all, df_country])
    df_all['baseline'] = df_all.groupby('Province_State')['Confirmed'].transform(lambda x: x.rolling(10, 1).mean()) 
    if country in ('Russia', 'Japan'):
        counts = df_all.groupby(by='Province_State', as_index=False).count()[['Province_State', 'Country_Region']]
        max_occ = max(counts['Country_Region'])
        provinces = counts[counts['Country_Region'] == max_occ]['Province_State'].tolist()
        df_all = df_all[df_all['Province_State'].isin(provinces)]

    df_dt = df_all[df_all['date'] == dt]
    df_dt.reset_index(inplace=True)    
    

def process_country_data(country, is_country, dt, num_partitions, use_province_state=False, provinceState=None):
    df_all = pd.DataFrame()
    for _date in allDates:
        path = os.path.join(PATH,'.'.join([_date, 'csv']))
        df = pd.read_csv(path, sep=',',header='infer',index_col=None)
        try:
            df_country = df[df['Country_Region'] == country]
        except KeyError:
            continue
        df_country['date']= _date
        df_all = pd.concat([df_all, df_country])

    # Compute baselines
    # XXX
    if country in ('Russia', 'Japan'):
        counts = df_all.groupby(by='Province_State', as_index=False).count()[['Province_State', 'Country_Region']]
        max_occ = max(counts['Country_Region'])
        provinces = counts[counts['Country_Region'] == max_occ]['Province_State'].tolist()
        df_all = df_all[df_all['Province_State'].isin(provinces)]
    df_all['baseline'] = df_all.groupby('Province_State')['Confirmed'].transform(lambda x: x.rolling(10, 1).mean()) 

    if use_province_state:
        df_all = df_all[df_all['Province_State'] == provinceState]
        
    all_results = list()
    all_single_best = list()
    df_dt = df_all[df_all['date'] == dt]
    if not df_dt.shape[0]:
        print('Date: {} not found in dataset'.format(dt))
        sys.exit()
    
    df_dt.reset_index(inplace=True)
    g = df_dt['Confirmed'].to_numpy().astype('float')
    h = df_dt['baseline'].to_numpy().astype('float')
    if g.shape[0]:
        all_results.append(solverSWIG_DP.OptimizerSWIG(num_partitions, g, h)())
        all_single_best.append(solverSWIG_LTSS.OptimizerSWIG(g, h)())
        print('OPTIMAL PARTITION')
        print('=================')
        for ind,result in enumerate(all_results[-1][0]):
            if is_country:
                allAdmins = df_dt.iloc[list(result)]['Province_State'].to_list()
            else:
                allAdmins = df_dt.iloc[list(result)]['Admin2'].to_list()
            admins = sorted([d for d in allAdmins if type(d) != float])
            score = SCORE_FN(g, h, result)
            print('Region: {}: score: {}'.format(ind+1, score))
            print('prov/states: {!r}'.format(admins))
        print('SINGLE BEST')
        print('===========')
        print('score: {}'.format(SCORE_FN(g, h, all_single_best[-1][0])))
        print('prov/states: {!r}'.format(sorted([d for d in df_dt.iloc[list(all_single_best[-1][0])]['Admin2'].to_list() if type(d) != float])))
        return all_single_best, all_results, df_dt, g, h
    return None, None, df_dt, g, h

def plot_spatial_data(df_dt, g, h, dt, country, num_partitions, single_best, results, plot_partition=True, infer_map_region=False, part_num_thresh=0):
    # Basemap stuff
    coordMap = {'Japan': dict(lat_0=36.2048, lon_0=138.2529, width=2E6, height=2.3E6),
              'Russia': dict(lat_0=61.5240, lon_0=105.3188, width=10E6, height=6.3E6),
              'US': dict(lat_0=37.0902, lon_0=-95.7129, width=8E6, height=5.0E6)
              }
    if not infer_map_region:
        coords = coordMap[country]

    if infer_map_region:
        des_lat = df_dt['Lat'].describe()
        des_long = df_dt['Long_'].describe()
        lat_range = des_lat['max']-des_lat['min']
        long_range = des_long['max']-des_long['min']
        # lat_center = .5*(des_lat['min']+des_lat['max'])
        # long_center = .5*(des_long['min']+des_long['max'])
        lat_center = des_lat['mean']
        long_center = des_long['mean']
        # Overwrite coords
        coords=dict(lat_0=lat_center, lon_0=long_center,
                    width=20.E4*long_range, height=2.85E5*lat_range)

        # coords['width'] = 8E6
        # coords['height'] = 5.0E6

    fig = plt.figure(figsize=(8, 8))    
    m = Basemap(projection='lcc', resolution='h',
                lat_0=coords['lat_0'], lon_0=coords['lon_0'],
                width=coords['width'], height=coords['height'])
    m.shadedrelief()
    m.drawcoastlines(color='gray')
    m.drawcountries(linewidth=0.5, color='gray')
    m.drawstates(color='gray')
    
    colors = list(plt.rcParams['axes.prop_cycle'])
    num_colors = len(colors)

    if (plot_partition):
        scores = [SCORE_FN(g, h, results[0][i]) for i,_ in enumerate(results[0])]
        # Desired range 1-50 ish
        mn = min(scores)
        dil = max(scores) + (-mn)
        rng = 49
        sizes = [10+((rng+100)/dil)*(s+(-mn)) for s in scores]

        top_parts = list(np.argsort(scores))[part_num_thresh:]
        
        for ind, row in df_dt.iterrows():
            part_num = [i for i,p in enumerate(results[0]) if ind in p][0]
            if part_num in top_parts:
                sze = sizes[part_num]
                m.scatter(row.Long_, row.Lat, latlon=True,
                          c=colors[part_num%num_colors]['color'], s=sze,
                          cmap='Reds', alpha=0.95)
    else:
        for ind, row in df_dt.iterrows():
            sze = 50
            if ind in single_best[0]:
                m.scatter(row.Long_, row.Lat, latlon=True,
                          c=colors[0]['color'], s=sze,
                          cmap='Reds', alpha=0.95)
        
    # create colorbar and legend
    # plt.colorbar(label=r'$\log_{10}({\rm population})$')
    plt.clim(3, 7)

    # make legend with dummy points
    if (plot_partition):
        # for ind,_ in enumerate(results[0]):
        for top_parts_ind, ind in enumerate(top_parts):
            sze = sizes[ind]
            plt.scatter([], [], c=colors[ind%num_colors]['color'], alpha=0.95, s=sze,
                        label='Region: {:2d} Score: {:>4.2f}'.format(1+part_num_thresh+top_parts_ind, round(scores[ind], 2)))
        plt.legend(scatterpoints=1, frameon=False,
                   labelspacing=1, loc='lower left')
        plt.title('JHU CSSE COVID-19 Dataset {} Confirmed Cases {} Subsets: {}'.format(country, dt, num_partitions))
        path_str = '{}_best_{}_thresh'.format(num_partitions, part_num_thresh)
    else:
            sze = 50
            plt.scatter([], [], c=colors[0]['color'], alpha=0.95, s=sze,
                        label='Max Region Score: {:>4.2f}'.format(SCORE_FN(g,h,single_best[0])))
            plt.legend(scatterpoints=1, frameon=False,
               labelspacing=1, loc='lower left');
            plt.title('JHU CSSE COVID-19 Dataset {} Confirmed Cases: {} Single Clust'.format(country, dt))
            path_str = 'single_best'
    path = '{}_{}_{}'.format(country, dt, path_str)
    # plt.show()
    plt.savefig(path)
    plt.close()

parser = argparse.ArgumentParser(description='Generate by-country or by-state (US) cluster figures')
parser.add_argument('ent', metavar='Entity/region', type=str,
                    help='Country or US state specification')
parser.add_argument('T', metavar='T', type=int,
                    help='number of partitions')
parser.add_argument('-d', nargs='+', default=['09-01-2020', '12-01-2020', '03-01-2021'],
                     metavar='Index of dates in mm-dd-yyyy formatfor graph rendering DEFAULT: (09-01-2020, 12-01-2020, 03-01-2021)',
                     type=str)
parser.add_argument('-p', metavar='index to start cluster', default=0)
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('-c', dest='c', action='store_true', default=True)
group.add_argument('-no-c', dest='c', action='store_false', default=False)

args = parser.parse_args(sys.argv[1:])

use_province_state = False
provinceState = None
country = args.ent
starting_part = args.p
infer_map_region = False

if not args.c:
    use_province_state = True
    provinceState = args.ent
    infer_map_region = True
    country = 'US'

allDates = sorted([fn.split('.')[0] for fn in os.listdir(PATH) if fn.endswith('csv')])
for dt in args.d:
    all_single_best, all_results, df_dt, g, h = process_country_data(country, args.c, dt, args.T,
                                                                     use_province_state=use_province_state,
                                                                     provinceState=provinceState)
    plot_spatial_data(df_dt, g, h, dt, args.ent, args.T, all_single_best[0], all_results[0],
                      plot_partition=True, infer_map_region=infer_map_region, part_num_thresh=starting_part)


