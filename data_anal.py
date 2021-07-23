import numpy as np
import pandas as pd
import os
import datetime
import solverSWIG_DP
import solverSWIG_LTSS
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

def rational_score(a,b,p):
    asum = np.sum(a[p])
    bsum = np.sum(b[p])
    return asum*asum/bsum

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
    # XXX
    if country in ('Russia', 'Japan'):
        counts = df_all.groupby(by='Province_State', as_index=False).count()[['Province_State', 'Country_Region']]
        max_occ = max(counts['Country_Region'])
        provinces = counts[counts['Country_Region'] == max_occ]['Province_State'].tolist()
        df_all = df_all[df_all['Province_State'].isin(provinces)]

    # Boost data
    centroids = ['Los Angeles', 'New York City', 'Miami-Date', 'Broward']
    base_mean = 1e5
    means = [ 2e6, 2e6, 1e6, 1e6]
    mean_rate = [.05, .05, .05, .05]
    

    df_dt = df_all[df_all['date'] == dt]
    df_dt.reset_index(inplace=True)    
    

def process_country_data(country, dt, num_partitions, use_province_state=False, provinceState=None):
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
    df_dt.reset_index(inplace=True)
    g = df_dt['Confirmed'].to_numpy().astype('float')
    h = df_dt['baseline'].to_numpy().astype('float')
    if g.shape[0]:
        all_results.append(solverSWIG_DP.OptimizerSWIG(num_partitions, g, h)())
        all_single_best.append(solverSWIG_LTSS.OptimizerSWIG(g, h)())
        print('OPTIMAL PARTITION')
        print('=================')
        for ind,result in enumerate(all_results[-1][0]):
            admins = sorted([d for d in df_dt.iloc[list(result)]['Admin2'].to_list() if type(d) != float])
            score = SCORE_FN(g, h, result)
            print('{}: score: {}: {}'.format(ind, score, set(result).issubset(set(all_single_best[-1][0]))))
            print('prov/states: {!r}'.format(admins))
        print('SINGLE BEST')
        print('===========')
        print('score: {}'.format(SCORE_FN(g, h, all_single_best[-1][0])))
        print('prov/states: {!r}'.format(sorted([d for d in df_dt.iloc[list(all_single_best[-1][0])]['Admin2'].to_list() if type(d) != float])))
        return all_single_best, all_results, df_dt, g, h
    return None, None, df_dt, g, h

def plot_spatial_data(df_dt, g, h, dt, single_best, results, plot_partition=True, infer_map_region=False, part_num_thresh=0):
    # Basemap stuff
    # Draw the map background
    coords = {'Japan': dict(lat_0=36.2048, lon_0=138.2529, width=2E6, height=2.3E6),
              'Russia': dict(lat_0=61.5240, lon_0=105.3188, width=10E6, height=6.3E6),
              'US': dict(lat_0=37.0902, lon_0=-95.7129, width=8E6, height=5.0E6)
              }[country]

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
        
    # 3. create colorbar and legend
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
            plt.title('JHU CSSE COVID-19 Dataset {} Confirmed Cases: {}'.format(country, dt))
            path_str = 'single_best'
    path = '{}_{}_{}'.format(country, dt, path_str)
    plt.show()
    import pdb; pdb.set_trace()
    plt.savefig(path)
    plt.close()

### DRIVER ###
# if __name__ == '__main__':
if False:
    allDates = sorted([fn.split('.')[0] for fn in os.listdir(PATH) if fn.endswith('csv')])    
    countries = ('Japan','US')
    num_partitionss = (4,4)
    dts = [allDates[i] for i in (99, 287, 408)]
    # dts = [allDates[i] for i in (287, 408)]
    for country in countries:
        for num_partitions in num_partitionss:
            for dt in dts:
                all_single_best, all_results, df_dt, g, h = process_country_data(country, dt, num_partitions)
                plot_spatial_data(df_dt, g, h, dt, all_single_best[0], all_results[0], plot_partition=True)
                plot_spatial_data(df_dt, g, h, dt, all_single_best[0], all_results[0], plot_partition=False)

# if False:
if __name__ == '__main__':
    allDates = sorted([fn.split('.')[0] for fn in os.listdir(PATH) if fn.endswith('csv')])
    country = 'US'
    num_partitions = 4
    dt = allDates[408]
    print('dt: {}'.format(dt))
    all_single_best, all_results, df_dt, g, h = process_country_data(country,
                                                                     dt,
                                                                     num_partitions,
                                                                     use_province_state=True,
                                                                     provinceState='Minnesota')
    plot_spatial_data(df_dt,
                      g,
                      h,
                      dt,
                      all_single_best[0],
                      all_results[0],
                      plot_partition=True,
                      infer_map_region=True,
                      part_num_thresh=0)
    
