import numpy as np
import pandas as pd

for filepart in ['detection_power','precision','recall','overlap','primary','secondary','distinguish']:
    df1 = pd.read_csv('exp4a_'+filepart+'.csv')
    df2 = pd.read_csv('exp4_'+filepart+'.csv')
    df1_low = df1[df1.index % 3 == 0].round(decimals=4)
    df1_med = df1[df1.index % 3 == 1].round(decimals=4)
    df1_high = df1[df1.index % 3 == 2].round(decimals=4)
    df2_low = df2[df2.index % 3 == 0].round(decimals=4)
    df2_med = df2[df2.index % 3 == 1].round(decimals=4)
    df2_high = df2[df2.index % 3 == 2].round(decimals=4)
    df_low = pd.concat([df1_low,df2_low]).sort_values(by='q1').drop_duplicates(subset='q1')
    df_med = pd.concat([df1_med,df2_med]).sort_values(by='q1').drop_duplicates(subset='q1')
    df_high = pd.concat([df1_high,df2_high]).sort_values(by='q1').drop_duplicates(subset='q1')
    df_low['epsilon'] = (df_low['q1']-1).round(decimals=2)
    df_med['epsilon'] = (df_med['q1']-1).round(decimals=2)
    df_high['epsilon'] = (df_high['q1']-1).round(decimals=2)
    df_low.loc[:,['epsilon','rp2','rp3','mcd2','mcd3']].to_csv('exp4_low_'+filepart+'.csv',index=False)
    df_med.loc[:,['epsilon','rp2','rp3','mcd2','mcd3']].to_csv('exp4_med_'+filepart+'.csv',index=False)
    df_high.loc[:,['epsilon','rp2','rp3','mcd2','mcd3']].to_csv('exp4_high_'+filepart+'.csv',index=False)
