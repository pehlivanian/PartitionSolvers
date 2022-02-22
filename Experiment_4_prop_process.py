import numpy as np
import pandas as pd
import matplotlib.pyplot as plot

labelMap = {'rp2': 'KULL',
            'rp3': 'PART',
            'mcd2': 'EBP',
            'mcd3': 'MCD'
            }
stats = ['detection_power','precision','recall','overlap','primary','secondary','distinguish']
titleMap = dict(zip(stats, ('Detection Power', 'Precision', 'Recall', 'Overlap', 'Primary cluster detection',
                     'Secondary cluster detection', 'Cluster differentiation')))
ylabelMap = dict(zip(stats, ('detection power', 'accuracy', 'accuracy', 'accuracy', 'accuracy',
                             'accuracy', 'accuracy')))
columns = ('rp2', 'rp3', 'mcd2', 'mcd3')

for filepart in stats:
    df = pd.read_csv('exp4b_'+filepart+'.csv')
    df_low = df[df.index % 3 == 0].round(decimals=4)
    df_med = df[df.index % 3 == 1].round(decimals=4)
    df_high = df[df.index % 3 == 2].round(decimals=4)

    # Save
    # df_low.loc[:,['epsilon','rp2','rp3','mcd2','mcd3']].to_csv('exp4b_low_'+filepart+'.csv',index=False)
    # df_med.loc[:,['epsilon','rp2','rp3','mcd2','mcd3']].to_csv('exp4b_med_'+filepart+'.csv',index=False)
    # df_high.loc[:,['epsilon','rp2','rp3','mcd2','mcd3']].to_csv('exp4b_high_'+filepart+'.csv',index=False)

    # Plot
    xaxis = df_med['prop'].values
    for column in columns:
        yaxis = df_med[column].values
        plot.plot(xaxis,yaxis, label=labelMap[column])
    plot.title('{}'.format(titleMap[filepart]))
    plot.xlabel('cluster proportion')
    plot.ylabel(ylabelMap[filepart])
    plot.xticks(xaxis, rotation=90)
        
    plot.legend()
    plot.grid(True)
    plot.show()
    plot.close()
    

