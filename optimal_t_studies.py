import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('exp0_scores.csv')
# df = pd.read_csv('null_scores.csv')
df = df[[x for x in df.columns if x.startswith('rp')]]
df_diff = df.diff(axis=1)
df_diff['rp2'] = df['rp2']
df = df_diff

def fit(z):
    y = np.log(z)
    X = np.log(range(2,11)).reshape(-1,1)
    clf = linear_model.LinearRegression(fit_intercept=True)
    try:
        clf.fit(X,y)
    except ValueError:
        return np.nan
    residuals = y - clf.predict(X)
    return residuals.argmin() + 1

optimal_t = list()
for rowNum in range(500):
    optimal_t.append(fit(df.iloc[rowNum,:].values))
    print('rowNum: {} Optimal t: {}'.format(rowNum,
                                            fit(df.iloc[rowNum,:].values)))

