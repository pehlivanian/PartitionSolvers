import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('exp0_scores.csv')
df_diff = df.diff(axis=1)
df_diff['rp2'] = df['rp2']
df_diff['epsilon'] = df['epsilon']
df = df_diff
print(df)

def fit(z):
    y = np.log(z).reset_index(drop=True)
    X = np.log(range(2,11)).reshape(-1,1)
    clf = linear_model.LinearRegression()
    clf.fit(X,y)
    residuals = y - clf.predict(X)
    return residuals.argmin() + 1
    
