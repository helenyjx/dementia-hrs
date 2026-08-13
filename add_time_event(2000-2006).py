from operator import index

import pandas as pd
import numpy as np
from scipy.stats import argus


def first_1_else_last_0_with_value(row,year=2000):
    ones = row[row == 1]
    if not ones.empty:
        if ones.index[0] == None:
            col = ones.index[0]
        else:
            col = int(ones.index[0]) - year
        return pd.Series([col, 1])

    zeros = row[row == 0]
    if not zeros.empty:
        if zeros.index[-1] == None:
            col = zeros.index[-1]
        else:
            col = int(zeros.index[-1]) - year
        return pd.Series([col, 0])

    return pd.Series([np.nan, np.nan])

base_dir = ('/Users/jiaxinying/Desktop/dementia_benchmark/')
n_folds = 10
y = pd.read_csv(base_dir+'resources/read_only/all_y_new.csv')
y_2000_2006 = y.iloc[:,:5]
y_2000_2006[['time','event']] = y_2000_2006.iloc[:,1:5].apply(first_1_else_last_0_with_value,year=2000, axis=1)

y_2008_2014 = y.iloc[:,[0,5,6,7,8]]
y_2008_2014[['time','event']] = y_2008_2014.iloc[:,1:5].apply(first_1_else_last_0_with_value,year=2008, axis=1)

for fold in range(1,n_folds+1):
    df = pd.read_csv(base_dir+f"interval/2000-2006train_{fold}.csv")
    df = pd.merge(df,y_2000_2006[['hhidpn','time','event']],on='hhidpn',how='left')
    df.to_csv(base_dir+f"interval/2000-2006train_new_{fold}.csv",index=False)
    df = pd.read_csv(base_dir + f"interval/2000-2006val_{fold}.csv")
    df = pd.merge(df, y_2000_2006[['hhidpn','time','event']], on='hhidpn', how='left')
    df.to_csv(base_dir + f"interval/2000-2006val_new_{fold}.csv",index=False)

df = pd.read_csv(base_dir+'interval/2000(00-06y)encode.csv')
df = pd.merge(df,y_2000_2006[['hhidpn','time','event']],on='hhidpn',how='left')
df.to_csv(base_dir + f"interval/2000(00-06y)encode_new.csv",index=False)

df = pd.read_csv(base_dir+'interval/2008(08-14y)encode.csv')
df = pd.merge(df,y_2008_2014[['hhidpn','time','event']],on='hhidpn',how='left')
df.to_csv(base_dir + f"interval/2008(08-14y)encode_new.csv",index=False)
