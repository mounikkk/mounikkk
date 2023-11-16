import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy
from sklearn.feature_selection import SelectPercentile
from sklearn import preprocessing

def entropy3(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def inf_gain(data):
    xtrain, xtest, ytrain, ytest = train_test_split(data.drop(columns=data.columns[-1],axis=1),data.iloc[:,-1:],test_size=0.3)
    mutual_info = mutual_info_classif(xtrain, ytrain)
    inf=entropy3(xtrain)
    gain_r=mutual_info/inf
    mutual_info1 = pd.Series(gain_r)
    mutual_info1.index = xtrain.columns
    mutual_info1.sort_values(ascending=False)
    sel_top_cols = SelectPercentile(mutual_info_classif, percentile=85)
    sel_top_cols.fit(xtrain.fillna(0), ytrain)
    sel_list = xtrain.columns[sel_top_cols.get_support()]
    for i in xtrain.columns:
        if i not in list(sel_list):
            data.drop(i,inplace=True,axis=1)
    return data