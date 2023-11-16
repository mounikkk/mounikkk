import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from FS.ga import jfs   # change this to switch algorithm


# load data
data  = pd.read_csv('phishing.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of particles
T    = 100   # maximum number of iterations
opts = {'k':k, 'fold':fold, 'N':N, 'T':T}
print("fs")
# perform feature selection
fmdl = jfs(feat, label, opts)
sf   = fmdl['sf']
print("done fs")
# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k)
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("Accuracy:", 100 * Acc)

mdl2      = RandomForestClassifier(n_estimators=100)
mdl2.fit(x_train, y_train)

# accuracy
y_pred    = mdl2.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("RFC Accuracy:", 100 * Acc)

mdl3      = xgb.XGBClassifier(random_state = 69)
mdl3.fit(x_train, y_train, eval_metric='rmse')

# accuracy
y_pred    = mdl3.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("xgb Accuracy:", 100 * Acc)
# number of selected features
num_feat = fmdl['nf']
print("Feature Size:", num_feat)