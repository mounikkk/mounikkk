import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from FS.pso import jfs1   # change this to switch algorithm
import matplotlib.pyplot as plt
from numpy.random import seed
seed(0)

# load data
data  = pd.read_csv('phishing.csv')
data  = data.values
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.2, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}

# parameter
k    = 5     # k-value in KNN
N    = 10    # number of particles
T    = 100   # maximum number of iterations
opts = {'k':k, 'fold':fold, 'N':N, 'T':T}

# perform feature selection
fmdl = jfs1(feat, label, opts)
sf   = fmdl['sf']
print(sf)
# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train   = xtrain[:, sf]
y_train   = ytrain.reshape(num_train)  # Solve bug
x_valid   = xtest[:, sf]
print("valid",x_valid)
y_valid   = ytest.reshape(num_valid)  # Solve bug

mdl       = KNeighborsClassifier(n_neighbors = k) 
mdl.fit(x_train, y_train)

# accuracy
y_pred    = mdl.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("KNN Accuracy:", 100 * Acc)

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

print(sf)
# plot convergence
curve   = fmdl['c']
curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSO')
ax.grid()
plt.show()