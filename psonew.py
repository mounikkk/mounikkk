import numpy as np
from numpy.random import rand
from sklearn.model_selection import train_test_split
from FS.functionHO import Fun
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from gain_ratio import inf_gain
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from numpy.random import seed
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


seed(0)
def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()
    return X

def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    return Xbin


def init_velocity(lb, ub, N, dim):
    V = np.zeros([N, dim], dtype='float')
    Vmax = np.zeros([1, dim], dtype='float')
    Vmin = np.zeros([1, dim], dtype='float')
    # Maximum & minimum velocity
    for d in range(dim):
        Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
        Vmin[0, d] = -Vmax[0, d]

    for i in range(N):
        for d in range(dim):
            V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()

    return V, Vmax, Vmin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def roulette_wheel(prob):
    num = len(prob)
    C   = np.cumsum(prob)
    P   = rand()
    for i in range(num):
        if C[i] > P:
            index = i
            break
    return index

lb = 0
ub = 1

#pso parameters:
w        = 0.9  # inertia weight
c1       = 2  # acceleration factor
c2       = 2  # acceleration factor
thres    = 0.5

k        = 5     # k-value in KNN
N        = 10    # number of chromosomes
T        = 100   # maximum number of generations

#genetic parameters:
CR       = 0.8
MR       = 0.01
max_iter = 100

data  = inf_gain()
data  = data.values
#encoder = OrdinalEncoder()
#data    = encoder.fit_transform(data)
feat  = np.asarray(data[:, 0:-1])
label = np.asarray(data[:, -1])

# split data into train & validation (70 -- 30)
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
fold = {'xt':xtrain, 'yt':ytrain, 'xv':xtest, 'yv':ytest}
opts = {'k':k, 'fold':fold, 'N':N, 'T':T, 'w':w, 'c1':c1, 'c2':c2}

dim = np.size(xtrain, 1)
if np.size(lb) == 1:
    ub = ub * np.ones([1, dim], dtype='float')
    lb = lb * np.ones([1, dim], dtype='float')

# Initialize position
X = init_position(lb, ub, N, dim)
X = binary_conversion(X, thres, N, dim)
y1=X.copy()

# Fitness at first iteration
fit  = np.zeros([N, 1], dtype='float')
Xgb  = np.zeros([1, dim], dtype='int')
fitG = float('inf')

V, Vmax, Vmin = init_velocity(lb, ub, N, dim)

Xpb   = np.zeros([N, dim], dtype='float')
fitP  = float('inf') * np.ones([N, 1], dtype='float')
curve = np.zeros([1, max_iter], dtype='float')
t     = 0
for i in range(N):
    fit[i, 0] = Fun(xtrain, ytrain, X[i, :], opts)
    if fit[i, 0] < fitP[i, 0]:
        Xpb[i, :] = X[i, :]
        fitP[i, 0] = fit[i, 0]
    if fitP[i, 0] < fitG:
        Xgb[0, :] = Xpb[i, :]
        fitG = fitP[i, 0]
# Pre
curve = np.zeros([1, max_iter], dtype='float')
curve[0, t] = fitG.copy()
t+=1
while t < max_iter:
    #probability
    inv_fit = 1 / (1 + fit)
    prob = inv_fit / np.sum(inv_fit)

    # Number of crossovers
    Nc = 0
    for i in range(N):
        if rand() < CR:
            Nc += 1

    x1 = np.zeros([Nc, dim], dtype='int')
    x2 = np.zeros([Nc, dim], dtype='int')
    x3 = np.zeros([Nc, dim], dtype='int')
    x3 = np.concatenate((x3,y1),axis=0)
    for i in range(Nc):
        # Parent selection
        k1 = roulette_wheel(prob)
        k2 = roulette_wheel(prob)
        P1 = X[k1, :].copy()
        P2 = X[k2, :].copy()
        # Random one dimension from 1 to dim
        index = np.random.randint(low=1, high=dim - 1)
        # Crossover
        x1[i, :] = np.concatenate((P1[0:index], P2[index:]))
        x2[i, :] = np.concatenate((P2[0:index], P1[index:]))

    x3=np.concatenate((x3,x1),axis=0)
    x3=np.concatenate((x3,x2),axis=0)
    for i in range(Nc):
        for d in range(dim):
            if rand() < MR:
                x1[i, d] = 1 - x1[i, d]

            if rand() < MR:
                x2[i, d] = 1 - x2[i, d]

    # Merge two group into one
    Xnew = np.concatenate((x1, x2), axis=0)
    Xnew = np.concatenate((Xnew, x3), axis=0)

    # Fitness
    for i in range(N):
        fit[i, 0] = Fun(xtrain, ytrain, Xnew[i, :], opts)
        if fit[i, 0] < fitP[i, 0]:
            Xpb[i, :] = X[i, :]
            fitP[i, 0] = fit[i, 0]
        if fitP[i, 0] < fitG:
            Xgb[0, :] = Xpb[i, :]
            fitG = fitP[i, 0]

    curve[0, t] = fitG.copy()
    #print("itertion:",t+1)
    #print("best iteration" ,curve[0 ,t])
    t+=1
    for i in range(N):
        for d in range(dim):
            # Update velocity
            r1 = rand()
            r2 = rand()
            V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - Xnew[i, d]) + c2 * r2 * (Xgb[0, d] - Xnew[i, d])
            # Boundary
            V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
            # Update position
            X[i, d] = Xnew[i, d] + V[i, d]
            # Boundary
            X[i, d] = boundary(Xnew[i, d], lb[0, d], ub[0, d])

# Best feature subset
Gbin       = binary_conversion(Xgb, thres, 1, dim)
Gbin       = Gbin.reshape(dim)
pos        = np.asarray(range(0, dim))
sel_index  = pos[Gbin == 1]
#print(sel_index)
num_feat   = len(sel_index)
#print(num_feat)


# model with selected features
num_train = np.size(xtrain, 0)
num_valid = np.size(xtest, 0)
x_train = xtrain[:, sel_index]
y_train = ytrain.reshape(num_train)  # Solve bug
x_valid = xtest[:, sel_index]
y_valid = ytest.reshape(num_valid)  # Solve bug
mdl = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p=2)
mdl.fit(x_train, y_train)

# accuracy
y_pred = mdl.predict(x_valid)

Acc = np.sum(y_valid == y_pred) / num_valid
print("KNN Accuracy:", 100 * Acc)

report = classification_report(y_valid ,y_pred)
print("Report:", report)

mdl2      = RandomForestClassifier(n_estimators=100)
mdl2.fit(x_train, y_train)

# accuracy
y_pred    = mdl2.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("RFC Accuracy:", 100 * Acc)

mdl3      = xgb.XGBClassifier(random_state = 50)
mdl3.fit(x_train, y_train, eval_metric='rmse')
# accuracy
y_pred    = mdl3.predict(x_valid)
Acc       = np.sum(y_valid == y_pred)  / num_valid
print("xgb Accuracy:", 100 * Acc)

curve   = curve.reshape(np.size(curve,1))
x       = np.arange(0, opts['T'], 1.0) + 1.0

fig, ax = plt.subplots()
ax.plot(x, curve, 'o-')
ax.set_xlabel('Number of Iterations')
ax.set_ylabel('Fitness')
ax.set_title('PSOGA')
ax.grid()
plt.show()