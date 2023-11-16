import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('phishing.csv')
dfmodel=data.copy()
list1=[]
list1.append(data.columns)
scaler=RobustScaler()
for i in list1:
    dfmodel[i]=scaler.fit_transform(dfmodel[i])
feat=dfmodel.drop(columns=['Result'])
label=dfmodel['Result']
xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3,random_state=4)
mdl = KNeighborsClassifier(n_neighbors = k, metric = 'minkowski', p=2)
mdl.fit(x_train, y_train)

# accuracy
y_pred = mdl.predict(x_valid)

print(data.columns)