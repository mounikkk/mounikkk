import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from flask import Flask, request, render_template

app = Flask(__name__)


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/", methods=['POST', 'GET'])
def perform():
    # uploading csv files to csvuploads folder
    # obj= request.files['csvfile']
    # csv_path = "./csvuploads/"+obj.filename
    # obj.save(csv_path)

    global ds, algo, data
    if request.method == "POST":
        ds = request.form['ds']
        algo = request.form['algo']

    sfl = []
    if ds == "ionosphere":
        data = pd.read_csv("ionosphere.csv")
        for col_name in data.columns:
            sfl.append(col_name)
        data = data.values

    if ds == "heart":
        data = pd.read_csv("heart.csv")
        for col_name in data.columns:
            sfl.append(col_name)
        data = data.values

    if ds == "iris":
        data = pd.read_csv("iris.csv")
        for col_name in data.columns:
            sfl.append(col_name)
        data = data.values

    if ds == "nuclear":
        data = pd.read_csv("nuclear.csv")
        for col_name in data.columns:
            sfl.append(col_name)
        data = data.values

    if ds == "diabetes":
        data = pd.read_csv("diabetes.csv")
        for col_name in data.columns:
            sfl.append(col_name)
        data = data.values


    feat = np.asarray(data[:, 0:-1])
    label = np.asarray(data[:, -1])

    # splitting data
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.3, stratify=label)
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

    final = []

    if algo == "pso":
        # pso parameters
        N = 10  # number of particles
        T = 100  # maximum number of iterations
        opts = {'fold': fold, 'N': N, 'T': T}
        from FS.pso import jfs1
        # perform feature selection
        fmdl = jfs1(feat, label, opts)
        sf = fmdl['sf']
        for i in sf:
            final.append(sfl[i])

    if algo == "ga":
        # GA parameters
        k = 5
        N = 10  # number of particles
        T = 100  # maximum number of iterations
        opts = {'k': k, 'fold': fold, 'N': N, 'T': T}
        from FS.ga import jfs
        # perform feature selection
        fmdl = jfs(feat, label, opts)
        sf = fmdl['sf']
        for i in sf:
            final.append(sfl[i])

    if algo == "cuckoo":
        # Cuckoo search parameter
        k = 5
        N = 10  # number of particles
        T = 100  # maximum number of iterations
        opts = {'k': k, 'fold': fold, 'N': N, 'T': T}
        from FS.cs import jfs
        # perform feature selection
        fmdl = jfs(feat, label, opts)
        sf = fmdl['sf']
        for i in sf:
            final.append(sfl[i])

    if algo == "hho":
        # Harris Hawk Algorithm parameter
        N = 10  # number of particles
        T = 100  # maximum number of iterations
        opts = {'fold': fold, 'N': N, 'T': T}
        from FS.hho import jfs
        # perform feature selection
        fmdl = jfs(feat, label, opts)
        sf = fmdl['sf']
        for i in sf:
            final.append(sfl[i])

    if algo == "ff":
        # Firefly Algorithm parameter
        N = 10  # number of particles
        T = 100  # maximum number of iterations
        opts = {'fold': fold, 'N': N, 'T': T}
        from FS.fa import jfs
        # perform feature selection
        fmdl = jfs(feat, label, opts)
        sf = fmdl['sf']
        for i in sf:
            final.append(sfl[i])

    if algo == "hybrid":
        xtrain, xtest, ytrain, ytest = train_test_split(data.drop(columns=data.columns[-1], axis=1), data.iloc[:, -1:],
                                                        test_size=0.3)
        from FS.hybridforgui import execute
        sf = execute(xtrain, xtest, ytrain, ytest, data)
        for i in sf:
            final.append(sfl[i])


    return render_template('index.html', final=final)


if __name__ == '__main__':
    app.run(debug=True)
