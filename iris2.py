import pandas as pd
import numpy as np


def load_iris(path="dataset/iris.csv", split_train_test=None):
    iris = pd.read_csv(path)
    X = iris.iloc[:, :4].values
    Y = iris.iloc[:, -1].values
    classes = np.unique(Y)
    iTrain = np.empty((0,), dtype=np.int)
    iTest = np.empty((0,), dtype=np.int)
    if split_train_test:
        for i in classes:
            data = np.where(Y == i)[0]
            split = int(len(data) * split_train_test)
            iTrain = np.concatenate((iTrain, data[:split]))
            iTest = np.concatenate((iTest, data[split:]))
        return X[iTrain], Y[iTrain], X[iTest], Y[iTest]
    return X, Y
