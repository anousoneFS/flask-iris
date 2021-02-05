import pandas as pd
import numpy as np


def iris_load(path="dataset/iris.csv", split_train_test=None):
    iris = pd.read_csv(path)

    X = iris.iloc[:, :4].values
    Y = iris.iloc[:, -1].values

    iTrain = np.empty((0,), dtype=np.int)
    iTest = np.empty((0,), dtype=np.int)

    if split_train_test:
        for i in np.unique(Y):  # 3 class
            data = np.where(Y == i)[0]
            num_train = int(len(data) * split_train_test)
            iTrain = np.concatenate((iTrain, data[:num_train]))
            iTest = np.concatenate((iTest, data[num_train:]))
        return X[iTrain], Y[iTrain], X[iTest], Y[iTest]
    return X, Y
