import numpy as np
import matplotlib.pyplot as plt
from test2 import iris_load


def knn(Xtrain, Ytrain, Xtest, k=1):
    Ztest = []
    for i in Xtest:
        d = np.sqrt(np.sum((Xtrain - i) ** 2, axis=1))
        ind = np.argsort(d)
        y = Ytrain[ind[:k]]
        (value, count) = np.unique(y, return_counts=True)
        m = np.argmax(count)
        Ztest.append(value[m])
    return Ztest


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = iris_load(split_train_test=0.05)
    Ytrain = np.array(["A", "B", "B", "A", "A", "C", "A", "B"])
    Xtrain = np.array([[1, 1], [3, 3], [3, 4], [1, 2], [2, 1], [10, 9], [3, 2], [4, 5]])
    Xtest = np.array([[3, 2, 2, 4]])

    result = knn(x_train, y_train, Xtest)
    print(result)
