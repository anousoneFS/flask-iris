from iris2 import load_iris
import numpy as np


def knn(Xtrain, Ytrain, Xtest, k=1):
    Ztest = []
    for i in Xtest:
        d = np.sqrt(np.sum((Xtrain - i) ** 2, axis=1))
        idx = np.argsort(d)
        (value, count) = np.unique(Ytrain[idx[:k]], return_counts=True)
        ind = np.argmax(count)
        Ztest.append(value[ind])
    return Ztest


if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = load_iris(split_train_test=0.9)
    result = knn(xtrain, ytrain, xtest)

    print(xtest)
    print(result)
    print(ytest) 
