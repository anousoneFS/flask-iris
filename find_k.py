from iris2 import load_iris
from knn2 import knn
import numpy as np

Xtrain, Ytrain, Xtest, Ytest = load_iris(split_train_test=0.8)
n = 0
result = []
# Ztest = knn(Xtrain, Ytrain, Xtest, 8)
for k in range(1, len(Xtest) + 1):
    Ztest = knn(Xtrain, Ytrain, Xtest, k)
    for id in range(len(Ytest)):
        if Ztest[id] == Ytest[id]:
            n += 1
    result.append(n)
    n = 0
print(f"k = {np.argmax(result)+1}")
print(result)
# print(Ztest)
# print(Ytest)
