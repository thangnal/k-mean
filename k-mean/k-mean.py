import numpy as np
from numpy import linalg as la
import random


def nomalize(X):
    length = len(X)
    for column in range(len(X[0])):
        for row in range(length):
            x = X[row][column]
            feature = X[:, column]
            average = sum(feature) / float(length)
            x = (x - average) / float(max(feature) - min(feature))
            X[row][column] = x
    return X


class K_Mean:
    X = np.loadtxt("testIris.txt", delimiter=",", usecols=(0, 1, 2, 3))
    X = nomalize(X)
    m = X.shape[0]

    # load the test dataset
    test_X = np.loadtxt("test_new.txt", delimiter=",", usecols=(0, 1, 2, 3))
    test_X = nomalize(test_X)
    test_X = np.append(np.ones((test_X.shape[0], 1)), test_X, axis=1)

    def __init__(self, k=3, iteration=100, length = len(X)):
        self.k = k
        self.iteration = iteration
        self.length = length

    def clustering(self):
        random_3 = random.sample(range(len(self.X)), self.k)
        u1 = self.X[random_3[0]]
        u2 = self.X[random_3[1]]
        u3 = self.X[random_3[2]]

        for iter in range(self.iteration):
            cluster_u1 = []
            cluster_u2 = []
            cluster_u3 = []
            for input in self.X:
                check = [la.norm(input - u1), la.norm(input - u2), la.norm(input - u3)]
                min_distance = min(check)
                if check.index(min_distance) == 0:
                    cluster_u1.append(input)
                if check.index(min_distance) == 1:
                    cluster_u2.append(input)
                if check.index(min_distance) == 2:
                    cluster_u3.append(input)

            cluster_u1 = np.array(cluster_u1)
            cluster_u2 = np.array(cluster_u2)
            cluster_u3 = np.array(cluster_u3)
            u1 = cluster_u1.mean(0)
            u2 = cluster_u2.mean(0)
            u3 = cluster_u3.mean(0)








a = np.array([1,2,3])
b = np.array([1,2,4])
print b - a
print la.norm(np.subtract(b,a))

K_Mean().clustering()

