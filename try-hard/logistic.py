import numpy as np

c = np.array([1,2,3,4])
d = np.array([5,6, 7,8])
print np.shape(d)
print np.shape(c)
print len(d)
e = np.reshape(d, (len(d), 1))
print np.shape(e)
print e
print c.dot(e)

def sigmoid(Z):
    den = 1 + e**(-Z)
    sig = 1/den
    return sig

def compute_cost(theta, X, y):
    m = X.shape[0]

    theta = np.reshape(theta, (len(theta), 1))

    cost = (1/m) * (-np.transpose(y) * np.log(sigmoid(X.dot(theta))) - np.transpose(1 - y) * np.log(1 - sigmoid(X.dot(theta))))

    grad = np.transpose(sigmoid(X.dot(theta))).dot(X)