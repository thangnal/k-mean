import numpy as np


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


class Neural:
    X = np.loadtxt("testIris.txt", delimiter=",", usecols=(0, 1, 2, 3))
    X = nomalize(X)
    X1 = np.append(np.ones((X.shape[0], 1)), X, axis=1)
    Y = np.loadtxt("testIris.txt", dtype="string", delimiter=",", usecols=4)
    m = X1.shape[0]

    # load the test dataset
    test_X = np.loadtxt("test_new.txt", delimiter=",", usecols=(0, 1, 2, 3))
    test_X = nomalize(test_X)
    test_X = np.append(np.ones((test_X.shape[0], 1)), test_X, axis=1)

    def __init__(self, epsilon=0.0001, numb_hidden_layer=1, alpha=0.1, iteration=200, init_epsilon=1, lambd=0.1,
                 hidden_layer_size=4, class_size=3):
        self.epsilon = epsilon
        self.numb_hidden_layer = numb_hidden_layer
        self.alpha = alpha
        self.iteration = iteration
        self.init_epsilon = init_epsilon
        self.lambd = lambd
        self.hidden_layer_size = hidden_layer_size
        self.class_size = class_size

    def sigmoid(self, z):
        return (1. / (1 + np.exp(-z)))

    def init_theta(self):
        # randomlize theta
        # theta1 = np.random.rand(self.hidden_layer_size, self.X.shape[1] + 1) * (2 * self.init_epsilon) - self.init_epsilon  # 4x5
        # theta2 = np.random.rand(self.class_size, self.hidden_layer_size + 1) * (2 * self.init_epsilon) - self.init_epsilon  # 3x5
        theta1 = np.array([[-1.70235426, -6.7527924, -4.86441856, -7.95722746, -8.53557407],
                           [9.92483188, 1.77877472, 1.73231308, 1.0528257, -0.19372525],
                           [3.58722429, -0.96241458, -0.05234676, -2.79911259, -3.4200588],
                           [8.0712906, 0.15295613, 2.43377193, -2.93110332, -3.80633998]])
        theta2 = np.array([[-39.27245838, 43.91924993, 13.66286656, 27.62912945, -14.59155094],
                           [-5.71493782, -18.48399989, 0.08345194, 16.04071569, -6.44473763],
                           [27.95123889, -23.83396336, -23.13656333, -28.44302731, 13.40755956]])

        return theta1, theta2

    def forward(self, W1, W2, X):
        a1 = X
        z2 = np.dot(a1, np.transpose(W1))
        a2 = self.sigmoid(z2)
        a2_bias = np.column_stack([np.ones((self.m, 1)), a2])
        z3 = np.dot(a2_bias, np.transpose(W2))
        a3 = self.sigmoid(z3)
        return a2_bias, a3
        # a3 is hypothesis function which we need to find

    def Y_binary(self, flower):
        y = []
        # trainning output as binary
        for i in range(0, len(self.Y)):
            if (self.Y[i] == flower):
                y.append(1)
            else:
                y.append(0)
        y = np.array(y)
        y.shape = (len(y), 1)

        return y

    #output as binary of all 3 flowers
    def Y_binary_flowers(self):
        iris_setosa = self.Y_binary("Iris-setosa")
        iris_versicolor = self.Y_binary("Iris-versicolor")
        iris_virginica = self.Y_binary("Iris-virginica")
        y = np.append(np.append(iris_setosa, iris_versicolor, axis=1), iris_virginica, axis=1)
        return y

    def backward(self):
        epoch = 0
        D2 = D1 = 0
        W1 = self.init_theta()[0]
        W2 = self.init_theta()[1]
        while epoch < self.iteration:
            epoch += 1
            # self.forward()[1] is a3
            a3 = self.forward(W1, W2, self.X1)[1]
            a2 = self.forward(W1, W2, self.X1)[0]
            d3 = a3 - self.Y_binary_flowers()  # Y is vectorized of y [150x3]
            d2 = np.dot(d3, self.init_theta()[1]) * (a2 * (1 - a2))  # [150x5)

            D2 = D2 + np.dot(np.transpose(d3), a2)
            D1 = D1 + np.dot(np.transpose(d2[:, 1:]), self.X1)

            derivative2 = 1. / (len(self.X)) * (D2 + self.lambd * W2)
            derivative1 = 1. / (len(self.X)) * (D1 + self.lambd * W1)

            W2 -= self.alpha * derivative2
            W1 -= self.alpha * derivative1

            cost = self.cost_function(a3, W1, W2)
            print cost
            if cost < 0.001:
                return W1, W2

    def cost_function(self, predicted_output, W1, W2):
        y = self.Y_binary_flowers()
        J = (-1. / self.m) * np.sum((y * np.log(predicted_output) + (1 - y) * np.log(1 - predicted_output)))
        return J

    def predict_flowers(self):
        W1 = self.backward()[0]
        W2 = self.backward()[1]
        a3 = self.forward(W1, W2, self.test_X)[1]
        test_output = []

        for i in range(len(a3)):
            max_rate = max(a3[i])
            item = [a3[i][0], a3[i][1], a3[i][2]]
            if item.index(max_rate) == 0:
                test_output.append("Iris-setosa")
            elif item.index(max_rate) == 1:
                test_output.append("Iris-versicolor")
            elif item.index(max_rate) == 2:
                test_output.append("Iris-virginica")

        test_output = np.array(test_output)
        trainning_output = np.loadtxt("testIris.txt", dtype="string", delimiter=",", usecols=(4))
        count = 0
        for i in range(len(trainning_output)):
            if test_output[i] == trainning_output[i]:
                count += 1

        print "Flowers prediction rate is: " + str((float(count) / len(trainning_output)) * 100) + "%"


neural = Neural()
neural.predict_flowers()
