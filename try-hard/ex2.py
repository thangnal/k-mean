import numpy as np
import sys
sys.setrecursionlimit(100000)


# load the test dataset
test_X = np.loadtxt("test_new.txt", delimiter=",", usecols=(0, 1, 2, 3))
test_X = np.append(np.ones((test_X.shape[0],1)), test_X, axis= 1)
# number of dataset
m = float (len(test_X))
initial_theta = np.array([[0, 0, 0, 0, 0]])
initial_theta.shape = (5, 1)


def trainning_y_binary(flower):
    trainning_y = []
    # trainning output as string
    trainning_output = np.loadtxt("testIris.txt", dtype="string", delimiter=",", usecols=(4))
    # trainning output as binary
    for i in range(0, len(trainning_output)):
        if (trainning_output[i] == flower):
            trainning_y.append(1)
        else:
            trainning_y.append(0)
    trainning_y = np.array(trainning_y)
    trainning_y.shape = (len(trainning_y), 1)

    return trainning_y


def flower_percentage(theta, test_X, flower):
    count = 0
    predict_y = sigmoid(test_X.dot(theta))
    predict_y[predict_y >= 0.5] = 1
    predict_y[predict_y < 0.5] = 0
    trainning_y = trainning_y_binary(flower)
    for i in range(len(predict_y)):
        if predict_y[i] == trainning_y[i]:
            count += 1
    percent = (float(count) / len(test_X)) * 100
    compute_cost(theta, test_X, trainning_y)
    print str(flower) + " percentage: " + str(round(percent, 3)) + "%"

def sigmoid(z):
    den = 1.0 + np.exp(-z)
    d = 1.0 / den
    return d

def compute_cost(theta, trainning_X, y):
    y = np.array(y)
    a = -np.transpose(y).dot(np.log(sigmoid(trainning_X.dot(theta))))
    b = np.transpose(1 - y).dot(np.log(1 - sigmoid(trainning_X.dot(theta))))
    J = (1.0 / m) * (a - b)

    return J

def compute_grad(theta, flower):
    iteration = 30000
    learning_rate = 0.1

    trainning_X = np.loadtxt("testIris.txt", delimiter=",", usecols=(0, 1, 2, 3))
    trainning_X = np.append(np.ones((trainning_X.shape[0], 1)), trainning_X, axis=1)

    trainning_y = trainning_y_binary(flower)

    while iteration != 0:
        theta = theta - (learning_rate / m) * np.transpose(trainning_X).dot(sigmoid(trainning_X.dot(theta)) - trainning_y)
        iteration -= 1

    return np.array(theta)

def predict_flowers(test_X, theta):
    count = 0
    predictSetosa = sigmoid(test_X.dot(theta_list[0]))
    predictColor = sigmoid(test_X.dot(theta_list[1]))
    predictVeronica= sigmoid(test_X.dot(theta_list[2]))

    test_output = []
    for i in range(len(predictSetosa)):
        predict = [predictSetosa[i], predictColor[i], predictVeronica[i]]
        max_rate = max(predict)
        if predict.index(max_rate) == 0:
            test_output.append("Iris-setosa")
        elif predict.index(max_rate) == 1:
            test_output.append("Iris-versicolor")
        elif predict.index(max_rate) == 2:
            test_output.append("Iris-virginica")

    test_output = np.array(test_output)
    trainning_output = np.loadtxt("testIris.txt", dtype="string", delimiter=",", usecols=(4))
    for i in range(len(trainning_output)):
        if test_output[i] == trainning_output[i]:
            count += 1

    print "Flowers prediction rate is: " + str((float (count) / m) * 100) + "%"

# store thetas value
theta_list = np.array([compute_grad(initial_theta, "Iris-setosa"),
                       compute_grad(initial_theta, "Iris-versicolor"),
                       compute_grad(initial_theta, "Iris-virginica")])

flower_percentage(theta_list[0], test_X, "Iris-setosa")
flower_percentage(theta_list[1], test_X, "Iris-versicolor")
flower_percentage(theta_list[2], test_X, "Iris-virginica")

predict_flowers(test_X, theta_list)
