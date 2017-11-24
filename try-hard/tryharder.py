import numpy as np
from numpy.linalg import inv

irisY = []

readBezdekFile = np.loadtxt("BezdekIris.txt", delimiter=",", usecols= (0,1,2,3))
testIris = []
for i in range(0, len(readBezdekFile)):
    testIris.append(np.append(1,readBezdekFile[i]))


def percentage(input, flower):
    count = 0
    theta = tryhard(flower)

    global irisY

    for i in range(0,len(irisY)):
        a = np.reshape(theta, (len(theta), 1))
        if irisY[i] == 1:
            output = input[i].dot(a)
            if output > 0.5:
                count += 1
        if irisY[i] == 0:
            output = input[i].dot(a)
            if output < 0.5:
                count += 1

    percentage = (float(count) / len(irisY)) * 100
    print count
    print str(flower) + " percentage: " + str(round(percentage, 3)) + "%"

def tryhard(flower):
    irisX = []
    global irisY
    irisY = []
    trainningdata = np.loadtxt("testIris.txt", delimiter=",", usecols= (0,1,2,3))
    output = np.loadtxt("testIris.txt", dtype="string", delimiter=",", usecols=(4))

    for i in range(0, len(output)):
        irisX.append(np.append(1, trainningdata[i]))
        if(output[i] == flower):
            irisY.append(1)
        else:
            irisY.append(0)

    irisXTranspose = np.transpose(irisX)
    theta = np.dot(np.dot(inv(np.dot(irisXTranspose, irisX)),irisXTranspose), irisY)

    return theta

percentage(testIris, "Iris-versicolor")
percentage(testIris, "Iris-setosa")
percentage(testIris, "Iris-virginica")