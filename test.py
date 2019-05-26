# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
import numpy as np
import utils
import main

def testUUniFast(n):

    a = np.empty((n,3))

    for i in range(n):
        a[i,:] = utils.UUniFast(3,1.0)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a[:,0], a[:,1], a[:,2])

    ax.set_xlabel('Task 0')
    ax.set_ylabel('Task 1')
    ax.set_zlabel('Task 2')

    plt.show()

# testUUniFast(1000)

# print(utils.isFeasible(np.asarray([[1.2,0.3],[8.0,2.0],[11.0,3.0] ]), 3) + ' should be true')
# print(utils.isFeasible(np.asarray([[1.2,0.3],[8.0,2.0],[11.0,7.0] ]), 3) + ' should be false')

# X, y, y_hyp = utils.generateDataSet(3000, 32, U=0.60, sample_set_utility=False, include_hyperbolic_labeling=True)
# print(X)
# print(y)
# print(y_hyp)

def testDT():
    X, y, _ = utils.generateDataSet(10000, 16)
    X_test, y_test, y_hyp = utils.generateDataSet(2000, 16, include_hyperbolic_labeling=True)

    model, depth = main.trainDecisionTree(X,y)
    y_pred = model.predict(X)
    trainingError = np.mean(y_pred != y)
    y_pred = model.predict(X_test)
    testError = np.mean(y_pred != y_test)
    hyperbolicError = np.mean(y_hyp != y_test)

    print('depth: ', depth)
    print('Training error: ', trainingError)
    print('Test error: ', testError)
    print('Hyperbolic error: ', hyperbolicError)

def testRF():
    X, y, _ = utils.generateDataSet(10000, 16)
    X_test, y_test, y_hyp = utils.generateDataSet(2000, 16, include_hyperbolic_labeling=True)


    model, depth = main.trainRandomForrest(X,y)
    y_pred = model.predict(X)
    trainingError = np.mean(y_pred != y)
    y_pred = model.predict(X_test)
    testError = np.mean(y_pred != y_test)
    hyperbolicError = np.mean(y_hyp != y_test)

    print('depth: ', depth)
    print('Training error: ', trainingError)
    print('Test error: ', testError)
    print('Hyperbolic error: ', hyperbolicError)

def testKNN():
    X, y, _ = utils.generateDataSet(10000, 16)
    X_test, y_test, y_hyp = utils.generateDataSet(2000, 16, include_hyperbolic_labeling=True)


    model, k = main.trainKNN(X,y)
    y_pred = model.predict(X)
    trainingError = np.mean(y_pred != y)
    y_pred = model.predict(X_test)
    testError = np.mean(y_pred != y_test)
    hyperbolicError = np.mean(y_hyp != y_test)

    print('k: ', k)
    print('Training error: ', trainingError)
    print('Test error: ', testError)
    print('Hyperbolic error: ', hyperbolicError)

def testNeuralNet():
    X, y, _ = utils.generateDataSet(10000, 16)
    X_test, y_test, y_hyp = utils.generateDataSet(1000, 16, include_hyperbolic_labeling=True)


    model, _ = main.trainNeuralNetwork(X,y)
    y_pred = model.predict(X)
    trainingError = np.mean(y_pred != y)
    y_pred = model.predict(X_test)
    testError = np.mean(y_pred != y_test)
    hyperbolicError = np.mean(y_hyp != y_test)

    print('Training error: ', trainingError)
    print('Test error: ', testError)
    print('Hyperbolic error: ', hyperbolicError)

testDT()
testRF()
testKNN()
testNeuralNet()
