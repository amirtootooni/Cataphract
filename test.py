from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import utils

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

# X = utils.generateDataSet(50, 32, U=0.99, sample_set_utility=False)
# print(X)
# Ye = utils.labelDataSet(X)
# print(Ye)
# Yh = utils.labelDataSet(X, useRTA=False)
# print(Yh)