import numpy as np
import random

# Threshold for determining the quality of R before stopping value iteration
VI_THRESHOLD = 0.0001

def UUniFast(n, U_set):
    sumU = U_set
    U_tasks = np.random.rand(n)


    for i in range(1,n):
        nextSumU = sumU * (U_tasks[i-1]**(1.0/(n-i)))
        U_tasks[i-1] = sumU - nextSumU
        sumU = nextSumU

    U_tasks[n-1] = sumU
    return U_tasks

def generateTaskSet(n, U_set):
    taskSet = np.empty((n,2)) # A task is defined by two features Period (@0) and Computation time (@1)
    taskSet[:, 0] = np.random.rand(n) # randomly initialize periods
    taskSet[:, 1] = taskSet[:, 0] * UUniFast(n, U_set) # initialize computation time based on utility
    return taskSet

# because the use of random Utility will never be exactly 1
# could change random.random() to int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1) if need be
def generateDataSet(N, n, U=0.0, sample_set_utility=True):
    X = np.empty((N, n, 2))
    
    for i in range(N):
        X[i,:,:] = generateTaskSet(n, random.random() if sample_set_utility else U)

    return X

def labelDataSet(X, useRTA=True):
    N, n, _= X.shape
    Y = np.zeros(N)

    for i in range(N):
        Y[i] = RTALabeling(X[i, :, :], n) if useRTA else hyperbolicBoundLabeling(X[i, :, :])
    
    return Y

# Exact test for schedulability under RM (responce time analysis)
def RTALabeling(taskset, n):
    # sort taskset such that index is inversely proportional to priority 
    # sort min to max based on task period
    ts = np.sort(taskset, axis=0)

    for i in range(1, n):
        I = 0.0
        P = ts[i, 0]
        C = ts[i, 1]

        while(True):
            R = I + C
            if (R > P): 
                return False
            I = np.sum(np.ceil(R/ts[0:i, 0])*ts[0:i, 1])
            if (abs(I + C - R) < VI_THRESHOLD):
                break
    
    return True

def hyperbolicBoundLabeling(taskset):
    return np.prod(taskset[:,1]/taskset[:,0] + 1) <= 2
