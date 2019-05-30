import numpy as np
import random
from threading import Thread
from math import ceil

# Threshold for determining the quality of R before stopping value iteration
VI_THRESHOLD = 0.000001
# for making threads for data generation
BATCH_SIZE = 512

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
    taskSet[:, 0] = np.random.randint(1,high=10000, size=n) # randomly initialize periods (as integers)
    taskSet[:, 1] = taskSet[:, 0] * UUniFast(n, U_set) # initialize computation time based on utility
    return taskSet

# because the use of random Utility will never be exactly 1
# could change random.random() to int.from_bytes(os.urandom(8), byteorder="big") / ((1 << 64) - 1) if need be
# also lables data and returns the lables
def generateDataSet(N, n, U=1.0, sample_set_utility=True, include_hyperbolic_labeling=False):
    X = np.empty((N, n, 2))
    y_rta = np.empty(N)
    y_hyp = np.empty(N)

    def taskSetMaker(X, y_rta, y_hyp, size, n, U, sample_set_utility):
        for i in range(0, size):
            X[i,:,:] = generateTaskSet(n, random.uniform(n(2**(1.0/n)-1), 1.0) if sample_set_utility else U)
            y_rta[i] = RTALabeling(X[i, :, :], n)
            if include_hyperbolic_labeling:
                y_hyp[i] = hyperbolicBoundLabeling(X[i, :, :])

    num_threads = ceil(N/BATCH_SIZE)
    threads = []
    for i in range(num_threads):
        start = i*BATCH_SIZE
        end = start + (BATCH_SIZE if N - i*BATCH_SIZE >= BATCH_SIZE else N - i*BATCH_SIZE)
        threads.append(Thread(target=taskSetMaker, args = (X[start:end,:,:], y_rta[start:end], y_hyp[start:end], end-start, n, U, sample_set_utility)))
        threads[i].start()

    for t in threads:
        t.join()

    # want a task set to be represented as an array of features for training models
    X_flat = np.reshape(X,(N, n*2))
    return X_flat, y_rta, y_hyp  


# Exact test for schedulability under RM (responce time analysis)
def RTALabeling(taskset, n):
    # sort taskset such that index is inversely proportional to priority 
    # sort min to max based on task period
    ts = np.asarray(taskset)
    ts = np.sort(ts, axis=0)

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
