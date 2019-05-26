import numpy as np
import utils
import matplotlib.pyplot as plt
from threading import Thread

# sklearn Statistical models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree.export import export_graphviz

import graphviz
import os


# os.environ["PATH"] += os.pathsep + 'C:/Users/Amirhossein/Downloads/graphviz-2.38/release/bin'

# sklearn Cross validation tools
from sklearn.model_selection import cross_val_score

NUM_CV_FOLDS = 4

# k-fold cross validation to find the optimal tree depth for a given set and return the model
def trainDecisionTree(X, y, max_depth=5, step_size=1):
    depths = range(1, max_depth+1, step_size)
    scores = np.empty(len(depths))

    def validator(X, y, scores, depth, i):
        model = DecisionTreeClassifier(max_depth=depth, criterion='entropy', random_state=1)
        scores[i] = cross_val_score(model, X, y, cv=NUM_CV_FOLDS).mean()
    
    threads = []
    i = 0
    for d in depths:
        threads.append(Thread(target=validator, args = (X, y, scores, d, i)))
        threads[i].start()
        i = i + 1

    for t in threads:
        t.join()

    best_depth = depths[np.argmax(scores)]

    model = DecisionTreeClassifier(max_depth=best_depth, criterion='entropy', random_state=1)
    model = model.fit(X,y)

    # dot_data = export_graphviz(model, out_file=None) 
    # graph = graphviz.Source(dot_data)
    # graph.render("decisionTree")

    return model, best_depth

def trainRandomForrest(X, y, max_depth=5, step_size=1):
    depths = range(1, max_depth+1, step_size)
    scores = np.empty(len(depths))

    def validator(X, y, scores, depth, i):
        model = RandomForestClassifier(max_depth=depth, n_estimators=100)
        scores[i] = cross_val_score(model, X, y, cv=NUM_CV_FOLDS).mean()
    
    threads = []
    i = 0
    for d in depths:
        threads.append(Thread(target=validator, args = (X, y, scores, d, i)))
        threads[i].start()
        i = i + 1

    for t in threads:
        t.join()

    best_depth = depths[np.argmax(scores)]

    model = RandomForestClassifier(max_depth=best_depth, n_estimators=100)
    model = model.fit(X,y)

    return model, best_depth

def trainkNN(X, y, max_k=10, step_size=1):
    ks = range(2, max_k+1, step_size)
    scores = np.empty(len(ks))

    def validator(X, y, scores, k, i):
        model = KNeighborsClassifier(n_neighbors=k)
        scores[i] = cross_val_score(model, X, y, cv=NUM_CV_FOLDS).mean()

    threads = []
    i = 0
    for k in ks:
        threads.append(Thread(target=validator, args = (X, y, scores, k, i)))
        threads[i].start()
        i = i + 1

    for t in threads:
        t.join()
    
    best_k = ks[np.argmax(scores)]
    model = KNeighborsClassifier(n_neighbors=best_k)
    model = model.fit(X,y)
    return model, best_k

def trainNeuralNetwork(X,y):
    model = MLPClassifier(learning_rate='adaptive', early_stopping=True, max_iter=100, batch_size=50)
    model = model.fit(X,y)
    return model, 0


def analyzeModels(num_models, training_size, test_size, num_tasks):
    training_errors = np.zeros((4, num_models))
    test_errors = np.zeros((4, num_models))
    hyperbolic_errors = np.zeros(num_models)
    models = [trainDecisionTree, trainRandomForrest, trainkNN, trainNeuralNetwork]
    
    for i in range(num_models):
        X, y, _ = utils.generateDataSet(training_size, num_tasks)
        X_test, y_test, y_hyp = utils.generateDataSet(test_size, num_tasks, include_hyperbolic_labeling=True)

        hyperbolic_errors[i] = np.mean(y_hyp != y_test)*100

        for j in range(4):
            model, _ = models[j](X,y)
            y_pred = model.predict(X)
            training_errors[0, j] = np.mean(y_pred != y)*100
            y_pred = model.predict(X_test)
            test_errors[0, j] = np.mean(y_pred != y_test)*100

    print('Training error: ', training_errors)
    print('Test error: ', test_errors)
    print('Hyperbolic error: ', hyperbolic_errors)

    fig, ax = plt.subplots()
    ax.set_title('Performance of different models')
    ax.set_ylabel('Error Percentage')
    ax.set_ylim(0, 30)
    ax.boxplot([hyperbolic_errors, test_errors[:, 0], test_errors[:, 1], test_errors[:, 2], test_errors[:, 3]])
    ax.set_xticklabels(['Hyperbolic Bound', 'Decision Tree', 'Random Forrest', 'kNN', 'Neural Network'], rotation=18, fontsize=8)
    ax.set_xlabel('Model used to determine Feasibility')
    plt.savefig('Performance.png')

    fig, ax = plt.subplots()
    ax.set_title('Training error of different models')
    ax.set_ylabel('Training Error Percentage')
    ax.set_ylim(0, 30)
    ax.boxplot([training_errors[:, 0], training_errors[:, 1], training_errors[:, 2], training_errors[:, 3]])
    ax.set_xticklabels(['Decision Tree', 'Random Forrest', 'kNN', 'Neural Network'], rotation=18, fontsize=8)
    ax.set_xlabel('Model')
    plt.savefig('TrainingErrors.png')
