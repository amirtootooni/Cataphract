import numpy as np
import utils

# sklearn Statistical models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# sklearn Cross validation tools
from sklearn.model_selection import cross_val_score

NUM_FOLDS_VALIDATION = 5

# k-fold cross validation to find the optimal tree depth for a given set and return the model
def trainDecisionTree(X, y, max_depth=5, step_size=1):

    best_score = 0.0
    best_depth = 1
    for i in range(1, max_depth+1, step_size):
        model = DecisionTreeClassifier(max_depth=i, criterion='entropy', random_state=1)
        score = cross_val_score(model, X, y, cv=NUM_FOLDS_VALIDATION).mean()
        if (score > best_score):
            best_score = score
            best_depth = i
    
    model = DecisionTreeClassifier(max_depth=best_depth, criterion='entropy', random_state=1)
    model.fit(X,y)

    return model, best_depth

def trainRandomForrest(X, y, max_depth=5, step_size=1):
    best_score = 0.0
    best_depth = 1
    
    for i in range(1, max_depth+1, step_size):
        model = RandomForestClassifier(max_depth=i, n_estimators=100)
        score = cross_val_score(model, X, y, cv=NUM_FOLDS_VALIDATION).mean()
        if (score > best_score):
            best_score = score
            best_depth = i
    
    
    model = RandomForestClassifier(max_depth=best_depth, n_estimators=100)
    model.fit(X,y)

    return model, best_depth
