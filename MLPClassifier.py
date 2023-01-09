import warnings

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import X_PATH, Y_PATH, RESULTS_MLP


def preprocess_data(X, y):
    """ preprocessing specifically for boosting tree """

    return X, y

def optimal_parameters(X, y):
    """Train the boosting tree model using x-fold cross validation """

    # the parameters to be optimized
    hidden_layer_range = list(np.arange(10, 200, 10))
    learning_rate_range = list(np.arange(0.0001, 0.0011, 0.0002))

    kf = KFold(n_splits=10)

    scores = []

    for n_layers in hidden_layer_range:
        for learning_rate in learning_rate_range:
            print(f'at n_layers = {n_layers}, learning_rate = {learning_rate}')
            mlp = MLPClassifier(hidden_layer_sizes=n_layers, learning_rate_init=learning_rate, max_iter=300)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=ConvergenceWarning, module="sklearn"
                )
                cv_scores = cross_val_score(mlp, X, y, cv=kf, n_jobs=-1, verbose=1, scoring='recall')
            scores.append((np.mean(cv_scores), n_layers, learning_rate))

    return(scores)

def analyse_data(results, n_layers, learning_rate):
    """ analyse the data """
    fig = plt.figure()
    plt.title('parameters vs the results')
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('learning rate')
    ax.set_ylabel('n_layers')
    ax.set_zlabel('recall accuracy')
    ax.scatter(results[:,2], results[:,1],  results[:,0])
    plt.show()

    plt.title('ROC curve of MLPClassifier')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    kf = KFold(n_splits=10)
    mlp = MLPClassifier(hidden_layer_sizes=n_layers, learning_rate_init=learning_rate, max_iter=300)
    for i_train, i_test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[i_train], X[i_test], y[i_train], y[i_test]    
        mlp.fit(X_train,y_train)
        y_prob = mlp.predict_proba(X_test)
        print(y_prob)

        fpr, tpr, _  = roc_curve(y_test, y_prob[:, 0])
        print(f'auc score {roc_auc_score(y_test, y_prob[:,1])}')

        plt.scatter(tpr, fpr, label='MLP Classifier')
        # from https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
        plt.plot([0, 1], [0, 1], transform=ax.transAxes, label='random guess')
        plt.legend()

        break

    plt.show()

    
 

    


if __name__=="__main__":
    X_data = pd.read_csv(X_PATH)
    y_data = pd.read_csv(Y_PATH)

    X_attributes = X_data.columns.tolist()
    X = X_data.to_numpy()[:,1:]
    y = y_data.to_numpy()[:,1].astype('int')
    print(X)
    print(y)

    X, y = preprocess_data(X, y)
    

    # print('training model')
    # scores = optimal_parameters(X, y)
    # print(scores)
    # score, n_layers, learning_rate = max(scores)
    # print(f'optimal parameters are n_estimators: {n_layers}, learning_rate: {learning_rate} \
    #     with score {score}')

    # optimal parameters found: n_estimators: 66, learning_rate: 0.1
    n_layers = 190
    learning_rate = 0.0005
    analyse_data(np.array(RESULTS_MLP), n_layers, learning_rate)




