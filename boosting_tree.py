from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import X_PATH, Y_PATH, RESULTS_BOOSTING_TREE
from test_performance import calc_accuracy



def preprocess_data(X, y):
    """ preprocessing specifically for boosting tree """
    return X, y

def optimal_parameters(X, y):
    """Train the boosting tree model using x-fold cross validation """

    # the parameters to be optimized
    n_estimators_range = list(range(2, 101, 2))
    learning_rate_range = list(np.arange(0.1, 1.1, 0.1))

    kf = KFold(n_splits=10)

    scores = []

    for n_estimators in n_estimators_range:
        for learning_rate in learning_rate_range:
            print(f'at n_estimators = {n_estimators}, learning_rate = {learning_rate}')
            abc = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
            cv_scores = cross_val_score(abc, X, y, cv=kf, n_jobs=-1, verbose=1, scoring='recall')
            scores.append((np.mean(cv_scores), n_estimators, learning_rate))

    print(scores)
    return max(scores)

def analyse_data(results, n_estimators, learning_rate, X, y):
    """ analyse the data """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plt.title('parameters vs the results')
    ax.set_ylabel('n estimators')
    ax.set_xlabel('learning rate')
    ax.set_zlabel('recall accuracy')
    ax.scatter(results[:,2], results[:,1],  results[:,0])
           
    plt.show()

    plt.title('ROC curve of Boosting Tree')
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')    
    kf = KFold(n_splits=10)
    abc = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    for i_train, i_test in kf.split(X, y):
        X_train, X_test, y_train, y_test = X[i_train], X[i_test], y[i_train], y[i_test]    
        abc.fit(X_train,y_train)
        y_prob = abc.predict_proba(X_test)
        print(y_prob)

        fpr, tpr, _  = roc_curve(y_test, y_prob[:, 0])
        print(f'auc score {roc_auc_score(y_test, y_prob[:,1])}')

        plt.scatter(tpr, fpr, label='AdaBooster')
        # from https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
        plt.plot([0, 1], [0, 1], transform=ax.transAxes, label='Random Guess')
        plt.legend()
        break

    
    plt.show()

    ax = fig.add_subplot()


    

    


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
    # score, n_estimators, learning_rate = optimal_parameters(X, y)
    # print(f'optimal parameters are n_estimators: {n_estimators}, learning_rate: {learning_rate} \
    #     with score {score}')

    # optimal parameters found: n_estimators: 66, learning_rate: 0.1
    n_estimators = 66
    learning_rate = 0.1
    analyse_data(np.array(RESULTS_BOOSTING_TREE), n_estimators, learning_rate, X, y)




