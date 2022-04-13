
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression

def get_lr_score(X_train, y_train, X_test, y_test, sqrt=True, suppress_warnings=True):

    if suppress_warnings:
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    score_train = lr.score(X_train, y_train)
    score_test = lr.score(X_test, y_test)

    if sqrt:
        score_train = np.sqrt(score_train)
        score_test = np.sqrt(score_test)

    return score_train, score_test

def get_lr_score_components(components_train, y_train, components_test, y_test, n_to_check=1, cumulative=False, sqrt=True):
    '''components_[train/test] is a list of n_samples x n_components arrays. Length of list is number of datasets'''

    n_datasets = len(components_train)
    scores_train = []
    scores_test = []

    for i_components in range(n_to_check):
        
        # build the data
        if cumulative:
            X_train = np.concatenate([components_train[i][:,:i_components+1] for i in range(n_datasets)], axis=1)
            X_test = np.concatenate([components_test[i][:,:i_components+1] for i in range(n_datasets)], axis=1)
        else:
            X_train = np.concatenate([components_train[i][:,[i_components]] for i in range(n_datasets)], axis=1)
            X_test = np.concatenate([components_test[i][:,[i_components]] for i in range(n_datasets)], axis=1)

        # print(X_train.shape, X_test.shape)
        score_train, score_test = get_lr_score(X_train, y_train, X_test, y_test, sqrt=sqrt)
        scores_train.append(score_train)
        scores_test.append(score_test)

    return np.array(scores_train), np.array(scores_test)
