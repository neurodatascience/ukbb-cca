
import warnings
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_lr_score(X_train, y_train, X_test, y_test, method='R2', sqrt=True, suppress_warnings=True):

    if suppress_warnings:
        warnings.filterwarnings('ignore', category=RuntimeWarning)

    valid_methods = ['R2', 'MSE']
    if method not in valid_methods:
        raise ValueError(f'Invalid method: {method}. Valid methods are in {valid_methods}')

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    if method == 'R2':
        score_train = lr.score(X_train, y_train)
        score_test = lr.score(X_test, y_test)
        if sqrt:
            score_train = np.sqrt(score_train)
            score_test = np.sqrt(score_test)
    elif method == 'MSE':
        squared = not sqrt # if sqrt is True, computes RMSE, else MSE
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)
        score_train = mean_squared_error(y_train, y_train_pred, squared=squared)
        score_test = mean_squared_error(y_test, y_test_pred, squared=squared)

    return score_train, score_test

def get_lr_score_components(components_train, y_train, components_test, y_test, method='R2', always_include_learn=None, always_include_test=None, n_to_check=1, cumulative=False, sqrt=True):
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

        if always_include_learn is not None:
            X_train = np.concatenate([X_train, np.concatenate(always_include_learn, axis=1)], axis=1)
            X_test = np.concatenate([X_test, np.concatenate(always_include_test, axis=1)], axis=1)
            # print(f'X_train with always_include_learn: {X_train.shape}')
            # print(f'X_test with always_include_test: {X_test.shape}')

        # print(X_train.shape, X_test.shape)
        score_train, score_test = get_lr_score(X_train, y_train, X_test, y_test, method=method, sqrt=sqrt)
        scores_train.append(score_train)
        scores_test.append(score_test)

    return np.array(scores_train), np.array(scores_test)
