"""
    This file is to train data with a machine learning model
"""
# Let's import libraries
import pickle
import pandas as pd

from xgboost import XGBRegressor
from sklearn import linear_model
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from preprocess import Preprocessor, build_dataset, save_dataset

import warnings
warnings.filterwarnings("ignore")

class Model(BaseEstimator, RegressorMixin):
    '''
    scikit-learn estimator for the Rossmann's stores prediction
    Parameters
    ----------
    alpha : float
        The regularization parameter for ridge and lasso regression
    max_iter : int
        The number of iterations / epochs to do on the data.
    solver : 'xgb' | 'lasso' | 'ridge' | 'linear'
    '''

    def __init__(self, max_iter=2000, solver='xgb', alpha=0.1):
        self.max_iter = max_iter
        self.alpha = alpha
        self.solver = solver
        self.model = None
      #  assert self.solver in ['xgb', 'lasso', 'ridge', 'linear']
        assert self.solver in ['xgb', 'lasso', 'ridge', 'linear']

    def fit(self, X, y):
        '''
        Fit method
        Input: ndarray, shape (n_samples, n_features) # The features
        Output: y  ndarray, shape (n_samples,)        # The target
        '''

        if self.solver == 'xgb':
            self.model = XGBRegressor(objective="reg:squarederror")
            self.model.fit(X, y)

        elif self.solver == 'lasso':
            self.model = linear_model.Lasso(alpha=self.alpha, max_iter=self.max_iter)
            self.model.fit(X, y)

        elif self.solver == 'ridge':
            self.model = linear_model.Ridge(alpha=self.alpha, max_iter=self.max_iter)
            self.model.fit(X, y)

        elif self.solver == 'linear':
            self.model = linear_model.LinearRegression()
            self.model.fit(X, y)

        return self

    def predict(self, X):
        '''Prediction method
            Input:   X : ndarray, shape (n_samples, n_features) # The features
            Output:  y_pred : ndarray, shape (n_samples,)       # The predicted target
        '''
        return self.model.predict(X)


if __name__ == "__main__":

    # load data
    print('Loading data...')
    data = build_dataset('train')
    train_data, valid_data = train_test_split(data, test_size=0.2, random_state=42)
    preprocessor = Preprocessor()
    print('Preprocessor initialization finished')
    preprocessor.fit(train_data)
    print('Preprocessor fitting finished')
    train_data = preprocessor.transform(train_data)
    valid_data = preprocessor.transform(valid_data)
    save_dataset(pd.concat([train_data, valid_data]), './data/train_preprocessed.csv')

    X_train = train_data.drop(['item_id', 'item_cnt_month'], axis=1)
    X_valid = valid_data.drop(['item_id', 'item_cnt_month'], axis=1)
    y_train = train_data['item_id']
    y_valid = valid_data['item_id']

    print('Training model on', len(X_train), 'samples')
    print('Validating model on', len(X_valid), 'samples')
    print('Training model on features: ', X_train.columns.tolist())

    # model selection with grid search
    solvers = ['xgb', 'lasso', 'ridge', 'linear']
    best_score, best_model = 0, (None, None)
    for solver in solvers:
        print('Solver:', solver)
        model = Model(solver=solver)
        model.fit(X_train, y_train)
        model_r2 = model.score(X_valid, y_valid)
        print('r2:', model_r2)
        preds = model.predict(X_valid)
        model_mse = mean_squared_error(y_valid, preds)
        print('mse:', model_mse)

        # keep track of best model
        if model_r2 > best_score:
            best_model = (solver, model)
            best_score = model_r2

    # save best model
    print('Best solver:', best_model[0])
    print('Saving best model to pickle file')
    model_file = open('model.pkl', 'wb')
    model = pickle.dump(best_model[1], model_file)
    model_file.close()
    print('Done!')