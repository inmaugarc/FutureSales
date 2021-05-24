"""
    This file is to predict data with a machine learning model
"""
import pickle
import numpy as np
import pandas as pd
from preprocess import Preprocessor, build_dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from training import Model


def check_model_drift(df_train, df_test, threshold):
    '''
    Function to change whether or not the data distribution of the test set is the same as the distribution of the
    training set. To do so, we will run a random forest model that will try to predict if a data point belongs to the
    training set or to the test set. If the model is too confident at discriminating the two sets, the sales prediction
    model needs to be retrained.
    Input:df_train: training set
          df_test: test set
          threshold: threshold of confidence above which the sales predictor has to be retrained
    '''

    df_train['isTrain'] = True
    df_train.drop(columns=['sales', 'Customers'], inplace=True)

    df_test['isTrain'] = False
    df_full = pd.concat([df_train, df_test])

    clf = RandomForestClassifier(n_estimators=3, random_state=42)
    X = df_full.drop(columns=['isTrain'])
    y = df_full['isTrain']
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')

    df_test.drop(columns=['isTrain'], inplace=True)

   # if scores.mean() >= threshold:
   #     print("WARNING: predictions will be meaningless because the distribution of the data has changed since the "
   #           "last time the model was trained")


if __name__ == "__main__":

    test_data = build_dataset('test')

    output = {'Id': test_data['Id'].tolist()}

    preprocessor = Preprocessor()
    test_data = preprocessor.transform(test_data)
    test_data.drop(columns=['Id'], inplace=True)

    threshold = 0.8
    train_data = pd.read_csv('../../Rossmann/data/train_preprocessed.csv')
    check_model_drift(train_data, test_data, threshold)

    model_file = open('model.pkl', 'rb')
    model = pickle.load(model_file)
    model_file.close()

    preds = model.predict(test_data)
    # sales have been scaled using logarithm during preprocessing, so we need to scale them back using exponential
    output['Predictions'] = np.exp(preds)

    output_df = pd.DataFrame(output)

    # save predictions to csv file
    preds_file = 'predictions.csv'
    output_df.to_csv(preds_file, index=False)

    print('Predictions saved to:', preds_file)