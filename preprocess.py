'''
    This file is to prepare data for a machine learning model, so it includes:
    - data pre-processing
    - feature engineering (creation, deletion, extraction, scaling)
'''
# Let's import libraries
import pandas as pd  # to handle dataframes
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import seaborn as sns
import datetime  # manipulating date formats
import warnings

warnings.filterwarnings("ignore")


# Let's build a function to differentiate the categorical and numerical columns of the dataset
def __cat_or_num__(df):
    '''
    Function that clasifies the kind of columns (categorical or numerical) that exist in a dataset
    Input:
            df: dataset where we want to clasify columns
    Output:
            num_cols,cat:cols: it returns a dictionary with the category of every column

    '''
    cat_cols = []
    num_cols = []

    num_cols = [i for i in df.columns if df.dtypes[i] != 'object']
    cat_cols = [i for i in df.columns if df.dtypes[i] == 'object']

    return cat_cols, num_cols


# get information about the cols and features
def show_cols_info(df):
    '''
    Function that shows the summary of the information of the columns of the datasets
    Input:
            df: dataset we want to show information
    Output:
            none: it prints a summary and the main values of every column

    '''

    for column in df.columns:
        # show a summary of the important information of the column
        df.describe([column]).show()
        # show the different values of the field
        df.select([column]).distinct().show()


def __load_data__(type):
    '''
    Function that loads data into Pandas dataframes
    Input: type of dataset (train/test) we want to load
    Output: df dataset is loaded into memory
    '''
    if type == 'train':
        items = pd.read_csv(os.path.join('./data/', 'items.csv'), low_memory=False, parse_dates=True)
        categories = pd.read_csv(os.path.join('./data/', 'item_categories.csv'), low_memory=False, parse_dates=True)
        data = pd.read_csv(os.path.join('./data/', 'sales_train.csv'), low_memory=False, parse_dates=True)
        shops = pd.read_csv(os.path.join('./data/', 'shops.csv'), low_memory=False, parse_dates=True)

    elif type == 'test':
        items = pd.read_csv(os.path.join('./data/', 'items.csv'), low_memory=False, parse_dates=True)
        categories = pd.read_csv(os.path.join('./data/', 'item_categories.csv'), low_memory=False, parse_dates=True)
        data = pd.read_csv(os.path.join('./data/', 'test.csv'), low_memory=False, parse_dates=True)
        shops = pd.read_csv(os.path.join('./data/', 'shops.csv'), low_memory=False, parse_dates=True)

    else:
        raise ValueError('dataset type must be "train" or "test"')

    # merge dataframes to get store information in both training and testing sets
    ##df_store = pd.read_csv(os.path.join('../../Rossmann/data/', 'store.csv'), low_memory=False, parse_dates=True)
    df = items.merge(categories, on='item_category_id')
    df = df.merge(data,on='item_id')
    df = df.merge(shops,on='shop_id')

    return df


def __handle_categorical__(df):
    '''
    Function to select categorical variables and encode them
    Input: df: Pandas dataframe containing categorical variables
    Output: modified StateHoliday, StoreType and Assortment columns
    '''

    # define a dict where the keys are the element to encode, and the values are the targets
    mappings = {'0': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4}

    # now we can encode the categorical features
    df['StateHoliday'].replace(mappings, inplace=True)
    df['StoreType'].replace(mappings, inplace=True)
    df['Assortment'].replace(mappings, inplace=True)

    return df


def __preprocess_features__(df):
    '''
    Function to add information to the date such as month, day, etc ...
    Input:  df: Pandas dataframe
    Output: df: Pandas dataframe

    '''

    # Now let's change some formats
    # Let's format the date column
    df['date'] = pd.to_datetime(df['date'])

    # Let's keep only items with price>0
    df = df[df['item_price']>0]

    # The train set has information daily grained but we are asked to
    # predict total sales for every product and store in the next month
    # so we need to aggregate data by month
    df_month = df[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]
    df_month = df_month.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], \
                                                    as_index=False)
    df_month = df_month.agg({'item_price': ['sum', 'mean'], 'item_cnt_day': ['sum', 'mean', 'count']})
    # Rename features
    df_month.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price',
                             'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']
    df = df_month
    # Let's augment with some more information

    df['Day'] = df['date'].apply(lambda x: x.day)
    df['Month'] = df['date'].apply(lambda x: x.month)
    df['Year'] = df['date'].apply(lambda x: x.year)
    df['DayOfWeek'] = df['date'].apply(lambda x: x.dayofweek)
    df['Week'] = df['date'].apply(lambda x: x.weekofyear)

    df['is_quarter_start'] = df['date'].dt.is_quarter_start
    df['is_quarter_end'] = df['date'].dt.is_quarter_end
    df['is_month_start'] = df['date'].dt.is_month_start
    df['is_month_end'] = df['date'].dt.is_month_end
    df['is_year_start'] = df['date'].dt.is_year_start
    df['is_year_end'] = df['date'].dt.is_year_end

    df.is_quarter_start = df.is_quarter_start.apply(lambda x: 0 if x == False else 1)
    df.is_quarter_end = df.is_quarter_end.apply(lambda x: 0 if x == False else 1)
    df.is_month_start = df.is_month_start.apply(lambda x: 0 if x == False else 1)
    df.is_month_end = df.is_month_end.apply(lambda x: 0 if x == False else 1)
    df.is_year_start = df.is_year_start.apply(lambda x: 0 if x == False else 1)
    df.is_year_end = df.is_year_end.apply(lambda x: 0 if x == False else 1)

    df.drop(['date'], inplace=True, axis=1)

    # Let's eliminate outliers
    # Those are items with price >100000 and items with item_cnt_day>1001
    df = df[df.item_price < 100000]
    df = df[df.item_cnt_day < 1001]

    # There are prices below 0, let's impute with the median

    median_value = df[(df.shop_id == 32) & (df.item_id == 2973) & (df.date_block_num == 4) \
                      & (df.item_price > 0)].item_price.median()
    df.loc[df.item_price < 0, 'item_price'] = median_value

    # Let's add the total sales for an item
    df['sales'] = df['item_price'] * df['item_cnt_day']

    return df


def build_dataset(type):
    '''
    Function to build the dataset
    Input:  df: Pandas dataframe
    Output: df: Pandas dataframe
    '''
    df = __load_data__(type)
    print('Data loaded')
  #  df = __handle_categorical__(df)
    df = __preprocess_features__(df)

    return df


def save_dataset(df, filename):
    '''
    Function to save a dataset to a csv file
    Input: df: dataset to save
    Output: filename: file to save the dataset
    '''

    path = os.path.join('', filename)
    df.to_csv(path, index=False)
    print('Saving dataset to:', filename)


class Preprocessor:
    def __init__(self):
        self.data_stats = {}
        pass

    def fit(self, data):
        '''
        We suppose that the train and test data come from the same distribution, so we need to fit the preprocessing
        on the train data and later transform the test data with respect to the distribution of the train data.
        This method saves train data statistics that will be needed to fill missing values

        Input data: data set from which statistics are saved
        '''

        print('Fitting data...')
        # save the mean of this column for transform
     #   self.data_stats['MonthsSinceCompetitionMean'] = math.floor(data['MonthsSinceCompetition'].mean())
        self.data_stats['timestamp'] = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

        # save data_stats to pickle file.
        # this file will be necessary when preprocessing test data
        print('Saving data stats to pickle file...')
        data_stats_file = open('data_stats.pkl', 'wb')
        pickle.dump(self.data_stats, data_stats_file)
        data_stats_file.close()
        print('Fitting data done.')

    def transform(self, data):
        """
        Fills missing values with means saved from training data and scales target
        :param data: dataset to transform
        """

        # if object has not been fit prior to transform call, load data stats from pickle file
        if not self.data_stats:
            data_stats_file = open('data_stats.pkl', 'rb')
            self.data_stats = pickle.load(data_stats_file)
            data_stats_file.close()

        print('Transforming data with training data statistics saved on:', self.data_stats['timestamp'])

        print('Transforming data done!')

        return data


def __get_most_important_features__(X_train, y_train, n_features):
    '''
    Perform feature selection using XGBoost algorithm
    '''
    model = XGBClassifier()
    model.fit(X_train, y_train)
    sorted_idx = np.argsort(model.feature_importances_)[::-1]
    sorted_idx = sorted_idx[:n_features]
    return X_train.columns[sorted_idx]


# unit test
if __name__ == "__main__":
    data = build_dataset('train')
    preprocessor = Preprocessor()
    preprocessor.fit(data)
    data = preprocessor.transform(data)