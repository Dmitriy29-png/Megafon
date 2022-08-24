import pandas as pd
import numpy as np

import dask.dataframe as dd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from sklearn.metrics import f1_score, classification_report

from sklearn.feature_selection import SelectFromModel


def preprocess_data_train(prep_data_df, FEATURES_DATA):
    prep_data_df = prep_data_df.drop('Unnamed: 0', axis=1)
    prep_data_df = prep_data_df.sort_values('buy_time')
    prep_data_df['not_first_offer'] = prep_data_df.duplicated('id').astype(int)

    prep_data_df['date'] = pd.to_datetime(prep_data_df['buy_time'], unit='s')

    features_data_df = dd.read_csv(FEATURES_DATA, sep='\t')
    features_data_df = features_data_df.drop('Unnamed: 0', axis=1)
    train_list_index = list(prep_data_df['id'].unique())
    features_data_df = features_data_df.loc[features_data_df['id'].isin(train_list_index)].compute()
    features_data_df = features_data_df.sort_values(by="buy_time")

    prep_data_df = prep_data_df.sort_index()

    prep_data_df = prep_data_df.sort_values(by="buy_time")

    result_data = pd.merge_asof(prep_data_df, features_data_df, on='buy_time', by='id', direction='nearest')

    result_data ['week_on_month'] = result_data ['date'].apply(lambda x: pd.to_datetime(x).day // 7)
    result_data ['day'] = result_data ['date'].apply(lambda x: pd.to_datetime(x).day)
    result_data ['month'] = result_data ['date'].apply(lambda x: pd.to_datetime(x).month)
    result_data  = result_data .drop('date', axis=1)
    result_data = result_data .drop('buy_time', axis=1)
    
    return result_data, train_list_index


def preprocess_data_test(prep_data_df, FEATURES_DATA, train_list_index):
    prep_data_df = prep_data_df.drop('Unnamed: 0', axis=1)
    prep_data_df = prep_data_df.sort_values('buy_time')
    prep_data_df['not_first_offer'] = (prep_data_df['id'].isin(train_list_index)).astype(int)

    prep_data_df['date'] = pd.to_datetime(prep_data_df['buy_time'], unit='s')

    features_data_df = dd.read_csv(FEATURES_DATA, sep='\t')
    features_data_df = features_data_df.drop('Unnamed: 0', axis=1)
    train_list_index = list(prep_data_df['id'].unique())
    features_data_df = features_data_df.loc[features_data_df['id'].isin(train_list_index)].compute()
    features_data_df = features_data_df.sort_values(by="buy_time")

    prep_data_df = prep_data_df.sort_index()

    prep_data_df = prep_data_df.sort_values(by="buy_time")

    result_data = pd.merge_asof(prep_data_df, features_data_df, on='buy_time', by='id', direction='nearest')

    result_data['week_on_month'] = result_data['date'].apply(lambda x: pd.to_datetime(x).day // 7)
    result_data['day'] = result_data['date'].apply(lambda x: pd.to_datetime(x).day)
    result_data['month'] = result_data['date'].apply(lambda x: pd.to_datetime(x).month)
    result_data = result_data.drop('date', axis=1)
    result_data = result_data.drop('buy_time', axis=1)
    
    return result_data


def select_type_cols(merged_data):
    X_nunique = merged_data.apply(lambda x: x.nunique(dropna=False))
    f_all = set(X_nunique.index.tolist())
    f_const = set(X_nunique[X_nunique == 1].index.tolist())
    f_other = f_all - f_const
    f_binary = set(merged_data.loc[:, f_other].columns[(
            (merged_data.loc[:, f_other].max() == 1) & \
            (merged_data.loc[:, f_other].min() == 0) & \
            (merged_data.loc[:, f_other].isnull().sum() == 0))])
    f_other = f_other - f_binary
    f_categorical = set(X_nunique.loc[f_other][X_nunique.loc[f_other] <= 5].index.tolist())
    f_other = f_other - f_categorical
    f_numeric = (merged_data[f_other].fillna(0).astype(int).sum() - merged_data[f_other].fillna(0).sum()).abs()
    f_numeric = set(f_numeric[f_numeric > 0].index.tolist())
    f_other = f_other - f_numeric
    f_numeric = f_numeric | f_other

    return f_all, f_binary, f_categorical, f_numeric


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            return X[self.columns]
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError("DataFrame не содердит следующие колонки: %s" % cols_error)