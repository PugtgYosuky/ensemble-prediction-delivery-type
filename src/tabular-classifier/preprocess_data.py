import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

# balance datasets
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest
from sklearn.impute import SimpleImputer


class DropColumns(BaseEstimator):
    """ Class to drop columns from the dataset"""
    def __init__(self, columns=[], threshold=0.8):
        self.threshold = threshold
        self.columns = columns
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns += list(X.columns[X.nunique() == 1])
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X)
        preprocessed = X.drop(self.columns, axis=1)
        return preprocessed



def dataset_balance(X ,y , model, seed=42):
    if model == "SMOTE":
        smote = SMOTE(random_state=seed)
        X_resampled, y_resampled = smote.fit_resample(X, y) 

    elif model == "SMOTETomek":
        smotetomek = SMOTETomek(random_state=seed)
        X_resampled, y_resampled = smotetomek.fit_resample(X, y)

    elif model == "SMOTEENN":
        smoteenn = SMOTEENN(random_state=seed)
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)

    elif model == "RandomUnderSampler":
        smoteenn = RandomUnderSampler(random_state=seed)
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)

    return X_resampled, y_resampled


def create_fit_pipeline(config, X, y, seed=42):
    """ Creates the pipeline and fits the training data"""
    # set parameters
    norm_model_name = config.get('norm_model', 'Standard') # try to get 'norm_model' from config files, uses 'Standart' if not founded

    # select which scaler to use
    if norm_model_name == 'MinMax':
        print('MinMaxScaler')
        norm_model = MinMaxScaler()
    elif   norm_model_name == 'Robust':
        print('RobustScaler')
        norm_model = RobustScaler()
    else:
        print('StandardScaler')
        norm_model = StandardScaler()

    # columns types
    numerical_columns = config.get('numeric_features')
    drop = DropColumns(config.get('columns_to_drop', []))
    # pipeline of Preprocessing(Normalization, Feature Selection, Variance Selection, Scaler)

    # !APENAS normalizar as features numericas, as restantes já estao codificadas
    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', norm_model, numerical_columns)
        ],
        remainder='passthrough',
        verbose_feature_names_out = False # dont use the prefix in the out features
    )
    pipeline = Pipeline(steps=[
        ('remove_cols', drop),
        # ('feature_selection', SelectKBest(k=config.get('number_best_features', 'all'))),
        # ('variance_select', VarianceThreshold(threshold=config.get('variance_threshold', 0))),
        ('scaler_preprocessor', preprocessor)
    ])

    y_transformed = y
    X_transformed = pipeline.fit_transform(X, y_transformed)
    return pipeline, X_transformed, y_transformed
    
