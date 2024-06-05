import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import sys
import datetime
import json
import pprint
import time
from sklearn.model_selection import StratifiedKFold
import shap

from sklearn import set_config
set_config(transform_output='pandas')

from utils import *
from preprocess_data import * 
from run_models import *

import warnings
warnings.filterwarnings('ignore')

plt.rcParams["figure.figsize"] = (20,12)
plt.rcParams['axes.grid'] = True
plt.style.use('fivethirtyeight')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['lines.linewidth'] = 3

def get_indexes(config):
    indexes = []
    for i in range(1, 3+1):
        train_indexes = np.load(os.path.join('..' , 'cv_indexes', f'train_indexes_{i}.npy'))
        test_indexes = np.load(os.path.join('..' , 'cv_indexes', f'test_indexes_{i}.npy'))

        aux = [train_indexes, test_indexes]
        indexes.append(aux)
    return indexes

def main(seed=42):
    experiences = os.listdir(os.path.join('logs'))
    # if logs is empty
    if len(experiences) == 0:
        exp = 'exp00'
    else:
        # in case the logs files doesn't contain the name "exp"
        try:
            num = int(sorted(experiences)[-1][3:]) + 1
            # in case num smaller than 10 to register the files in a scale of 00 to 09
            if num < 10:
                exp = f'exp0{num}'
            # otherwise
            else:
                exp = f'exp{num}'
        except:
            exp = f'exp{datetime.datetime.now()}'
    
     # crate the new experience folder
    LOGS_PATH = os.path.join('logs', exp)
    os.makedirs(LOGS_PATH)
    GRID_PATH = os.path.join(LOGS_PATH, 'grid_search')
    os.makedirs(GRID_PATH)
    PREDICTIONS_PATH = os.path.join(LOGS_PATH, 'predictions')
    os.makedirs(PREDICTIONS_PATH)
    IMPORTANCES_PATH = os.path.join(LOGS_PATH, 'importances')
    os.makedirs(IMPORTANCES_PATH)
    CM_PATH = os.path.join(LOGS_PATH, 'confusion-matrix')
    os.makedirs(CM_PATH)

    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    # save config in logs
    with open(os.path.join(LOGS_PATH, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=1)

    data = pd.read_csv(config['dataset'])
    data = data.sort_values(by='Processo')
    data = data.reset_index(drop=True)
    process_info = data.pop('Processo')
    X = data.copy()
    y = X.pop(config['target_column'])

    metadata = pd.read_csv(os.path.join('..', 'dataset', 'metadata_images_merge.csv'))
    metadata = metadata.sort_values(by='Processo')
    metadata = metadata.reset_index(drop=True)
    metadata_class = metadata.pop('Classificação')

    pipeline, X_transformed, y_transformed = create_fit_pipeline(config, X, y, seed=seed)

    models_to_test = config.get('models_names', [['LogisticRegression', {}]])
    use_grid_search = config.get('grid_search', False)
    balance_method = config.get('balance_dataset', None)

    start_total = time.time()
    best_models_results = pd.DataFrame()
    SHAP_values_per_fold = []
    ix_test = []

    for fold, (train_indexes, test_indexes) in enumerate(get_indexes(config)):
        print(f'############## {fold+1}-FOLD ##############')

        x_train = X_transformed.iloc[train_indexes]
        y_train = y_transformed.iloc[train_indexes]
        x_test = X_transformed.iloc[test_indexes]
        y_test = y_transformed.iloc[test_indexes]

        ix_test.append(test_indexes)

        if balance_method:
            x_train, y_train = dataset_balance(x_train, y_train, balance_method)

        start = time.time()
        count = 0
        for model_name, params in models_to_test:
            # create and train model
            model = instantiate_model(model_name, params, seed=seed)
            print('Model:', model_name)
            start_model = time.time()
            model.fit(x_train, y_train)
            end_model = time.time()
            # predict
            y_pred = model.predict(x_test)
            y_pred_proba = model.predict_proba(x_test)
            y_pred_proba = y_pred_proba[:, 1]

            if model_name not in  ['MLPClassifier', 'KNeighborsClassifier', 'GaussianNB', 'SVC', 'LogisticRegression']:
                importances = pd.DataFrame()
                importances['Features'] = model.feature_names_in_
                importances['Importance'] = model.feature_importances_
                importances.to_csv(os.path.join(IMPORTANCES_PATH, f'model_{model_name}_{count}_fold_{fold+1}.csv'), index=False)
            if config.get('test_shap', False) & len(models_to_test) == 1:
                    explainer = shap.Explainer(model, x_train)
                    shap_values = explainer.shap_values(x_test)
                    for SHAPs in shap_values:
                        SHAP_values_per_fold.append(SHAPs)


            metrics_df = train_predict_model(PREDICTIONS_PATH, y_test, y_pred, y_pred_proba, fold, f'{model_name}_{count}', params, end_model - start_model, process_info=process_info[test_indexes])
            cm = metrics.confusion_matrix(y_test, y_pred)
            np.savetxt(os.path.join(CM_PATH, f'model_{model_name}_{count}_fold_{fold}_cm.csv'), cm, delimiter=',')
            best_models_results = pd.concat([best_models_results, metrics_df], ignore_index=True)
            count += 1

        end = time.time()
        print(f'\n[{fold+1}-fold] Time to test all models: ', (end-start)/60, 'minutes')

    # plot overall shap values
    if config.get('test_shap', False):
        new_index = [ix for ix_test_fold in ix_test for ix in ix_test_fold]
        shap.summary_plot(np.array(SHAP_values_per_fold), X_transformed.reindex(new_index), show=False, max_display=15)
        plt.savefig(os.path.join(LOGS_PATH, 'shap_values.png'), transparent=True)

    best_models_results.to_csv(os.path.join(LOGS_PATH, 'model_metrics.csv'), index=False)
    ranking = get_best_models_config(best_models_results, config.get('num_best_models', 3))

    end_total = time.time()
    print(f'TOTAL TIME - {config["kfolds"], 10}-FOLDS:', (end_total - start_total) / 60, 'minutes')
    print("\n\n BEST CONFIGS \n")
    print(ranking)
    ranking.to_csv(os.path.join(LOGS_PATH, 'models_mean_results.csv'), index=False)

if __name__ == '__main__':
    main(seed=1257623)