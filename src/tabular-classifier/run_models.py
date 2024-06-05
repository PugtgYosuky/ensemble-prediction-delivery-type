import time
import os
from utils import *
from preprocess_data import *
import pandas as pd
import pprint
import json
from sklearn.model_selection import GridSearchCV

def grid_search(model, params, x_train, y_train, model_name, kfold, GRID_PATH):
    """ Performes grid search to find the best model for the parameters"""
    scoring = {
        'f1_weighted' : 'f1_weighted',
        'balanced_accuracy' : 'balanced_accuracy',
        'matthews_corrcoef' : 'matthews_corrcoef',
        'recall_weighted' : 'recall_weighted',
        'precision_weighted' : 'precision_weighted',
        'roc_auc' : 'roc_auc'

    }
    start = time.time()
    grid_search = GridSearchCV(
        estimator   = model, 
        param_grid  = params,
        scoring     = scoring,
        refit       = 'matthews_corrcoef',
        cv          = kfold,
        verbose     = 2,
        
    )

    # fitting the model    
    grid_search.fit(x_train, y_train)
    end = time.time()
    print(f'Time to test grid search {model_name}: {(end - start) / 60} minutes')

    # store the results of grid search
    cv_results = grid_search.cv_results_

    # create a pandas dataframe with the parameters and its means of the model tested
    df = pd.DataFrame(cv_results)
    df.to_csv(os.path.join(GRID_PATH, f'{model_name}_grid_search_{time.time()}.json'), index=False)
    
    metrics = ['f1_weighted','matthews_corrcoef','balanced_accuracy','precision_weighted','recall_weighted', 'roc_auc']
    metrics_names = [f'mean_test_{metric}' for metric in metrics]
    # saves the results
    results = pd.DataFrame()
    results['model'] = [model.__class__.__name__] * len(cv_results['params'])
    results['config'] = cv_results['params']
    results['fold'] = 0
    for metric, cv_name in zip(metrics, metrics_names):
        results[metric] = cv_results[cv_name]

    results['time(minutes)'] = cv_results['mean_fit_time'] + cv_results['mean_score_time']
    results['count'] = 5
    sorted_results = results.sort_values(by=['matthews_corrcoef', 'f1_weighted'], ascending=False)
    sorted_results = sorted_results.iloc[:min(len(sorted_results), 5)] # selects up to 5 models (the best ones)
    return sorted_results


def get_best_models_config(data, best_num=5):
    """ returns the best config for each model tested according to the cross-validation results"""
    compare = data.groupby(['model', 'config']).mean()
    compare['count'] = data.groupby(['model', 'config'])['fold'].count()
    metrics = ['roc_auc_score','f1_weighted','matthews_corrcoef', 'balanced_accuracy', 'precision_weighted', 'recall_weighted', 'specificity', 'NPV', 'tp', 'fp', 'tn', 'fn']
    aux_std = data.groupby(['model', 'config']).std()
    for metric in metrics:
        compare[f'std_{metric}'] = aux_std[metric]
    compare = compare.reset_index()
    compare = compare.sort_values(['count', 'roc_auc_score'], ascending=False)
    return compare


def train_predict_model(save_path, y_test, y_pred, y_proba, fold, model_name, params, total_time, process_info=None):
    """ Calculates the metrics of the predictions"""
    predictions = pd.DataFrame()
    predictions['y_test'] = y_test
    predictions['y_pred'] = y_pred
    predictions['y_pred_proba'] = y_proba
    if process_info is not None:
        predictions['Processo'] = list(process_info)
    predictions.to_csv(os.path.join(save_path, f'{model_name}_fold{fold+1}_predictions.csv'), index=False)
    # calculate metrics
    model_metrics = calculate_metrics(y_test, y_pred, y_proba)
    model_metrics['time(minutes)'] = (total_time) / 60 # add fitting time to the evaluation
    
    model_metrics['fold'] = fold + 1 # save fold
    model_metrics['config'] = json.dumps(params) # save config 
    model_metrics['model'] = model_name
    pprint.pprint(model_metrics)
    metrics = pd.DataFrame(model_metrics, index=[0])
    # reorganize dataframe
    metrics = metrics[['model', 'config', 'fold', 'roc_auc_score', 'f1_weighted','matthews_corrcoef', 'balanced_accuracy','precision_weighted','recall_weighted', 'specificity', 'NPV', 'tp', 'fp', 'tn', 'fn', 'time(minutes)']]
    return metrics