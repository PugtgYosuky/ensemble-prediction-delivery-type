import json
import numpy as np
from sklearn import metrics
import os
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

def set_global_determinism(seed):
    """
    Fix seeds for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

class NpEncoder(json.JSONEncoder):
    # from https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)
    

def write_metrics(y_test, y_pred, y_proba, path, time, prefix):
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    results =  {
          'balanced_accuracy' : metrics.balanced_accuracy_score(y_test, y_pred),
          'roc_auc_score' : metrics.roc_auc_score(y_test, y_proba),
          'recall_weighted' : metrics.recall_score(y_test, y_pred, average='weighted'), 
          'f1_weighted' : metrics.f1_score(y_test, y_pred, average='weighted'), 
          'precision_weighted' : metrics.precision_score(y_test, y_pred, average='weighted'), 
          'matthews_corrcoef' : metrics.matthews_corrcoef(y_test, y_pred),
          'specificity' : tn / (tn+fp),
          'NPV' : tn / (tn + fn),
          'tp' : tp,
          'fp' : fp,
          'fn' : fn,
          'tn' : tn,
          'time_seconds' : time
    }

    with open(os.path.join(path, f'{prefix}_results_metrics.json'), 'w') as file:
        json.dump(results, file, indent=1, cls=NpEncoder)


def write_predictions(test_data, y_pred, y_proba, path, prefix):
    # writes predictions
    test_df = pd.DataFrame()
    y_test = test_data.classes
    test_df['number'] = np.arange(len(y_test))
    test_df[f'y_{prefix}'] = y_test
    test_df['y_pred'] = y_pred
    test_df['y_proba'] = y_proba
    test_df['image'] = test_data.filenames
    test_df.to_csv(os.path.join(path, f'{prefix}_prediction.csv'), index=False)
    return test_df

def plot_evolution(history, path, prefix=''):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(path, f'{prefix}_train_results.png'))
    plt.close()