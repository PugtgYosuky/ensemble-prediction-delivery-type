
import numpy as np

from sklearn import metrics

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def instantiate_model(model_name, settings={}, seed=42):
    """ Instantiates the classifier given it's name and settings"""
    if model_name == 'LogisticRegression':
        model = LogisticRegression(**settings, random_state=seed)

    elif model_name == 'RandomForestClassifier':
        model = RandomForestClassifier(**settings, random_state=seed)

    elif model_name == 'KNeighborsClassifier':
        model = KNeighborsClassifier(**settings)

    elif model_name == 'DecisionTreeClassifier':
        model = DecisionTreeClassifier(**settings, random_state=seed)

    elif model_name == 'GaussianNB':
        model = GaussianNB(**settings)
    
    elif model_name == 'SVC':
        model = SVC(**settings, random_state=seed, probability=True)

    elif model_name == 'AdaBoostClassifier':
        model = AdaBoostClassifier(**settings, random_state=seed)

    elif model_name == 'XGBClassifier':
        model = XGBClassifier(**settings, random_state=seed)
    
    elif model_name == 'MLPClassifier':
        model = MLPClassifier(**settings, random_state=seed)
    
    return model


def calculate_metrics (y_test, y_pred, y_proba):
    """ calculates the metrics of the predictions """
    tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()
    return {
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
          'tn' : tn
    }


def majority_voting(results, weights):
    """ Function from the professor slides"""
    aux = np.bincount(results, weights=weights)
    return np.argmax(aux)