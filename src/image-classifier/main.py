import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import sys
import json
import pprint
import random
import time
import seaborn as sns
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# deep architectures
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3

from utils import *

# plot parameters
plt.rcParams["figure.figsize"] = (20,12)

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# fix seeds: code from https://stackoverflow.com/questions/36288235/how-to-get-stable-results-with-tensorflow-setting-random-seed


def get_layer(layer_config):
    """
    Converts a layers form the config file into a keras layer
    """
    layer_config = layer_config.copy()
    layer_name = layer_config.pop('type')
    layer_func = getattr(tf.keras.layers, layer_name)
    layer = layer_func(**layer_config)
    return layer

def get_model(config, width, height):
    """
    Instantiates a keras model
    """
    loss = getattr(tf.keras.losses, config['loss'])
    optimizer = getattr(tf.keras.optimizers.legacy, config['optimizer'])
    model_name = config.get('deep-model', None)

    
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(width, height, 3)))
    base_model = None
    model.add(layers.RandomFlip(mode='horizontal'))
    model.add(layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)))
    #model.add(layers.RandomContrast(factor=0.2))
    model.add(layers.RandomRotation(factor=0.2))
    if config.get('resizing-width', None):
        width = config.get('resizing-width')
        height = config.get('resizing-height')
        model.add(layers.Resizing(width=width, height=height))
    if model_name is None:
        for layer in config["layers"]:
            model.add(get_layer(layer))
    else:
        # add depp-model architecture
        if model_name == 'vgg16':
            base_model = VGG16(
                weights=config.get('weights', 'imagenet'), 
                include_top=False,
                pooling = 'max',
                input_shape=(width, height, 3),
                )
        elif model_name == 'vgg19':
            base_model = VGG19(
                weights=config.get('weights', 'imagenet'), 
                include_top=False,
                pooling = 'max',
                input_shape=(width, height, 3),
                )
        elif model_name == 'resnet50': 
            base_model = ResNet50(
                weights=config.get('weights', 'imagenet'), 
                include_top=False,
                pooling = 'max',
                input_shape=(width, height, 3),
                )
        elif model_name == "xception":
            base_model = Xception(
                weights=config.get('weights', 'imagenet'), 
                include_top=False,
                pooling = 'max',
                input_shape=(width, height, 3),
                )  
        elif model_name == 'inception':
            base_model = InceptionV3(
                weights=config.get('weights', 'imagenet'), 
                include_top=False,
                pooling='max',
                input_shape=(width, height, 3)
            )
        if config.get('transfer-learning', True):
            base_model.trainable = False
        else:
            base_model.trainable = True
        model.add(base_model)
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(
        optimizer = optimizer(learning_rate=config['learning_rate']),
        loss = loss(),
        metrics = [config['metrics'], tf.keras.metrics.AUC()],
    )
    print(model.summary())
    return model, base_model


def fit_and_predict_dataset(config, train_data, validation_data, test_data, width, height, prefix='', path='', seed=42):
    """
    Train and predict given a certain dataset
    """

    model, base_model = get_model(config, width, height)
    loss = getattr(tf.keras.losses, config['loss'])
    early_stopping_loss = EarlyStopping(
        monitor='val_loss',
        patience = config['patience'],
        restore_best_weights = True
    )
    best_model_path = os.path.join(path, 'best_model.h5')
    model_checkpoint = ModelCheckpoint(filepath=best_model_path,
                        save_best_only=True,
                        monitor='val_loss',
                        mode='min',
                        verbose=1)
    start = time.time()
    with tf.device('/device:GPU:0'):
        history = model.fit(
            train_data,
            steps_per_epoch = len(train_data),
            epochs = config['epochs'],
            callbacks = [early_stopping_loss, model_checkpoint],
            validation_data=validation_data,
            validation_steps=len(validation_data)
        )
        with open(os.path.join(path, f'{prefix}_history.json'), 'w') as outfile:
            json.dump(history.history, outfile, indent=1)

        model.save(os.path.join(path, f'{prefix}_model.h5'))
        with open(os.path.join(path, f'{prefix}_model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))

        plot_evolution(history, path)

        if config.get('transfer-learning', True) and config.get('fine-tuning', False):
            # fine tune the deep architecture
            with open(os.path.join(path, f'prefix_history-prev-finetuning.json'), 'w') as outfile:
                json.dump(history.history, outfile, indent=1)
            base_model.trainable = True

            model.compile(
                optimizer = tf.keras.optimizers.Adam(learning_rate=config.get('learning_rate', 0.001)),
                loss = loss(),
                metrics = [config['metrics'], tf.keras.metrics.AUC()]
            )

            early_stopping_loss = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            model_checkpoint = ModelCheckpoint(filepath=best_model_path,
                        save_best_only=True,
                        monitor='val_loss',
                        mode='min',
                        verbose=1)

            history = model.fit(
                train_data,
                steps_per_epoch = len(train_data),
                epochs = config['epochs-finetuning'],
                callbacks = [early_stopping_loss, model_checkpoint],
                validation_data=validation_data,
                validation_steps=len(validation_data)
            )

            with open(os.path.join(path, f'{prefix}_fine_tuning_history.json'), 'w') as outfile:
                json.dump(history.history, outfile, indent=1)

            model.save(os.path.join(path, f'{prefix}_fine_tuning_model.h5'))
            with open(os.path.join(path, f'{prefix}_fine_tuning_model_summary.txt'), 'w') as f:
                model.summary(print_fn=lambda x: f.write(x + '\n'))

            plot_evolution(history, path, prefix='fine_tuning')

        
        end = time.time()

            # load best model
        model = load_model(best_model_path)

        # test predictions
        predictions = model.predict(test_data)
        preds = (predictions > 0.5).astype(int)
        # train predictions
        predictions_train = model.predict(train_data)
        preds_train = (predictions_train > 0.5).astype(int)

        # save test metrics
        write_metrics(test_data.classes, y_pred=preds, y_proba=predictions, path=path, time=end-start, prefix='test')
        write_predictions(test_data=test_data, y_pred=preds, y_proba=predictions, path=path, prefix='test')

        # save train metrics
        write_metrics(train_data.classes, y_pred=preds_train, y_proba=predictions_train, path=path, time=end-start, prefix='train')
        write_predictions(test_data=train_data, y_pred=preds_train, y_proba=predictions_train, path=path, prefix='train')

        print(metrics.classification_report(test_data.classes, preds))
        print(metrics.matthews_corrcoef(test_data.classes, preds))

        # test confusion matrix
        cm = metrics.confusion_matrix(test_data.classes, preds)
        fig, ax = plt.subplots(1, figsize=(20, 20))
        sns.set(font_scale=1.5)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Vaginal Delivery', 'Cesarian Delivery'])
        disp.plot(ax=ax)
        ax.grid(False)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(path, f'test_confusion_matrix.png'))
        plt.close()

        # train confusion matrix
        cm = metrics.confusion_matrix(train_data.classes, preds_train)
        fig, ax = plt.subplots(1, figsize=(20, 20))
        sns.set(font_scale=1.5)
        disp = metrics.ConfusionMatrixDisplay(cm, display_labels=['Vaginal Delivery', 'Cesarian Delivery'])
        disp.plot(ax=ax)
        ax.grid(False)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(path, f'train_confusion_matrix.png'))
        plt.close()


        # Calculate Area Under the Curve (AUC)
        fpr, tpr, thresholds = metrics.roc_curve(test_data.classes, predictions)
        roc_auc = metrics.auc(fpr, tpr)

        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')

        # Set labels and title
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(path, 'roc_curve.png'))
        plt.close()


if __name__ == '__main__':
    seed = 1257623
    set_global_determinism(seed)
    # disable Tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    config_path = sys.argv[1]
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    exps_path = os.listdir(os.path.join(f'logs-{config.get("deep-model", "cnn")}'))
    if not os.path.exists(exps_path):
        os.makedirs(exps_path)
    
    if len(exps_path) < 9:
        exp = f'exp0{len(exps_path)+1}'
    else:
        exp = f'exp{len(exps_path)+1}'

    LOGS_PATH = os.path.join(exps_path, exp)
    os.makedirs(LOGS_PATH)


    with open(os.path.join(LOGS_PATH, 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=1)

    width = config.get('width', 224)
    height = config.get('height', 224)
    train_datagen = ImageDataGenerator(rescale=1/255.0)
    validation_datagen = ImageDataGenerator(rescale=1/255.0)
    test_datagen = ImageDataGenerator(rescale=1/255.0)
    dataset = config.get('dataset', 'abdomen_dataset')
    train_data = train_datagen.flow_from_directory(
        os.path.join('..', dataset, 'train'),
        batch_size = config['batch_size'],
        class_mode='binary', 
        target_size=(width, height, ),
        classes=['Vaginal Birth', 'Cesarean Birth']
        )
    validation_data = validation_datagen.flow_from_directory(
        os.path.join('..', dataset, 'validation'),
        batch_size = config['batch_size'],
        class_mode='binary', 
        target_size=(width, height),
        classes=['Vaginal Birth', 'Cesarean Birth']
        )
    test_data = test_datagen.flow_from_directory(
        os.path.join('..', dataset, 'test'),
        batch_size = config['batch_size'],
        class_mode='binary', 
        target_size=(width, height), 
        classes=['Vaginal Birth', 'Cesarean Birth']
    )

    
    # train test
    fit_and_predict_dataset(config, train_data, validation_data, test_data, width=width, height=height, prefix=f'',path=LOGS_PATH, seed=seed)

