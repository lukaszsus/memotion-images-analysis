# ## Plots
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
from contextlib import redirect_stdout
from datetime import datetime

from deep_learning.vgg_model import VggModel
from settings import DATA_PATH

DEEP_LEARNING_PARENT_DIR = "deep_learning"
DEEP_LEARNING_PARENT_DIR = os.path.join(DATA_PATH, DEEP_LEARNING_PARENT_DIR)


def create_dirs():
    if not os.path.exists(DEEP_LEARNING_PARENT_DIR):
        os.makedirs(DEEP_LEARNING_PARENT_DIR)

    dirs = ["plots", "models", "tables", "models_summaries"]
    for d in dirs:
        path = os.path.join(DEEP_LEARNING_PARENT_DIR, d)
        if not os.path.exists(path):
            os.makedirs(path)


def plot_acc_loss(history, name):
    train_losses = history['train_loss']
    val_losses = history['test_loss']
    train_acc = history['train_acc']
    val_acc = history['test_acc']

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, '--o', label='train')
    plt.plot(epochs, val_losses, '--o', label='validation')
    plt.xlabel('num of epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.title('loss')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc, '--o', label='train')
    plt.plot(epochs, val_acc, '--o', label='validation')
    plt.xlabel('num of epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('accuracy')

    dirpath = os.path.join(DEEP_LEARNING_PARENT_DIR, "plots")

    plt.savefig('{}/{}-{}.png'.format(dirpath, name, datetime.now().strftime('%Y-%m-%d-t%H-%M')), bbox_inches='tight')


def prepare_and_plot_confusion_matrix(name, predictions, y_test, class_names):
    conf_matrix = tf.math.confusion_matrix(y_test, predictions)
    if tf.is_tensor(conf_matrix):
        conf_matrix = conf_matrix.numpy()
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1)
    _plot_confusion_matrix(conf_matrix, name, class_names)


def _plot_confusion_matrix(confusion_matrix, name, class_names, figsize=(12, 12)):
    df_cm = pd.DataFrame(confusion_matrix, index=class_names, columns=class_names)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt=".2f", cmap="Blues")
    bottom, top = heatmap.get_ylim()
    heatmap.set_ylim(bottom + 0.5, top - 0.5)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    dirpath = os.path.join(DEEP_LEARNING_PARENT_DIR, "plots")

    plt.savefig('{}/{}-cmatrix-{}.png'.format(dirpath, name, datetime.now().strftime('%Y-%m-%d-t%H-%M')), bbox_inches='tight')


def save_summary(model, dir):
    with open('{}/{}-summary.txt'.format(dir, model.name), 'w') as f:
        with redirect_stdout(f):
            model.summary()


def load_model(model_params, model_weights_path, train_dataset, x_test, y_test):
    """
    Warning: It is required that model trains for one epoch before loading weights form file.
    """
    model = VggModel(model_params)
    model.fit(train_dataset=train_dataset, x_test=x_test, y_test=y_test, epochs=1)
    model.load_weights(filepath=model_weights_path)
    return model
