from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(y_test, y_pred, class_names, cmap=plt.cm.Greens):
    cm = confusion_matrix(y_test, np.array(y_pred).flatten())
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
         xticklabels=class_names, yticklabels=class_names,
         title='Confusion matrix', ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation_mode="anchor")

    thresh = cm.max() - (cm.max() - cm.min()) / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
              ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.set_ylim(ax.get_ylim()[0] + 0.5, ax.get_ylim()[1] - 0.5)

    plt.ylim(plt.ylim()[0] - 0.5, plt.ylim()[1] + 0.5)
    plt.grid(False)
    plt.show()