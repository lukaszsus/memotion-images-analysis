import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np


class BaseClassifier():
    def __init__(self, x, y, x_feature_names, y_label_names):
        self.x = np.array(x)
        self.y = np.array(y)
        self.feature_names = x_feature_names
        self.label_names = y_label_names

    def predict(self, clf):
        clf.fit(self.x, self.y)
        y_pred = clf.predict(self.x)
        return clf, y_pred

    def plot_confusion_matrix(self, y_pred, cmap=plt.cm.Greens):
        cm = confusion_matrix(self.y, np.array(y_pred).flatten())
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=self.label_names, yticklabels=self.label_names,
               title='Confusion matrix', ylabel='True label', xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation_mode="anchor")

        thresh = cm.max() - (cm.max() - cm.min()) / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.grid(False)
        plt.show()

    def count_basic_metrics(self, y_pred):
        acc = accuracy_score(self.y, y_pred)
        f1 = f1_score(self.y, y_pred, average='weighted')
        return acc, f1

    def show_basic_metrics(self, y_pred):
        acc, f1 = self.count_basic_metrics(y_pred)
        print(f'-------------------')
        print(f'Accuracy: {round(acc * 100, 2)}%')
        print(f'F-score:  {round(f1 * 100, 2)}%')