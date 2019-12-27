import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_predict


class BaseClassifier():
    """
    Base classifier for all implemented classifiers - parent class.
    All implemented classifiers needs to have implementation of its methods.
    """

    def __init__(self, x, y, x_feature_names, y_label_names, cv_parts=5):
        """
        :param x: list of x values/features in shape (number of instances x number of features); x_train
        :param y: list of labels, 1D vector of numbers from 0 to N-1, where N is number of labels; y_train
        :param x_feature_names: list of feature names = x data column headers
        :param y_label_names: list of N label names
        :param cv_parts: number of cross-validation parts; number of subsets to which dataset is going to be divided
        """
        self.x = np.array(x)
        self.y = np.array(y)
        self.feature_names = x_feature_names
        self.label_names = y_label_names
        self.cv_parts = cv_parts

    def _predict(self, clf, x_test=None):
        """
        Given classifier returns predictions for whole x_train data (self.x)
        :param clf: instance of classifier
        :param x_test: test part of dataset (default is x_train)
        :return: y predictions for given x data
        """
        if x_test is None:
            x_test = self.x

        clf.fit(self.x, self.y)
        y_pred = clf.predict(x_test)
        return clf, y_pred

    def _cross_val_predict(self, clf, one_vs_rest: bool = False):
        """
        Given classifier returns predictions on cross-validated dataset.
        :param clf: instance of classifier
        :param one_vs_rest: whether use one vs rest classification or not
        :return: y predictions for cross-validated x
        """
        if one_vs_rest:
            clf = OneVsRestClassifier(clf, n_jobs=-1)

        y_pred = cross_val_predict(clf, self.x, self.y, cv=self.cv_parts)
        return y_pred

    def plot_confusion_matrix(self, y_pred, y_test=None, label='', cmap=plt.cm.Greens):
        """
        Plots normalized confusion matrix.
        :param label: name of classifier to print e.g. decission tree / kNN
        :param y_pred: predictions for given earlier x
        :param y_test: true y_test labels; needed if using 'predict' method with given x_test
        :param cmap: matplotlib color map
        """
        if y_test is None:
            y_test = self.y

        cm = confusion_matrix(y_test, np.array(y_pred).flatten())
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        if len(label) > 0:
            label = f'for instance of {label}\n'

        ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
               xticklabels=self.label_names, yticklabels=self.label_names,
               title=f'Confusion matrix of {self.__class__.__name__} class\n{label}',
               ylabel='True label', xlabel='Predicted label')

        plt.setp(ax.get_xticklabels(), rotation_mode="anchor")

        thresh = cm.max() - (cm.max() - cm.min()) / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], '.2f'), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        plt.grid(False)
        plt.show()
        return cm

    def count_basic_metrics(self, y_pred, y_test=None, f1_type='macro'):
        """
        Counts matrics - accuracy and f1_score
        :param y_pred: predictions for given earlier x
        :param y_test: true y_test labels; needed if using 'predict' method with given x_test
        :param f1_type: f1-score type
        :return: value of accuracy and f1-score
        """
        if y_test is None:
            y_test = self.y
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=f1_type)
        return acc, f1

    def show_basic_metrics(self, y_pred, y_test=None, label=None):
        """
        Prints basic metrics for classifier.
        :param label: name of classifier to print e.g. decission tree / kNN
        :param y_pred: predictions for given earlier x
        :param y_test: true y_test labels; needed if using 'predict' method with given x_test
        :return: prints in console values of metrics from count_basic_metrics method
        """
        acc, f1 = self.count_basic_metrics(y_pred, y_test)
        print(f'----------------------------------')
        if label is not None:
            print(f'Metrics for {label}:')
        print(f'accuracy: {round(acc * 100, 2)}%')
        print(f'f-score:  {round(f1 * 100, 2)}%')
