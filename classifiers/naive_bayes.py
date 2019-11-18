import os
import pickle as pkl
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB, ComplementNB
import numpy as np

from user_settings import PROJECT_PATH
from classifiers.classifiers_utils import plot_confusion_matrix


class NaiveBayes():
    """
    Class implementing multiple kinds of Naive Bayes algorithm.
    """
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

    def gaussian_navie_bayes(self):
        """
        Simple Gaussian Naive Bayes.
        :return: y predicted for given x (x train in that case)
        """
        gnb = GaussianNB()
        y_pred = gnb.fit(self.x, self.y).predict(self.x)
        return y_pred

    def complement_navie_bayes(self):
        """
        Non-negative x-values needed.
        :return: y predicted for given x (x train in that case)
        """
        cnb = ComplementNB()
        y_pred = cnb.fit(self.x, self.y).predict(self.x)
        return y_pred


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    nb = NaiveBayes(x, y)
    y_pred_gaussian = nb.gaussian_navie_bayes()
    plot_confusion_matrix(y, y_pred_gaussian, y_feature_names)

    # y_pred_complement = nb.complement_navie_bayes()
    # plot_confusion_matrix(y, y_pred_complement, y_feature_names)