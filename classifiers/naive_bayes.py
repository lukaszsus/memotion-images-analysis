import os
import pickle as pkl
from sklearn import datasets
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_val_predict
from sklearn.naive_bayes import GaussianNB, ComplementNB
import numpy as np

from classifiers.base_classifier import BaseClassifier
from user_settings import PROJECT_PATH


class NaiveBayes(BaseClassifier):
    """
    Class implementing multiple kinds of Naive Bayes algorithm.
    """
    def __init__(self, x, y, x_feature_names, y_label_names, cv_parts=5):
        super().__init__(x, y, x_feature_names, y_label_names, cv_parts=cv_parts)

    def gaussian_navie_bayes(self, x_test=None):
        """
        Simple Gaussian Naive Bayes.
        :return: y predicted for given x (x train in that case)
        """
        gnb = GaussianNB()
        gnb, y_pred = self._predict(gnb, x_test)
        return y_pred

    def crossval_gaussian_navie_bayes(self):
        """
        Simple Gaussian Naive Bayes using cross validation.
        :return: y predicted for given x cross-validated parts.
        """
        gnb = GaussianNB()
        y_pred = self._cross_val_predict(gnb)
        return y_pred

    # def complement_navie_bayes(self):
    #     """
    #     Non-negative x-values needed.
    #     :return: y predicted for given x (x train in that case)
    #     """
    #     cnb = ComplementNB()
    #     cnb, y_pred = self.predict(cnb)
    #     return y_pred


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    nb = NaiveBayes(x, y, x_feature_names, y_feature_names)
    y_pred = nb.crossval_gaussian_navie_bayes()
    nb.show_basic_metrics(y_pred)
    nb.plot_confusion_matrix(y_pred)

