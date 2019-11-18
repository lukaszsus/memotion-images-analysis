from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np


class NaiveBayes():
    def __init__(self, x, y, x_names, y_names):
        self.x = np.array(x)
        self.y = np.array(y)
        self.features = x_names
        self.lables = y_names

    def gaussian_navie_bayes(self):
        gnb = GaussianNB()
        y_pred = gnb.fit(self.x, self.y).predict(self.x)
        return y_pred



if __name__ == "__main__":

    iris = datasets.load_iris()
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print("Number of mislabeled points out of a total %d points : %d" % (iris.data.shape[0],(iris.target != y_pred).sum()))

    print(iris.target.shape)