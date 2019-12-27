import os
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from classifiers.base_classifier import BaseClassifier
from user_settings import PROJECT_PATH


class NearestNeighbours(BaseClassifier):
    """
    Class implementing basic kNN algorithm.
    """
    def __init__(self, x, y, x_feature_names, y_label_names, cv_parts=5, n_neigh=3):
        """
        Child of BaseClassifier __init__()
        :param n_neigh: number of neighbours to consider
        """
        super().__init__(x, y, x_feature_names, y_label_names, cv_parts)
        self.num_neighbors = n_neigh

    def knn(self, x_test=None):
        """
        Basic kNN.
        :return: y predict default for x_train (but can be for x_test if given)
        """
        knn = KNeighborsClassifier(n_neighbors=self.num_neighbors)
        knn, y_pred = self._predict(knn, x_test)
        return y_pred

    def crossval_knn(self, one_vs_rest: bool = False):
        """
        Basic kNN using cross validation.
        :param one_vs_rest: whether to use one vs rest classification or not
        :return: y predict for cross-validated dataset
        """
        knn = KNeighborsClassifier(n_neighbors=self.num_neighbors)
        y_pred = self._cross_val_predict(knn, one_vs_rest)
        return y_pred


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    n_neighbours = 3
    nn = NearestNeighbours(x, y, x_feature_names, y_feature_names, n_neigh=n_neighbours)
    y_pred_knn = nn.crossval_knn()
    nn.plot_confusion_matrix(y_pred_knn)
    nn.show_basic_metrics(y_pred_knn)