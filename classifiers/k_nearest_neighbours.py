import os
import pickle as pkl
from sklearn.neighbors import KNeighborsClassifier
from classifiers.base_classifier import BaseClassifier
from user_settings import PROJECT_PATH


class NearestNeighbours(BaseClassifier):
    def __init__(self, x, y, x_feature_names, y_label_names, n_neigh=3):
        super().__init__(x, y, x_feature_names, y_label_names)
        self.num_neighbours = n_neigh

    def knn(self):
        knn = KNeighborsClassifier(n_neighbors=self.num_neighbours)
        knn, y_pred = self.predict(knn)
        return y_pred


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    n_neighbours = 3
    nn = NearestNeighbours(x, y, x_feature_names, y_feature_names, n_neighbours)
    y_pred_knn = nn.knn()

    nn.plot_confusion_matrix(y_pred_knn)
    nn.show_basic_metrics(y_pred_knn)