import os

from sklearn.neural_network import MLPClassifier
import pickle as pkl
from classifiers.base_classifier import BaseClassifier
from settings import PROJECT_PATH


class NeuralNetwork(BaseClassifier):
    """
    Class implementing Neural Network methods from sklearn (it's based on BaseClassifier).
    """
    def __init__(self, x, y, x_feature_names, y_label_names, cv_parts=5, hidden_neurons=(100,), activation_fun='relu'):
        """
        Child of BaseClassifier __init__()
        :param hidden_neurons: number of layers and neurons in them; in tuple format
               eg. (100, 200) - 2 hidden layers with sizes 100 and 200
        :param activation_fun: activation function, one of: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}
        """
        super().__init__(x, y, x_feature_names, y_label_names, cv_parts=cv_parts)
        self.hn = hidden_neurons
        self.activation_fun = activation_fun

    def mlp(self, x_test=None):
        """
        Simple implementation of MLP with given params.
        :return: y predict default for x_train (but can be for x_test if given)
        """
        mlp = MLPClassifier(hidden_layer_sizes=self.hn, activation=self.activation_fun, max_iter=500)
        mlp, y_pred = self._predict(mlp)
        return y_pred

    def crossval_mlp(self, one_vs_rest: bool = False):
        """
        Simple implementation of MLP with cross-validated dataset
        :param one_vs_rest: whether to use one vs rest classification
        :return: y predictions for cross-validated x
        """
        mlp = MLPClassifier(hidden_layer_sizes=self.hn, activation=self.activation_fun, max_iter=500)
        y_pred = self._cross_val_predict(mlp, one_vs_rest)
        return y_pred


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    hn, activation = (100, ), 'relu'
    nn = NeuralNetwork(x, y, x_feature_names, y_feature_names, hidden_neurons=hn, activation_fun=activation)

    y_pred_mlp = nn.crossval_mlp()
    nn.plot_confusion_matrix(y_pred_mlp)
    nn.show_basic_metrics(y_pred_mlp)