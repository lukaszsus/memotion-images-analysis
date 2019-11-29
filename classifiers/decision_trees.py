import os
import matplotlib.pyplot as plt
import numpy
from sklearn import tree
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier

from classifiers.base_classifier import BaseClassifier
from user_settings import PROJECT_PATH

from sklearn.tree import export_graphviz, export_text, DecisionTreeClassifier
from subprocess import call


class DecisionTree(BaseClassifier):
    """
    Class implementing decision tree based classifiers.
    """
    def __init__(self, x, y, x_feature_names, y_label_names, cv_parts=5, max_depth=4, min_samples_leaf=5):
        """
        Child of BaseClassifier __init__()
        :param max_depth: max possible depth of tree
        :param min_samples_leaf: minimum samples that tree has to have in its leaves
        """
        super().__init__(x, y, x_feature_names, y_label_names, cv_parts)
        self.max_depth = max_depth
        self.min_samples = min_samples_leaf

    def decision_tree(self, x_test=None):
        """
        Simple decision tree - CART.
        :return: instance of classifier, y predict default for x_train (but can be for x_test if given)
        """
        clf = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples)
        clf, y_pred = self._predict(clf, x_test)
        return clf, y_pred

    def crossval_decision_tree(self):
        """
        CART decision tree with cross-validated dataset.
        :return: y predict for cross-validated dataset
        """
        clf = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples)
        y_pred = self._cross_val_predict(clf)
        return y_pred

    def random_forest(self, num_estimators=10, x_test=None):
        """
        Random forest for given parameters in constructor.
        :param num_estimators: number of simple trees that makes a forest
        :return: instance of classifier, y predict default for x_train (but can be for x_test if given)
        """
        clf = RandomForestClassifier(n_estimators=num_estimators)
        clf, y_pred = self._predict(clf, x_test)
        return clf, y_pred

    def crossval_random_forest(self, num_estimators=10):
        """
        Random forest run for cross-validated dataset.
        :param num_estimators: number of trees in forest
        :return: y predict for cross-validated dataset
        """
        clf = RandomForestClassifier(n_estimators=num_estimators)
        y_pred = self._cross_val_predict(clf)
        return y_pred

    def save_and_plot_tree(self, clf, fname='decision_tree', path='./tree_visualizations'):
        """
        Plots tree returned from _predict methods.
        :param clf: instance of classifier
        :param fname: file name of .dot and .png graph/plot
        :param path: path from current directory to directory containing saved files
        :return: two saved files in given directory and shows tree on plot
        """
        file_name = os.path.join(path, fname)

        export_graphviz(clf, out_file=f'{file_name}.dot', filled=True, rounded=True,
                        special_characters=True, feature_names=self.feature_names, class_names=self.label_names)
        call(['dot', '-Tpng', f'{file_name}.dot', '-o', f'{file_name}.png', '-Gdpi=600'])

        plt.figure(figsize=(20, 10))
        plt.imshow(plt.imread(f'{file_name}.png'))
        plt.axis('off')
        plt.show()

    def show_tree_structure_in_terminal(self, clf):
        """
        Prints tree structure in console.
        :param clf: instance of classifier returned by _predict method
        :return: prints tree structure
        """
        r = export_text(clf, feature_names=self.feature_names)
        print(r)


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'datasets_pkl/dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    max_depth, min_samples_leaf = 3, 5
    dt = DecisionTree(x, y, x_feature_names, y_feature_names, max_depth=max_depth, min_samples_leaf=min_samples_leaf)

    tree, y_pred_tree = dt.decision_tree()
    dt.plot_confusion_matrix(y_pred_tree)
    dt.show_basic_metrics(y_pred_tree)

    # y_pred_forest = dt.crossval_random_forest(num_estimators=15)
    # dt.plot_confusion_matrix(y_pred_forest)
    # dt.show_basic_metrics(y_pred_forest)

    file_name = f'decission_tree_depth{max_depth}_samples{min_samples_leaf}'
    dt.save_and_plot_tree(tree, file_name)
