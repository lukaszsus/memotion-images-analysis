import os
import matplotlib.pyplot as plt
from sklearn import tree
import pickle as pkl

from sklearn.ensemble import RandomForestClassifier

from classifiers.base_classifier import BaseClassifier
from user_settings import PROJECT_PATH

from sklearn.tree import export_graphviz, export_text
from subprocess import call


class DecisionTree(BaseClassifier):
    def __init__(self, x, y, x_feature_names, y_label_names, max_depth=4, min_samples_leaf=5):
        super().__init__(x, y, x_feature_names, y_label_names)
        self.max_depth = max_depth
        self.min_samples = min_samples_leaf

    def decision_tree(self):
        clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples)
        clf, y_pred = self.predict(clf)
        return clf, y_pred

    def random_forest(self, num_estimators=10):
        clf = RandomForestClassifier(n_estimators=num_estimators)
        clf, y_pred = self.predict(clf)
        return clf, y_pred

    def save_and_plot_tree(self, clf, fname='decision_tree', path='./tree_visualizations'):
        file_name = os.path.join(path, fname)

        export_graphviz(clf, out_file=f'{file_name}.dot', filled=True, rounded=True,
                        special_characters=True, feature_names=self.feature_names, class_names=self.label_names)
        call(['dot', '-Tpng', f'{file_name}.dot', '-o', f'{file_name}.png', '-Gdpi=600'])

        plt.figure(figsize=(20, 10))
        plt.imshow(plt.imread(f'{file_name}.png'))
        plt.axis('off')
        plt.show()

    def show_tree_structure_in_terminal(self, clf):
        r = export_text(clf, feature_names=self.feature_names)
        print(r)


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    max_depth, min_samples_leaf = 3, 5

    dt = DecisionTree(x, y, x_feature_names, y_feature_names, max_depth, min_samples_leaf)

    tree, y_pred_tree = dt.decision_tree()
    dt.plot_confusion_matrix(y_pred_tree)
    dt.show_basic_metrics(y_pred_tree)

    forest, y_pred_forest = dt.random_forest(num_estimators=8)
    dt.plot_confusion_matrix(y_pred_forest)
    dt.show_basic_metrics(y_pred_forest)

    # file_name = f'decission_tree_depth{max_depth}_samples{min_samples_leaf}'
    # dt.save_and_plot_tree(tree, file_name)
