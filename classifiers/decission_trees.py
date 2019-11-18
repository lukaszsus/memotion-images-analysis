import os
import matplotlib.pyplot as plt
from sklearn import tree
import pickle as pkl
from classifiers.classifiers_utils import plot_confusion_matrix
from user_settings import PROJECT_PATH

from sklearn.tree import export_graphviz, export_text
from subprocess import call
from IPython.display import Image


class DecissionTree():
    def __init__(self, x, y, max_depth=4, min_samples_leaf=5):
        self.x = x
        self.y = y
        self.max_depth = max_depth
        self.min_samples = min_samples_leaf

    def decision_tree(self):
        clf = tree.DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples)
        clf = clf.fit(x, y)
        y_pred = clf.predict(x)
        return clf, y_pred

    @staticmethod
    def save_and_plot_tree(clf, x_labels, y_labels, fname='decision_tree', path='./tree_visualizations'):
        file_name = os.path.join(path, fname)

        export_graphviz(clf, out_file=f'{file_name}.dot', filled=True, rounded=True,
                        special_characters=True, feature_names=x_labels, class_names=y_labels)
        call(['dot', '-Tpng', f'{file_name}.dot', '-o', f'{file_name}.png', '-Gdpi=600'])

        plt.figure(figsize=(20, 10))
        plt.imshow(plt.imread(f'{file_name}.png'))
        plt.axis('off')
        plt.show()

    @staticmethod
    def show_tree_structure_in_terminal(clf, x_labels):
        r = export_text(clf, feature_names=x_labels)
        print(r)


if __name__ == "__main__":

    # You need to have dataset_pics.pkl file in main repository.
    dataset_path = os.path.join(PROJECT_PATH, 'dataset_pics.pkl')
    with open(dataset_path, "rb") as f:
        x, y, x_feature_names, y_feature_names = pkl.load(f)

    max_depth, min_samples_leaf = 4, 5
    dt = DecissionTree(x, y, max_depth, min_samples_leaf)
    tree, y_pred_tree = dt.decision_tree()

    plot_confusion_matrix(y, y_pred_tree, y_feature_names)
    # file_name = f'decission_tree_depth{max_depth}_samples{min_samples_leaf}'
    # DecissionTree.save_and_plot_tree(tree, x_feature_names, y_feature_names, file_name)
