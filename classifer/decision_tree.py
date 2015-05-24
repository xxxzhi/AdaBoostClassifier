from sklearn import tree
from base import BaseClassifier

__author__ = 'houzhi'

class DecisionTreeClassifier(BaseClassifier):
    """
    depending the sklearn DecisionTreeClassifier
    """
    def __init__(self):
        super(DecisionTreeClassifier, self).__init__()
        self.clf = tree.DecisionTreeClassifier()

    def train(self, x, y):
        self.clf.fit(x, y)


    def predict(self, x):
        """
        predict
        :param x:
        :return:
        """
        return self.clf.predict(x)