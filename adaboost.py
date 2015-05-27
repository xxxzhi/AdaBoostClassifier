#!/usr/bin/python
# -*- coding: utf-8 -*-
import dbm
from sklearn.datasets import load_iris

from classifer.base import BaseClassifier
from classifer.decision_tree import DecisionTreeClassifier
import numpy as np


class AbsAdaBoostClassifier(BaseClassifier):



    def __init__(self, num_rounds):
        super(AbsAdaBoostClassifier, self).__init__()
        self.num_rounds = num_rounds
        self.clf = None

    def create_classifer(self, index=0):
        """
        create a new classifer.
        :return: BaseClassifier
        """
        pass

    def process_alpha(self, index, alpha):
        """
        after a successful classifier training. this method will be called

        :param index
        :param alpha
        :return:
        """
        print '------------alpha-----------'
        print alpha

    def train(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        num_rows = len(x)
        classifiers = []
        alphas = []
        weights = np.ones(num_rows) * 1.0 / num_rows
        for n in range(self.num_rounds):
            error = 0.
            random_indices = AbsAdaBoostClassifier.resample(weights)
            resampled_entries_x = []
            resampled_entries_y =[]
            for i in range(num_rows):
                resampled_entries_x.append(x[random_indices[i]])
                resampled_entries_y.append(y[random_indices[i]])

            print 'round ' + str(n + 1) + ' training...'
            weak_classifier = self.create_classifer(n)
            print len(resampled_entries_x)
            weak_classifier.train(resampled_entries_x, resampled_entries_y)

            # training and calculate the rate of error
            classifications = weak_classifier.predict(x)
            error = 0
            for i in range(len(classifications)):
                predicted = classifications[i]
                error += (predicted != np.argmax(y[i])) * weights[i]

            print 'Error', error

            if error == 0.:
                alpha = 4.0
            elif error > 0.7:
                print 'Discarding learner'
                print n
                continue  # discard classifier with error > 0.5
            else:
                alpha = 0.5 * np.log((1 - error) / error)

            self.process_alpha(n, alpha)
            alphas.append(alpha)
            classifiers.append(weak_classifier)
            print 'weak learner added'

            for i in range(num_rows):
                if np.size(y[i]) > 1:
                    ry = np.argmax(y[i])
                else:
                    ry = y[i]
                h = classifications[i]
                h = (-1 if h == 0 else 1)
                ry = (-1 if ry == 0 else 1)
                weights[i] = weights[i] * np.exp(-alpha * h * ry)
            sum_weights = sum(weights)
            print 'Sum of weights', sum_weights
            normalized_weights = [float(w) / sum_weights for w in weights]
            weights = normalized_weights
        print alphas
        print '----------weight----------'
        self.clf = zip(alphas, classifiers)

    def predict(self, x):
        """

        @:param x array-like features
        @:return labels
        """

        result_list = []
        weight_list = []
        for (weight, classifier) in self.clf:
            res = classifier.predict(x)
            result_list.append(res)
            weight_list.append(weight)

        res =[]
        for i in range(len(result_list[0])):
            result_map = {}
            for j in range(len(result_list)):
                if not result_map.has_key(str(result_list[j][i])):
                    result_map[str(result_list[j][i])] = 0
                result_map[str(result_list[j][i])] = result_map[str(result_list[j][i])] + weight_list[j]

            cur_max_value = -10000000000000000000000
            max_key = ''
            for key in result_map:
                if result_map[key] > cur_max_value:
                    cur_max_value = result_map[key]
                    max_key = key

            res.append(int(max_key))
        return np.asarray(res)

    def load(self, weight=[]):
        """
        reload model base on weight of each classifier
        :param weight:
        :return:
        """
        classifiers = []
        for index in range(len(weight)):
            classifiers.append(self.create_classifer(index))
        self.clf = zip(weight, classifiers)

    @staticmethod
    def resample(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        result_arr = np.searchsorted(t, np.random.rand(len(weights))*s)
        # add all dataset
        # result_arr.append(np.arange(0, len(weights),step=1))
        return result_arr




class DecisionAdaBoostClassifier(AbsAdaBoostClassifier):


    def __init__(self, num_rounds):
        super(DecisionAdaBoostClassifier, self).__init__(num_rounds)

    def create_classifer(self, index=0):
        return DecisionTreeClassifier()

    def process_alpha(self, index, alpha):
        super(DecisionAdaBoostClassifier, self).process_alpha(index, alpha)
        db = dbm.open('params.pag','c')
        key = 'alpha_'+str(index)
        db[key] = str(alpha)
        db.close()

def test():
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02
    # Load data
    iris = load_iris()
    import matplotlib.pyplot as plt

    # We only take the two corresponding features
    pairidx = 0
    pair =[0,1]
    X = iris.data[:, pair]
    y = iris.target

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Train
    clf = DecisionAdaBoostClassifier(num_rounds=3)
    # clf = DecisionTreeClassifier()
    # print X
    print y
    clf.train(X, y)

    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    print '----'
    print iris.data[:1,  ]
    values = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(values)
    print Z
    print Z.shape
    print xx.shape
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlabel(iris.feature_names[pair[0]])
    plt.ylabel(iris.feature_names[pair[1]])
    plt.axis("tight")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                    cmap=plt.cm.Paired)

    plt.axis("tight")

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend()
    plt.show()




if __name__ == '__main__':
    test()
