# encoding=utf-8
__author__ = 'houzhi'


class BaseClassifier(object):

    def __init__(self):
        print 'baseClassifier'

    def train(self, x, y):
        """
        训练
        :param x:
        :param y:
        :return:
        """
        raise NotImplementedError()


    def predict(self, x):
        """
        预测
        :param x:
        :return:
        """
        raise NotImplementedError()