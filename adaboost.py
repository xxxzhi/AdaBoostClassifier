#!/usr/bin/python
# -*- coding: utf-8 -*-

import scipy.io as sio
from scipy import stats
import numpy as np
import math
import random
import decisiontree
import csv
from numpy.random import rand
from numpy import cumsum, sort, sum, searchsorted


def train(
    training_data,
    attr_names,
    label_name,
    num_attr,
    num_rounds,
    ):
    num_rows = len(training_data)
    classifiers = []
    alphas = []
    weights = np.ones(num_rows) * 1.0 / num_rows
    for n in range(num_rounds):
        error = 0.
        random_indices = resample(weights)
        resampled_entries = []

        for i in range(num_rows):
            resampled_entries.append(training_data[random_indices[i]])

        default = decisiontree.mode(resampled_entries, -1)
        print 'round ' + str(n + 1) + ' training...'
        weak_classifier = decisiontree.learn_decision_tree(
            resampled_entries,
            attr_names,
            default,
            label_name,
            0,
            num_attr,
            )

        classifications = decisiontree.classify(training_data,
                weak_classifier)
        error = 0
        for i in range(len(classifications)):
            predicted = classifications[i]
            error += (predicted != training_data[i][-1]) * weights[i]

        print 'Error', error

        if error == 0.:
            alpha = 4.0
        elif error > 0.5:
            print 'Discarding learner'
            continue  # discard classifier with error > 0.5
        else:
            alpha = 0.5 * np.log((1 - error) / error)

        alphas.append(alpha)
        classifiers.append(weak_classifier)
        print 'weak learner added'

        for i in range(num_rows):
            y = training_data[i][-1]
            h = classifications[i]
            h = (-1 if h == 0 else 1)
            y = (-1 if y == 0 else 1)
            weights[i] = weights[i] * np.exp(-alpha * h * y)
        sum_weights = sum(weights)
        print 'Sum of weights', sum_weights
        normalized_weights = [float(w) / sum_weights for w in weights]
        weights = normalized_weights
    return zip(alphas, classifiers)


def classify(weight_classifier, example):
    classification = 0
    for (weight, classifier) in weight_classifier:
        if str(classifier.predict(example)) == str(1):
            ex_class = 1
        else:
            ex_class = -1
        classification += weight * ex_class
    return (1 if classification > 0 else 0)


def resample(weights):
    t = cumsum(weights)
    s = sum(weights)
    return searchsorted(t,rand(len(weights))*s)


def main():
    train_filename = './Datasets/66_SAMPLED.csv'
    test_filename = './Datasets/66_test.csv'

    # specify numeric attributes in input file

    num_attr = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        ]
    (attr_names, training_data) = \
        decisiontree.read_data(train_filename, num_attr)
    (attr_names, test_data) = decisiontree.read_data(test_filename,
            num_attr)
    label_name = 'IsBadBuy'

    # read in ids

    ids = []
    for line in open('./Datasets/ids.csv'):
        line = line.replace('"', '').strip()
        ids.append(line)

    ret = train(training_data, attr_names, label_name, num_attr, 5)
    print 'adaboost building complete!'

    print ret

    predictions = []

    for row in test_data:
        predictions.append(classify(ret, row))

    res_file = open('adaboost_res.csv', 'w')
    writer = csv.writer(res_file)
    header = ('RefId', 'IsBadBuy')
    writer.writerow(header)
    for i in range(len(predictions)):
        writer.writerow([ids[i], predictions[i]])
    res_file.close()


if __name__ == '__main__':
    main()
