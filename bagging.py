#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import random
import decisiontree
import csv



def train(training_data, attr_names, label_name, num_attr, num_rounds):
    num_rows = len(training_data)
    classifiers = []
    alphas = []
    for n in range(num_rounds):
        resampled_entries = []
        random_indices = random.sample(list(range(0, num_rows)),
                num_rows)
        for i in range(num_rows):
            resampled_entries.append(training_data[random_indices[i]])
        default = decisiontree.mode(resampled_entries, -1)
        print 'round ' + str(n + 1) + ' training...'
        weak_classifier = \
             decisiontree.learn_decision_tree(
            resampled_entries,
            attr_names,
            default,
            label_name,
            0,
            num_attr,
            )

        classifiers.append(weak_classifier)
    return classifiers


def classify(classifiers, example):
    classification = 0
    vote1 = 0
    vote0 = 0
    for weak_classifier in classifiers:
        if str(weak_classifier.predict(example)) == str(1):
            vote1 += 1
        else:
            vote0 += 1
    return (1 if vote1 > vote0 else 0)

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
    (attr_names, training_data) = decisiontree.read_data(train_filename,
            num_attr)
    (attr_names, test_data) = decisiontree.read_data(test_filename,
            num_attr)
    label_name = 'IsBadBuy'
    
    # read in ids
    
    ids = []
    for line in open('./Datasets/ids.csv'):
        line = line.replace('"', '').strip()
        ids.append(line)
    
    ret = train(training_data, attr_names, label_name, num_attr, 5)
    print 'bagging procedure complete!'
    
    print ret
    
    predictions = []
    
    for row in test_data:
        predictions.append(classify(ret, row))
    
    res_file = open('bagging_res.csv', 'w')
    writer = csv.writer(res_file)
    header = ('RefId', 'IsBadBuy')
    writer.writerow(header)
    for i in range(len(predictions)):
        writer.writerow([ids[i], predictions[i]])
    res_file.close()


if __name__ == '__main__':
    main()
