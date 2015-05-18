#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import Counter
import math
import random
import operator
import copy
import csv


class Node:

    def predict(self, e):
        if isinstance(self, Leaf):
            return self.result
        else:
            if self.numeric:
                if e[self.attr] <= self.splitval and '<=' \
                    in self.branches:
                    return self.branches['<='].predict(e)
                elif '>' in self.branches:
                    return self.branches['>'].predict(e)
                else:
                    return '0'
            else:
                try:
                    out = self.branches[e[self.attr]].predict(e)
                except:
                    return '0'
                return out


class Leaf(Node):

    def __init__(self, target_class):
        self.result = target_class

    def makeBranch(self):
        self.__class__ = Branch
        self.result = None


class Branch(Node):

    def __init__(self, attr_arr):
        self.attr = attr_arr[0]
        self.splitval = attr_arr[1]
        self.numeric = attr_arr[2]
        self.attr_name = attr_arr[3]
        self.mode = attr_arr[4]
        self.branches = {}

    def makeLeaf(self, target):
        self.__class__ = Leaf
        self.result = target

    def addBranch(
        self,
        val,
        subtree,
        default,
        ):

        self.branches[val] = subtree


def get_freq(examples, attributes, target_class):
    freq = {}
    a = attributes.index(target_class)
    for row in examples:
        if row[a] in freq:
            freq[row[a]] += 1
        else:
            freq[row[a]] = 1
    return freq


def entropy(examples, attributes, target):

    dataEntropy = 0.0
    frequencies = get_freq(examples, attributes, target)
    for freq in frequencies.values():
        dataEntropy += -freq / len(examples) * math.log(freq
                / len(examples), 2)
    return dataEntropy


def gain(
    examples,
    attributes,
    attr,
    target_attr,
    num_attr,
    ):

    current_entropy = entropy(examples, attributes, target_attr)
    subset_entropy = 0.0
    i = attributes.index(attr)
    best = 0

    if num_attr[i]:
        order = sorted(examples, key=operator.itemgetter(i))
        subset_entropy = current_entropy
        for j in range(len(order)):
            if j == 0 or j == len(order) - 1 or order[j][-1] == order[j
                    + 1][-1]:
                continue

            current_split_ent = 0.0
            subsets = [order[0:j], order[j + 1:]]

            for subset in subsets:
                setProb = len(subset) / len(order)
                current_split_ent += setProb * entropy(subset,
                        attributes, target_attr)

            if current_split_ent < subset_entropy:
                best = order[j][i]
                subset_entropy = current_split_ent
    else:
        value_freq = get_freq(examples, attributes, attr)

        for (val, freq) in value_freq.iteritems():
            value_prob = freq / sum(value_freq.values())
            data_sub = [entry for entry in examples if entry[i] == val]
            subset_entropy += value_prob * entropy(data_sub,
                    attributes, target_attr)

    return [current_entropy - subset_entropy, best]


def select_attribute(
    examples,
    attributes,
    target_class,
    num_attr,
    ):

    best = False
    bestCut = None
    max_gain = 0
    for a in attributes[:-1]:
        (new_gain, cut_at) = gain(examples, attributes, a,
                                  target_class, num_attr)
        if new_gain > max_gain:
            max_gain = new_gain
            best = attributes.index(a)
            bestCut = cut_at
    return [best, bestCut]


def check_one_class(examples):
    first_class = examples[0][-1]
    for e in examples:
        if e[-1] != first_class:
            return False
    return True


def mode(examples, index):
    L = [e[index] for e in examples]
    return Counter(L).most_common()[0][0]


def split_tree(
    examples,
    attr,
    splitval,
    num_attr,
    ):

    isNum = num_attr[attr]
    positive_count = 0

    if isNum:
        subsets = {'<=': [], '>': []}
        for row in examples:
            if row[-1] == '1':
                positive_count += 1
            if row[attr] <= splitval:
                subsets['<='].append(row)
            elif row[attr] > splitval:
                subsets['>'].append(row)
    else:
        subsets = {}
        for row in examples:
            if row[-1] == '1':
                positive_count += 1
            if row[attr] in subsets:
                subsets[row[attr]].append(row)
            else:
                subsets[row[attr]] = [row]
    negative_count = len(examples) - positive_count
    if positive_count > negative_count:
        majority = '1'
    else:
        majority = '0'

    out = {
        'splitOn': splitval,
        'branches': subsets,
        'numeric': isNum,
        'mode': majority,
        }
    return out


def learn_decision_tree(
    examples,
    attributes,
    default,
    target,
    iteration,
    num_attr,
    ):

    iteration += 1

    if iteration > 10:
        return Leaf(default)
    if not examples:
        tree = Leaf(default)
    elif check_one_class(examples):
        tree = Leaf(examples[0][-1])
    else:
        best_attr = select_attribute(examples, attributes, target,
                num_attr)
        if best_attr is False:
            tree = Leaf(default)
        else:

            split_examples = split_tree(examples, best_attr[0],
                    best_attr[1], num_attr)
            best_attr.append(split_examples['numeric'])
            best_attr.append(attributes[best_attr[0]])
            best_attr.append(split_examples['mode'])
            tree = Branch(best_attr)
            for (branch_lab, branch_examples) in \
                split_examples['branches'].iteritems():
                if not branch_examples:
                    break
                sub_default = mode(branch_examples, -1)
                subtree = learn_decision_tree(
                    branch_examples,
                    attributes,
                    sub_default,
                    target,
                    iteration,
                    num_attr,
                    )
                tree.addBranch(branch_lab, subtree, sub_default)
    return tree


def classify(examples, dt):
    classifications = []
    for row in examples:
        classifications.append(dt.predict(row))
    return classifications


def read_data(filename, numeric_attributes):
    with open(filename, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')  # delimiter is ',' seperating fields
        attr = datareader.next()  # header
        data = []
        for row in datareader:
            data.append(row)
    if data[-1] == ['']:
        del data[-1]
    for row in data:
        for i in range(len(numeric_attributes) - 1):
            if numeric_attributes[i]:
                row[i] = int(row[i])
    return (attr, data)


def main():
    target = 'IsBadBuy'
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
    (attr_names, train_data) = read_data(train_filename, num_attr)
    (attr_names, test_data) = read_data(test_filename, num_attr)
    default = mode(train_data, -1)
    learned_tree = learn_decision_tree(
        train_data,
        attr_names,
        default,
        target,
        0,
        num_attr,
        )

    predictions = classify(test_data, learned_tree)

    # read in label

    ids = []
    for line in open('./Datasets/ids.csv'):
        line = line.replace('"', '').strip()
        ids.append(line)

    res_file = open('decision_tree_res.csv', 'w')
    writer = csv.writer(res_file)
    header = ('RefId', 'IsBadBuy')
    writer.writerow(header)
    for i in range(len(predictions)):
        writer.writerow([ids[i], predictions[i][-1]])
    res_file.close()
    print 'Done'


if __name__ == '__main__':
    main()
