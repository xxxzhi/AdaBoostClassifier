#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

# from sklearn.ensemble import BaggingClassifier     #???

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.qda import QDA
import csv


def main(training_file, test_file):  # , result_file):

    # input training and test dataset

    with open(training_file, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')  # delimiter is ',' seperating fields
        datareader.next()  # skip header and read only data
        training_data = []
        for row in datareader:
            training_data.append(row)
    with open(test_file, 'rb') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')  # delimiter is ',' seperating fields
        datareader.next()  # skip header and read only data
        test_data = []
        for row in datareader:
            test_data.append(row)

    # get label 'IsBadBuy' column and remove 'RefId' for both datasets, store test_data_refid for submission

    tr = [line[2::] for line in training_data]
    tr_label = [int(line[1]) for line in training_data]
    refid = [line[0] for line in test_data]
    test = [line[1::] for line in test_data]

   # print data

    j = 0
    for i in range(len(tr)):
        month = tr[i][j].index('/', 0, len(tr[i][j]))
        year = tr[i][j].rfind('/', 0, len(tr[i][j]))
        tr[i].append((tr[i][j])[0:month])
        tr[i][j] = tr[i][j][year + 1::]

    j = 0
    for i in range(len(test)):
        month = test[i][j].index('/', 0, len(test[i][j]))
        year = test[i][j].rfind('/', 0, len(test[i][j]))
        test[i].append((test[i][j])[0:month])
        test[i][j] = test[i][j][year + 1::]

    # fill empty number values with median of values in its column

    numbers_columns = [
        3,
        12,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        29,
        31,
        ]

    for j in numbers_columns:
        column_median = median([row[j] for row in tr])
        for i in range(len(tr)):
            if tr[i][j] == '' or tr[i][j] == 'NULL':
                tr[i][j] = column_median
            else:
                tr[i][j] = float(tr[i][j])

    for j in numbers_columns:
        column_median = median([row[j] for row in test])
        for i in range(len(test)):
            if test[i][j] == '' or test[i][j] == 'NULL':
                test[i][j] = column_median
            else:
                test[i][j] = float(test[i][j])

    for column in [
        0,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        24,
        25,
        26,
        27,
        28,
        30,
        32,
        ]:
        categories = {}
        count = 0
        for row in tr:
            cat = row[column]
            if not cat in categories:
                categories[cat] = count
                count += 1
        tr = categorize(tr, column, categories)
        test = categorize(test, column, categories)

    with open('training_new.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(tr)

    with open('test_new.csv', 'w') as fp:
        a = csv.writer(fp, delimiter=',')
        a.writerows(test)

    print 'Done'


def categorize(data, column, categories):
    new_cat = max(categories.values()) + 1
    for row_index in range(len(data)):
        cate = data[row_index][column]
        if not cate in categories:
            categories[cate] = new_cat
            new_cat += 1
        else:
            cate_num = categories[cate]
        data[row_index][column] = cate_num
    return data


def median(column_values):
    sortedLst = sorted(column_values)
    lstLen = len(column_values)
    index = (lstLen - 1) // 2
    if lstLen % 2:
        return sortedLst[index]
    else:
        return (sortedLst[index] + sortedLst[index + 1]) / 2.0


if __name__ == '__main__':
    main('./Datasets/training.csv', './Datasets/test.csv')  # ,'./Datasets/res.csv')

			
