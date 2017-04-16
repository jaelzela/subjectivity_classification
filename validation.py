# -*- coding: utf-8 -*-
"""
This module includes functions to validate a classifier.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

import sys
import collections
from nltk.classify import NaiveBayesClassifier, DecisionTreeClassifier, MaxentClassifier, SklearnClassifier
from nltk.metrics import precision, recall, f_measure
from sklearn.model_selection import check_cv
from sklearn.svm import LinearSVC
from statistics import mean

CLASSIFIERS = dict(naive_bayes=NaiveBayesClassifier,
                   decision_tree=DecisionTreeClassifier,
                   maximum_entropy=MaxentClassifier,
                   svm=SklearnClassifier(LinearSVC()))


def split(x1, x2, n_folds=5):
    cv1 = check_cv(n_folds, x1)
    cv1_iter = list(cv1.split(x1, None, None))
    cv2 = check_cv(n_folds, x2)
    cv2_iter = list(cv2.split(x2, None, None))

    cv_iter = []
    for i in range(len(cv1_iter)):
        train1, test1 = cv1_iter[i]
        train2, test2 = cv2_iter[i]

        x1_train = [x1[index] for index in train1]
        x1_test = [x1[index] for index in test1]
        x2_train = [x2[index] for index in train2]
        x2_test = [x2[index] for index in test2]

        cv_iter.append((x1_train + x2_train, x1_test + x2_test))

    return cv_iter


def cross_validation(x1, x2, folds=5, classifier='naive_bayes'):
    scores = []
    for train, test in split(x1, x2, n_folds=folds):
        sys.stdout.write('.')
        sys.stdout.flush()
        scores.append(train_and_score(CLASSIFIERS[classifier], train, test))

    print ""

    precisions1 = []
    recalls1 = []
    f_measures1 = []
    precisions2 = []
    recalls2 = []
    f_measures2 = []

    for score in scores:
        precisions1.append(score[0][0])
        recalls1.append(score[0][1])
        f_measures1.append(score[0][2])

        precisions2.append(score[0][0])
        recalls2.append(score[0][1])
        f_measures2.append(score[0][2])

    measures = []
    measures.append([mean(precisions1), mean(recalls1), mean(f_measures1)])
    measures.append([mean(precisions2), mean(recalls2), mean(f_measures2)])

    return measures


def train_and_score(classifier, train, test):
    #clf = classifier.train(train, algorithm='MEGAM')
    #clf = classifier.train(train, binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=3)
    clf = classifier.train(train)

    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test):
        refsets[label].add(i)
        observed = clf.classify(feats)
        testsets[observed].add(i)

    measures = []
    for key in refsets.keys():
        measures.append([precision(refsets[key], testsets[key]),
                       recall(refsets[key], testsets[key]),
                       f_measure(refsets[key], testsets[key])])

    return measures

## MAXENT ##
#[[0.9326739868595826, 0.8821004914799031, 0.9053205966625409], [0.9326739868595826, 0.8821004914799031, 0.9053205966625409]]

## SVM ##
#[[0.9421653548454361, 0.8795292629682221, 0.908083947111019], [0.9421653548454361, 0.8795292629682221, 0.908083947111019]]