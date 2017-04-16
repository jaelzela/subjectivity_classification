# -*- coding: utf-8 -*-
"""
This module includes functions to evaluate a NaiveBayesClassifier.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

from feature_extraction import bigram_feats, feature_extraction
from validation import cross_validation
from datasets import g2crowd, pweb


def evaluate_classifier(featx, dataset, type=0):
    """
        type 0 : character ngrams
        type 1 : word ngrams
    """
    subfeats, objfeats = feature_extraction(featx, dataset, type=type)

    print cross_validation(subfeats, objfeats, folds=5, classifier='maximum_entropy')


g2crowd_dataset = g2crowd()
pweb_dataset = pweb()

dataset = dict()
dataset['sub'] = g2crowd_dataset['sub']
dataset['obj'] = g2crowd_dataset['obj'] + pweb_dataset['obj']

evaluate_classifier(bigram_feats, dataset, type=1)
