# -*- coding: utf-8 -*-
"""
This module includes functions to evaluate a NaiveBayesClassifier.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

from feature_extraction import bigram_feats, feature_extraction
from validation import cross_validation
from datasets import g2crowd, pweb


def evaluate_classifier(featxs, datasets):
    """
        type 0 : character ngrams
        type 1 : word ngrams
    """
    subfeats = []
    objfeats = []
    for dataset in datasets:
        subfeats += [(feature_extraction(featxs, sen), 'sub') for sen in dataset.sents('sub', punctuation=False)]
        objfeats += [(feature_extraction(featxs, sen), 'obj') for sen in dataset.sents('obj', punctuation=False)]

    subfeats = subfeats[:len(objfeats)]

    print cross_validation(subfeats, objfeats, folds=5, classifier='decision_tree')


evaluate_classifier([bigram_feats], [g2crowd, pweb])
