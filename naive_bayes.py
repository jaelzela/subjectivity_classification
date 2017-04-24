# -*- coding: utf-8 -*-
"""
This module includes functions to evaluate a NaiveBayesClassifier.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

from feature_extraction import bigram_feats, bag_of_words, tf_idf, feature_extraction
from validation import cross_validation
from datasets import g2crowd, pweb


def evaluate_classifier(featxs, datasets):

    subfeats, objfeats = feature_extraction(featxs, datasets)

    print '\ncross validation NB'
    print cross_validation(subfeats, objfeats, folds=5, classifier='naive_bayes')


if __name__ == "__main__":
    evaluate_classifier([tf_idf], [g2crowd, pweb])
