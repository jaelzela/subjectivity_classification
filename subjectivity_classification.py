# -*- coding: utf-8 -*-
"""
This module evaluates all classifier with different features.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

from feature_extraction import feature_extraction, bag_of_words, bigram_feats, tf_idf, part_of_speech
from validation import cross_validation
from datasets import g2crowd, pweb


if __name__ == "__main__":
    subfeats, objfeats = feature_extraction([bag_of_words, bigram_feats, tf_idf, part_of_speech], [g2crowd, pweb], stopwords=False, punctuation=False)

    print '\ncross validation NB'
    print cross_validation(subfeats, objfeats, folds=5, classifier='naive_bayes')
    print '\ncross validation SVM'
    print cross_validation(subfeats, objfeats, folds=5, classifier='svm')
    print '\ncross validation ME'
    print cross_validation(subfeats, objfeats, folds=5, classifier='maximum_entropy')
    #print '\ncross validation DT'
    #print cross_validation(subfeats, objfeats, folds=5, classifier='decision_tree')
    #print '\ncross validation RF'
    #print cross_validation(subfeats, objfeats, folds=5, classifier='random_forest')
    #print '\ncross validation MLP'
    #print cross_validation(subfeats, objfeats, folds=5, classifier='mlp_nn')
    #print '\ncross validation KNN'
    #print cross_validation(subfeats, objfeats, folds=5, classifier='k_neighbors')
