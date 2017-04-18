# -*- coding: utf-8 -*-
"""
This module includes functions to extract features from text.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures, NgramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
import string


def bigram_feats(text, score_fn=BigramAssocMeasures.pmi, n_best=200):
    bigram_finder = BigramCollocationFinder.from_words(text)
    n_grams = bigram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def trigram_feats(text, score_fn=TrigramAssocMeasures.pmi, n_best=200):
    trigram_finder = TrigramCollocationFinder.from_words(text)
    n_grams = trigram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def quadgram_feats(text, score_fn=NgramAssocMeasures.pmi, n_best=200):
    #n_grams = list(ngrams(characters, n)) + list(ngrams(characters, n-1)) + list(ngrams(characters, n-2))
    quadgram_finder = QuadgramCollocationFinder.from_words(text)
    n_grams = quadgram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def multigram_feats(text):
    grams = bigram_character_feats(text)
    grams.update(trigram_character_feats(text))
    grams.update(quadgram_character_feats(text))
    return grams


def feature_extraction(featxs, words):
    features = dict()

    for featx in featxs:
        features.update(featx(words))

    return features