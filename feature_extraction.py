# -*- coding: utf-8 -*-
"""
This module includes functions to extract features from text.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures, NgramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
import string


def bigram_feats(characters, score_fn=BigramAssocMeasures.pmi, n_best=200):
    bigram_finder = BigramCollocationFinder.from_words(characters)
    n_grams = bigram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def trigram_feats(characters, score_fn=TrigramAssocMeasures.pmi, n_best=200):
    trigram_finder = TrigramCollocationFinder.from_words(characters)
    n_grams = trigram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def quadgram_feats(characters, score_fn=NgramAssocMeasures.pmi, n_best=200):
    #n_grams = list(ngrams(characters, n)) + list(ngrams(characters, n-1)) + list(ngrams(characters, n-2))
    quadgram_finder = QuadgramCollocationFinder.from_words(characters)
    n_grams = quadgram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def multigram_feats(characters):
    grams = bigram_character_feats(characters)
    grams.update(trigram_character_feats(characters))
    grams.update(quadgram_character_feats(characters))
    return grams


def feature_extraction(featx, dataset, type=0):
    subjectives = dataset['sub']
    objectives = dataset['obj']

    subfeats = []
    objfeats = []
    if type == 0:
        print '** Character Ngram **'
        for sub in subjectives:
            for sen in sent_tokenize(sub['like']):
                subfeats.append((featx(sen.encode('utf8').translate(None, string.punctuation)), 'sub'))
            for sen in sent_tokenize(sub['dislike']):
                subfeats.append((featx(sen.encode('utf8').translate(None, string.punctuation)), 'sub'))

        for obj in objectives:
            for sen in sent_tokenize(obj['description']):
                objfeats.append((featx(sen.encode('utf8').translate(None, string.punctuation)), 'obj'))
    elif type == 1:
        print '** Word Ngram **'
        for sub in subjectives:
            for sen in sent_tokenize(sub['like']):
                subfeats.append((featx(word_tokenize(sen.encode('utf8').translate(None, string.punctuation))), 'sub'))
            for sen in sent_tokenize(sub['dislike']):
                subfeats.append((featx(word_tokenize(sen.encode('utf8').translate(None, string.punctuation))), 'sub'))

        for obj in objectives:
            for sen in sent_tokenize(obj['description']):
                objfeats.append((featx(word_tokenize(sen.encode('utf8').translate(None, string.punctuation))), 'obj'))

    subfeats = subfeats[:len(objfeats)]

    print 'Num Sub Sentences', len(subfeats)
    print 'Num Obj Sentences', len(objfeats)

    return subfeats, objfeats