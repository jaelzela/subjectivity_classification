# -*- coding: utf-8 -*-
import collections
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from nltk.util import ngrams
from nltk.metrics import precision, recall, f_measure, BigramAssocMeasures, TrigramAssocMeasures, NgramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import io
import string


def bigram_character_feats(characters, score_fn=BigramAssocMeasures.pmi, n_best=200):
    bigram_finder = BigramCollocationFinder.from_words(characters)
    n_grams = bigram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def trigram_character_feats(characters, score_fn=TrigramAssocMeasures.pmi, n_best=200):
    trigram_finder = TrigramCollocationFinder.from_words(characters)
    n_grams = trigram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def quadgram_character_feats(characters, score_fn=NgramAssocMeasures.pmi, n_best=200):
    #n_grams = list(ngrams(characters, n)) + list(ngrams(characters, n-1)) + list(ngrams(characters, n-2))
    quadgram_finder = QuadgramCollocationFinder.from_words(characters)
    n_grams = quadgram_finder.nbest(score_fn, n_best)
    return dict([(n_gram, True) for n_gram in n_grams])


def multigram_character_feats(characters):
    grams = bigram_character_feats(characters)
    grams.update(trigram_character_feats(characters))
    grams.update(quadgram_character_feats(characters))
    return grams


def evaluate_classifier(featx, dataset, type=0):
    """
    type 0 : character ngrams
    type 1 : word ngrams
    """
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

    subcutoff = len(subfeats)*3/4
    objcutoff = len(objfeats)*3/4

    print 'Num Sub Sentences', len(subfeats)
    print 'Num Obj Sentences', len(objfeats)

    trainfeats = subfeats[:subcutoff] + objfeats[:objcutoff]
    testfeats = subfeats[subcutoff:] + objfeats[objcutoff:]

    print '\tTrain', len(subfeats[:subcutoff]), len(objfeats[:objcutoff])
    print '\tTest', len(subfeats[subcutoff:]), len(objfeats[objcutoff:]), '\n'

    classifier = NaiveBayesClassifier.train(trainfeats)
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(testfeats):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print 'accuracy:', accuracy(classifier, testfeats)
    print 'sub precision:', precision(refsets['sub'], testsets['sub'])
    print 'sub recall:', recall(refsets['sub'], testsets['sub'])
    print 'sub f-measure:', f_measure(refsets['sub'], testsets['sub'])
    print 'obj precision:', precision(refsets['obj'], testsets['obj'])
    print 'obj recall:', recall(refsets['obj'], testsets['obj'])
    print 'obj f-measure:', f_measure(refsets['obj'], testsets['obj'])
    classifier.show_most_informative_features()


with io.open('g2crowd_apis.json', 'r', encoding='utf8') as data_file:
    apis = json.load(data_file, encoding='utf8')
    data_file.close()

with io.open('pweb_apis.json', 'r', encoding='utf8') as data_file:
    pweb_apis = json.load(data_file, encoding='utf8')
    data_file.close()

with io.open('g2crowd_reviews.json', 'r', encoding='utf8') as data_file:
    reviews = json.load(data_file, encoding='utf8')
    data_file.close()

dataset = dict()
dataset['sub'] = reviews
dataset['obj'] = apis + pweb_apis

evaluate_classifier(bigram_character_feats, dataset, type=1)
