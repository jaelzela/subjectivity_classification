# -*- coding: utf-8 -*-
"""
This module includes functions to retrieve datasets.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

import json
import io
from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation as signs
from nltk.corpus import stopwords as stopswords_corpus

stops = set(stopswords_corpus.words('english'))


class CategorizedDataset(object):

    def __init__(self, name, categories, encoding='utf8'):
        if name is None:
            raise AttributeError('File name is required.')
        if categories is None:
            raise AttributeError('Categories name is required.')

        if type(name) != type(categories):
            raise AttributeError('Categories do not match with file name.')

        self.__name = name
        self._cateories = categories
        self.__encoding = encoding
        self._dataset = dict()
        self._load()

    def _load(self):
        if type(self.__name) is list:
            for i in range(len(self.__name)):
                with io.open(self.__name[i], 'r', encoding=self.__encoding) as data_file:
                    data = json.load(data_file, encoding=self.__encoding)
                    data_file.close()

                self._dataset[self._cateories[i]] = data
        elif type(self.__name) is str:
            with io.open(self.__name, 'r', encoding=self.__encoding) as data_file:
                data = json.load(data_file, encoding=self.__encoding)
                data_file.close()

            self._dataset[self._cateories] = data
        else:
            raise AttributeError('File name should be a string or list of strings.')


class G2CrowdDataset(CategorizedDataset):

    def __init__(self, name, categories, encoding='utf8'):
        super(G2CrowdDataset, self).__init__(name, categories, encoding=encoding)

    def raw(self, category):
        raw_text = []

        if category not in self._cateories:
            return raw_text

        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                raw_text.append(review['like'])
                raw_text.append(review['dislike'])
            elif category in 'obj,objective':
                raw_text.append(review['description'])

        return raw_text

    def raw_sents(self, category):
        raw_text = []

        if category not in self._cateories:
            return raw_text

        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                for sen in sent_tokenize(review['like']):
                    raw_text.append(sen)
                for sen in sent_tokenize(review['dislike']):
                    raw_text.append(sen)
            elif category in 'obj,objective':
                for sen in sent_tokenize(review['description']):
                    raw_text.append(sen)

        return raw_text

    def words(self, category, stopwords=True, punctuation=True):
        result = []

        if category not in self._cateories:
            return result

        words_text = []
        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                words_text.append(word_tokenize(review['like']))
                words_text.append(word_tokenize(review['dislike']))
            elif category in 'obj,objective':
                words_text.append(word_tokenize(review['description']))

        result = words_text

        if not punctuation:
            words_text = result
            result = [word for word in words_text if word not in signs]

        if not stopwords:
            words_text = result
            result = [word for word in words_text if word.lower() not in stops]

        return result

    def sents(self, category, stopwords=True, punctuation=True):
        result = []

        if category not in self._cateories:
            return result

        sents_text = []
        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                for sen in sent_tokenize(review['like']):
                    sents_text.append(word_tokenize(sen))
                for sen in sent_tokenize(review['dislike']):
                    sents_text.append(word_tokenize(sen))
            elif category in 'obj,objective':
                for sen in sent_tokenize(review['description']):
                    sents_text.append(word_tokenize(sen))

        result = sents_text

        if not punctuation:
            sents_text = result
            result = [[word for word in sen if word not in signs] for sen in sents_text]

        if not stopwords:
            sents_text = result
            result = [[word for word in sen if word.lower() not in stops] for sen in sents_text]

        return result


class PWebDataset(CategorizedDataset):

    def __init__(self, name, categories, encoding='utf8'):
        super(PWebDataset, self).__init__(name, categories, encoding=encoding)

    def raw(self, category):
        raw_text = []

        if category not in self._cateories:
            return raw_text

        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                raw_text.append(review['review'])
            elif category in 'obj,objective':
                raw_text.append(review['description'])

        return raw_text

    def raw_sents(self, category):
        raw_text = []

        if category not in self._cateories:
            return raw_text

        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                for sen in sent_tokenize(review['review']):
                    raw_text.append(sen)
            elif category in 'obj,objective':
                for sen in sent_tokenize(review['description']):
                    raw_text.append(sen)

        return raw_text

    def words(self, category, stopwords=True, punctuation=True):
        result = []

        if category not in self._cateories:
            return result

        words_text = []
        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                words_text.append(word_tokenize(review['review']))
            elif category in 'obj,objective':
                words_text.append(word_tokenize(review['description']))

        result = words_text

        if not punctuation:
            words_text = result
            result = [word for word in words_text if word not in signs]

        if not stopwords:
            words_text = result
            result = [word for word in words_text if word.lower() not in stops]

        return result

    def sents(self, category, stopwords=True, punctuation=True):
        result = []

        if category not in self._cateories:
            return result

        sents_text = []
        for review in self._dataset[category]:
            if category in 'sub,subjetive':
                for sen in sent_tokenize(review['review']):
                    sents_text.append(word_tokenize(sen))
            elif category in 'obj,objective':
                for sen in sent_tokenize(review['description']):
                    sents_text.append(word_tokenize(sen))

        result = sents_text

        if not punctuation:
            sents_text = result
            result = [[word for word in sen if word not in signs] for sen in sents_text]

        if not stopwords:
            sents_text = result
            result = [[word for word in sen if word.lower() not in stops] for sen in sents_text]

        return result


g2crowd = G2CrowdDataset(['g2crowd_reviews.json', 'g2crowd_apis.json'], ['sub', 'obj'])

pweb = PWebDataset(['pweb_apis.json'], ['obj'])