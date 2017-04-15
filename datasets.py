# -*- coding: utf-8 -*-
"""
This module includes functions to retrieve datasets.
"""

# Author: Jael Zela <jael.ruiz@students.ic.unicamp.br>

import json
import io


def g2crowd():
    with io.open('g2crowd_apis.json', 'r', encoding='utf8') as data_file:
        apis = json.load(data_file, encoding='utf8')
        data_file.close()

    with io.open('g2crowd_reviews.json', 'r', encoding='utf8') as data_file:
        reviews = json.load(data_file, encoding='utf8')
        data_file.close()

    dataset = dict()
    dataset['sub'] = reviews
    dataset['obj'] = apis

    return dataset


def pweb():
    with io.open('pweb_apis.json', 'r', encoding='utf8') as data_file:
        apis = json.load(data_file, encoding='utf8')
        data_file.close()

    with io.open('pweb_reviews.json', 'r', encoding='utf8') as data_file:
        reviews = json.load(data_file, encoding='utf8')
        data_file.close()

    dataset = dict()
    dataset['sub'] = reviews
    dataset['obj'] = apis

    return dataset