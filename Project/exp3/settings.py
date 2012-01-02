#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings.py: A module that contain some settings."""

NAME = "No_US"
PERCENTAGE = 100

TRAIN_FILE = "data/train.json.txt.pos." + str(PERCENTAGE)
DEV_FILE = "data/dev.json.txt.pos." + str(PERCENTAGE)
TEST_FILE = "data/test.json.txt.pos." + str(PERCENTAGE)
CLEAN = True

CACHE = True
TRAIN = None
DEV = None
TEST = None
GRAMS = None
LANGUAGES = None
TAGS = None

__train_fs = None
__dev_fs = None
__test_fs = None


MAP = {
'User_de-N': 'German',
'User_es-N': 'Spanish',
'User_fr-N': 'French',
'User_ru-N': 'Russian',
'User_nl-N': 'Dutch',
}

n = 4
LEN_1SENT = 1
NUM_SENTS = 1
COMMENT_SIZE = 20
LAMBDAS = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
#LAMBDAS = [0.01, 3, 10, 30, 100]
#LAMBDAS = [30, 100, 300]
#LAMBDAS = [3]
