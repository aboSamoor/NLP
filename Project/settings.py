#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings.py: A module that contain some settings."""

PERCENTAGE = 50

TRAIN_FILE = "data/train.json.txt.pos." + str(PERCENTAGE)
DEV_FILE = "data/dev.json.txt.pos." + str(PERCENTAGE)
TEST_FILE = "data/test.json.txt.pos." + str(PERCENTAGE)
CLEAN = True

CACHE = False
TRAIN = None
DEV = None
TEST = None
GRAMS = None
LANGUAGES = None
TAGS = None

__train_fs = None
__dev_fs = None
__test_fs = None



MAP_FOREIGN ={
'User_en-us': 'N',
'User_de-N': 'F',
'User_es-N': 'F',
'User_fr-N': 'F',
'User_ru-N': 'F',
'User_nl-N': 'F',
'User_pt-N': 'F',
'User_sv-N': 'F',
'User_it-N': 'F',
'User_pl-N': 'F',
'User_zh-N': 'F',
'User_no-N': 'F',
'User_fi-N': 'F',
'User_da-N': 'F',
'User_ja-N': 'F',
'User_hu-N': 'F',
'User_tr-N': 'F',
'User_ar-N': 'F',
'User_ko-N': 'F',
'User_yue-N':'F',
}

MAP_GROUPS ={
'User_en-us':  'EN-US',
'User_de-N': 'NGermanic',
'User_es-N': 'Roman',
'User_fr-N': 'Roman',
'User_ru-N': 'Uralic',
'User_nl-N': 'NGermanic',
'User_pt-N': 'Roman',
'User_sv-N': 'NGermanic',
'User_it-N': 'Roman',
'User_pl-N': 'Uralic',
'User_zh-N': 'Asian',
'User_no-N': 'NGermanic',
'User_fi-N': 'Uralic',
'User_da-N': 'NGermanic',
'User_ja-N': 'Asian',
'User_hu-N': 'Uralic',
'User_ko-N': 'Asian',
'User_yue-N':  'Asian',
}

MAP_POPULAR = {
'User_en-us':  'EN-US',
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
