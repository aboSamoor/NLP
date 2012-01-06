#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings.py: A module that contain some settings."""

NAME = "NON_NATIVE"
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



MAP ={
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


n = 4
LEN_1SENT = 1
NUM_SENTS = 1
COMMENT_SIZE = 20
LAMBDAS = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
#LAMBDAS = [0.01, 3, 10, 30, 100]
#LAMBDAS = [30, 100, 300]
#LAMBDAS = [3]
