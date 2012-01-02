#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""settings.py: A module that contain some settings."""

NAME = "ALL"
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
"User_en-us": "US-EN",
"User_de-N": "German",
"User_es-N": "Spanish",
"User_fr-N": "French",
"User_ru-N": "Russian",
"User_nl-N": "Dutch",
"User_pt-N": "Portugese",
"User_sv-N": "Swdish",
"User_it-N": "Italian",
"User_pl-N": "Polish",
"User_zh-N": "Mandarin",
"User_no-N": "Norwagien",
"User_fi-N": "Finnish",
"User_da-N": "Danish",
"User_ja-N": "Japanese",
"User_hu-N": "Hungarian",
"User_tr-N": "Turkish",
"User_ar-N": "Arabic",
"User_ko-N": "Korean",
"User_yue-N": "Cantonese"
}

n = 4
LEN_1SENT = 1
NUM_SENTS = 1
COMMENT_SIZE = 20
LAMBDAS = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
#LAMBDAS = [0.01, 3, 10, 30, 100]
#LAMBDAS = [30, 100, 300]
#LAMBDAS = [3]
