#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
import json
import re
import pdb

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

def user_groups(users, languages):
  language_groups = dict([(lang, []) for lang in languages])
  lang_size = dict([(lang, {'comms_num':0, 'comms_size':0}) for lang in languages])
  languages = set(languages)
  native = re.compile(r'.*-N$')
  for user in users:
    user_langs = set(users[user]['langs'])
    many = len(filter(lambda x: native.match(x), user_langs)) > 1
    common = user_langs.intersection(languages)
    if len(common) != 1:
      continue
    if 'en'
    not_one_lang = if 'User_en-N' in 
    if many or not_one_lang:
      logging.debug("%s is ecluded because his languages are %s" ,user, user_langs)
      continue
    language_groups[list(common)[0]].append(user)
    lang_size[list(common)[0]]['comms_num'] += users[user]['comms_num']
    lang_size[list(common)[0]]['comms_size'] += users[user]['comms_size']
  for lang in lang_size:
    print '%s\t%d\t%d\t%d' % (lang, len(language_groups[lang]), lang_size[lang]['comms_num'], lang_size[lang]['comms_size'])
  return language_groups


def main(options, args):
  users = json.load(open(options.filename, 'r'))
  languages = ["User_en-us",
               "User_de-N",
               "User_es-N",
               "User_fr-N",
               "User_ru-N",
               "User_nl-N",
               "User_zh-N",
               "User_it-N",
               "User_pt-N",
               "User_pl-N",
               "User_ja-N",
               "User_sv-N",
               "User_fi-N",
               "User_tr-N",
               "User_no-N",
               "User_yue-N",
               "User_ar-N",
               "User_hu-N",
               "User_da-N",
               "User_ko-N"]
  lang_groups = user_groups(users, languages)
  json.dump(lang_groups, open('lang_users.json.txt', 'w'), indent=2)

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

