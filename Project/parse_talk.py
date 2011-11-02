#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
import re
import json

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


_no_new_user = '[^]|[]*'
_no_new_user2 = '[^][]*?'
_user = '[[]{2}User:%s\|%s[]]{2}' % (_no_new_user, _no_new_user2)
_named_user = '[[]{2}User:(?P<username>%s)\|%s[]]{2}' % (_no_new_user, _no_new_user2)
_talk_page = '[[]{2}User talk:%s\|%s[]]{2}' % (_no_new_user, _no_new_user2)

_hour = '\d{1,2}:\d{1,2}'
_day = '\d{1,2}'
_month = '[A-Za-z]*?'
_year = '\d{,4}'
_zone = '[A-Z]{,6}'
_time = '%s,\s*?%s\s*?%s\s*?%s\s*?\(%s\)' % (_hour, _day, _month, _year, _zone)
_named_time = '(?P<time>%s,\s*?%s\s*?%s\s*?%s\s*?\(%s\))' % (_hour, _day,
                                                             _month, _year, _zone)

_sep = '[\(\),\'\"\s]*?'
_all = '%s%s%s%s%s' % (_user, _sep, _talk_page, _sep, _time)
ALL = re.compile('%s%s%s%s%s' % (_named_user, _sep, _talk_page, _sep, _named_time))
_no_talk = '%s%s%s' % (_user, _sep, _time)
NO_TALK = re.compile('%s%s%s' % (_named_user, _sep, _named_time))
_no_time = '%s%s%s' % (_user, _sep, _talk_page)
NO_TIME = re.compile('%s%s%s' % (_named_user, _sep, _talk_page))
SIGNATURE = re.compile('%s|%s|%s' % (_all, _no_talk, _no_time))



def sections(text):
  SECTION_HEADERS = r'={2,}.*={2,}'
  # Split page by sections, do not report the sections
  return re.split(SECTION_HEADERS, text)


def parse_signature(text):
  patterns = [ALL, NO_TALK, NO_TIME]
  for pattern in patterns:
    match = pattern.match(text)
    if match:
      return match.groupdict()


def split(compiled_pattern, text, groups=[]):
  results = []
  start = 0
  for match in compiled_pattern.finditer(text):
    user_groupdict = {}
    groupdict = match.groupdict()
    for group in groups:
      user_groupdict[group] = groupdict.get(group, None)
    user_groupdict['match'] = text[match.start():match.end()]
    segment = text[start:match.start()]
    results.append((segment, user_groupdict))
    start = match.end()
  if start < len(text):
    user_groupdict = {}
    for group in groups:
      user_groupdict[group] = None
    user_groupdict['match'] = None
    results.append((text[start:], user_groupdict))
  return results


def users_comments(text):
  comments = split(SIGNATURE, text)
  parsed_comments = []
  for segment, matches in comments:
    if matches['match']:
      parsed_comments.append((segment, parse_signature(matches['match'])))
  return parsed_comments


def parse_page(text):
  sects = sections(text)
  cmmts = []
  for section in sects:
    cmmts.extend(users_comments(section))
  return cmmts


def main(options, args):
  text = open(options.filename, 'r').read()
  text = text.decode('utf-8')
  result = parse_page(text)
  json.dump(result, open(options.filename+'.res', 'w'), indent=2)


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

