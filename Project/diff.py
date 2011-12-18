#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
import  re
import nltk

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

tokenizer = None


def sentence_tokenize(text):
  """
  The splitter divides the text depending on the.
  """
  # A set of punctuation characters to split a senetence
  END_OF_STATEMENT = '[\n!?.]'

  # The following + matches \n\n, ???, or !!.
  # The splitter should be followed by space otherwise we will split I.B.M
  SPLITTER = '%s+\s+' % END_OF_STATEMENT

  # To report the splitter matched we have to surround the regex by ()
  temp = re.split(r'(%s)' % SPLITTER, text)

  # we need to combine the splitters to their original statements
  sentences = []
  if len(temp) > 1:
    for i in xrange(0, len(temp)-1, 2):
      sentences.append('%s%s' % (temp[i], temp[i+1]))
  else:
    sentences = temp
  return sentences


def nltk_sentence_tokenize(text):
  global tokenizer
  if not tokenizer:
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
  return tokenizer.tokenize(text)


def main(options, args):
  pass
  

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

