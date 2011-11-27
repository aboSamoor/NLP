#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rami_learnign.py: Contains the ML algorithms and the baseline used to solve
the native language attribution problem."""

from optparse import OptionParser
import logging
import json
import nltk
import os
from nltk.classify.api import MultiClassifierI
from nltk import FreqDist
from random import choice
from nltk.classify import accuracy
from nltk import NgramModel
from nltk.probability import LidstoneProbDist, SimpleGoodTuringProbDist
import functools
import pdb
from cPickle import dump, load

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

TRAIN = [(comment, label) for (comment, label) in json.load(open('data/train.0.0-0.1.tok.json.txt', 'r'))]
DEV = [(comment, label) for (comment, label) in json.load(open('data/dev.0.0-0.1.tok.json.txt', 'r'))]
TEST = [(comment, label) for (comment, label) in json.load(open('data/test.0.0-0.1.tok.json.txt', 'r'))]
LANGUAGES = list(set([label for comment,label in TRAIN]))

SENTENCE_SPLITTER =  nltk.data.load('tokenizers/punkt/english.pickle')
WORD_TOKENIZER = nltk.TreebankWordTokenizer()

class Memoized(object):
  """Decorator that caches a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned, and
  not re-evaluated.
  """

  __cache = {}

  def __init__(self, func):
    self.func = func
    self.key = (func.__module__, func.__name__)

  def __call__(self, *args):
    try:
      return Memoized.__cache[self.key][args]
    except KeyError:
      value = self.func(*args)
      if self.key in Memoized.__cache:
        Memoized.__cache[self.key][args] = value
      else:
        Memoized.__cache[self.key] = {args : value}
      return value
    except TypeError:
      # uncachable -- for instance, passing a list as an argument.
      # Better to not cache than to blow up entirely.
      return self.func(*args)

  def __get__(self, obj, objtype):
    """Support instance methods."""
    return functools.partial(self.__call__, obj)

  @staticmethod
  def reset():
    Memoized.__cache = {}


class Serialized(object):
  """Decorator that serliazes a function's return value each time it is called.
  If called later with the same arguments, the cached value is returned, and
  not re-evaluated.
  """

  __CACHE = 'cache'
  if not os.path.isdir('cache'):
    os.mkdir(Serialized.__CACHE)

  def __init__(self, func):
    self.func = func

  def _path(self, name):
    filename = '.'.join([self.func.__module__, self.func.__name__, name, 'ser'])
    return os.path.join(Serialized.__CACHE, filename)

  def __call__(self, name, *args):
    data = None
    filename = self._path(name)
    try:
      fh = open(filename, 'r')
      data = load(fh)
      fh.close()
    except:
      data = self.func(*args)
      fh = open(filename, 'w')
      dump(data, fh)
      fh.close()
    return data

  def __get__(self, obj, objtype):
    """Support instance methods."""
    return functools.partial(self.__call__, obj)


def _estimator(fdist, bins):
  return SimpleGoodTuringProbDist(fdist)


def tokenize(text):
  sentences = [sentence for sentence in SENTENCE_SPLITTER.tokenize(text)]
  tokens = [WORD_TOKENIZER.tokenize(sentence) for sentence in sentences]
  return tokens


def tokenize_comments(samples):
  for comment,label in samples:
      yield (tokenize(comment), label)


class Random(MultiClassifierI):

  @property
  @Memoized
  def _train_labels(self):
    return [label for (comment, label) in TRAIN]

  def classify(self, unlabeled_problem):
    return choice(self._train_labels)

  @Memoized
  def labels(self):
    return list(set(self._train_labels))


class MostCommon(MultiClassifierI):
  
  @property
  @Memoized
  def _train_labels(self):
    return [label for (comment, label) in TRAIN]

  @property
  @Memoized
  def _labels_dist(self):
    return FreqDist(self._train_labels)
    
  def classify(self, unlabeled_problem):
    return self._labels_dist.items()[0][0]

  @Memoized
  def labels(self):
    return list(set(self._train_labels))


class Perplexity(MultiClassifierI):
  def __init__(self):
    self._model = language_ngrams('Unigram_TRAIN', 1, TRAIN)

  def labels(self):
    return LANGUAGES

  def classify(self, comment):
    words = sum(comment, [])
    if not words:
      return None
    probabilities = []
    for label in self.labels():
      p = 0
      for statement in comment:
        p += self._model[label].entropy(statement)
      p /= len(words)
      probabilities.append((p,label))
    return min(probabilities)[1]
      
  

@Serialized
def language_ngrams(n, training):
  language_ngrams = {}
  languages = {}
  for language in LANGUAGES:
    languages[language] = []
  for comment, language in training:
    words_of_a_comment = [word for statement in comment for word in statement]
    languages[language].extend(words_of_a_comment)
  for language in LANGUAGES:
    language_ngrams[language] = NgramModel(n, languages[language], _estimator)
  return language_ngrams


def maxent_featureset(unlabeled_token):
  pass


def main(options, args):
#  random_classifier = Random()
#  logging.debug("random_classifier labels are:\n %s", str(random_classifier.labels()))
#  logging.info("random_classifier accuracy is: %3.3f", accuracy(random_classifier, TEST))
#  most_common = MostCommon()
#  logging.debug("MostCommon_classifier labels are:\n %s", str(most_common.labels()))
#  logging.info("MostCommon_classifier accuracy is: %3.3f", accuracy(most_common, TEST))
  perp = Perplexity()
  logging.debug("Perplexity_classifier labels are:\n %s", str(perp.labels()))
  logging.info("Perplexity_classifier accuracy is: %3.3f", accuracy(perp, TEST))


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)
