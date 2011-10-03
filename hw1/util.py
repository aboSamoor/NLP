#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
import os
import copy
import functools
import json
import nltk
from nltk.tag import StanfordTagger
from nltk.probability import FreqDist
from nltk.probability import GoodTuringProbDist, ELEProbDist, MLEProbDist

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"


MODEL_PATH = "stanford-postagger-full-2011-09-14/models/left3words-wsj-0-18.tagger"
JAR_PATH = "stanford-postagger-full-2011-09-14/stanford-postagger.jar"
STEMMER = nltk.WordNetLemmatizer()
TAGGER = StanfordTagger(path_to_model=MODEL_PATH, path_to_jar=JAR_PATH)


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


def get_text(filename):
  abs_filename = os.path.abspath(filename)
  fdesc = open(abs_filename, "r")
  raw_text = fdesc.read()
  fdesc.close()
  return raw_text


def write_text(text, filename):
  abs_filename = os.path.abspath(filename)
  fdesc = open(abs_filename, "w")
  fdesc.write(text)
  fdesc.close()


def log_level():
  return logging.getLogger().getEffectiveLevel()

class Document(nltk.Text):
  def __init__(self, filename, tokens=[]):
    self._filename = filename
    self._tokens = tokens

  @property
  @Memoized
  def text(self):
    fdesc = open(self.name, "r")
    raw_text = fdesc.read()
    fdesc.close()
    logging.info("%s is read." % self.name)
    return raw_text

  @property
  def name(self):
    return os.path.abspath(self._filename)

  @property
  def tokens(self):
    if not self._tokens:
      sentences = nltk.sent_tokenize(self.text)
      tokens = [nltk.word_tokenize(sentence) for sentence in sentences]
      self._tokens = tokens
    return self._tokens

  @tokens.setter
  def tokens(self, value):
    self._tokens = value
    
  @property
  def words(self):
    for sentence in self.tokens:
      for word in sentence:
        yield word

  @staticmethod
  def dump_text(content, ext):
    fdesc = open("%s.%s" % (self.name, ext), "w")
    fdesc.write(content)
    fdesc.close()
    
  def dump_json(self, object_, ext):
    content = json.dumps(object_, indent=2)
    name = "%s.%s.json" % (self.name, ext)
    fdesc = open(name, "w")
    fdesc.write(content)
    fdesc.close()
    logging.debug("%s is saved." % name)

  @property
  @Memoized
  def pos_tagged_tokens(self):
    try:
      pos_tagged_tokens = json.load(open("%s.pos.json" % self.name, "r"))
      logging.info("POS tags are loaded from %s.pos.json" % self.name)
      return pos_tagged_tokens
    except:
      pos_tagged_tokens = TAGGER.batch_tag(self.tokens)
      logging.info("POS tags are calcualted for %s." % self.name)
      if log_level() == logging.DEBUG:
        self.dump_json(pos_tagged_tokens, "pos")
      return pos_tagged_tokens

  @property
  @Memoized
  def histogram(self):
    logging.info("Frequency distribution is calculated.")
    return FreqDist(self.words)

  def exists(self, word):
    return word in self.histogram
  
  @Memoized
  def language_model(self, level):
    #estimator = lambda fdist, bins: LidstoneProbDist(fdist, 0.2)
    estimator = lambda fdist, bins: GoodTuringProbDist(fdist)
    #estimator = lambda fdist, bins: MLEProbDist(fdist)
    model = nltk.NgramModel(level, self.words, estimator)
    logging.info("Ngram model of length %d is calculated for %s."
                 % (level, self.name))
    return model
  
  def ngram_prob(self, level, word, context):
    try:
      model = self.language_model(level)
      return model.prob(word, context)
    except:
      return 0.0

  def get_labeled_featureset(self):
    raise NotImplementedError
  
  @property
  def problem_set(self):
    raise NotImplementedError
