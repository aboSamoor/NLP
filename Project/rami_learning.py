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
from nltk.corpus import stopwords
from nltk.classify.maxent import MaxentClassifier
from multiprocessing import Pool, cpu_count

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

__ranges = [(0.0, 0.1)]
TRAIN = []
DEV = []
TEST = []
for segment in __ranges:
  start, end = segment
  TRAIN.extend(json.load(open('data/train.%.1f-%.1f.tok.json.txt' %(start, end), 'r')))
  DEV.extend(json.load(open('data/dev.%.1f-%.1f.tok.json.txt' % (start, end), 'r')))
  TEST.extend(json.load(open('data/test.%.1f-%.1f.tok.json.txt' % (start, end), 'r')))

TRAIN = filter(lambda x: x[0][0], TRAIN)
DEV = filter(lambda x: x[0][0] and x[1], DEV)
TEST = filter(lambda x: x[0][0] and x[1], TEST)

__train_fs = []
__dev_fs = []

LANGUAGES = list(set([label for comment,label in TRAIN]))

SENTENCE_SPLITTER =  nltk.data.load('tokenizers/punkt/english.pickle')
WORD_TOKENIZER = nltk.TreebankWordTokenizer()
EN_STOPWORDS = stopwords.words('english') 

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
    probabilities = []
    for label in self.labels():
      probabilities.append((comment_perplexity(comment, self._model[label]), label))
    return min(probabilities)[1]


def comment_perplexity(comment, model):
  e = 0.0
  words = sum(comment, [])
  for statement in comment:
    e += model.entropy(statement)
  e /= len(words)
  return e


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


UNIGRAMS = language_ngrams('Unigram_TRAIN', 1, TRAIN)
def featureset(sample):
  comment, label = sample
  features = {}
  words = sum(comment, [])
  size_= sum([len(word) for word in words])
  features['stmt_len'] = len(words)/float(len(comment))
  features['word_len'] = size_/float(len(words))
  features['size'] = size_ 
  size_= sum([len(word) for word in words])
  dist = FreqDist([word.lower() for word in words])
  for word in EN_STOPWORDS:
    features[word] = dist.get(word, 0)/float(len(words))
  features['alwayson'] = True
#  for language in LANGUAGES:
#    features['perp_%s' % language] = comment_perplexity(comment, UNIGRAMS[language])
  return (features, label)


@Serialized
def samples_featuresets(samples):
  p = Pool(cpu_count())
  return p.map(featureset, samples)


def normalize(featuresets, feature):
  values = [fs.get(feature, 0) for fs, label in featuresets]
  range_ = max(values) - min(values)
  avg = sum(values)/float(len(values))
  for i in range(len(featuresets)):
    old_value = featuresets[i][0].get(feature, 0)
    new_value = (old_value-avg)/range_
    featuresets[i][0][feature] = new_value


def normalize_featuresets(featuresets):
  fs, label = featuresets[0]
  def isnumeric(x):
    return type(x) in {type(1), type(1.0), type(1L)}
  numeric_features = [f for f,v in filter(lambda fv: isnumeric(fv[1]), fs.items())]
  for feature in numeric_features:
    normalize(featuresets, feature)


def sigma(lambda_):
  return (1.0/lambda_)**(0.5)


def pick_lambda(train, dev):
  global __train_fs, __dev_fs
#  lambdas = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100]
  lambdas = [0.1, 10]
#  lambdas = [1]
  __train_fs = train
  __dev_fs = dev
  if len(lambdas) ==1:
    return maxent_classifier(lambdas[0])
  else:
    p = Pool(cpu_count())
    return p.map(maxent_classifier, lambdas)


def maxent_classifier(lambda_):
  sigma_ = sigma(lambda_)
  maxent = maxentropy('maxent_%4.4f' % lambda_, __train_fs, sigma_)
  logging.info("Finished training the classifier lambda=%f ..." % lambda_)
  dev_acc = accuracy(maxent, __dev_fs)
  logging.info("MaxEnt_classifier lambda=%f accuracy on DEV is: %3.5f",
                lambda_, dev_acc)
  train_acc = accuracy(maxent, __train_fs)
  logging.info("MaxEnt_classifier lambda=%f accuracy on TRAIN is: %3.5f",
                lambda_, train_acc)
  return (lambda_, dev_acc, train_acc)


@Serialized
def maxentropy(samples_fs, sigma=0):
  maxent = MaxentClassifier.train(samples_fs, 'MEGAM', gaussian_prior_sigma=sigma)
  return maxent


def main(options, args):
  logging.info("Training Size: %d\t DEV size: %d\t TEST size:%d"
               % (len(TRAIN), len(DEV), len(TEST)))
#  random_classifier = Random()
#  logging.debug("random_classifier labels are:\n %s", str(random_classifier.labels()))
#  logging.info("random_classifier accuracy is: %3.3f", accuracy(random_classifier, TEST))

#  most_common = MostCommon()
#  logging.debug("MostCommon_classifier labels are:\n %s", str(most_common.labels()))
#  logging.info("MostCommon_classifier accuracy is: %3.3f", accuracy(most_common, TEST))

#  perp = Perplexity()
#  logging.debug("Perplexity_classifier labels are:\n %s", str(perp.labels()))
#  logging.info("Perplexity_classifier accuracy is: %3.3f", accuracy(perp, DEV))

  logging.info("Started calculating the featuresets ...")
  train_fs = samples_featuresets('TRAIN_FS', TRAIN)
  normalize_featuresets(train_fs)
  logging.info("Finished calculating the featuresets of training...")
  dev_fs = samples_featuresets('DEV_FS', DEV)
  normalize_featuresets(train_fs)
  logging.info("Finished calculating the featuresets of development...")
#  maxent = maxentropy('maxent_%4.4f' % 1.0, train_fs, sigma(1))
  stats = pick_lambda(train_fs, dev_fs)
  text = '\n'.join([', '.join([str(num) for num in stat]) for stat in stats])
  print '\n---------------------\n',text

  result = [maxent.classify(fs) for fs,label in dev_fs]
  gold = [label for fs,label in dev_fs]
  cm = nltk.ConfusionMatrix(gold, result)
  print cm.pp(sort_by_count=True, show_percents=True, truncate=20)

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)
