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
from nltk import FreqDist, ngrams
from random import choice
from nltk.classify import accuracy
from nltk import NgramModel
from nltk.probability import LidstoneProbDist, SimpleGoodTuringProbDist
import functools
import pdb
from cPickle import dump, load
from nltk.corpus import stopwords
from nltk.classify.maxent import MaxentClassifier, TypedMaxentFeatureEncoding
from multiprocessing import Pool, cpu_count
import perceptron
from numpy import array
from sklearn import linear_model, svm
from math import log
from settings import *

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def prune_data(samples):
  """Decide which samples are valid"""
  def is_valid(sample):
    comment, label = sample
    cond_1 = len(comment) >= NUM_SENTS and len(comment[0]) >= LEN_1SENT
    cond_2 = label in ACCEPTED_LANGUAGES
    if cond_1 and cond_2:
      return True
    return False
  samples = filter(is_valid, samples)
  _, labels = zip(*samples)
  dist = FreqDist(labels)
  smallest_number = dist.items()[-1][1]
  new_samples = []
  counts = {}
  for lang in ACCEPTED_LANGUAGES:
    counts[lang] = 0
  for sample in samples:
    comment, label = sample
    if counts[label] <= smallest_number:
      new_samples.append(sample)
      counts[label] += 1
  return new_samples


TRAIN = json.load(open('data/train.json.txt.pos.20', 'r'))
DEV  = json.load(open('data/dev.json.txt.pos.20', 'r'))
TEST = json.load(open('data/test.json.txt.pos.20', 'r'))

TRAIN = prune_data(TRAIN) 
DEV = prune_data(DEV)
TEST = prune_data(TEST)

__train_fs = []
__dev_fs = []

LANGUAGES = list(set([label for comment,label in TRAIN]))
TAGS = list(set([tag for comment,label in DEV for statement in comment for word,tag in statement]))

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


class Scikit(MultiClassifierI):
  def __init__(self, lambda_=1.0):
    labels = self.labels()
    self._map = dict(zip(labels, range(len(labels))))
    self._rmap = dict(zip(range(len(labels)), labels))
    C = 1.0/lambda_
#    self._clf = linear_model.LogisticRegression(C=C)
    self._clf = svm.SVC(C=C)
    self._fx = None

  def _fsets2dataset(self, samples):
    self._fx = samples[0][0].keys()
    X = []
    y = []
    for sample in samples:
      fset, label = sample
      v = [fset[l] for l in self._fx]
      y.append(self._map[label])
      X.append(v)
    return (array(X), array(y))

  def labels(self):
    return LANGUAGES

  def classify(self, fset):
    v = array([fset[l] for l in self._fx])
    class_ = self._clf.predict(v)
    return self._rmap[class_[0]]

  def batch_classify(self, fsets):
    v = array([[fset[l] for l in self._fx] for fset in fsets])
    classes = self._clf.predict(v)
    return [self._rmap[res] for res in classes]

  def train(self, samples):
    X, y = self._fsets2dataset(samples)
    self._clf.fit(X, y)
    return self


def similarity(freqdist, sequence, n):
  """Calculate the log of the counts of the n-grams of a sequence against a
     precomputed a frequency distribution"""
  similarity = 0.0
  ngrams_ = ngrams(sequence, n, True, True, '<P>')
  for ngram in ngrams_:
    count = freqdist.get(ngram, 1.0)
    similarity += log(count, 2)
  return similarity/len(sequence)


def comment_similarity(model, comment, n):
  """calculate the comment similarity with a language distribution for a give
      n-gram"""
  words_similarity = 0.0
  tags_similarity = 0.0
  char_similarity = 0.0
  for statement in comment:
    words,tags = zip(*statement)
    statement_text = ' '.join(words)
    size = float(len(words))
    char_similarity += similarity(model["chars"], statement_text, n)
    words_similarity += similarity(model["words"], words, n)
    tags_similarity += similarity(model["tags"], tags, n)
  return words_similarity, tags_similarity, char_similarity
    

def comment_perplexity(comment, model):
  e = 0.0
  for statement in comment:
    e += model.entropy(statement)
  return e


@Serialized
def language_ngrams(n, training):
  language_ngrams = {}
  languages = {}
  for language in LANGUAGES:
    languages[language] = []
  for comment, language in training:
    words_of_a_comment = [word for statement in comment for word,tag in statement]
    languages[language].extend(words_of_a_comment)
  for language in LANGUAGES:
    language_ngrams[language] = NgramModel(n, languages[language], _estimator)
  return language_ngrams


@Serialized
def language_ngrams_tags(n, training):
  language_ngrams = {}
  languages = {}
  for language in LANGUAGES:
    languages[language] = []
  for comment, language in training:
    tags_of_a_comment = [tag for statement in comment for word,tag in statement]
    languages[language].extend(tags_of_a_comment)
  for language in LANGUAGES:
    language_ngrams[language] = NgramModel(n, languages[language], _estimator)
  return language_ngrams


@Serialized
def language_distribution(n, training):
  """Calculate the ngrams distribution up to n"""
  language_dist = {}
  for language in LANGUAGES:
    language_dist[language] = {"words": FreqDist(),
                               "tags": FreqDist(),
                               "chars": FreqDist()}
  for comment, language in training:
    for statement in comment:
      words, tags = zip(*statement)
      statement_text = ' '.join(words)
      for i in range(1, n+1):
        language_dist[language]["words"].update(nltk.ngrams(words, i))
        language_dist[language]["tags"].update(nltk.ngrams(tags, i))
        language_dist[language]["chars"].update(nltk.ngrams(statement_text, i))
  return language_dist

n = 4
GRAMS = language_distribution('Model_TRAIN', n, TRAIN)

def featureset(sample):
  comment, label = sample
  features = {}
  tags = map(lambda statement: map(lambda (w,t):t, statement), comment)
  words = map(lambda statement: map(lambda (w,t):w, statement), comment)
  words = sum(words, [])
  tags = sum(tags, [])
  size_= sum([len(word) for word in words])
  features['stmt_len'] = len(words)/float(len(comment))
  features['word_len'] = size_/float(len(words))
  features['size'] = size_
#  tags_dist = FreqDist(sum(tags, []))
#  for tag in TAGS:
#    features[tag] = tags_dist.get(tag, 0)
  dist = FreqDist([word.lower() for word in words])
  for word in EN_STOPWORDS:
    features[word] = dist.get(word, 0)/float(len(words))
  features['alwayson'] = 1.0
  for language in LANGUAGES:
    for i in range(1,n+1):
      word_sim, tag_sim, char_sim = comment_similarity(GRAMS[language], comment, i)
      features['w_sim_%d_%s' % (i, language)] = word_sim
      features['t_sim_%d_%s' % (i, language)] = tag_sim
      features['c_sim_%d_%s' % (i, language)] = char_sim
  return (features, label)


@Serialized
def samples_featuresets(samples):
  p = Pool(cpu_count()/4)
  fs = p.map(featureset, samples)
  normalize_featuresets(fs)
  return fs


def normalize(featuresets, feature):
  values = [fs.get(feature, 0) for fs, label in featuresets]
  range_ = max(values) - min(values)
  avg = sum(values)/float(len(values))
  for i in range(len(featuresets)):
    old_value = featuresets[i][0].get(feature, 0)
    new_value = old_value
    if range_ != 0:
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
  lambdas = [0.01, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
#  lambdas = [0.01, 3, 10, 30, 100]
#  lambdas = [30, 100, 300]
#  lambdas = [3]
  __train_fs = train
  __dev_fs = dev
  if len(lambdas) ==1:
    return [classifier(lambdas[0])]
  else:
    p = Pool(cpu_count())
    return p.map(classifier, lambdas)


def classifier(lambda_):
  clf = get_classifier('%f' % lambda_, __train_fs, lambda_)
  logging.info("Finished training the classifier lambda=%f ..." % lambda_)
  dev_acc = accuracy(clf, __dev_fs)
  logging.info("classifier lambda=%f accuracy on DEV is: %3.5f",
                lambda_, dev_acc)
  train_acc = accuracy(clf, __train_fs)
  logging.info("classifier lambda=%f accuracy on TRAIN is: %3.5f",
                lambda_, train_acc)
#  clf.show_most_informative_features()
  result = [clf.classify(fs) for fs,label in __dev_fs]
  gold = [label for fs,label in __dev_fs]
  cm = nltk.ConfusionMatrix(gold, result)
  cf_text = cm.pp(sort_by_count=True, show_percents=True, truncate=20)
  open('cm.%.4f.log' % lambda_, 'w').write(cf_text)
  return (lambda_, dev_acc, train_acc)


@Serialized
def get_classifier(samples_fs, lambda_=1):
#  sigma_ = sigma(lambda_)
#  clf = MaxentClassifier.train(samples_fs, 'IIS',
#                                  gaussian_prior_sigma=sigma_)
  clf = Scikit(lambda_).train(samples_fs)
  return clf


def stats(samples):
  _, labels = zip(*samples)
  dist = FreqDist(labels)
  size = len(samples)
  for key in dist.keys():
    print key, dist[key]/float(size)

def main(options, args):
  
  random_classifier = Random()
  logging.debug("random_classifier labels are:\n %s", str(random_classifier.labels()))
  logging.info("random_classifier accuracy is: %3.3f", accuracy(random_classifier, TEST))

  most_common = MostCommon()
  logging.debug("MostCommon_classifier labels are:\n %s", str(most_common.labels()))
  logging.info("MostCommon_classifier accuracy is: %3.3f", accuracy(most_common, TEST))

#  perp = Perplexity()
#  logging.debug("Perplexity_classifier labels are:\n %s", str(perp.labels()))
#  logging.info("Perplexity_classifier accuracy is: %3.3f", accuracy(perp, DEV))

  logging.info("Started calculating the featuresets ...")
  train_fs = samples_featuresets('TRAIN_FS', TRAIN)
  logging.info("Finished calculating the featuresets of training...")
  dev_fs = samples_featuresets('DEV_FS', DEV)
  logging.info("Finished calculating the featuresets of development...")
  results = pick_lambda(train_fs, dev_fs)
  logging.info("Training Size: %d\t DEV size: %d\t TEST size:%d"
               % (len(TRAIN), len(DEV), len(TEST)))
  print "stats about training data"
  stats(TRAIN)
  print "stats about dev data"
  stats(DEV)
  text = '\n'.join([', '.join([str(num) for num in stat]) for stat in results])
  print '\n---------------------\n',text


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)
