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
from sklearn.feature_selection import RFE
from math import log
from settings import *
from re import compile
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"
EN_STOPWORDS = stopwords.words('english') 


def is_ascii(char):
  return ord(char) <= 128


def replace_non_ascii(char):
  if not is_ascii(char):
    return '*'
  return char


def clean_word(tagged_word):
  word, tag = tagged_word
  return (''.join([replace_non_ascii(c) for c in word]), tag)


def replace_NNP(tagged_word):
  word, tag = tagged_word
  if tag in ['NNP', 'NNPS']:
    return (tag, tag)
  return (word,tag)


def _map(func, list_, num_processors=2):
  p = Pool(num_processors)
  results = p.map(func, list_)
  p.close()
  p.join()
  return results


def clean_comment(sample):
  comment, label = sample
  new_comment = [[replace_NNP(tagged_word) for tagged_word in statement] for statement in comment]
  new_comment_2 = [[clean_word(tagged_word) for tagged_word in statement] for statement in new_comment]
  return new_comment_2, label


def to_string_comment(sample):
  text = ""
  comment, label = sample
  for statement in comment:
    text += ' '.join([word for word,tag in statement])
    text += ' '
  text += '\n'+label
  return text


def to_string_samples(samples):
  results = _map(to_string_comment, samples, (cpu_count()/6)+1)
  return '\n\n'.join(results)


def prune_data(samples):
  """Decide which samples are valid"""
  def is_valid(sample):
    comment, label = sample
    cond_1 = len(comment) >= 1 and len(comment[0]) >= 1
    cond_2 = label in MAP
    cond_3 = sum([len(statement) for statement in comment]) >= COMMENT_SIZE
    if cond_1 and cond_2 and cond_3:
      return True
    return False

  samples = filter(is_valid, samples)

  comments, labels = zip(*samples)
  labels = [MAP[l] for l in labels]
  samples = zip(comments, labels)

  # Balance data
  dist = FreqDist(labels)
  smallest_number = dist.items()[-1][1]

  new_samples = []
  counts = {}
  for lang in dist.keys():
    counts[lang] = 0

  for sample in samples:
    comment, label = sample
    if counts[label] <= smallest_number:
      new_samples.append(sample)
      counts[label] += 1

  # clean data
  if CLEAN:
    new_samples = _map(clean_comment, new_samples)
#    open('clean_data.txt', 'w').write(to_string_samples(new_samples))
  return new_samples


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
  if not os.path.isdir(__CACHE):
    os.mkdir(__CACHE)

  def __init__(self, func):
    self.func = func

  def _path(self, name):
    filename = '.'.join([self.func.__module__, self.func.__name__, name, 'ser'])
    return os.path.join(Serialized.__CACHE, filename)

  def __call__(self, name, *args):
    data = None
    filename = self._path(name)
    try:
      if not CACHE:
        raise Exception
      fh = open(filename, 'r')
      data = load(fh)
      fh.close()
    except:
      data = self.func(*args)
      if CACHE:
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
    self._clf = linear_model.LogisticRegression(C=C)
#    self._clf = svm.SVC(C=C)
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


  def show_most_informative_features(self, samples):
    X, y = self._fsets2dataset(samples)
    rfe = RFE(self._clf, 1)
    rfe.fit(X, y)
    ranking = rfe.ranking_
    if len(ranking) != len(self._fx):
      logging.error("Both feature ranking and features should have the same"
                     "length %d != %d", len(ranking), len(self._fx))
    fx_ranking = []
    for i in range(len(self._fx)):
      fx_ranking.append((ranking[i], self._fx[i]))
    self._clf.fit(X, y)
    return '\n'.join(['\t'.join([str(y),str(x)]) for x,y in sorted(fx_ranking)])


class Pybrain(MultiClassifierI):
  def __init__(self):
    self._ds = None
    self._net = None
    self._trainer = None
    labels = self.labels()
    self._map = dict(zip(labels, range(len(labels))))
    self._rmap = dict(zip(range(len(labels)), labels))

  def samples_to_Xy(self, samples):
    if not self._fx:
      self._fx = samples[0][0].keys()
    converter = lambda fset:[fset[l] for l in self._fx]
    fsets, labels = zip(*samples)
    X = map(converter, fsets)
    return zip(X, labels) 
    
    
  def train(self, samples):
    self._fx = samples[0][0].keys()
    self._ds = ClassificationDataSet(len(self._fx))
    for sample in self.samples_to_Xy(samples):
      fvec, label = sample
      self._ds.addSample(fvec, [self._map[label]])
    self._ds._convertToOneOfMany()
    self._net = buildNetwork(self._ds.indim,
                            self._ds.outdim, 
                            self._ds.outdim, 
                            self._ds.outdim, bias=True, outclass=SoftmaxLayer)
    self._trainer = BackpropTrainer(self._net, self._ds, verbose=True,
                                    )
    error = self._trainer.trainEpochs(40)
#    logging.info('Error: %f' % error)
    return self
    
  def labels(self):
    return LANGUAGES

  def batch_classify(self, samples):
    ds = ClassificationDataSet(len(self._fx))
    for sample in samples:
      fvec = [sample[l] for l in self._fx]
      ds.addSample(fvec, [0])
    results = self._trainer.testOnClassData(ds)
    return [self._rmap[r] for r in results]
      
      
def similarity(freqdist, sequence, n):
  """Calculate the log of the counts of the n-grams of a sequence against a
     precomputed a frequency distribution"""
  similarity = 0.0
  ngrams_ = ngrams(sequence, n, True, True)
  for ngram in ngrams_:
    count = freqdist[n].get(ngram, 1.0)
    similarity += log(count, 2)
  return similarity/len(sequence)


def comment_similarity(model, comment, n):
  """calculate the comment similarity with a language distribution for a give
      n-gram"""
  words_similarity = 0.0
  tags_similarity = 0.0
  char_similarity = 0.0
  w_size_similarity = 0.0
  size = float(len(comment))
  for statement in comment:
    words,tags = zip(*statement)
    sizes = [len(word) for word in words]
    statement_text = ' '.join(words)
    char_similarity += similarity(model["chars"], statement_text, n)
    words_similarity += similarity(model["words"], words, n)
    tags_similarity += similarity(model["tags"], tags, n)
    w_size_similarity += similarity(model["w_sizes"], sizes, n)
  return [value/size for value in 
          [words_similarity, tags_similarity, char_similarity, w_size_similarity]]
    

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
    language_dist[language] = {"words": dict(zip(range(1, n+1), [FreqDist() for i in range(1, n+1)])),
                               "tags": dict(zip(range(1, n+1), [FreqDist() for i in range(1, n+1)])),
                               "chars": dict(zip(range(1, n+1), [FreqDist() for i in range(1, n+1)])),
                               "w_sizes": dict(zip(range(1, n+1), [FreqDist() for i in range(1, n+1)]))}
  for comment, language in training:
    for statement in comment:
      words, tags = zip(*statement)
      sizes = [len(word) for word in words]
      statement_text = ' '.join(words)
      for i in range(1, n+1):
        language_dist[language]["words"][i].update(nltk.ngrams(words, i))
        language_dist[language]["tags"][i].update(nltk.ngrams(tags, i))
        language_dist[language]["chars"][i].update(nltk.ngrams(statement_text, i))
        language_dist[language]["w_sizes"][i].update(nltk.ngrams(sizes, i))
  return language_dist


def featureset(sample):
  comment, label = sample
  features = {}
#  tags = map(lambda statement: map(lambda (w,t):t, statement), comment)
#  words = map(lambda statement: map(lambda (w,t):w, statement), comment)
#  words = sum(words, [])
#  tags = sum(tags, [])
#  size_= sum([len(word) for word in words])
#  features['stmt_len'] = len(words)/float(len(comment))
#  features['word_len'] = size_/float(len(words))
#  features['size'] = size_
#  tags_dist = FreqDist(sum(tags, []))
#  for tag in TAGS:
#    features[tag] = tags_dist.get(tag, 0)
#  dist = FreqDist([word.lower() for word in words])
#  num_stop_words = float(sum([dist.get(word, 0) for word in EN_STOPWORDS]))
#  features['prob_stop_words'] = num_stop_words/len(words)
#  for word in EN_STOPWORDS:
#    features[word] = dist.get(word, 0)/float(len(words))
  features['alwayson'] = 1.0
  for language in LANGUAGES:
    for i in range(1,n+1):
      word_sim, tag_sim, char_sim, w_s_sim = comment_similarity(GRAMS[language], comment, i)
      features['w_sim_%d_%s' % (i, language)] = word_sim
      features['t_sim_%d_%s' % (i, language)] = tag_sim
      features['c_sim_%d_%s' % (i, language)] = char_sim
      features['s_sim_%d_%s' % (i, language)] = w_s_sim
  return (features, label)


@Serialized
def samples_featuresets(samples):
  fs = _map(featureset, samples, (cpu_count()/8)+1)
  normalize_featuresets(fs)
  return fs


def normalize(featuresets, feature):
  values = array([fs.get(feature, 0) for fs, label in featuresets])
  std = values.std()
  avg = values.mean()
  for i in range(len(featuresets)):
    old_value = featuresets[i][0].get(feature, 0)
    new_value = old_value
    if std != 0:
      new_value = (old_value-avg)/std
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
  __train_fs = train
  __dev_fs = dev
  if len(LAMBDAS) == 1:
    return [classifier(LAMBDAS[0])]
  else:
    result = _map(classifier, LAMBDAS, (cpu_count()/8)+1)
    return result
    

def classifier(lambda_):
  clf = get_classifier('%f' % lambda_, __train_fs, lambda_)
  logging.debug("Finished training the classifier lambda=%f ..." % lambda_)
  dev_acc = accuracy(clf, __dev_fs)
  logging.debug("classifier lambda=%f accuracy on DEV is: %3.5f",
                lambda_, dev_acc)
  train_acc = accuracy(clf, __train_fs)
  logging.debug("classifier lambda=%f accuracy on TRAIN is: %3.5f",
                lambda_, train_acc)
#  clf.show_most_informative_features()
  result = [clf.classify(fs) for fs,label in __dev_fs]
  gold = [label for fs,label in __dev_fs]
  cm = nltk.ConfusionMatrix(gold, result)
  cf_text = cm.pp(sort_by_count=True, show_percents=True, truncate=20)
  return (lambda_, dev_acc, train_acc, cf_text, clf)


@Serialized
def get_classifier(samples_fs, lambda_=1):
#  sigma_ = sigma(lambda_)
#  clf = MaxentClassifier.train(samples_fs, 'MEGAM',
#                                  gaussian_prior_sigma=sigma_)
  clf = Scikit(lambda_).train(samples_fs)
#  clf = Pybrain().train(samples_fs)
  return clf


def stats(samples):
  _, labels = zip(*samples)
  dist = FreqDist(labels)
  size = len(samples)
  for key in dist.keys():
    logging.info("%s\t%f" % (key, dist[key]/float(size)))


def main(options, args):

  global TRAIN, DEV, TEST, MAP, GRAMS, LANGUAGES, TAGS, __train_fs, __dev_fs, __test_fs

  exps_MAPS ={'g': MAP_GROUPS, 'p': MAP_POPULAR, 'f': MAP_FOREIGN, 'n': MAP_POP_NoEN}
  MAP = exps_MAPS[options.exp]
  logging.info("Eperiment mode is %s" % options.exp)

  TRAIN = json.load(open(TRAIN_FILE, 'r'))
  DEV  = json.load(open(DEV_FILE, 'r'))
  TEST = json.load(open(TEST_FILE, 'r'))
  logging.info("Data is loaded")

  TRAIN = prune_data(TRAIN) 
  DEV = prune_data(DEV)
  TEST = prune_data(TEST)
  logging.info("Data is filtered")
  
  TAGS = list(set([tag for comment,label in DEV for statement in comment for word,tag in statement]))
  LANGUAGES = list(set([label for comment,label in TRAIN]))
  GRAMS = language_distribution('Model_TRAIN', n, TRAIN)


  logging.info("Ngrams calculated up to: %d\n" % n)
  logging.info("ACCEPTED_LANGUAGES: %s\n" % str(MAP.keys()))
  logging.info("LANGUAGES_MAP: %s\n" % str(MAP))
  logging.info("Comment Minimum Size: %d" % COMMENT_SIZE)
  logging.info("Percentage of data used: %d" % PERCENTAGE)
  if CLEAN:
    logging.info("NPP(S) are replaced by their tags and non ascii characters are replaced by a special character: %d" % CLEAN)
 
 
  logging.info("Training Size: %d\t DEV size: %d\t TEST size:%d"
               % (len(TRAIN), len(DEV), len(TEST)))
  total_size = float(len(TRAIN) + len(DEV) + len(TEST))
  logging.info("Training Size: %f\t DEV size: %f\t TEST size:%f"
               % (len(TRAIN)/total_size, len(DEV)/total_size, len(TEST)/total_size))
  
 
  random_classifier = Random()
  logging.debug("random_classifier labels are:\n %s", str(random_classifier.labels()))
  logging.info("random_classifier accuracy is: %3.3f", accuracy(random_classifier, TEST))

  most_common = MostCommon()
  logging.debug("MostCommon_classifier labels are:\n %s", str(most_common.labels()))
  logging.info("MostCommon_classifier accuracy is: %3.3f", accuracy(most_common, TEST))

#  perp = Perplexity()
#  logging.debug("Perplexity_classifier labels are:\n %s", str(perp.labels()))
#  logging.info("Perplexity_classifier accuracy is: %3.3f", accuracy(perp, DEV))

  logging.debug("Started calculating the featuresets ...")
  train_fs = samples_featuresets('TRAIN_FS', TRAIN)
  logging.info("Features calculated:\n%s\n" %(",".join([str(key) for key in train_fs[0][0].keys()])))
  logging.debug("Finished calculating the featuresets of training...")
  dev_fs = samples_featuresets('DEV_FS', DEV)
  logging.debug("Finished calculating the featuresets of development...")
  results = pick_lambda(train_fs, dev_fs)

  logging.info("stats about training data")
  stats(TRAIN)
  logging.info("stats about dev data")
  stats(DEV)
  lambdas, dev_acc, train_acc, cfms, clfs = zip(*results)
  reduced_results = zip(lambdas, dev_acc, train_acc)
  text = '\n'.join([', '.join([str(num) for num in stat]) for stat in reduced_results])
  logging.info("Results:")
  logging.info('\n'+text)

  max_index = dev_acc.index(max(dev_acc))
  best_lambda = lambdas[max_index]
  cfm = cfms[max_index]
  logging.info("best lambda is %f", best_lambda)
  logging.info("Confusion Matrix of the best classifier on the dec data:\n%s\n" % cfm)


  test_fs = samples_featuresets('TEST_FS', TEST)
  result = [clfs[max_index].classify(fs) for fs,label in test_fs]
  gold = [label for fs,label in test_fs]
  acc = len(filter(lambda (x,y): x==y, zip(result, gold)))/float(len(gold))
  logging.info("Accuracy on test is %f", acc)
  cm = nltk.ConfusionMatrix(gold, result)
  cf_text = cm.pp(sort_by_count=True, show_percents=True, truncate=20)
  logging.info("Confusion Matrix of the best classifier on the test data:\n%s\n" % cf_text)
  
  #This introduced a bug, where raw_coef are modified when they should not
#  logging.info('Important features\n%s\n', clfs[max_index].show_most_informative_features(dev_fs))

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-e", "--experiment", dest="exp", help="Experiment Class")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()
  root = logging.getLogger()
  while root.handlers:
    root.removeHandler(root.handlers[0])
  numeric_level = getattr(logging, options.log.upper(), None)
  file_log = '.'.join([options.filename, options.exp, 'log'])
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT, filename=file_log)
  main(options, args)
