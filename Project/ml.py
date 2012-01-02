#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ml.py: Multiclass classifiers."""

from optparse import OptionParser
import logging
from random import choice
from nltk import FreqDist
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SoftmaxLayer
from sklearn import linear_model, svm
from sklearn.feature_selection import RFE
from nltk.classify.maxent import MaxentClassifier, TypedMaxentFeatureEncoding
from nltk.classify.api import MultiClassifierI
from numpy import array
from cPickle import dump, load
from util import *

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


class Random(MultiClassifierI):

  def __init__(self, trainset):
    self.trainset = trainset

  @property
  @Memoized
  def _train_labels(self):
    return [label for (comment, label) in self.trainset]

  def classify(self, unlabeled_problem):
    return choice(self._train_labels)

  @Memoized
  def labels(self):
    return list(set(self._train_labels))


class MostCommon(MultiClassifierI):

  def __init__(self, trainset):
    self.trainset = trainset

  @property
  @Memoized
  def _train_labels(self):
    return [label for (comment, label) in self.trainset]

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
  def __init__(self, trainset):
    self._model = language_ngrams('Unigram_TRAIN', 1, trainset)

  @Memoized
  def labels(self):
    return list(set(self._train_labels))

  def classify(self, comment):
    words = sum(comment, [])
    probabilities = []
    for label in self.labels():
      probabilities.append((comment_perplexity(comment, self._model[label]), label))
    return min(probabilities)[1]
  
  @property
  @Memoized
  def _train_labels(self):
    return [label for (comment, label) in self.trainset]

  def comment_perplexity(self, comment):
    e = 0.0
    for statement in comment:
      e += self._model.entropy(statement)
    return e


class Scikit(MultiClassifierI):
  def __init__(self, lambda_=1.0):
    self.labels = []
    self._map = {}
    self._rmap = {}
    C = 1.0/lambda_
    self._clf = linear_model.LogisticRegression(C=C)
#    self._clf = svm.SVC(C=C)
#    self._clf = svm.LinearSVC(C=C)
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
    return self.labels

  def classify(self, fset):
    v = array([fset[l] for l in self._fx])
    class_ = self._clf.predict(v)
    return self._rmap[class_[0]]

  def batch_classify(self, fsets):
    v = array([[fset[l] for l in self._fx] for fset in fsets])
    classes = self._clf.predict(v)
    return [self._rmap[res] for res in classes]

  def train(self, samples):
    self.labels = list(set([label for fset,label in samples]))
    self._map = dict(zip(self.labels, range(len(self.labels))))
    self._rmap = dict(zip(range(len(self.labels)), self.labels))
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
    self.labels = []
    self._map = {}
    self._rmap = {}

  def samples_to_Xy(self, samples):
    if not self._fx:
      self._fx = samples[0][0].keys()
    converter = lambda fset:[fset[l] for l in self._fx]
    fsets, labels = zip(*samples)
    X = map(converter, fsets)
    return zip(X, labels) 
    
    
  def train(self, samples):
    self.labels = list(set([label for fset,label in samples]))
    self._map = dict(zip(labels, range(len(labels))))
    self._rmap = dict(zip(range(len(labels)), labels))
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
    return self.labels

  def batch_classify(self, samples):
    ds = ClassificationDataSet(len(self._fx))
    for sample in samples:
      fvec = [sample[l] for l in self._fx]
      ds.addSample(fvec, [0])
    results = self._trainer.testOnClassData(ds)
    return [self._rmap[r] for r in results]
      

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
