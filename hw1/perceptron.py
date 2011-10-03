#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""percepton.py: This module afford a ML algorithm and its training method."""

from optparse import OptionParser
import logging
import util
import copy
import random
from nltk.classify.api import ClassifierI
import pdb
import json
__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"


EPOCHS = 10 


class Perceptron(ClassifierI):
  def __init__(self, labels_weights):
    self.labels_weights = labels_weights
    self.labels = labels_weights.keys()
    self.learning_rate = 0.01

  def labels(self):
    return self.labels

  def _predict(self, featureset, label_weights):
#    pdb.set_trace()
    sum = 0.0
    for feature, value in featureset.iteritems():
      if isinstance(value, (str, unicode)):
        stored_key = "%s_%s" % (feature, value)
        if stored_key in label_weights:
          sum += label_weights[stored_key] * 1.0
      else:
        sum += label_weights[feature] * value
    return sum + label_weights["bias"]

  def classify(self, featureset):
    predictions = []
    for label in self.labels:
      predictions.append((self._predict(featureset, self.labels_weights[label]),
                         label))
    return max(predictions)[1]

  def _increase_label_weights(self, featureset, label_weights):
    label_weights["bias"] += self.learning_rate * 1.0
    for feature, value in featureset.iteritems():
      if isinstance(value, (unicode, str)):
        stored_key = "%s_%s" % (feature, value)
        if stored_key in label_weights:
          label_weights[stored_key] += self.learning_rate * 1.0
      else:
        label_weights[feature] += self.learning_rate * value

  def _decrease_label_weights(self, featureset, label_weights):
    label_weights["bias"] -= self.learning_rate * 1.0
    for feature, value in featureset.iteritems():
      if isinstance(value, (unicode, str)):
        stored_key = "%s_%s" % (feature, value)
        if stored_key in label_weights:
          label_weights[stored_key] -= self.learning_rate * 1.0
      else:
        label_weights[feature] -= self.learning_rate * value

  def update(self, featureset, correct_label):
    for label in self.labels:
      if label != correct_label:
        self._decrease_label_weights(featureset, self.labels_weights[label])
    self._increase_label_weights(featureset, self.labels_weights[correct_label])

  def dump_json(self):
    content = json.dumps(self.labels_weights, indent=2)
    fdesc = open("perceptron.json", "w")
    fdesc.write(content)
    fdesc.close()
    logging.debug("perceptron.json is saved.")
    
    
def train(labeled_featuresets):
  weights = {}
  weights = {"bias": 0.0}
  labels_weights = {}
  # For numerical values the perceptron algorithm should be straight forward
  for featureset, label in labeled_featuresets:
    labels_weights[label] = None
    for feature, value in featureset.iteritems():
      if isinstance(value, (unicode, str)):
        weights['%s_%s' % (feature, value)] = 0.0
      else:
        weights[feature] = 0.0

  for label in labels_weights:
    labels_weights[label] = copy.deepcopy(weights)

  classifier = Perceptron(labels_weights)
  classifier.learning_rate = 100.0/len(labeled_featuresets) 

  logging.info("The perceptron algorithm will be trained %s times over the set "
               "with learning rate %f" % (EPOCHS, classifier.learning_rate))
  for epoch in range(EPOCHS):
    random.shuffle(labeled_featuresets)
    for featureset, label in labeled_featuresets:
      if classifier.classify(featureset) != label:
        classifier.update(featureset, label)

  logging.info("Perceptron classfier is trained")
  if util.log_level() == logging.DEBUG:
    classifier.dump_json()

  return classifier
