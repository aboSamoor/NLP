#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Hw1.py: Read the data of the Hw1."""


from optparse import OptionParser
import logging
import os
import pprint
import nltk
import copy
import json
import random
import pdb
import perceptron
import util
from util import Memoized

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

class Prepositions(util.Document):
  SUPPORTED_PREPOSITIONS = ["in", "of", "on"]
  BLANK_POSITION = "OOOO"
  STATEMENT_START = "<S>"
  STATEMENT_END = "."
  STATEMENT_START_TAG = "SS"
  STATEMENT_END_TAG = "."

  def __init__(self, filename, tokens=[]):
    super(Prepositions, self).__init__(filename, tokens)
    
  def get_featureset(self, statement):
    index = map(lambda tagged_word: tagged_word[1] == Prepositions.BLANK_POSITION, statement).index(True)
    size = len(statement)
    features = {}
    features["pWord"] = Prepositions.STATEMENT_START
    features["nWord"] = Prepositions.STATEMENT_END
    features["pTag"] = Prepositions.STATEMENT_START_TAG
    features["nTag"] = Prepositions.STATEMENT_END_TAG
    label, tag = statement[index]
    if index > 0:
      word, tag = statement[index - 1]
      features["pTag"] = tag
      features["pWord"] = util.STEMMER.lemmatize(word) 
    if index < size - 1:
      word, tag = statement[index + 1]
      features["nTag"] = tag
      features["nWord"] = util.STEMMER.lemmatize(word)
      features["nChar"] = features["nWord"][0]
    features["pBigram"] = self.ngram_prob(2, label, (features["pWord"])) 
    features["nBigram"] = self.ngram_prob(2, features["nWord"], (label)) 
    features["Trigram"] = self.ngram_prob(3, features["nWord"],
                                                 (features["pWord"], label))
    return features

  def _get_problems(self, statement):
    problems = []
    new_words = []
    for i in range(len(statement)):
      if statement[i][0].lower() in Prepositions.SUPPORTED_PREPOSITIONS:
        label = statement[i][0]
        new_words = copy.deepcopy(statement)
        new_words[i] = (label, Prepositions.BLANK_POSITION)
        problems.append((new_words, label))
    return problems
  
  @property
  @Memoized
  def problem_set(self):
    problem_set = []
    for stmt in self.pos_tagged_tokens: 
      problem_set.extend(self._get_problems(stmt))
    logging.info("%d Problems calculated for %s." % (len(problem_set), self.name))
    if util.log_level() == logging.DEBUG:
      self.dump_json(problem_set, "pset")
    return problem_set

  def get_labeled_featureset(self):
    labeled_featureset = []
    for labeled_problem in self.problem_set:
      problem, label  = labeled_problem
      labeled_featureset.append((self.get_featureset(problem), label))
    logging.info("%d labeled featureset is calculated for %s."
                 % (len(labeled_featureset), self.name))
    if util.log_level() == logging.DEBUG:
      self.dump_json(labeled_featureset, "lfs")
    return labeled_featureset

class Baseline1(Prepositions):
  def __init__(self, filename):
    super(Baseline1, self).__init__(filename)
  
  def labels(self):
    return Prepositions.SUPPORTED_PREPOSITIONS
  
  def classify(self, featureset):
    choices = []
    for preposition in Prepositions.SUPPORTED_PREPOSITIONS:
      P = self.ngram_prob(3, featureset["nWord"], (featureset["pWord"], preposition))
      choices.append((P, preposition))
    return max(choices)[1]
  
  def batch_classify(self, featuresets):
     return [self.classify(fs) for fs in featuresets]

class Baseline2(Baseline1): 
  def __init__(self, filename):
    super(Baseline1, self).__init__(filename)
  
  @property
  @Memoized
  def tags_document(self):
    pos_tokens = []
    for sentence in self.pos_tagged_tokens:
      sentence_tags = []
      for tagged_word in sentence:
        word = tagged_word[0]
        tag = tagged_word[1]
        if tag == "IN" and word in Prepositions.SUPPORTED_PREPOSITIONS:
          sentence_tags.append(word)
        else:
          sentence_tags.append(tag)
      pos_tokens.append(sentence_tags)
    return util.Document("%s.tag" % self.name, tokens=pos_tokens)
  
  def classify(self, featureset):
    choices = []
    for preposition in Prepositions.SUPPORTED_PREPOSITIONS:
      P = self.tags_document.ngram_prob(3, featureset["nTag"], (featureset["pTag"], preposition))
      choices.append((P, preposition))
    return max(choices)[1]

  

def main(options, args):
  logging.info("processing started ...")
  training_document = Prepositions(options.filename)
  testing_document = Prepositions(options.testfile)
  baseline1_document = Baseline1(options.filename)
  baseline2_document = Baseline2(options.filename)

  training_set = training_document.get_labeled_featureset()
  classifier = perceptron.train(training_set)

  testing_set = testing_document.get_labeled_featureset()
  logging.info("Accuracy of the Perceptron Classifier: %f",
               nltk.classify.accuracy(classifier, testing_set))

  errors = open('errors', 'a')
  for fs, label in testing_set:
    if classifier.classify(fs) != label:
      errors.write(str((fs, label))+'\n')

  logging.info("Training NaiveBayes classifier")
  bayes_classifier = nltk.NaiveBayesClassifier.train(training_set)
  logging.info("Accuracy of the NaiveBayes Classifier: %f",
               nltk.classify.accuracy(bayes_classifier, testing_set))

  logging.info("Accuracy of the Baseline1: %f",
                nltk.classify.accuracy(baseline1_document, testing_set))
  
  logging.info("Accuracy of the Baseline2: %f",
                nltk.classify.accuracy(baseline2_document, testing_set))

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-t", "--test", dest="testfile", help="Test file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()
  
  
  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)
