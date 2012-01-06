#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""rami_learnign.py: Contains the ML algorithms and the baseline used to solve
the native language attribution problem."""

from optparse import OptionParser
import logging
import json
import nltk
import os
import sys
from random import shuffle
from nltk import FreqDist, ngrams
from nltk.classify import accuracy
from nltk import NgramModel
from nltk.probability import LidstoneProbDist, SimpleGoodTuringProbDist
from cPickle import dump, load
from nltk.corpus import stopwords
from multiprocessing import Pool, cpu_count
import perceptron
from numpy import array
from math import log
from re import compile
import util
from demonym import *
from ml import *
from util import *

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
  if word in DEMONYMS:
    return (tag, tag)
  return (word,tag)


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
  results = pool_map(to_string_comment, samples, (cpu_count()/6)+1)
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

  shuffle(samples)
  comments, labels = zip(*samples)
  labels = [MAP[l] for l in labels]
  samples = zip(comments, labels)
  new_samples = samples

  if BALANCED:
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
    new_samples = pool_map(clean_comment, new_samples)
#    open('clean_data.txt', 'w').write(to_string_samples(new_samples))
  return new_samples


def _estimator(fdist, bins):
  return SimpleGoodTuringProbDist(fdist)

      
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
  words = map(lambda statement: map(lambda (w,t):w, statement), comment)
  words = sum(words, [])
#  tags = sum(tags, [])
  size_= sum([len(word) for word in words])
  features['stmt_len'] = len(words)/float(len(comment))
  features['word_len'] = size_/float(len(words))
  features['size'] = size_
#  tags_dist = FreqDist(sum(tags, []))
#  for tag in TAGS:
#    features[tag] = tags_dist.get(tag, 0)
  dist = FreqDist([word.lower() for word in words])
#  num_stop_words = float(sum([dist.get(word, 0) for word in EN_STOPWORDS]))
#  features['prob_stop_words'] = num_stop_words/len(words)
  for word in EN_STOPWORDS:
    features[word] = dist.get(word, 0)/float(len(words))
  features['alwayson'] = 1.0
  for language in LANGUAGES:
    for i in range(1,n+1):
      word_sim, tag_sim, char_sim, w_s_sim = comment_similarity(GRAMS[language], comment, i)
      features['w_sim_%d_%s' % (i, language)] = word_sim
      features['t_sim_%d_%s' % (i, language)] = tag_sim
      features['c_sim_%d_%s' % (i, language)] = char_sim
#     features['s_sim_%d_%s' % (i, language)] = w_s_sim
  return (features, label)


@Serialized
def samples_featuresets(samples):
  fs = pool_map(featureset, samples, (cpu_count()/8)+1)
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
    result = pool_map(classifier, LAMBDAS, (cpu_count()/8)+1)
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

  logging.info("Experiment started")
  global TRAIN, DEV, TEST, MAP, GRAMS, LANGUAGES, TAGS, NAME
  global  __train_fs, __dev_fs, __test_fs

  logging.info("Eperiment mode is %s" % NAME)

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
  
 
  random_classifier = Random(TRAIN)
  logging.debug("random_classifier labels are:\n %s", str(random_classifier.labels()))
  logging.info("random_classifier accuracy is: %3.3f", accuracy(random_classifier, TEST))

  most_common = MostCommon(TRAIN)
  logging.debug("MostCommon_classifier labels are:\n %s", str(most_common.labels()))
  logging.info("MostCommon_classifier accuracy is: %3.3f", accuracy(most_common, TEST))

#  perp = Perplexity(TRAIN)
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
  logging.info('Important features\n%s\n', clfs[max_index].show_most_informative_features(dev_fs))

if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-c", "--conf", dest="conf", help="Configuration file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  conf_path = os.path.dirname(os.path.abspath(options.conf))
  sys.path.append(conf_path)
  from settings import *
  util.CACHE_FOLDER = conf_path
  util.CACHE = CACHE

  root = logging.getLogger()
  while root.handlers:
    root.removeHandler(root.handlers[0])
  numeric_level = getattr(logging, options.log.upper(), None)
  file_log = '.'.join([options.filename, NAME, 'log'])
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT, filename=file_log)

  main(options, args)
