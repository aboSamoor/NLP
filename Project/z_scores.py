#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""template.py: Description of what the module does."""

from optparse import OptionParser
import logging
from numpy import *
from cPickle import dump, load
import json
import pdb
import os
import util
from math import log
from util import *

__author__ = "Rami Al-Rfou"
__email__ = "rmyeid@gmail.com"

LOG_FORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"

CACHE = True

def indices(list_, item):
  inds = []
  for i in range(len(list_)):
    if list_[i] == item:
      inds.append(i)
  return inds


def z_score(list_):
  m = float(mean(array(list_)))
  s = float(std(array(list_)))
  if s ==0:
    return [0 for x in list_]
  return [abs((x-m)/s) for x in list_]


def anomalies(list_, threshold):
  z_scores = z_score(list_)
  anomaly = []
  for x in z_scores:
    if x>threshold:
      anomaly.append(1)
    else:
      anomaly.append(0)
  return zip(list_, anomaly)


def featureset(distributions, elements, size):
  featureset = set()
  for lang in distributions.keys():
    featureset.update(distributions[lang][elements][size].keys())
  return featureset


def freqdist_prob(freqdist, item):
  count = freqdist.get(item, 0.0)
  return count/float(freqdist.N())

    
def valid_feature(distributions, elements, size, item):
  empty = 0
  languages = sorted(distributions.keys())
  for lang in languages:
    count = distributions[lang][elements][size].get(item, 0.0)
    if count < 2:
      empty += 1
  if empty > len(languages)/2:
    return False
  return True


def features_probs(distributions, elements, size):
  features = featureset(distributions, elements, size)
  languages = sorted(distributions.keys())
  features_distribution = {}
  for feature in features:
    if valid_feature(distributions, elements, size, feature):
      feature_probs = []
      for lang in languages:
        prob = freqdist_prob(distributions[lang][elements][size], feature)
        feature_probs.append(prob)
      features_distribution[feature] = feature_probs
  return features_distribution


def freqdist_logcount(freqdist, item):
  return log(freqdist.get(item, 1.0), 2.0)


def features_log_count(distributions, elements, size):
  features = featureset(distributions, elements, size)
  languages = sorted(distributions.keys())
  features_distribution = {}
  for feature in features:
    feature_probs = []
    for lang in languages:
      prob = freqdist_logcount(distributions[lang][elements][size], feature)
      feature_probs.append(prob)
    features_distribution[feature] = feature_probs
  return features_distribution

@Serialized
def features_languages(distributions):
  languages = sorted(distributions.keys())
  elements = distributions[languages[0]].keys()
  sizes = distributions[languages[0]][elements[0]].keys()
  probabilities = {}
  for element in elements:
    for size in sizes:
      probabilities[str((element,str(size)))] = features_log_count(distributions, element, size)
  return probabilities


def interesting(probabilities, threshold):
  results = []
  for element_size in probabilities:
    features = probabilities[element_size]
    i = 0
    j = 0
    for feature in features:
      abnormal = anomalies(features[feature], threshold)
      probs, flags = zip(*abnormal)
      flags_indices = indices(flags, 1)
      if len(flags_indices) > 0:
        values = [x for (x,y) in abnormal if y==1]
        result = zip(flags_indices, values)
        i += 1
        results.append((feature,result))
      else:
        j += 1
    logging.info("%d features found in %s: %d interesting", i+j, str(element_size), i)
  return results


def important_features(probabilities):
  results = dict(zip(probabilities.keys(), [[] for i in probabilities]))
  for element_size in probabilities:
    features = probabilities[element_size]
    for feature in features:
      prob_list = features[feature]
      num_zeros = len(filter(lambda x: x<=1, prob_list))
      if num_zeros <= len(prob_list)/2:
        zscores = z_score(prob_list) 
        max_zscore = max(zscores)
        index_max_zscore = zscores.index(max_zscore)
        results[element_size].append((max_zscore,index_max_zscore, feature, prob_list))
  important_results = dict(zip(probabilities.keys(), [[] for i in probabilities]))
  for element_size in probabilities:
    results[element_size].sort()
    important_results[element_size] = results[element_size][-10:]
  return important_results


def main(options, args):
  util.CACHE_FOLDER = os.path.dirname(os.path.abspath(options.filename))
  distributions = load(open(options.filename, 'r'))
  logging.info("File has been read")
  languages = sorted(distributions.keys())
  map_ = dict(zip(range(len(languages)), languages))
  probabilities = features_languages('features_probs', distributions)
  logging.info("Probabilities are caclulated for each feature")
#  pdb.set_trace()
  facts = important_features(probabilities)
  logging.info("facts are caclulated for each feature")
  renamed_facts = dict(zip([k for k in facts], [[] for k in facts]))
  for element_size in facts:
    features = facts[element_size]
    for feature in features:
      max_zscore,index_max_zscore, feature_name, prob_list = feature
      renamed = {}
      for i in range(len(prob_list)):
        renamed[map_[i]] = prob_list[i]
      featured_language = map_[index_max_zscore]
      renamed_facts[element_size].append((feature_name, featured_language, renamed))
  json.dump(renamed_facts, open(options.filename+'.facts', 'w'))
  logging.info("facts are dumped")


if __name__ == "__main__":
  parser = OptionParser()
  parser.add_option("-f", "--file", dest="filename", help="Input file")
  parser.add_option("-l", "--log", dest="log", help="log verbosity level",
                    default="INFO")
  (options, args) = parser.parse_args()

  numeric_level = getattr(logging, options.log.upper(), None)
  logging.basicConfig(level=numeric_level, format=LOG_FORMAT)
  main(options, args)

